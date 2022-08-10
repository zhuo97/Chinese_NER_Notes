import torch
import torch.nn as nn

"""
构建一个通用的 CRF 层，同时这个 CRF 层支持 batch 输入
主要参考代码：https://github.com/CLUEbenchmark/CLUENER2020/blob/master/bilstm_crf_pytorch/crf.py
"""


class CRF(nn.Module):
    def __init__(self, tagset_size, tag_dict, device, is_bert):
        """
        :param tagset_size: 标签数量
        :param tag_dict: 标签对应的编码
        :param device: 是否使用 gpu
        :param is_bert: 是否以 bert 模型作为 base
        """
        super(CRF, self).__init__()

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        if is_bert:
            self.START_TAG = "[CLS]"
            self.STOP_TAG = "[SEP]"
        self.tag_dict = tag_dict
        self.tagset_size = tagset_size
        self.device = device

        """
        初始化转移矩阵，transitions[i][j] 表示从状态 j 转移到状态 i 的概率
        转移矩阵的第i行表示其他状态转移到状态i的概率；因为其他状态不可能转移到<START>，所以<START>所在行被赋值为-10000
        转移矩阵的第j列表示状态j转移到其他状态的概率；因为<STOP>不可能转移到其他状态，所以<STOP>所在列被赋值为-10000
        """
        self.transitions = torch.randn(self.tagset_size, self.tagset_size)
        self.transitions.detach()[self.tag_dict[self.START_TAG], :] = -10000
        self.transitions.detach()[:, self.tag_dict[self.STOP_TAG]] = -10000
        self.transitions = self.transitions.to(device)
        self.transitions = nn.Parameter(self.transitions)

    def _score_sentence(self, feats, tags, lengths):
        """
        :param feats: (batch_size, padded_seq_len, tagset_size) batch内每一行表示第i个时刻各个状态的发射分数
        :param tags: (batch_size, seq_len)  batch内每一行表示序列的真实标签
        :param lengths: (batch_size, ) batch内每一行表示序列的真实长度
        :return: 返回真实路径的分数
        """

        # 拼接上<START>和<STOP>标签编码
        batch_size = tags.shape[0]
        start = torch.LongTensor([self.tag_dict[self.START_TAG]]).to(self.device)
        start = start.unsqueeze(0).repeat(batch_size, 1)
        stop = torch.LongTensor([self.tag_dict[self.STOP_TAG]]).to(self.device)
        stop = stop.unsqueeze(0).repeat(batch_size, 1)
        pad_start_tag = torch.cat([start, tags], dim=1)
        pad_stop_tag = torch.cat([tags, stop], dim=1)

        # 对pad的部分都转为<STOP>标签编码
        for i in range(batch_size):
            pad_stop_tag[i, lengths[i]:] = self.tag_dict[self.STOP_TAG]

        score = torch.FloatTensor(batch_size).to(self.device)
        # 这里的 i 表示第 i 个样本
        for i in range(batch_size):
            rows = torch.LongTensor(range(lengths[i])).to(self.device)
            # 得到第i个样本的所有真实转移分数总和
            # 举例，tensor[0, [0, 0], [0, 1]] 取 batch_id=0 的 (0, 0)、(0, 1) 位置的两个元素
            # tensor[[1, 2], [3, 4]] 同理，取 (1, 3)、(2, 4) 位置的两个元素
            # 这里即表示 transition[位置0, 位置1, ..., <STOP>], [<START>, 位置0, 位置1, ...]] 得到 (位置0, <START>), (位置1, 位置0)...
            transition_score = torch.sum(
                self.transitions[
                    pad_stop_tag[i, :lengths[i]+1], pad_start_tag[i, :lengths[i]+1]
                ])
            # 得到第i个样本的所有真实发射分数总和
            # feats[i, rows, tags[i, :lengths[i]] 类似取第i个样本的每个时刻的真实标签
            emission_score = torch.sum(feats[i, rows, tags[i, :lengths[i]]])
            # 得到第i个样本的真实路径分数
            score[i] = transition_score + emission_score

        return score

    def _forward_alg(self, feats, lengths):
        """
        :param feats: (batch_size, padded_seq_len, tagset_size) batch内每一行表示第i个时刻各个状态的发射分数
        :param
        : (batch_size, ) batch内每一行表示序列的真实长度
        :return: 返回所有路径的总分
        """
        # 假设batch_size=1且均为列向量，计算方式为：alpha = forward_var.T + transition_matrix + emission_var
        # forward_var.T 为 (tagset_size, 1) 经过 broadcast 为 (tagset_size, tagset_size)
        # emission_var 为 (1, tagset_size) 经过 broadcast 为 (tagset_size, tagset_size)
        # logsumexp(alpha, dim=1) 为 [tagset_size, 1]

        # 初始化变量，<START> 标签
        batch_size, padded_seq_len, tagset_size = feats.shape
        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dict[self.START_TAG]] = 0.0

        # [batch_size, padded_seq_len+1, tagset_size]
        # 忽略 batch 这个维度，每一行表示的是每个时刻，元素(i, j)表示的是在 i 时刻到达状态 j 的所有路径总和
        forward_var = torch.zeros(
            batch_size,
            padded_seq_len + 1,
            tagset_size,
            dtype=torch.float,
            device=self.device
        )

        # 为所有样本初始化初始状态
        forward_var[:, 0, :] = init_alphas.unsqueeze(0).repeat(batch_size, 1)
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(batch_size, 1, 1)

        # 这里的 i 表示第 i 个时刻
        for i in range(padded_seq_len):
            # [batch_size, tagset_size]
            emit_score = feats[:, i, :]
            # [batch_size, tagset_size, tagset_size]
            emit_score = emit_score.unsqueeze(2).repeat(1, 1, tagset_size)
            current_tag_var = forward_var[:, i, :].clone()
            # [batch_size, tagset_size, tagset_size]，这里对于每个样本要进行转置
            current_tag_var = current_tag_var.unsqueeze(2).repeat(1, 1, tagset_size).transpose(2, 1)
            next_tag_var = current_tag_var + transitions + emit_score
            # torch.logsumexp(next_tag_var, dim=2): [batch_size, tagset_size]
            forward_var[:, i+1, :] = torch.logsumexp(next_tag_var, dim=2)
        # [batch_size, tagset_size]，每个样本只取真实长度对应的那一行
        forward_var = forward_var[range(batch_size), lengths, :]
        terminal_var = forward_var + self.transitions[self.tag_dict[self.STOP_TAG]].unsqueeze(0).repeat(batch_size, 1)
        alpha = torch.logsumexp(terminal_var, dim=1)

        return alpha

    def calculate_loss(self, feats, tags, lengths):
        forward_score = self._forward_alg(feats, lengths)
        gold_score = self._score_sentence(feats, tags, lengths)
        score = forward_score - gold_score
        return score.mean()

    def obtain_labels(self, feats, id2tag, lengths):
        batch_best_path = []
        batch_best_score = []
        for feat, length in zip(feats, lengths):
            best_path, best_score = self._viterbi_decode(feat[:length])
            batch_best_path.append([id2tag[tag_id] for tag_id in best_path])
            batch_best_score.append(best_score)
        return batch_best_path, batch_best_score

    def _viterbi_decode(self, feat):
        """
        :param feat: [seq_len, tagset_size] 这里传入的是单个样本，以及对应的真实路径长度
        :return:
        """
        back_pointers = []

        # 存储到当前时刻的所有状态所对应的最大路径分数
        init_var = torch.FloatTensor(1, self.tagset_size).to(self.device).fill_(-10000.0)
        init_var[0][self.tag_dict[self.START_TAG]] = 0
        forward_var = init_var

        # 遍历每一个时刻
        for feat_t in feat:
            # feat_t: [1, tagset_size]
            # 第 i 行表示上一个时刻的所有状态转移到当前第 i 个状态的所有得分
            next_var_t = \
                forward_var.expand(self.tagset_size, self.tagset_size) + self.transitions
            # 这里 bptrs_t 的 shape 为 [batch_size]，设第 i 个的值为 k，则表示到当前时刻状态 i 的最大路径需要从上一时刻的状态 k 发射
            _, bptrs_t = torch.max(next_var_t, dim=1)
            # next_var_t[range(len(bptrs_t)), bptrs_t]: [1, tagset_size]
            forward_var = next_var_t[range(len(bptrs_t)), bptrs_t] + feat_t
            back_pointers.append(bptrs_t)

        # [tagset_size]
        terminal_var = forward_var + self.transitions[self.tag_dict[self.STOP_TAG]]
        _, best_tag_id = torch.max(terminal_var, dim=0)
        best_tag_id = best_tag_id.item()
        best_path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(back_pointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())

        start = best_path.pop()
        assert start == self.tag_dict[self.START_TAG]
        best_path.reverse()
        return best_path, best_path_score
