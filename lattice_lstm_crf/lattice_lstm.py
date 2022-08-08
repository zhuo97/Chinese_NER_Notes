import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch.nn import init


# 计算词的cell state
class WordLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 输入 x 对应的权重矩阵，一共有三个
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3*hidden_size)
        )
        # hidden state 对应的权重矩阵，一共有三个
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3*hidden_size)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3*hidden_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal(self.weight_ih.data)
        # shape: (hidden_size, hidden_size)
        weight_hh_data = torch.eye(self.hidden_size)
        # shape: (hidden_size, 3*hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        计算词的cell state，例如，词=南京市，x为"南京市"的emb，
        h和c为"南"的hidden state和cell state；"南京市"的cell state会在
        计算"市"的cell state时进行融合
        """
        # input_: (num_words, word_emb_dim)
        # h_0, c_0: (1, hidden_size)
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        # bias_batch: (1, 3*hidden_size)
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
        # wh_b: (1, 3*hidden_size)
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        # wi: (num_words, 3*hidden_size)
        wi = torch.mm(input_, self.weight_ih)
        # f,i,g: (num_words, hidden_size)
        f, i, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)
        # c_l: (num_words, hidden_size)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)

        return c_1

    def __repr__(self):
        s = "{name}({input_size}, {hidden_size})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


# 计算字的hidden state和cell state；
# 对于每个字需要区分是否是结尾字，如果不是结尾字，同LSTM原本计算公式；
# 如果是结尾字，在计算结尾字的cell state时，需要融合对应词汇的cell state
class MultiInputLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3*hidden_size)
        )
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3*hidden_size)
        )
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size)
        )
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3*hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("alpha_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal(self.weight_ih.data)
        init.orthogonal(self.alpha_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        self.alpha_weight_hh.data.set_(alpha_weight_hh_data)

        if self.use_bias:
            init.constant(self.bias.data, val=0)
            init.constant(self.alpha_bias.data, val=0)

    def forward(self, input_, c_input, hx):
        """
        :param input_: (1, input_size)
        :param c_input: list, num_words 个 (1, hidden_size)
        :param hx: 2 个 (1， hidden_size)
        :return:
        """
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        assert batch_size == 1
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(self.input_size, self.weight_ih)
        i, o, g = torch.split(wh_b+wi, split_size_or_sections=self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_num = len(c_input)

        if c_num == 0:
            f = 1 - i
            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            # 竖着拼接，shape: (num_words, hidden_size)
            c_input_var = torch.cat(c_input, dim=0)
            # 根据公式，需要额外增加一个门控用于计算词的加权系数，用以生成新的 cell state
            alpha_bias_batch = self.alpha_bias.unsqueeze(0).expand(batch_size, *self.alpha_bias.size())
            # shape: (1, hidden_size)
            alpha_wi = torch.addmm(alpha_bias_batch, self.input_size, self.alpha_weight_ih)
            # shape: (num_words, hidden_size)
            alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)
            # shape: (num_words, hidden_size)
            alpha = torch.sigmoid(alpha_wi + alpha_wh)
            # 这里是计算了所有词的 input gate，还需要拼接末尾字的 input gate
            # shape: (num_words+1, hidden_size)
            alpha = torch.exp(torch.cat([i, alpha], dim=0))
            # shape: (hidden_size)
            alpha_sum = alpha.sum(dim=0)
            # (num_words+1, hidden_size)
            alpha = torch.div(alpha, alpha_sum)
            # 拼接所有 cell state，shape: (num_words+1, hidden_size)
            merge_i_c = torch.cat([g, c_input_var], dim=0)
            c_1 = merge_i_c * alpha
            # (1, hidden_size)
            c_1 = c_1.sum(dim=0).unsqueeze(0)
            h_1 = o * torch.tanh(c_1)

        return h_1, c_1

    def __repr__(self):
        s = "{name}({input_size}, {hidden_size})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 word_drop,
                 word_alphabet_size,
                 word_emb_dim,
                 pretrain_word_emb=None,
                 left2right=True,
                 fix_word_emb=True,
                 gpu=True,
                 use_bias=True):
        super(LatticeLSTM, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.word_emb = nn.Embedding(word_alphabet_size, word_emb_dim)
        if pretrain_word_emb is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(pretrain_word_emb))
        else:
            self.word_emb.weight.data.copy_(torch.from_numpy(self.random_embedding(word_alphabet_size, word_emb_dim)))
        if fix_word_emb:
            self.word_emb.weight.requires_grad = False
        self.word_dropout = nn.Dropout(word_drop)

        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim, use_bias)
        self.word_rnn = WordLSTMCell(word_emb_dim, hidden_dim, use_bias)
        self.left2right = left2right
        if self.gpu:
            self.rnn = self.rnn.cuda()
            self.word_rnn = self.word_rnn.cuda()
            self.word_emb = self.word_emb.cuda()
            self.word_dropout = self.word_dropout.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input_, skip_input_list, hidden=None):
        """
        :param input_: (1, seq_len, input_size), input_size 即为 embedding dim
        :param skip_input_list: (skip_input, volatile_flag)
               skip_input: [[],...,[], [[25, 13], [2, 3]], ...], 25, 13 为 word id；2，3 为 word 长度
               第一个空 list 用于初始状态，中间也可能存在空 list，表示没有词以该字为起始
               共 seq_len 个；这个集合可以理解为首字词集合
        :param hidden:
        :return:
        """
        volatile_flag = skip_input_list[1]
        skip_input = skip_input_list[0]
        if not self.left2right:
            skip_input = convert_forward_gaz_to_backward(skip_input)
        # input_: (seq_len, 1, input_size)
        input_ = input_.transpose(1, 0)
        seq_len = input_.size(0)
        batch_size = input_.size(1)
        assert batch_size == 1
        hidden_out = [] # 储存 hidden state 的输出
        memory_out = [] # 储存 cell state 的输出
        if hidden:
            (hx, cx) = hidden
        else:
            hx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            cx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            if self.gpu:
                hx = hx.cuda()
                cx = cx.cuda()

        id_list = range(seq_len)
        if not self.left2right:
            id_list = list(reversed(id_list))
        # input_c_list 里面的每个元素为 list，用于存储以该字为结尾的词 cell state；
        # 如果没有词以该字结尾，则列表为空；这个集合可以理解为尾字词集合
        input_c_list = init_list_of_objects(seq_len)
        for t in id_list:
            # input_[t]: (1, hidden_size)
            (hx, cx) = self.rnn(input_[t], input_c_list[t], (hx, cx))
            hidden_out.append(hx)
            memory_out.append(cx)
            # 如果存在以t时刻为首的词，计算词的 cell state
            if skip_input[t]:
                num_words = len(skip_input[t][0])
                word_var = autograd.Variable(torch.LongTensor(skip_input[t][0]), volatile=volatile_flag)
                if self.gpu:
                    word_var = word_var.cuda()
                word_emb = self.word_emb(word_var)
                word_emb = self.word_drop_out(word_emb)
                ct = self.word_rnn(word_emb, (hx, cx))
                assert ct.size(0) == num_words
                # 将计算好的词 cell state 存入尾字词集合
                for idx in range(num_words):
                    length = skip_input[t][1][idx]
                    if self.left2right:
                        input_c_list[t+length-1].append(ct[idx, :].unsqueeze(0))
                    else:
                        input_c_list[t-length-1].append(ct[idx, :].unsqueeze(0))

        if not self.left2right:
            hidden_out = list(reversed(hidden_out))
            memory_out = list(reversed(memory_out))
        output_hidden, output_memory = torch.cat(hidden_out, dim=0), torch.cat(memory_out, dim=0)

        # (1, seq_len, hidden_size)
        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(size):
        list_of_objects.append(list())
    return list_of_objects


# 在双向lstm中，如果是从右到左进行计算，需要把首字词集合变为尾字词集合
def convert_forward_gaz_to_backward(forward_gaz):
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)

    for idx in range(length):
        if forward_gaz[idx]:
            assert len(forward_gaz[idx]) == 2
            num_words = len(forward_gaz[idx][0])
            for idy in range(num_words):
                word_id = forward_gaz[0][idy]
                word_length = forward_gaz[1][idy]
                new_pos = idx + word_length - 1
                if backward_gaz[new_pos]:
                    backward_gaz[new_pos][0].append(word_id)
                    backward_gaz[new_pos][1].append(word_length)
                else:
                    backward_gaz[new_pos] = [[word_id], [word_length]]
    return backward_gaz
















