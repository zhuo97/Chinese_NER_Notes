import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def get_pos_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """
    实现 transformer 的绝对位置编码，sinusoidal embeddings
    :param max_seq_len:
    :param embedding_dim:
    :param padding_idx:
    :param rel_pos_init:
        如果是0，那么从 -max_len 到 max_len 的相对位置编码就按 0-2*max_len 来初始化
        如果是1，那么就按 -max_len, max_len 来初始化
    :return:
    """
    rel_pos_num = 2 * max_seq_len + 1 # 相对位置
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(rel_pos_num,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len, max_seq_len+1,
                           dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)

    # shape: [2*max_seq_len+1, embedding_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(rel_pos_num, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(rel_pos_num, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0

    return emb


class FourPosFusionEmbedding(nn.Module):
    """
    FLAT 位置编码，计算得到公式中的 R_ij
    R_ij = RELU(W_r(concat(四种相对距离)))，输出的结果用于 Multi-Attn 的计算中
    """
    def __init__(self, fusion_method, pe, hidden_size, overall_max_seq_len):
        """
        :param fusion_method: 拼接方式
        :param pe_ss: 论文中提出的四种相对距离，s代表start，e代表end
        :param pe_se: 这里的四个输入为 get_pos_embedding 函数的返回结果
        :param pe_es:
        :param pe_ee:
        :param hidden_size:
        """
        super(FourPosFusionEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.pe = pe
        
        self.pos_fusion_forward = nn.Sequential(
            nn.Linear(self.hidden_size*4, self.hidden_size),
            nn.ReLU(inplace=True)
        )
        self.overall_max_seq_len = overall_max_seq_len

    def forward(self, pos_s, pos_e):
        """
        :param pos_s: shape: [batch_size, batch_max_seq_len]，start 向量
                      论文中叫 head 向量
        :param pos_e: shape: [batch_size, batch_max_seq_len]，end 向量
                      论文中叫 end 向量
        :return: shape: [batch_size, batch_max_seq_len, batch_max_seq_len, hidden_size]
        """
        batch = pos_s.size(0)
        batch_max_seq_len = pos_s.size(1)

        # shape: [batch_size, batch_max_seq_len, batch_max_seq_len]
        # 这里 +overall_max_seq_len 是因为 get_pos_embedding 的 rel_pos_init 是为 0，
        # 相对位置编码是按照 0-2*overall_max_seq_len 来初始化的
        # 第i行表示第i个token的start与其他token的start的距离
        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.overall_max_seq_len
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.overall_max_seq_len
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.overall_max_seq_len
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.overall_max_seq_len

        # shape: [batch_size, batch_max_seq_len, batch_max_seq_len, 1]
        pe_ss = pos_ss.view(size=[batch, batch_max_seq_len, batch_max_seq_len, -1])
        pe_se = pos_se.view(size=[batch, batch_max_seq_len, batch_max_seq_len, -1])
        pe_es = pos_es.view(size=[batch, batch_max_seq_len, batch_max_seq_len, -1])
        pe_ee = pos_ee.view(size=[batch, batch_max_seq_len, batch_max_seq_len, -1])
        
        pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
        # shape: [batch_size*batch_max_seq_len*batch_max_seq_len, 4]
        pe_4 = pe_4.view(size=[-1, 4])
        pe_unique, inverse_indices = torch.unique(pe_4, sorted=True, return_inverse=True, dim=0)
        pos_unique_embedding = self.pe(pe_unique)
        pos_unique_embedding = pos_unique_embedding.view([pos_unique_embedding.size(0), -1])
        pos_unique_embedding_after_fusion = self.pos_fusion_forward(pos_unique_embedding)
        rel_pos_embedding = pos_unique_embedding_after_fusion[inverse_indices]
        # shape: [batch_size, batch_max_seq_len, batch_max_seq_len, hidden_size]
        rel_pos_embedding = rel_pos_embedding.view(size=[batch, batch_max_seq_len, batch_max_seq_len, -1])
        
        return rel_pos_embedding







