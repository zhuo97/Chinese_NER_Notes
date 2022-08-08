import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 scaled,
                 attn_dropout,
                 layer_norm_dropout,
                 layer_norm_eps):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attn_layer = MultiHeadAttention(hidden_size,
                                                        num_heads,
                                                        scaled,
                                                        attn_dropout)
        self.feed_forward_layer = FeedForward(hidden_size,
                                              layer_norm_dropout,
                                              layer_norm_eps)

    def forward(self, inputs, pos_embedding, seq_mask):
        """
        :param inputs: [batch_size, batch_max_seq_len, hidden_size]
                       这里的 hidden_size 是指字向量/词向量的维度
        :param pos_embedding: [batch_size, batch_max_seq_len, batch_max_seq_len, hidden_size]
        :param seq_mask: [batch_size, batch_max_seq_len]
        :return:
        """
        outputs = self.multi_head_attn_layer(inputs, inputs, inputs, pos_embedding, seq_mask)
        outputs = self.feed_forward_layer(outputs, inputs)

        # shape: [batch_size, batch_max_seq_len, hidden_size]
        return outputs


class MultiHeadAttention(nn.Module):
    """
    仿照 transformer-xl 的相对位置编码计算公式进行计算
    把 R_ij 换成 FourPosFusionEmbedding 的计算结果
    """
    def __init__(self, hidden_size, num_heads, scaled=True, attn_dropout=None):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size

        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert self.per_head_size * self.num_heads == self.hidden_size

        # attention 的 q,k,v 变换矩阵
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)

        # 计算 R_ij 的权重
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)

        # 计算 A* 的权重
        self.u = nn.Parameter(
            torch.randn(self.num_heads, self.per_head_size), requires_grad=True
        )
        self.v = nn.Parameter(
            torch.randn(self.num_heads, self.per_head_size), requires_grad=True
        )

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, pos_embedding, seq_mask):
        """
        :param query: [batch_size, batch_max_seq_len, hidden_size]
        :param key:
        :param value:
        :param pos_embedding: [batch_size, batch_max_seq_len, batch_max_seq_len, hidden_size]
        :param seq_mask: [batch_size, batch_max_seq_len]
        :return:
        """
        batch_size = key.size(0)
        batch_max_seq_len = key.size(1)

        # 输入线性变换
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        rel_pos_embedding = self.w_r(pos_embedding)

        query = query.reshape(batch_size, batch_max_seq_len, self.num_heads, self.per_head_size)
        key = key.reshape(batch_size, batch_max_seq_len, self.num_heads, self.per_head_size)
        value = value.reshape(batch_size, batch_max_seq_len, self.num_heads, self.per_head_size)
        rel_pos_embedding = rel_pos_embedding.reshape(
            batch_size, batch_max_seq_len, batch_max_seq_len, self.num_heads, self.per_head_size
        )

        # shape: [batch_size, num_heads, batch_max_seq_len, per_head_size]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # shape: [batch, num_heads, per_head_size, batch_max_seq_len]
        key = key.transpose(-1, -2)

        # shape: [1, num_heads, 1, per_head_size]
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)

        # shape: [batch_size, num_heads, batch_max_seq_len, per_head_size]
        query_and_u_for_c = query + u_for_c

        # shape: [batch_size, num_heads, batch_max_seq_len, batch_max_seq_len]
        A_C = torch.matmul(query_and_u_for_c, key)

        # shape: [batch_size, num_heads, batch_max_seq_len, per_head_size, batch_max_seq_len]
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_for_b = query.view(batch_size, self.num_heads, batch_max_seq_len, 1, self.per_head_size)
        # shape: [batch_size, num_heads, batch_max_seq_len, 1, per_head_size]
        query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        # shape: [batch_size, num_heads, batch_max_seq_len, batch_max_seq_len]
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)

        # shape: [batch_size, num_heads, batch_max_seq_len, batch_max_seq_len]
        attn_score_raw = A_C + B_D

        # 计算 score
        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        # shape: [batch_size, 1, 1, batch_max_seq_len]
        seq_mask = 1 - seq_mask.long().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(seq_mask.bool(), -1e15)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        # shape: [batch_size, num_heads, batch_max_seq_len, batch_max_seq_len]
        attn_score = self.dropout(attn_score)

        # shape: [batch_size, num_heads, batch_max_seq_len, per_head_size]
        value_weight_sum = torch.matmul(attn_score, value)
        # shape: [batch_size, batch_max_seq_len, hidden_size]
        result = value_weight_sum.transpose(1, 2).contiguous().reshape(batch_size, batch_max_seq_len, self.hidden_size)

        return result


class FeedForward(nn.Module):
    def __init__(self, hidden_size, layer_norm_dropout, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(layer_norm_dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states
