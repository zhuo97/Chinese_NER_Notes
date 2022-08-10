import torch.nn as nn
from crf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_size):
        """
        :param vocab_size:
        :param embedding_dim:
        :param hidden_dim:
        :param out_size:
        :param device:
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, out_size)

    def forward(self, input_ids, lengths):
        # [batch_size, max_len, embedding_dim]
        embeds = self.embedding(input_ids)
        # 输入时可以不进行排序
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        scores = self.linear(lstm_out)
        return scores


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_dict, embedding_dim, hidden_dim, device, is_bert):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_dict = tag_dict
        self.id2tag = dict((id_, tag) for tag, id_ in self.tag_dict.items())
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.tagset_size = len(tag_dict)

        self.bilstm = BiLSTM(vocab_size, embedding_dim, hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, tag_dict, device, is_bert)

    def forward(self, input_ids, lengths):
        feats = self.bilstm(input_ids, lengths)
        batch_best_path, batch_best_score = self.crf.obtain_labels(feats, self.id2tag, lengths)
        return batch_best_path, batch_best_score

    def get_loss(self, input_ids, lengths, tags):
        feats = self.bilstm(input_ids, lengths)
        return self.crf.calculate_loss(feats, tags, lengths)


