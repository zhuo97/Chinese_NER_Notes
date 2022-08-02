import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 out_size,
                 embedding_weights):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_weights, False)

        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, out_size)

    def forward(self, input_ids, lenghs):
        # [batch_size, seq_len, embedding_size]
        embeds = self.embedding(input_ids)
        # 输入序列不进行排序
        packed = pack_padded_sequence(embeds, lenghs, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        scores = self.linear(lstm_out)
        return scores


class BiLSTM_CRF(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 label2id,
                 device,
                 pretrain_embeddings=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.label2id = label2id
        self.id2label = {id_: label for label, id_ in label2id.items()}
        self.num_labels = len(label2id)
        self.device = device

        self.bilstm = BiLSTM(embedding_size, hidden_size, self.num_labels, pretrain_embeddings)
        self.crf = CRF(self.num_labels, batch_first=True)

    def get_masks(self, input_, lengths):
        # input_: [batch_size, max_seq_len]
        batch_size, max_seq_len = input_.size(0), input_.size(1)
        masks = torch.ByteTensor(batch_size, max_seq_len).fill_(0)
        for i, seq_len in enumerate(lengths):
            masks[i, :seq_len] = 1
        return masks.to(self.device)

    def forward(self, input_ids, lengths, label_ids):
        # input_ids, label_ids: tensors, (batch_size, max_seq_len)
        # lengths: list, 元素是每个输入文本的实际长度
        # [batch_size, max_seq_len, num_labels]
        feats = self.bilstm(input_ids, lengths)
        label_masks = self.get_masks(input_ids, lengths)
        loss = self.crf(emissions=feats, tags=label_ids, mask=label_masks)
        return -1*loss

    def predict(self, input_ids, lengths):
        feats = self.bilstm(input_ids, lengths)
        label_masks = self.get_masks(input_ids, lengths)
        predicted_label_ids = self.crf.decode(feats, label_masks)
        predicted_labels = [[self.id2label[id_] for id_ in label_ids]
                            for label_ids in predicted_label_ids]
        return predicted_labels



