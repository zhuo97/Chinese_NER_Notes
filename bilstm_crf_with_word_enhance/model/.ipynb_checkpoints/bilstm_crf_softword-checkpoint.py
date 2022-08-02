import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTMSoftWord(nn.Module):
    def __init__(self,
                 hidden_size,
                 out_size,
                 word_enhance_size,
                 pretrain_embedding_weight):
        super(BiLSTMSoftWord, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrain_embedding_weight, False)
        self.softword_embedding = nn.Embedding(word_enhance_size, pretrain_embedding_weight.shape[-1])

        self.bilstm = nn.LSTM(input_size=pretrain_embedding_weight.shape[-1]*2,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, out_size)

    def forward(self, input_ids, lengths, softword_ids):
        # [batch_size, seq_len, embedding_size]
        embeds = self.embedding(input_ids)
        softword_embeds = self.softword_embedding(softword_ids)
        embeds = torch.cat([embeds, softword_embeds], axis=-1)

        # 输入序列不进行排序
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        scores = self.linear(lstm_out)
        return scores


class BiLSTM_CRF_SoftWord(nn.Module):
    def __init__(self,
                 hidden_size,
                 label2id,
                 device,
                 word_enhance_size,
                 pretrain_embedding_weight):
        super(BiLSTM_CRF_SoftWord, self).__init__()
        self.hidden_size = hidden_size
        self.label2id = label2id
        self.id2label = {id_: label for label, id_ in label2id.items()}
        self.num_labels = len(label2id)
        self.device = device

        self.bilstm = BiLSTMSoftWord(hidden_size=hidden_size,
                                     out_size=self.num_labels,
                                     word_enhance_size=word_enhance_size,
                                     pretrain_embedding_weight=pretrain_embedding_weight)

        self.crf = CRF(self.num_labels, batch_first=True)

    def get_masks(self, input_, lengths):
        # input_: [batch_size, max_seq_len]
        batch_size, max_seq_len = input_.size(0), input_.size(1)
        masks = torch.ByteTensor(batch_size, max_seq_len).fill_(0)
        for i, seq_len in enumerate(lengths):
            masks[i, :seq_len] = 1
        return masks.to(self.device)

    def forward(self, input_ids, lengths, softword_ids, label_ids):
        # input_ids, label_ids: tensors, (batch_size, max_seq_len)
        # lengths: list, 元素是每个输入文本的实际长度
        # [batch_size, max_seq_len, num_labels]
        feats = self.bilstm(input_ids, lengths, softword_ids)
        label_masks = self.get_masks(input_ids, lengths)
        loss = self.crf(emissions=feats, tags=label_ids, mask=label_masks)
        return -1*loss

    def predict(self, input_ids, lengths, softword_ids):
        feats = self.bilstm(input_ids, lengths, softword_ids)
        label_masks = self.get_masks(input_ids, lengths)
        predicted_label_ids = self.crf.decode(feats, label_masks)
        predicted_labels = [[self.id2label[id_] for id_ in label_ids]
                            for label_ids in predicted_label_ids]
        return predicted_labels



