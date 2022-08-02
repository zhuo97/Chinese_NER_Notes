import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTMSoftLexicon(nn.Module):
    def __init__(self,
                 hidden_size,
                 out_size,
                 word_enhance_size,
                 max_lexicon_length,
                 char_embedding_weights,
                 word_embedding_weights):
        """
        :param hidden_size:
        :param out_size:
        :param word_enhance_size:
        :param max_lexicon_length:
        :param char_embedding_weights:
        :param word_embedding_weights:
        """
        super(BiLSTMSoftLexicon, self).__init__()

        self.char_embedding = nn.Embedding.from_pretrained(char_embedding_weights, False)
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_weights, False)
        self.bilstm_input_size = \
            char_embedding_weights.shape[-1] + word_embedding_weights.shape[-1]*word_enhance_size
        self.bilstm = nn.LSTM(input_size=self.bilstm_input_size,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, out_size)
        self.word_enhance_size = word_enhance_size
        self.max_lexicon_length = max_lexicon_length
        self.word_embedding_size = word_embedding_weights.shape[-1]

    def forward(self, input_ids, lengths, softlexicon_ids, softlexicon_weights):
        """
        :param input_ids: 输入的字符 id，tensor，[batch_size, max_seq_len]
        :param lengths: 输入字符的真实长度，list，len(lengths) 为 batch_size
        :param softlexicon_ids: softlexicon 编码，tensor
                                shape:[batch_size, max_seq_len, word_enhance_size, max_lexicon_length]
                                对于每个句子的每个字符，需要标注其 B/E/M/S 所对应的词；同时，由于 B/E/M/S 对应的
                                词的数量不同，需要 pad/trunc 到 max_lexicon_length 长度
        :param softlexion_weight: 需要对 B/E/M/S 对应的词向量做加权求和，这里是权重
                                  shape: [batch_size, max_seq_len, word_enhance_size, max_lexicon_length]
        :return:
        """
        char_embeds = self.char_embedding(input_ids)

        batch_size = input_ids.shape[0]
        # shape: [batch_size, max_seq_len, word_enhance_size*max_lexicon_length]
        softlexicon_ids = softlexicon_ids.reshape(batch_size, -1, self.word_enhance_size*self.max_lexicon_length)
        softlexicon_weights = softlexicon_weights.reshape(batch_size, -1, self.word_enhance_size*self.max_lexicon_length)

        # shape: [batch_size, max_seq_len, word_enhance_size*max_lexicon_length, word_embedding_size]
        softlexicon_embeds = self.word_embedding(softlexicon_ids)
        # 对应位置元素相乘
        softlexicon_embeds = torch.mul(softlexicon_embeds, softlexicon_weights.unsqueeze(-1))
        softlexicon_embeds = softlexicon_embeds.reshape(batch_size,
                                                        -1,
                                                        self.word_enhance_size,
                                                        self.max_lexicon_length,
                                                        self.word_embedding_size)
        # shape: [batch_size, max_seq_len, word_enhance_size, word_embedding_size]
        softlexicon_embeds = torch.sum(softlexicon_embeds, dim=3)
        # shape: [batch_size, max_seq_len, word_enhance_size*word_embedding_size]
        softlexicon_embeds = softlexicon_embeds.reshape(batch_size, -1, self.word_enhance_size*self.word_embedding_size)

        embeds = torch.cat([char_embeds, softlexicon_embeds], dim=-1)

        # 输入序列不进行排序
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        scores = self.linear(lstm_out)
        return scores


class BiLSTM_CRF_SoftLexicon(nn.Module):
    def __init__(self,
                 hidden_size,
                 label2id,
                 device,
                 word_enhance_size,
                 max_lexicon_length,
                 char_embedding_weights,
                 word_embedding_weights):
        super(BiLSTM_CRF_SoftLexicon, self).__init__()
        self.hidden_size = hidden_size
        self.label2id = label2id
        self.id2label = {id_: label for label, id_ in label2id.items()}
        self.num_labels = len(label2id)
        self.device = device

        self.bilstm = BiLSTMSoftLexicon(hidden_size=hidden_size,
                                        out_size=self.num_labels,
                                        word_enhance_size=word_enhance_size,
                                        max_lexicon_length=max_lexicon_length,
                                        char_embedding_weights=char_embedding_weights,
                                        word_embedding_weights=word_embedding_weights)

        self.crf = CRF(self.num_labels, batch_first=True)

    def get_masks(self, input_, lengths):
        # input_: [batch_size, max_seq_len]
        batch_size, max_seq_len = input_.size(0), input_.size(1)
        masks = torch.ByteTensor(batch_size, max_seq_len).fill_(0)
        for i, seq_len in enumerate(lengths):
            masks[i, :seq_len] = 1
        return masks.to(self.device)

    def forward(self, input_ids, lengths, softlexicon_ids, softlexicon_weights, label_ids):
        # input_ids, label_ids: tensors, (batch_size, max_seq_len)
        # lengths: list, 元素是每个输入文本的实际长度
        # [batch_size, max_seq_len, num_labels]
        feats = self.bilstm(input_ids, lengths, softlexicon_ids, softlexicon_weights)
        label_masks = self.get_masks(input_ids, lengths)
        loss = self.crf(emissions=feats, tags=label_ids, mask=label_masks)
        return -1*loss

    def predict(self, input_ids, lengths, softlexicon_ids, softlexicon_weights):
        feats = self.bilstm(input_ids, lengths, softlexicon_ids, softlexicon_weights)
        label_masks = self.get_masks(input_ids, lengths)
        predicted_label_ids = self.crf.decode(feats, label_masks)
        predicted_labels = [[self.id2label[id_] for id_ in label_ids]
                            for label_ids in predicted_label_ids]
        return predicted_labels



