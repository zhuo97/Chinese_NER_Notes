import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertCRFSoftLexicon(nn.Module):
    def __init__(self, bert_config, num_labels, word_enhance_size, max_lexicon_length, word_embedding_weights):
        super(BertCRFSoftLexicon, self).__init__()
        self.word_enhance_size = word_enhance_size
        self.max_lexicon_length = max_lexicon_length
        self.word_embedding_size = word_embedding_weights.shape[-1]
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_weights, False)
        
        self.bert = BertModel.from_pretrained(bert_config)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.classifier = nn.Linear(self.bert.config.hidden_size+self.word_embedding_size*word_enhance_size, 
                                    num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, word_ids, attention_mask, soft_lexicon_ids, soft_lexicon_weights, label_ids=None):
        """
        Args:
            word_ids:
            attention_mask:
            soft_lexicon_ids: shape: [batch_size, max_seq_len, word_enhance_size, max_lexicon_length]
            soft_lexicon_weights: shape: [batch_size, max_seq_len, word_enhance_size, max_lexicon_length]
            label_ids:
        Returns:
        """
        # bert 计算逻辑
        outputs = self.bert(input_ids=word_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # soft lexicon 计算逻辑
        batch_size = word_ids.shape[0]
        # shape: [batch_size, max_seq_len, word_enhance_size*max_lexicon_length]
        softlexicon_ids = soft_lexicon_ids.reshape(batch_size, -1, self.word_enhance_size * self.max_lexicon_length)
        softlexicon_weights = soft_lexicon_weights.reshape(batch_size, -1,
                                                          self.word_enhance_size * self.max_lexicon_length)

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
        softlexicon_embeds = softlexicon_embeds.reshape(batch_size, -1,
                                                        self.word_enhance_size * self.word_embedding_size)

        embeds = torch.cat([sequence_output, softlexicon_embeds], dim=-1)

        logits = self.classifier(embeds)
        if label_ids is not None:
            loss = self.crf(emissions=logits, tags=label_ids, mask=attention_mask)
            outputs = (logits, -1*loss)
        else:
            outputs = (logits, )
        return outputs
