import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF


class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, word_ids, attention_mask, label_ids=None):
        outputs = self.bert(input_ids=word_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if label_ids is not None:
            loss = self.crf(emissions=logits, tags=label_ids, mask=attention_mask)
            outputs = (logits, -1*loss)
        else:
            outputs = (logits, )
        return outputs
