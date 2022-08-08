import copy
import torch
import torch.nn as nn
from torchcrf import CRF
from model.attention import TransformerEncoderLayer
from model.position import FourPosFusionEmbedding, get_pos_embedding


def seq_len_to_mask(seq_len, max_len=None):
    batch_size = seq_len.shape[0]
    max_len = int(max_len) if max_len else seq_len.max().long()
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))

    return mask.bool()


class FLAT(nn.Module):
    def __init__(self, config, vocab):
        super(FLAT, self).__init__()

        self.config = config
        pe = get_pos_embedding(config.overall_max_seq_len,
                               config.flat_model_size,
                               config.rel_pos_init)

        if config.pos_norm:
            pe_sum = pe.sum(dim=-1, keepdim=True)
            with torch.no_grad():
                pe = pe / pe_sum

        self.pe = nn.Embedding(config.overall_max_seq_len*2+1, config.flat_model_size, _weight=pe)
        
        self.pos_layer = FourPosFusionEmbedding(
            fusion_method=config.fusion_metod,
            pe=self.pe,
            hidden_size=config.flat_model_size,
            overall_max_seq_len=config.overall_max_seq_len
        )

        # 包含字向量和词向量
        self.lattice_embedding = nn.Embedding.from_pretrained(vocab.pretrain_embeddings)

        if config.use_bert:
            self.char_proj = nn.Linear(config.char_embedding_size+config.bert_embedding_size,
                                       config.flat_model_size)
        else:
            self.char_proj = nn.Linear(config.char_embedding_size, config.flat_model_size)
        self.word_proj = nn.Linear(config.word_embedding_size, config.flat_model_size)

        self.encoder_layer = TransformerEncoderLayer(hidden_size=config.flat_model_size,
                                                     num_heads=config.num_heads,
                                                     scaled=config.scaled,
                                                     attn_dropout=config.attn_dropout,
                                                     layer_norm_dropout=config.layer_norm_dropout,
                                                     layer_norm_eps=config.layer_norm_eps)

        self.output = nn.Linear(config.flat_model_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self,
                lattice,
                char_seq_len,
                num_words,
                pos_s,
                pos_e,
                label_ids=None,
                bert_embed_without_cls=None):
        """
        :param lattice: [batch_size, batch_max_seq_len]
            每个 seq 分为 char_id + word_id 两部分，
            对于小于 batch_max_seq_len 的 seq，在 word_id 后面进行 pad
        :param char_seq_len: tensor, shape: [batch_size], 记录 char 序列的真实长度
        :param num_words: tensor, shape: [batch_size], 记录 word 序列的真实长度
        :param pos_s:
        :param pos_e:
        :param label_ids:
        :param bert_embed_without_cls: shape: [batch_size, batch_max_char_seq_len, bert_embed_size]
        :return:
        """
        batch_size = lattice.shape[0]
        batch_max_seq_len = lattice.shape[1]
        batch_max_char_seq_len = int(char_seq_len.max())

        # shape: [batch_size, batch_max_seq_len, embedding_size]
        raw_char_word_embed = self.lattice_embedding(lattice)
        raw_char_embed = raw_char_word_embed.clone()
        if self.config.use_bert:
            bert_pad_length = batch_max_seq_len - batch_max_char_seq_len
            bert_embed = torch.cat([bert_embed_without_cls,
                                    torch.zeros(size=[batch_size,
                                                      bert_pad_length,
                                                      self.config.bert_embedding_size],
                                                device=self.config.device,
                                                requires_grad=False)], dim=-2)
            # shape: [batch_size, batch_max_seq_len, embedding_size + bert_embed_size]
            raw_char_embed = torch.cat([raw_char_embed, bert_embed], dim=-1)

        # shape: [batch_size, batch_max_seq_len, flat_model_size]
        char_embed = self.char_proj(raw_char_embed)
        # 对词部分置为0，仅保留 char 部分
        char_mask = seq_len_to_mask(char_seq_len, batch_max_seq_len)
        char_embed = char_embed.masked_fill(~(char_mask.unsqueeze(-1)), 0)

        # shape: [batch_size, batch_max_seq_len, flat_model_size]
        word_embed = self.word_proj(raw_char_word_embed)
        # 对字部分置为0, 仅保留 word 部分
        word_mask = seq_len_to_mask(char_seq_len+num_words) ^ char_mask
        word_embed = word_embed.masked_fill(~(word_mask.unsqueeze(-1)), 0)

        embed = char_embed + word_embed

        pos_embedding = self.pos_layer(pos_s, pos_e)

        char_word_mask = seq_len_to_mask(char_seq_len+num_words)
        encoded = self.encoder_layer(embed, pos_embedding, char_word_mask)

        # 只取 char token, shape: [batch_size, batch_max_char_seq_len, flat_model_size]
        encoded = encoded[:, :batch_max_char_seq_len, :]
        pred = self.output(encoded)

        mask = seq_len_to_mask(char_seq_len)
        if self.training:
            loss = self.crf(emissions=pred, tags=label_ids, mask=mask)
            return -loss, None
        else:
            if label_ids is not None:
                loss = self.crf(emissions=pred, tags=label_ids, mask=mask)
                pred_labels_ids = self.crf.decode(pred, mask)
                return -loss, pred_labels_ids
            else:
                pred_labels_ids = self.crf.decode(pred, mask)
                return None, pred_labels_ids
