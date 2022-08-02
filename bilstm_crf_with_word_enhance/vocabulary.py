import numpy as np
import importlib
from collections import Counter
from utils import load_pickle, save_pickle, normalize


class Vocabulary:
    def __init__(self,
                 model_dir,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 random_init=False,
                 random_init_emb_size=None):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_tokem = sep_token
        self.mask_token = mask_token
        self.special_token_num = 5
        self.word2idx = {}
        self.idx2word = None
        self.pretrain_embeddings = None
        self.pretrain_embeddings_size = None
        self.random_init = random_init
        self.random_init_emb_size = random_init_emb_size

        self._reset()
        self.model = getattr(importlib.import_module(model_dir), 'model')
        self._build_vocab()
        self._build_reverse_vocab()
        self._build_embeddings()

    def _reset(self):
        """
        初始化词典
        :return:
        """
        ctrl_symbols = [self.pad_token, self.unk_token, self.cls_token, self.sep_tokem, self.mask_token]
        for idx, symbol in enumerate(ctrl_symbols):
            self.word2idx[symbol] = idx

    def to_index(self, word):
        """
        将词转为数字
        :param word:
        :return:
        """
        if word in self.word2idx:
            return self.word2idx[word]
        if self.unk_token is not None:
            return self.word2idx[self.unk_token]
        else:
            return ValueError("word {} not in vocabulary".format(word))

    def to_word(self, idx):
        """
        将 id 转为对应的词
        :param idx:
        :return:
        """
        return self.idx2word[idx]

    def _build_vocab(self):
        start_idx = len(self.word2idx)
        for i, word in enumerate(self.model.key_to_index):
            self.word2idx[word] = start_idx + i

    def _build_reverse_vocab(self):
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def _build_embeddings(self):
        if self.random_init and self.random_init_emb_size is not None:
            self.pretrain_embeddings \
                = np.vstack((np.random.normal(0, 1,
                             size=(self.special_token_num+len(self.word2idx), self.random_init_emb_size))
                             )).astype(np.float32)
            self.pretrain_embeddings_size = self.random_init_emb_size
        else:
            self.pretrain_embeddings \
                = np.vstack((np.random.normal(0, 1, size=(self.special_token_num, self.model.vector_size)),
                             np.asarray(self.model.vectors)
                            )).astype(np.float32)
            self.pretrain_embeddings = np.apply_along_axis(normalize, 1, self.pretrain_embeddings)
            self.pretrain_embeddings_size = self.model.vector_size

    def save(self, file_path):
        mappings = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "pretrain_embeddings": self.pretrain_embeddings
        }
        save_pickle(data=mappings, file_path=file_path)

    def load_from_file(self, file_path):
        mappings = load_pickle(input_file=file_path)
        self.word2idx = mappings["word2idx"]
        self.idx2word = mappings["idx2word"]
        self.pretrain_embeddings = mappings["pretrain_embeddings"]

    def __len__(self):
        return len(self.idx2word)


"""
Vocabulary_bak 适用于从语料中构建词典，这里直接使用外部词向量来构建词典，弃用这个类，留作参考
"""
class Vocabulary_bak:
    def __init__(self,
                 max_size=0,
                 min_freq=0,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]"):
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_tokem = sep_token
        self.mask_token = mask_token
        self.word2idx = {}
        self.idx2word = None
        self.rebuild = True
        self.word_counter = Counter()
        self.reset()

    def reset(self):
        """
        初始化词典
        :return:
        """
        ctrl_symbols = [self.pad_token, self.unk_token, self.cls_token, self.sep_tokem, self.mask_token]
        for idx, symbol in enumerate(ctrl_symbols):
            self.word2idx[symbol] = idx

    def update(self, word_list):
        """
        依次增加序列中的词在词典中出现的频率
        :param word_list:
        :return:
        """
        self.word_counter.update(word_list)

    def add(self, word):
        """
        增加一个新词在词典中出现的频率
        :param word:
        :return:
        """
        self.word_counter[word] += 1

    def has_word(self, word):
        """
        检查词是否被记录
        :param word:
        :return:
        """
        return word in self.word2idx

    def to_index(self, word):
        """
        将词转为数字
        :param word:
        :return:
        """
        if word in self.word2idx:
            return self.word2idx[word]
        if self.unk_token is not None:
            return self.word2idx[self.unk_token]
        else:
            return ValueError("word {} not in vocabulary".format(word))

    def to_word(self, idx):
        """
        将 id 转为对应的词
        :param idx:
        :return:
        """
        return self.idx2word[idx]

    def build_vocab(self):
        max_size = min(self.max_size, len(self.word2idx)) if self.max_size else None
        words = self.word_counter.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self.word2idx:
            words = filter(lambda kv: kv[0] not in self.word2idx, words)
        start_idx = len(self.word2idx)
        self.word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False

    def build_reverse_vocab(self):
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def save(self, file_path):
        mappings = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word
        }
        if self.pretrain_embeddings is not None:
            mappings["pretrain_embeddings"] = self.pretrain_embeddings
        save_pickle(data=mappings, file_path=file_path)

    def load_from_file(self, file_path):
        mappings = load_pickle(input_file=file_path)
        self.word2idx = mappings["word2idx"]
        self.idx2word = mappings["idx2word"]
        if "pretrain_embeddings" in mappings.keys():
            self.pretrain_embeddings = mappings["pretrain_embeddings"]

    def clear(self):
        self.word_counter.clear()
        self.word2idx = None
        self.idx2word = None
        self.rebuild = True
        self.reset()

    def unknown_idx(self):
        """
        unk 对应的数字
        :return:
        """
        if self.unk_token is None:
            return None
        return self.word2idx[self.unk_token]

    def padding_idx(self):
        """
        pad 对应的数字
        :return:
        """
        if self.pad_token is None:
            return None
        return self.word2idx[self.pad_token]

    def __len__(self):
        return len(self.idx2word)
