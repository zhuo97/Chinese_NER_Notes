import importlib
import numpy as np
from collections import defaultdict
from copy import deepcopy
from utils import normalize

Soft2Idx = {
    "<None>": 0, # used for padding, cls, sep and None softword
    "S": 1,
    "M": 2,
    "B": 3,
    "E": 4,
}
MaxWordLen = 5
MaxLexiconLen = 5 # only keep topn words for softlexicon


class SoftLexiconVocab:
    def __init__(self, model_dir):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.none_token = "<None>"
        self.special_token_num = 3
        self.word2idx = {}
        self.idx2word = None
        self.word_freq = defaultdict(int)
        self.soft_lexicon_embeddings = None
        self.soft_lexicon_embeddings_size = None
        self._reset()

        self.model = getattr(importlib.import_module(model_dir), 'model')
        self._build_vocab()
        self._build_reverse_vocab()
        # 其实就是 word embeddings
        self._build_embeddings()
        self._build_word_freq()

    def _reset(self):
        """
        初始化词典
        :return:
        """
        ctrl_symbols = [self.pad_token, self.unk_token, self.none_token]
        for idx, symbol in enumerate(ctrl_symbols):
            self.word2idx[symbol] = idx

        self.word_freq.update({
            self.word2idx[self.pad_token]: 0,
            self.word2idx[self.unk_token]: 1,
            self.word2idx[self.none_token]: 1
        })

    def _build_vocab(self):
        start_idx = len(self.word2idx)
        for i, word in enumerate(self.model.key_to_index):
            self.word2idx[word] = start_idx + i

    def _build_reverse_vocab(self):
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def _build_embeddings(self):
        self.soft_lexicon_embeddings \
            = np.vstack((np.random.normal(0, 1, size=(self.special_token_num, self.model.vector_size)),
                         np.asarray(self.model.vectors)
                         )).astype(np.float32)
        self.soft_lexicon_embeddings = np.apply_along_axis(normalize, 1, self.soft_lexicon_embeddings)
        self.soft_lexicon_embeddings_size = self.model.vector_size


    def _build_word_freq(self):
        for word in self.model.key_to_index:
            self.word_freq[self.word2idx[word]] = self.model.get_vecattr(word, "count")

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

    # ex-softword 对于每个字，只统计其分词tag B/E/M/S，没有用到词汇本身信息
    # softlexicon 则是存储了每个分词tag B/E/M/S 所对应的词，既使用词边界信息和使用了词汇信息
    def build_soft_lexicon(self, sentence, verbose=False):
        sentence = sentence.replace(" ", "")
        default = {"B": set(), "M": set(), "E": set(), "S": set()}
        soft_lexicon = [deepcopy(default) for _ in range(len(sentence))]

        for i in range(len(sentence)):
            for j in range(i, min(i + MaxWordLen, len(sentence))):
                word = sentence[i: j + 1]
                if word in self.word2idx:
                    if j - i == 0:
                        soft_lexicon[i]["S"].add(word)
                    elif j - i == 1:
                        soft_lexicon[i]["B"].add(word)
                        soft_lexicon[j]["E"].add(word)
                    else:
                        soft_lexicon[i]["B"].add(word)
                        soft_lexicon[j]["E"].add(word)
                        for k in range(i + 1, j):
                            soft_lexicon[k]["M"].add(word)
            for tag, words in soft_lexicon[i].items():
                if not words:
                    soft_lexicon[i][tag].add(self.none_token)

        if verbose:
            print(sentence)
            print(soft_lexicon)

        for lexicon in soft_lexicon:
            for tag, words in lexicon.items():
                lexicon[tag] = [self.to_index(word) for word in words]

        return soft_lexicon

    def postproc_soft_lexicon(self, soft_lexicon):
        """
        1. 将单个字符的 B/E/M/S 字典拼成一个 4*MaxLexiconLen 长的 list，具体来说：
        如果 B/E/M/S 对应的词汇个数少于 MaxLexiconLen，则进行 padding
        否则按照词汇频率降序进行 trunc
        2. 对单个字符的 B/E/M/S 的对应的词的词频求和，得到每个字符对应的词集的总词频；
        归一后得到单个字符的 B/E/M/S 词向量的加权系数
        3. 将每个字符的计算结果合并成 seq list，权重也对应进行合并
        :param soft_lexicon: [{"B": set(), "M": set(), "E": set(), "S": set()},...,...]
                             len(soft_lexicon) 为 seq_len
        :param word_freq:
        :return: seq_ids: list[list[]], 内层list长度为 4*MaxLexicon，外层list长度为 seq_len
                 seq_weights: 类似 seq_ids
        """

        def helper(word_ids):
            num_words = len(word_ids)
            if num_words <= MaxLexiconLen:
                word_ids = word_ids + [self.to_index(self.pad_token)] * (MaxLexiconLen - num_words)
                weights = [self.word_freq.get(id_, 1) for id_ in word_ids]
                return word_ids, weights
            else:
                tmp = sorted([(id_, self.word_freq.get(id_, 1)) for id_ in word_ids],
                             key=lambda x: x[1], reverse=True)
                tmp = tmp[:MaxWordLen]
                tmp = list(zip(*tmp))
                return tmp[0], tmp[1]

        seq_ids, seq_weights = [], []

        for lexicon in soft_lexicon:
            lexicon_ids, lexicon_weights = [], []
            total_weights = 0
            for tag, word_ids in lexicon.items():
                word_ids, word_weights = helper(word_ids)
                lexicon_ids += word_ids
                lexicon_weights += word_weights
                total_weights += sum(word_weights)
            lexicon_weights = [i / total_weights for i in lexicon_weights]
            seq_ids.append(lexicon_ids)
            seq_weights.append(lexicon_weights)

        return seq_ids, seq_weights
