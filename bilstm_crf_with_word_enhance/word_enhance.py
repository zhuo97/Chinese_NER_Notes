import jieba
import json
import importlib
import numpy as np
from itertools import chain
from collections import defaultdict
from copy import deepcopy
from utils import load_pickle, save_pickle, normalize

Soft2Idx = {
    "<None>": 0, # used for padding, cls, sep and None softword
    "S": 1,
    "M": 2,
    "B": 3,
    "E": 4,
}
MaxWordLen = 5
MaxLexiconLen = 5 # only keep topn words for softlexicon


# 先分词，然后根据分词结果对每个字标注B/E/M/S
# B/E/M/S分别对应一个向量，使用时加上/拼接已有的 token embedding
# 缺点：只引入词边界信息，对词本身语义表达有限；分词误差
def build_softword(sentence, verbose=False):
    """
    :param sentence: raw sentence
    :param verbose:
    :return: same length as sentence, using word cut result from jieba
    """
    jieba.initialize()
    softword_index = []
    sentence = sentence.replace(" ", "")
    words = jieba.cut(sentence)
    for word in words:
        length = len(word)
        if length == 1:
            softword_index.extend("S")
        elif length == 2:
            softword_index.extend(["B", "E"])
        else:
            softword_index.extend(["B"] + (length-2) * ["M"] + ["E"])

    assert len(softword_index) == len(sentence), "softword len={} != sentence len={}".format(len(softword_index), len(sentence))

    if verbose:
        print(sentence)
        print("".join(softword_index))
    softword_index = [Soft2Idx[i] for i in softword_index]

    return softword_index


# 遍历词表，考虑每个字能匹配上的所有词汇信息，统计其是否为B/E/M/S
# 每个字输出一个 multi-hot 向量
# 遍历的时候需要一个词汇窗口，这里设置为MaxWordLen，只统计小于等于MaxWordLen的词汇
def build_ex_softword(sentence, vocab, verbose=False):
    sentence = sentence.replace(" ", "")
    ex_softword_index = [set() for _ in range(len(sentence))]

    for i in range(len(sentence)):
        for j in range(i, min(i+MaxWordLen, len(sentence))):
            word = sentence[i: j+1]
            if word in vocab.word2idx:
                if j-i == 0:
                    ex_softword_index[i].add("S")
                elif j-i == 1:
                    ex_softword_index[i].add("B")
                    ex_softword_index[j].add("E")
                else:
                    ex_softword_index[i].add("B")
                    ex_softword_index[j].add("E")
                    for k in range(i+1, j):
                        ex_softword_index[k].add("M")
    if verbose:
        print(sentence)
        print(ex_softword_index)

    # multi-hot encoding of B/M/E/S/None/set
    multi_hot_index = []
    default = [0, 0, 0, 0, 1]
    for index in ex_softword_index:
        if len(index) == 0:
            multi_hot_index.append(default)
        else:
            tmp = [0, 0, 0, 0, 0]
            for i in index:
                tmp[Soft2Idx[i]] = 1
            multi_hot_index.append(tmp)

    return multi_hot_index


class SoftLexiconVocab:
    def __init__(self, model_dir, data_dir):
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
        # self._build_word_freq(data_dir)
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
    '''
    # 这里只针对 msra 的训练数据
    def _build_word_freq(self, data_dir):
        weight_path = data_dir / "lexicon_weights.pkl"
        if weight_path.exists():
            print("Load soft lexicon weight from {}".format(weight_path))
            self.word_freq = load_pickle(weight_path)
        else:
            print("Create soft lexicon weight from msra train data")
            file_path = data_dir / "sentences.txt"
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    text = "".join(line.split(" "))

                    soft_lexicon = self.build_soft_lexicon(text)
                    for item in soft_lexicon:
                        for val in chain(*item.values()):
                            self.word_freq[val] += 1
            save_pickle(self.word_freq, weight_path)
    ''' 
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

