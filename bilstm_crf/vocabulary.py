from collections import Counter
from utils import save_pickle
from utils import load_pickle

class Vocabulary():
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
        # list，里面元素为 tuple (word, word_count)
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
        save_pickle(data=mappings, file_path=file_path)

    def load_from_file(self, file_path):
        mappings = load_pickle(input_file=file_path)
        self.word2idx = mappings["word2idx"]
        self.idx2word = mappings["idx2word"]

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
