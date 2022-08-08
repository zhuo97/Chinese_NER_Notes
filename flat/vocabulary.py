import numpy as np
import torch
import importlib
import collections
from utils import normalize


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self, w):
        '''
        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([sentence[i:j+1], i, j])

        return result


class Vocabulary:
    def __init__(self,
                 model_dir,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_tokem = sep_token
        self.mask_token = mask_token
        self.special_token_num = 5
        self.word2idx = {}
        self.idx2word = None
        self.word_list = []
        self.trie = Trie()
        self.pretrain_embeddings = None
        self.pretrain_embeddings_size = None

        self._reset()
        self.model = getattr(importlib.import_module(model_dir), 'model')
        self._build_vocab()
        self._build_reverse_vocab()
        self._build_embeddings()
        self._build_trie()

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
        self.pretrain_embeddings \
            = np.vstack((np.random.normal(0, 1, size=(self.special_token_num, self.model.vector_size)),
                         np.asarray(self.model.vectors)
                        )).astype(np.float32)
        self.pretrain_embeddings = np.apply_along_axis(normalize, 1, self.pretrain_embeddings)
        self.pretrain_embeddings = torch.tensor(self.pretrain_embeddings, dtype=torch.float32)
        self.pretrain_embeddings_size = self.model.vector_size

    def _build_trie(self):
        for i, word in enumerate(self.model.key_to_index):
            if len(word) > 1:
                self.trie.insert(word)

    def __len__(self):
        return len(self.idx2word)


if __name__ == "__main__":
    vocab = Vocabulary("pretrain_word_emb.ctb50")
    print("vocab size: ", len(vocab))
    print([word for word, id_ in vocab.word2idx.items() if len(word) != 1])

    sentence = "第一章提到过中文NER的普遍使用字符粒度的输入，从而避免分词错误/分词粒度和NER粒度不一致限制模型表现的天花板，以及词输入OOV的问题。"
    result = vocab.trie.get_lexicon(sentence)
    print(result)