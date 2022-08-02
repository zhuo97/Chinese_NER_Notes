import torch
from torch.utils.data import Dataset
from word_enhance import build_softword


class NERDataset(Dataset):
    def __init__(self, samples, vocab, label2id, device, max_len=512):
        self.vocab = vocab
        self.label2id = label2id
        self.id2label = {id_: label for label, id_ in label2id.items()}
        self.device = device
        self.data = self.process(samples, max_len)

    def process(self, samples, max_len):
        data = []
        for sample_dict in samples:
            words = sample_dict["words"]
            words = words[: max_len]
            word_ids = [self.vocab.to_index(word) for word in words]

            labels = sample_dict["labels"]
            labels = labels[: max_len]
            label_ids = [self.label2id[label] for label in labels]

            data.append((word_ids, label_ids))

        return data

    def get_tensor(self, input_list, lengths, batch_size, max_seq_len):
        inputs_tensor = \
            torch.LongTensor(batch_size, max_seq_len).fill_(self.vocab.to_index(self.vocab.pad_token))
        for i, seq in enumerate(input_list):
            seq_len = lengths[i]
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        word_ids = self.data[index][0]
        label_ids = self.data[index][1]

        return word_ids, label_ids

    def collate_fn(self, batch):
        word_ids_list = [data[0] for data in batch]
        label_ids_list = [data[1] for data in batch]

        lengths = [len(seq) for seq in word_ids_list]
        max_seq_len = max(lengths)
        batch_size = len(batch)

        word_ids_tensor = self.get_tensor(word_ids_list, lengths, batch_size, max_seq_len)
        label_ids_tensor = self.get_tensor(label_ids_list, lengths, batch_size, max_seq_len)

        return word_ids_tensor, lengths, label_ids_tensor


class NERDatasetSoftWord(Dataset):
    def __init__(self, samples, vocab, label2id, device, max_len=512):
        self.vocab = vocab
        self.label2id = label2id
        self.id2label = {id_: label for label, id_ in label2id.items()}
        self.device = device
        self.data = self.process(samples, max_len)

    def process(self, samples, max_len):
        data = []
        for sample_dict in samples:
            words = sample_dict["words"]
            words = words[: max_len]
            word_ids = [self.vocab.to_index(word) for word in words]

            labels = sample_dict["labels"]
            labels = labels[: max_len]
            label_ids = [self.label2id[label] for label in labels]

            text = sample_dict["text"]
            softword_ids = build_softword(text)
            softword_ids = softword_ids[: max_len]

            data.append((word_ids, label_ids, softword_ids))

        return data

    def get_tensor(self, input_list, lengths, batch_size, max_seq_len):
        inputs_tensor = \
            torch.LongTensor(batch_size, max_seq_len).fill_(self.vocab.to_index(self.vocab.pad_token))
        for i, seq in enumerate(input_list):
            seq_len = lengths[i]
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        word_ids = self.data[index][0]
        label_ids = self.data[index][1]
        softword_ids = self.data[index][2]

        return word_ids, label_ids, softword_ids

    def collate_fn(self, batch):
        word_ids_list = [data[0] for data in batch]
        label_ids_list = [data[1] for data in batch]
        softword_ids_list = [data[2] for data in batch]

        lengths = [len(seq) for seq in word_ids_list]
        max_seq_len = max(lengths)
        batch_size = len(batch)

        word_ids_tensor = self.get_tensor(word_ids_list, lengths, batch_size, max_seq_len)
        label_ids_tensor = self.get_tensor(label_ids_list, lengths, batch_size, max_seq_len)
        softword_ids_tensor = self.get_tensor(softword_ids_list, lengths, batch_size, max_seq_len)

        return word_ids_tensor, lengths, softword_ids_tensor, label_ids_tensor


class NERDatasetSoftLexicon(Dataset):
    def __init__(self, samples, char_vocab, soft_lexicon_vocab, label2id, device, max_len=512):
        self.char_vocab = char_vocab
        self.soft_lexicon_vocab = soft_lexicon_vocab
        self.label2id = label2id
        self.device = device
        self.data = self.process(samples, max_len)

    def process(self, samples, max_len):
        data = []
        for sample_dict in samples:
            words = sample_dict["words"]
            words = words[: max_len]
            word_ids = [self.char_vocab.to_index(word) for word in words]

            text = sample_dict["text"]
            softlexicon = self.soft_lexicon_vocab.build_soft_lexicon(text)
            softlexicon_ids, softlexicon_weights = self.soft_lexicon_vocab.postproc_soft_lexicon(softlexicon)

            labels = sample_dict["labels"]
            labels = labels[: max_len]
            label_ids = [self.label2id[label] for label in labels]

            data.append((word_ids, softlexicon_ids, softlexicon_weights, label_ids))

        return data

    def get_tensor(self, input_list, lengths, batch_size, max_seq_len, pad_token):
        inputs_tensor = torch.LongTensor(batch_size, max_seq_len).fill_(pad_token)
        for i, seq in enumerate(input_list):
            seq_len = lengths[i]
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)

    def pad_softlexicon(self, seq, max_seq_len, type="id"):
        if type == "weight":
            default_encoding = [0.0] * len(seq[0])
        else:
            default_encoding = [self.soft_lexicon_vocab.to_index(self.soft_lexicon_vocab.pad_token)] * len(seq[0])

        seq = seq[:max_seq_len]
        seq_len = len(seq)
        seq += [default_encoding] * (max_seq_len - seq_len)
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        word_ids = self.data[index][0]
        softlexicon_ids = self.data[index][1]
        softlexicon_weights = self.data[index][2]
        label_ids = self.data[index][3]

        return word_ids, softlexicon_ids, softlexicon_weights, label_ids

    def collate_fn(self, batch):
        word_ids_list = [data[0] for data in batch]
        softlexicon_ids_list = [data[1] for data in batch]
        softlexicon_weights_list = [data[2] for data in batch]
        label_ids_list = [data[3] for data in batch]

        lengths = [len(seq) for seq in word_ids_list]
        max_seq_len = max(lengths)
        batch_size = len(batch)

        word_ids_tensor = self.get_tensor(word_ids_list,
                                          lengths,
                                          batch_size,
                                          max_seq_len,
                                          self.char_vocab.to_index(self.char_vocab.pad_token))
        label_ids_tensor = self.get_tensor(label_ids_list, lengths, batch_size, max_seq_len, 0)

        softlexicon_ids_list = [self.pad_softlexicon(seq, max_seq_len, "id") for seq in softlexicon_ids_list]
        softlexicon_weights_list = [self.pad_softlexicon(seq, max_seq_len, "weight") for seq in
                                    softlexicon_weights_list]

        softlexicon_ids_tensor = torch.tensor(softlexicon_ids_list)
        softlexicon_weights_tensor = torch.tensor(softlexicon_weights_list, dtype=torch.float)

        return word_ids_tensor, lengths, softlexicon_ids_tensor, softlexicon_weights_tensor, label_ids_tensor




