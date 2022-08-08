import torch
from torch.utils.data import Dataset


class FLATDataset(Dataset):
    def __init__(self, samples, vocab, label2id, device, overall_max_char_seq_len=400, overall_max_seq_len=512):
        self.vocab = vocab
        self.label2id = label2id
        self.device = device
        self.data = self.process(samples, overall_max_char_seq_len, overall_max_seq_len)
        self.pad_token_id = vocab.to_index(vocab.pad_token)
        self.overall_max_char_seq_len = overall_max_char_seq_len
        self.overall_max_seq_len = overall_max_seq_len

    def process(self, samples, overall_max_char_seq_len, overall_max_seq_len):
        """
        :param samples:
        :param max_char_seq_len:
        :param max_seq_len:
        :return:
            data 由以下几个部分组成：
                lattice = char_id + word_id，char 长度限制在 max_char_seq_len，总长度限制在 max_seq_len
                char_seq_len: 标量，char字符长度
                pos_s: head 向量
                pos_e: tail 向量
                label_ids: 标签序列
        """
        data = []
        for sample_dict in samples:
            chars = sample_dict["chars"]
            chars = chars[: overall_max_char_seq_len]
            char_ids = [self.vocab.to_index(char) for char in chars]
            char_seq_len = len(char_ids)

            pos_s = [i for i in range(len(char_ids))]
            pos_e = [i for i in range(len(char_ids))]

            text = sample_dict["text"]
            text = text[: overall_max_char_seq_len]
            # [[word, start, end], ...]
            lexicon = self.vocab.trie.get_lexicon(text)
            words = []
            for word_info in lexicon:
                word, start, end = word_info
                words.append(word)
                pos_s.append(start)
                pos_e.append(end)
            word_ids = [self.vocab.to_index(word) for word in words]

            lattice = char_ids + word_ids
            lattice = lattice[: overall_max_seq_len]
            num_words = len(lattice) - char_seq_len
            pos_s = pos_s[: overall_max_seq_len]
            pos_e = pos_e[: overall_max_seq_len]

            labels = sample_dict["labels"]
            labels = labels[: overall_max_char_seq_len]
            label_ids = [self.label2id[label] for label in labels]

            data.append((lattice, char_seq_len, num_words, pos_s, pos_e, label_ids))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lattice = self.data[index][0]
        char_seq_len = self.data[index][1]
        num_words = self.data[index][2]
        pos_s = self.data[index][3]
        pos_e = self.data[index][4]
        label_ids = self.data[index][5]

        return lattice, char_seq_len, num_words, pos_s, pos_e, label_ids

    def get_tensor(self, input_list, lengths, batch_size, max_seq_len, pad_token):
        inputs_tensor = torch.LongTensor(batch_size, max_seq_len).fill_(pad_token)
        for i, seq in enumerate(input_list):
            seq_len = lengths[i]
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)

    def collate_fn(self, batch):
        lattice_list = [data[0] for data in batch]
        char_seq_len_list = [data[1] for data in batch]
        num_words_list = [data[2] for data in batch]
        pos_s_list = [data[3] for data in batch]
        pos_e_list = [data[4] for data in batch]
        label_ids_list = [data[5] for data in batch]

        lattice_lengths = [len(lattice) for lattice in lattice_list]
        batch_max_seq_len = max(lattice_lengths)
        batch_size = len(batch)
        batch_max_char_seq_len = max(char_seq_len_list)

        lattice_tensor = self.get_tensor(lattice_list, lattice_lengths, batch_size, batch_max_seq_len, self.pad_token_id)
        pos_s_tensor = self.get_tensor(pos_s_list, lattice_lengths, batch_size, batch_max_seq_len, self.pad_token_id)
        pos_e_tensor = self.get_tensor(pos_e_list, lattice_lengths, batch_size, batch_max_seq_len, self.pad_token_id)
        label_ids_tensor = self.get_tensor(label_ids_list, char_seq_len_list, batch_size, batch_max_char_seq_len, 0)

        char_seq_len_tensor = torch.tensor(char_seq_len_list).to(self.device)
        num_words_tensor = torch.tensor(num_words_list).to(self.device)

        return lattice_tensor, char_seq_len_tensor, num_words_tensor, pos_s_tensor, pos_e_tensor, label_ids_tensor


if __name__ == "__main__":
    import config
    from vocabulary import Vocabulary
    from processor import MSRAProcessor, ResumeProcessor
    from torch.utils.data import DataLoader

    vocab = Vocabulary(config.pretrain_word_emb_path)
    processor = ResumeProcessor(config.resume_data_dir)
    train_samples = processor.get_train_samples()
    train_dataset = FLATDataset(train_samples, vocab, config.resume_label2id, "cpu")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    for batch_samples in train_loader:
        lattice_tensor, char_seq_len_tensor, num_words_tensor, pos_s_tensor, pos_e_tensor, label_ids_tensor = batch_samples
        print(lattice_tensor)
        print(char_seq_len_tensor)
        print(num_words_tensor)
        print(pos_s_tensor)
        print(pos_e_tensor)
        print(label_ids_tensor)
        break



