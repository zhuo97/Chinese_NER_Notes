import random
import torch


class DatasetLoader():
    def __init__(self, data, batch_size, shuffle, vocab, label2id, seed, sort=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocab = vocab
        self.label2id = label2id
        self.seed = seed
        self.sort = sort
        self.examples = None
        self.features = None
        self.reset()

    def reset(self):
        self.examples = self.preprocess(self.data)
        if self.sort:
            self.examples = sorted(self.examples, key=lambda x: x[2], reverse=True)
        if self.shuffle:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        self.features = [self.examples[i:i+self.batch_size] for i in range(0, len(self.examples), self.batch_size)]
        print(f"{len(self.features)} batches created")

    def preprocess(self, data):
        """Preprocess the data and convert to ids"""
        processed = []
        for d in data:
            context = d["context"]
            tokens = [self.vocab.to_index(w) for w in context.split(" ")]
            lengths = len(tokens)
            text_tag = d["tag"]
            tag_ids = [self.label2id[tag] for tag in text_tag.split(" ")]
            processed.append((tokens, tag_ids, lengths))
        return processed

    def get_long_tensor(self, inputs_list, lengths, batch_size):
        max_token_len = max(lengths)
        inputs_tensor = torch.LongTensor(batch_size, max_token_len).fill_(0)
        for i, s in enumerate(inputs_list):
            inputs_tensor[i, :lengths[i]] = torch.LongTensor(s)
        return inputs_tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        batch = self.features[index]
        batch_size = len(batch)
        batch = list(zip(*batch))
        tokens_list, tags_list, lengths = list(batch[0]), list(batch[1]), list(batch[2])
        input_ids = self.get_long_tensor(tokens_list, lengths, batch_size)
        input_tags = self.get_long_tensor(tags_list, lengths, batch_size)
        return input_ids, input_tags, lengths


if __name__ == "__main__":
    from pathlib import Path
    from data_processer import CluenerProcessor

    data_path = Path("./data")
    processor = CluenerProcessor(data_path)
    processor.get_vocab()
    train_dataset = processor.get_train_examples()
    label2id = {
        "O": 0,
        "B-address": 1,
        "B-book": 2,
        "B-company": 3,
        'B-game': 4,
        'B-government': 5,
        'B-movie': 6,
        'B-name': 7,
        'B-organization': 8,
        'B-position': 9,
        'B-scene': 10,
        "I-address": 11,
        "I-book": 12,
        "I-company": 13,
        'I-game': 14,
        'I-government': 15,
        'I-movie': 16,
        'I-name': 17,
        'I-organization': 18,
        'I-position': 19,
        'I-scene': 20,
        "S-address": 21,
        "S-book": 22,
        "S-company": 23,
        'S-game': 24,
        'S-government': 25,
        'S-movie': 26,
        'S-name': 27,
        'S-organization': 28,
        'S-position': 29,
        'S-scene': 30,
        "<START>": 31,
        "<STOP>": 32
    }

    train_loader = DatasetLoader(
        data=train_dataset,
        batch_size=50,
        shuffle=False,
        seed=1234,
        sort=False,
        vocab=processor.vocab,
        label2id=label2id
    )

    for step, batch in enumerate(train_loader):
        input_ids, input_tags, lengths = batch
        print(input_ids)
        print(input_tags)
        print(lengths)
        break