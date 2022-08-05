import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, samples, label2id, bert_model, device, max_len=512, cls_token="[CLS]", pad_token=0):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.label2id = label2id
        self.id2label = {_id: _label for _label, _id in self.label2id.items()}
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.device = device
        self.data = self.process(samples, max_len)

    def process(self, samples, max_len):
        data = []
        for sample_dict in samples:
            words = sample_dict["words"]
            words = [self.cls_token] + words
            word_ids = self.tokenizer.convert_tokens_to_ids(words)
            word_ids = word_ids[:max_len]

            labels = sample_dict["labels"]
            # 因为添加了 CLS 标签
            labels = ["O"] + labels
            label_ids = [self.label2id[label] for label in labels]
            label_ids = label_ids[:max_len]

            data.append((word_ids, label_ids))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
       
        word_ids = self.data[index][0]
        label_ids = self.data[index][1]
        
        return word_ids, label_ids

    def get_tensor(self, input_list, seq_max_len, batch_size):
        inputs_tensor = torch.LongTensor(batch_size, seq_max_len).fill_(self.pad_token)
        for i, seq in enumerate(input_list):
            seq_len = len(seq)
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)

    def get_mask(self, input_list, seq_max_len, batch_size):
        mask = torch.BoolTensor(batch_size, seq_max_len).fill_(self.pad_token)
        for i, seq in enumerate(input_list):
            seq_len = len(seq)
            mask[i, :seq_len] = 1
        return mask.to(self.device)

    def collate_fn(self, batch):
        """
        需要注意的点：
        1. label 也需要 pad，和 input 共用 mask
        2. 这里不需要 token_type_ids，因为只要一个句子而非句子对
        3. 不需要自己生成 position_ids
        """
        
        word_ids_list = [data[0] for data in batch]
        label_ids_list = [data[1] for data in batch]

        seq_max_len = max([len(seq) for seq in word_ids_list])
        batch_size = len(batch)

        word_ids_tensor = self.get_tensor(word_ids_list, seq_max_len, batch_size)
        attention_mask = self.get_mask(word_ids_list, seq_max_len, batch_size)

        label_ids_tensor = self.get_tensor(label_ids_list, seq_max_len, batch_size)

        return word_ids_tensor, attention_mask, label_ids_tensor


class NERDatasetSoftLexicon(Dataset):
    def __init__(self, samples, bert_model, soft_lexicon_vocab, label2id, device, max_len=512, cls_token="[CLS]", pad_token="[PAD]"):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.soft_lexicon_vocab = soft_lexicon_vocab
        self.label2id = label2id
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.device = device
        self.data = self.process(samples, max_len)

    def process(self, samples, max_len):
        data = []
        for sample_dict in samples:
            words = sample_dict["words"]
            words = [self.cls_token] + words
            word_ids = self.tokenizer.convert_tokens_to_ids(words)
            word_ids = word_ids[:max_len]

            text = sample_dict["text"]
            softlexicon = self.soft_lexicon_vocab.build_soft_lexicon(text)
            softlexicon_ids, softlexicon_weights = self.soft_lexicon_vocab.postproc_soft_lexicon(softlexicon)

            labels = sample_dict["labels"]
            # 因为添加了 CLS 标签
            labels = ["O"] + labels
            label_ids = [self.label2id[label] for label in labels]
            label_ids = label_ids[:max_len]

            data.append((word_ids, softlexicon_ids, softlexicon_weights, label_ids))

        return data

    def get_tensor(self, input_list, lengths, batch_size, max_seq_len, pad_token):
        inputs_tensor = torch.LongTensor(batch_size, max_seq_len).fill_(pad_token)
        for i, seq in enumerate(input_list):
            seq_len = lengths[i]
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)
    
    def get_mask(self, input_list, max_seq_len, batch_size):
        mask = torch.BoolTensor(batch_size, max_seq_len).fill_(self.tokenizer.convert_tokens_to_ids(self.pad_token))
        for i, seq in enumerate(input_list):
            seq_len = len(seq)
            mask[i, :seq_len] = 1
        return mask.to(self.device)

    def pad_softlexicon(self, seq, max_seq_len, type="id"):
        if type == "weight":
            default_encoding = [0.0] * len(seq[0])
        else:
            default_encoding = [self.soft_lexicon_vocab.to_index(self.pad_token)] * len(seq[0])

        # 因为添加 CLS token
        seq = seq[:max_seq_len-1]
        seq = [default_encoding] + seq

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
                                          self.tokenizer.convert_tokens_to_ids(self.pad_token))
        attention_mask = self.get_mask(word_ids_list, max_seq_len, batch_size)

        label_ids_tensor = self.get_tensor(label_ids_list, 
                                           lengths, 
                                           batch_size, 
                                           max_seq_len, 
                                           self.label2id[self.pad_token])

        softlexicon_ids_list = [self.pad_softlexicon(seq, max_seq_len, "id") for seq in softlexicon_ids_list]
        softlexicon_weights_list = [self.pad_softlexicon(seq, max_seq_len, "weight") for seq in
                                    softlexicon_weights_list]
        
        softlexicon_ids_tensor = torch.tensor(softlexicon_ids_list).to(self.device)
        softlexicon_weights_tensor = torch.tensor(softlexicon_weights_list, dtype=torch.float).to(self.device)

        return word_ids_tensor, attention_mask, softlexicon_ids_tensor, softlexicon_weights_tensor, label_ids_tensor


class TestNERDataset(Dataset):
    def __init__(self, samples, bert_model, device, max_len=512, cls_token="[CLS]", pad_token=0):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.device = device
        self.data = self.process(samples, max_len)

    def process(self, samples, max_len):
        data = []
        for sample_dict in samples:
            words = sample_dict["words"]
            words = [self.cls_token] + words
            word_ids = self.tokenizer.convert_tokens_to_ids(words)
            word_ids = word_ids[:max_len]
            data.append((sample_dict["words"], word_ids))
            
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        words = self.data[index][0]
        word_ids = self.data[index][1]
        
        return words, word_ids

    def get_tensor(self, input_list, seq_max_len, batch_size):
        inputs_tensor = torch.LongTensor(batch_size, seq_max_len).fill_(self.pad_token)
        for i, seq in enumerate(input_list):
            seq_len = len(seq)
            inputs_tensor[i, :seq_len] = torch.LongTensor(seq)
        return inputs_tensor.to(self.device)

    def get_mask(self, input_list, seq_max_len, batch_size):
        mask = torch.BoolTensor(batch_size, seq_max_len).fill_(self.pad_token)
        for i, seq in enumerate(input_list):
            seq_len = len(seq)
            mask[i, :seq_len] = 1
        return mask.to(self.device)

    def collate_fn(self, batch):
        """
        需要注意的点：
        1. label 也需要 pad，和 input 共用 mask
        2. 这里不需要 token_type_ids，因为只要一个句子而非句子对
        3. 不需要自己生成 position_ids
        """
        
        words = [data[0] for data in batch]
        word_ids_list = [data[1] for data in batch]

        seq_max_len = max([len(seq) for seq in word_ids_list])
        batch_size = len(batch)

        word_ids_tensor = self.get_tensor(word_ids_list, seq_max_len, batch_size)
        attention_mask = self.get_mask(word_ids_list, seq_max_len, batch_size)
       
        return words, word_ids_tensor, attention_mask


if __name__ == "__main__":
    import config
    import torch
    from tqdm import tqdm
    from data_processor import Processor
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    processor = Processor(config.msra_data_dir)
    train_samples = processor.get_train_samples()
    train_dataset = NERDataset(train_samples, config.msra_label2id, config.bert_model, device)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn)
    for idx, batch_samples in enumerate(tqdm(train_dataloader)):
        word_ids, word_ids_attn_mask, label_ids, label_ids_attn_mask = batch_samples
