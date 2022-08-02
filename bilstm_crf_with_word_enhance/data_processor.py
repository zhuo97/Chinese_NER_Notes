import json

class CLUENERProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_samples(self):
        return self._create_samples(str(self.data_dir / "train.json"))

    def get_dev_samples(self):
        return self._create_samples(str(self.data_dir / "dev.json"))

    def get_test_samples(self):
        return self._create_samples(str(self.data_dir / "test.json"))

    def _create_samples(self, input_path):
        samples = []

        with open(input_path, "r") as f:
            for line in f.readlines():
                sample_dict = {}
                json_line = json.loads(line.strip())
                text = json_line["text"]
                words = list(text)
                label_entities = json_line.get("label", None)
                labels = ["O"] * len(words)

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index: end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = "S-" + key
                                else:
                                    labels[start_index] = "B-" + key
                                    labels[start_index+1: end_index+1] = ["I-" + key] * (len(sub_name)-1)

                sample_dict["text"] = text
                sample_dict["words"] = words
                sample_dict["labels"] = labels
                samples.append(sample_dict)
        return samples


class MSRAProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def get_train_samples(self):
        return self._create_samples(self.data_dir / "train")

    def get_dev_samples(self):
        return self._create_samples(self.data_dir / "dev")

    def get_test_samples(self):
        return self._create_samples(self.data_dir / "test")
        
    def _create_samples(self, input_path):
        samples = []
        
        sentence_file = open(input_path / "sentences.txt", "r")
        tag_file = open(input_path / "tags.txt")
        
        sentences = sentence_file.readlines()
        tags = tag_file.readlines()
        
        assert len(sentences) == len(tags)
        
        for sentence, tag in zip(sentences, tags):
            sample_dict = {}
            sentence = sentence.strip()
            tag = tag.strip()
            
            words = sentence.split(" ")
            text = "".join(words)
            labels = tag.split(" ")
            
            assert len(words) == len(labels)
            
            sample_dict["text"] = text
            sample_dict["words"] = words
            sample_dict["labels"] = labels
            samples.append(sample_dict)
        
        return samples
        
            


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("./data/msra")
    processor = MSRAProcessor(data_dir)
    train_samples = processor.get_train_samples()
    for sample_dict in train_samples:
        print(sample_dict["text"])
        print(sample_dict["words"])
        print(sample_dict["labels"])
        break
        