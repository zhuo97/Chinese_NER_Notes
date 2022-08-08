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
            sample_dict["chars"] = words
            sample_dict["labels"] = labels
            samples.append(sample_dict)

        return samples


class ResumeProcessor:
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
            sample_dict["chars"] = words
            sample_dict["labels"] = labels
            samples.append(sample_dict)

        return samples


if __name__ == "__main__":
    import config

    processor = ResumeProcessor(config.resume_data_dir)
    train_samples = processor.get_train_samples()
    lengths = []
    for sample_dict in train_samples:
        lengths.append(len(sample_dict["text"]))
    print("max length: ", max(lengths))
    print("avg length: ", sum(lengths)/len(lengths))
