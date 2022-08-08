import os

def load_dataset(data_path):
    dataset = []

    with open(data_path) as f:
        words, tags = [], []
        for line in f:
            if line != "\n":
                line = line.strip("\n")
                word, tag = line.split(" ")
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("An exception was raised, skippping a word: {}".format(e))
            else:
                if len(words) > 0:
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_sentences = open(os.path.join(save_dir, "sentences.txt"), "w")
    file_tags = open(os.path.join(save_dir, "tags.txt"), "w")

    for words, tags in dataset:
        file_sentences.write("{}\n".format(' '.join(words)))
        file_tags.write("{}\n".format(' '.join(tags)))

    file_sentences.close()
    file_tags.close()

    print("- done.")


def build_tags(data_dir, tags_file):
    data_types = ["train", "dev", "test"]
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, "tags.txt")
        with open(tags_path, "r") as f:
            for line in f:
                tag_seq = filter(len, line.strip().split(" "))
                tags.update(list(tag_seq))

    with open(tags_file, "w") as f:
        f.write("\n".join(tags))


if __name__ == "__main__":
    train_data_path = "./raw_data/train.char.bmes"
    dev_data_path = "./raw_data/dev.char.bmes"
    test_data_path = "./raw_data/test.char.bmes"

    train_dataset = load_dataset(train_data_path)
    save_dataset(train_dataset, "./train")

    dev_dataset = load_dataset(dev_data_path)
    save_dataset(dev_dataset, "./dev")

    test_dataset = load_dataset(test_data_path)
    save_dataset(test_dataset, "./test")

    build_tags("./", "./tags.txt")





