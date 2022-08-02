from pathlib import Path

cluner_data_dir = Path("./data/cluner")
msra_data_dir = Path("./data/msra")

cluner_output_dir = Path("./output/cluner")
msra_output_dir = Path("./output/msra")

random_init_emb_size = 128

pretrain_char_emb_path = "pretrain_word_emb.giga"
pretrain_word_emb_path = "pretrain_word_emb.ctb50"

msra_label2id = {
    '[PAD]': 0,
    'O': 1,
    'B-ORG': 2,
    'I-ORG': 3,
    'B-PER': 4,
    'I-PER': 5,
    'B-LOC': 6,
    'I-LOC': 7,
    '[CLS]': 8,
    '[SEP]': 9
}

cluner_label2id = {
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