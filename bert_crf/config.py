from pathlib import Path


cluener_data_dir = Path("./data/cluener")
msra_data_dir = Path("./data/msra")

cluener_output_dir = Path("./output/cluener")
msra_output_dir = Path("./output/msra")

cluener_predict_dir = Path("./predict/cluener")
msra_predict_dir = Path("./predict/msra")

log_dir = Path("./log")

cached_dir = Path("./cached_data")

bert_model = "bert-base-chinese"

pretrain_word_emb_path = "pretrain_word_emb.ctb50"

cluner_label2id = {
        '[PAD]': 0,
        "O": 1,
        "B-address": 2,
        "B-book": 3,
        "B-company": 4,
        'B-game': 5,
        'B-government': 6,
        'B-movie': 7,
        'B-name': 8,
        'B-organization': 9,
        'B-position': 10,
        'B-scene': 11,
        "I-address": 12,
        "I-book": 13,
        "I-company": 14,
        'I-game': 15,
        'I-government': 16,
        'I-movie': 17,
        'I-name': 18,
        'I-organization': 19,
        'I-position': 20,
        'I-scene': 21,
        "S-address": 22,
        "S-book": 23,
        "S-company": 24,
        'S-game': 25,
        'S-government': 26,
        'S-movie': 27,
        'S-name': 28,
        'S-organization': 29,
        'S-position': 30,
        'S-scene': 31
}

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