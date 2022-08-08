import torch
from pathlib import Path

cache_dir = Path("./cached_data")

msra_data_dir = Path("./data/msra")
msra_output_dir = Path("./output/msra")

resume_data_dir = Path("./data/ResumeNER")
resume_output_dir = Path("./output/ResumeNER")

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

resume_label2id = {
    '[PAD]': 0,
    "B-LOC": 1,
    "E-TITLE": 2,
    "E-LOC": 3,
    "B-TITLE": 4,
    "E-PRO": 5,
    "O": 6,
    "B-EDU": 7,
    "S-ORG": 8,
    "E-CONT": 9,
    "M-ORG": 10,
    "B-PRO": 11,
    "S-RACE": 12,
    "M-RACE": 13,
    "B-RACE": 14,
    "M-PRO": 15,
    "M-LOC": 16,
    "E-RACE": 17,
    "B-CONT": 18,
    "S-NAME": 19,
    "M-CONT": 20,
    "M-EDU": 21,
    "B-NAME": 22,
    "E-ORG": 23,
    "M-NAME": 24,
    "M-TITLE": 25,
    "B-ORG": 26,
    "E-NAME": 27,
    "E-EDU": 28
}

# train parameters
batch_size = 20
epochs = 30
learning_rate = 8e-4
overall_max_char_seq_len = 400
overall_max_seq_len = 512
gpu = "0"
if gpu != "":
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

# model parameters
flat_model_size = 160
rel_pos_init = 0  # 需要写死
pos_norm = False
four_pos_shared = True
learnable_position = False
fusion_metod = "ff_two"  # 未使用
use_bert = False
char_embedding_size = 0  # 初始化
word_embedding_size = 0  # 初始化
bert_embedding_size = 0  # 初始化
num_heads = 8
scaled = True
attn_dropout = 0
layer_norm_dropout = 0.15
layer_norm_eps = 1e-12



