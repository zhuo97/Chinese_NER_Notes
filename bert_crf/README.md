# 运行方式
1. `python run_bert_crf.py --do_train` 对模型进行训练，其他参数使用详见代码
2. 如果需要使用 soft lexicon 进行词汇增强，需要下载对应的词向量文件到 `pretain_word_emb` 目录，然后运行 `run_bert_crf_soft_lexicon.py`