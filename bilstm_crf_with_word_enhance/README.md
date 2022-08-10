# 运行方式
1. 下载 `gigaword_chn.all.a2b.uni.ite50.vec` 和 `ctb.50d.vec` 到 `data` 文件夹对应的目录下
2. 然后运行 `python run_lstm_crf.py --do_train` 进行模型的训练，训练基线模型 BiLSTM+CRF；这里可以通过参数来决定是否使用对抗训练、是否使用预训练词向量、使用什么数据集等
3. `run_lstm_crf_softword.py` 对应的是 softword 词汇增强方法，使用方法与基线模型类似
4. `run_lstm_crf_softlexicon.py` 对应的是 softlexicon 词汇增强方法，使用方法与基线模型类似
