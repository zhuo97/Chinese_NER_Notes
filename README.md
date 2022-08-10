# Chinese_NER_Notes
这个仓库主要包含两部分内容：
1. 传统的基线模型：bilstm+crf 和 bert+crf 
2. 采用词汇增强的方法的模型：lattice-lstm、flat；以及与模型无关的 soft-lexicon

这里采用 pytorch 的方式进行实现，具体如下：
1. bert_crf 和 bilstm_crf_with_word_enhance 两个文件夹包含了传统的基线模型实现以及结合 soft-lexicon 的实现；对于 bilstm-crf 也尝试使用了对抗训练的方式，如 fgm
2. 对于 lattice-lstm 和 flat 主要的重点放在对于模型结构的学习

这里用到的词向量文件如下（受限于资源，这里没有用到 bigram 的词向量），具体的下载地址在 https://github.com/jiesutd/LatticeLSTM
1. gigaword_chn.all.a2b.uni.ite50.vec
2. ctb.50d.vec

模型的介绍和使用参考各个文件夹下的说明，对应的笔记存放在 notes 文件夹下。

# 参考资料
1. https://github.com/CLUEbenchmark/CLUENER2020
2. https://github.com/LeeSureman/Flat-Lattice-Transformer
3. https://github.com/DSXiangLi/ChineseNER
4. https://github.com/LeeSureman/Batch_Parallel_LatticeLSTM
5. https://github.com/v-mipeng/LexiconAugmentedNER
6. https://github.com/jiesutd/LatticeLSTM
7. https://github.com/zzh-SJTU/NER_Chinese_medical
