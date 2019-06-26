# Chinese-Word-Segmentation
中文分词的学习总结记录。
中文分词《State-of-the-art Chinese Word Segmentation with Bi-LSTMs》
论文地址:https://www.aclweb.org/anthology/D18-1529
代码参考:https://github.com/mokeam/Chinese-Word-Segmentation-in-NLP, 我对其进行了一些修改，并手动实现了一些api,没有调包
此论文作者提出的模型相对简单，就是一个简单的堆叠BiLSTM，并且作者指出，中文分词不需要太复杂的模型，合理的训练以及调整就能达到很好的效果。
作者还列举了分词错误的原因：标注不一致，词汇外单词，并且前后缀也对分词造成了一定的影响。作者认为这些错误无法通过更复杂的模型避免，要想解决错误，只能让数据集更“完美”.
在ctb等数据集中，达到了95%+的准确率，
