import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'bert_RNN'
        self.train_path = '../dataset/CARUSERdata/train.txt'  # 训练集
        self.dev_path = '../dataset/CARUSERdata/dev.txt'  # 验证集
        self.test_path = '../dataset/CARUSERdata/test.txt'  # 测试集
        self.topic_class_list = [x.strip() for x in open(
            '../dataset/CARUSERdata/class.txt').readlines()]                                   # 类别名单
        self.save_path = '../saves/bert_RNN_saved_dict/' + self.model_name + '.ckpt'           # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')             # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_topic_classes = len(self.topic_class_list)             # 主题类别数
        self.num_sentiment_classes = 3                                  # 情感类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '../pre_train_model/bert_base_chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])
        return out
