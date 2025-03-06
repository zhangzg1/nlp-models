import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
PAD, CLS = '[PAD]', '[CLS]'                                       # padding符号, bert中综合信息符号
context_max_len = 64                                              # 输入文本的最大长度
num_topic_classes = 10                                            # 主题类别数
num_sentiment_classes = 3                                         # 情感类别数
hidden_size = 768                                                 # 隐藏层维度
bert_path = '../pre_train_model/bert_base_chinese'  # bert模型路径
checkpoint_path = '../saves/bert_saved_dict/bert.ckpt'  # 模型微调后的权重参数路径


class BertForClassification(nn.Module):
    def __init__(self, bert_model_path, num_topic_classes, num_sentiment_classes):
        super(BertForClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.fc_topic = nn.Linear(hidden_size, num_topic_classes)
        self.fc_sentiment = nn.Linear(hidden_size, num_sentiment_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        topic_out = self.fc_topic(pooled)
        sentiment_out = self.fc_sentiment(pooled)
        return topic_out, sentiment_out


def text_classify(context):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    # 加载基础模型Bert
    model = BertForClassification(bert_path, num_topic_classes, num_sentiment_classes)
    # 加载微调后的权重参数
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 文本预处理
    token = tokenizer.tokenize(context)
    token = [CLS] + token
    token_ids = tokenizer.convert_tokens_to_ids(token)
    mask = []
    if context_max_len:
        if len(token) < context_max_len:
            mask = [1] * len(token_ids) + [0] * (context_max_len - len(token))
            token_ids += ([0] * (context_max_len - len(token)))
        else:
            mask = [1] * context_max_len
            token_ids = token_ids[:context_max_len]
    token_ids = torch.tensor([token_ids]).long().to(device)
    mask = torch.tensor([mask]).long().to(device)

    # 模型预测
    topic_out, sentiment_out = model(token_ids, attention_mask=mask)
    # 使用 sigmoid 函数进行多分类预测
    topic_probabilities = torch.sigmoid(topic_out)
    # 根据概率阈值（例如 0.5）确定类别
    topic_predict = (topic_probabilities > 0.5).int()
    # 使用 softmax 函数进行单分类预测
    sentiment_probabilities = torch.softmax(sentiment_out, dim=1)
    # 获取预测的类别索引
    sentiment_predict = torch.argmax(sentiment_probabilities, dim=1)

    # 主题分类和情感分类的字典
    topic_dict = {"动力": 0, "价格": 1, "内饰": 2, "配置": 3, "安全性": 4, "外观": 5, "操控": 6, "油耗": 7, "空间": 8,
                  "舒适性": 9}
    sentiment_dict = {"负面": 0, "中性": 1, "正面": 2}

    # 将字典反转，便于从索引映射到标签
    topic_labels = {v: k for k, v in topic_dict.items()}
    sentiment_labels = {v: k for k, v in sentiment_dict.items()}

    # 将预测结果从索引映射到具体的类别标签
    topic_results = [topic_labels[i] for i in range(num_topic_classes) if topic_predict[0, i] == 1]
    sentiment_result = sentiment_labels[sentiment_predict.item()]

    return topic_results, sentiment_result


if __name__ == '__main__':
    context = "空间我倒是无所谓，主要就是森林人有更好的动力！！！！"
    topic_results, sentiment_result = text_classify(context)
    print(f"文本内容: {context}")
    print(f"主题类别预测: {topic_results}")
    print(f"情感类别预测: {sentiment_result}")
