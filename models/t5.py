import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 't5'
        self.train_path = '../dataset/DuReaderQG/train.json'  # 训练集
        self.dev_path = '../dataset/DuReaderQG/dev.json'  # 验证集
        self.save_path = '../saves/t5_saved_dict/' + self.model_name + '.ckpt'             # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')         # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 16                                            # mini-batch大小
        self.question_pad_size = 512                                    # 问题处理成的长度(短填长切)
        self.answer_pad_size = 32                                       # 答案处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.t5_path = '../pre_train_model/t5_base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.t5_path)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(config.t5_path)
        for param in self.t5.parameters():
            param.requires_grad = True
    def forward(self, question, answer):
        src_token_ids, src_mask = question[0], question[2]
        trg_token_ids, trg_mask = answer[0], answer[2]
        output = self.t5(input_ids=src_token_ids, attention_mask=src_mask,
                         decoder_input_ids=trg_token_ids[:, :-1], decoder_attention_mask=trg_mask[:, :-1])
        return output.logits
