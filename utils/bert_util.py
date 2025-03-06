import random
import torch
from tqdm import tqdm

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


# 构建训练、验证数据集
def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip().replace(" ", "")
                if not lin:
                    continue
                if '\t' not in lin:
                    continue
                content, label = lin.split('\t', 1)
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                parts = label.split('\t')
                topic_label = [0] * 10
                topic_dict = {"动力": 0, "价格": 1, "内饰": 2, "配置": 3, "安全性": 4, "外观": 5, "操控": 6, "油耗": 7,
                              "空间": 8, "舒适性": 9}
                for part in parts:
                    if '#' in part:
                        all_scores = 0
                        topic, scores = part.split('#')
                        topic_id = topic_dict[topic]
                        topic_label[topic_id] = 1
                        all_scores += int(scores)
                sentiment_label = 2 if all_scores > 0 else 0 if all_scores < 0 else 1  # 0:负面, 1:中性, 2:正面

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, topic_label, sentiment_label, seq_len, mask))

        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


# 构建数据迭代器
class Dataset_Iterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        topic_y = torch.LongTensor([_[1] for _ in datas]).to(self.device).float()
        sentiment_y = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        return (x, seq_len, mask), (topic_y, sentiment_y)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = Dataset_Iterater(dataset, config.batch_size, config.device)
    return iter


# 将数据集按照split_ratio划分为训练集和测试集
def split_dataset(input_file, train_file, valid_file, split_ratio_):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    random.shuffle(lines)  # 随机打乱数据
    split_index = int(len(lines) * split_ratio_)
    print("总共{}行，训练集:{}行， 测试集:{}行".format(len(lines), split_index, len(lines) - split_index))

    train_data = lines[:split_index]
    valid_data = lines[split_index:]

    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(train_data)
    print(f"训练集保存成功，位于:{train_file}")

    with open(valid_file, 'w', encoding='utf-8') as file:
        file.writelines(valid_data)
    print(f"测试集保存成功，位于:{valid_file}")
