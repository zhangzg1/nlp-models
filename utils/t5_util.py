import json
import torch
from tqdm import tqdm

PAD, UNK, S, E = '[PAD]', '[UNK]', '[CLS]', '[SEP]'  # padding符号，UNK未知符号，S句子起始符号，E句子终止符号


# 构建训练、验证数据集
def build_dataset(config):
    def load_dataset(path, question_pad_size=64, answer_pad_size=128):
        datas = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line)
                context = data['context']
                question = data['question']
                answer = data['answer']

                src = "question:" + question + " context:" + context
                trg = answer

                src_token = config.tokenizer.tokenize(src)
                src_token = [S] + src_token + [E]
                src_len = len(src_token)
                src_token_ids = config.tokenizer.convert_tokens_to_ids(src_token)

                trg_token = config.tokenizer.tokenize(trg)
                trg_token = [S] + trg_token + [E]
                trg_len = len(trg_token)
                trg_token_ids = config.tokenizer.convert_tokens_to_ids(trg_token)

                src_mask = []
                if question_pad_size:
                    if src_len < question_pad_size:
                        src_mask = [1] * len(src_token_ids) + [0] * (question_pad_size - src_len)
                        src_token_ids += ([0] * (question_pad_size - src_len))
                    else:
                        src_mask = [1] * question_pad_size
                        src_token_ids = src_token_ids[:question_pad_size - 1] + [src_token_ids[-1]]
                        src_len = question_pad_size

                trg_mask = []
                if answer_pad_size:
                    if trg_len < answer_pad_size:
                        trg_mask = [1] * len(trg_token_ids) + [0] * (answer_pad_size - trg_len)
                        trg_token_ids += ([0] * (answer_pad_size - trg_len))
                    else:
                        trg_mask = [1] * answer_pad_size
                        trg_token_ids = trg_token_ids[:answer_pad_size - 1] + [trg_token_ids[-1]]
                        trg_len = answer_pad_size

                datas.append((src_token_ids, src_len, src_mask, trg_token_ids, trg_len, trg_mask))

        return datas

    train = load_dataset(config.train_path, config.question_pad_size, config.answer_pad_size)
    dev = load_dataset(config.dev_path, config.question_pad_size, config.answer_pad_size)
    return train, dev


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
        src_token_ids = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        src_seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        src_mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        trg_token_ids = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trg_seq_len = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        trg_mask = torch.LongTensor([_[5] for _ in datas]).to(self.device)

        return (src_token_ids, src_seq_len, src_mask), (trg_token_ids, trg_seq_len, trg_mask)

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
