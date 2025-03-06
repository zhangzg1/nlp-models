import time
import torch
import numpy as np
from train_eval.train_bert import train, get_time_dif
from importlib import import_module
import argparse
from utils.bert_util import build_dataset, build_iterator, split_dataset

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='bert', type=str, help='choose a model: bert, bert_CNN, bert_RNN')
args = parser.parse_args()

if __name__ == '__main__':
    # 将数据集按比例划分为训练数据和验证数据
    np.random.seed(42)
    path_raw = "dataset/CARUSERdata/train_data.txt"
    path_train = "dataset/CARUSERdata/data/train.txt"
    path_valid = "dataset/CARUSERdata/data/dev.txt"
    split_ratio = 0.8
    split_dataset(path_raw, path_train, path_valid, split_ratio)

    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config()
    # 确保结果的可复现性
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 开始训练
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
