import time
import torch
import numpy as np
from train_eval.train_t5 import train, get_time_dif
from importlib import import_module
import argparse
from utils.t5_util import build_dataset, build_iterator

parser = argparse.ArgumentParser(description='Chinese Text QA')
parser.add_argument('--model', default='t5', type=str, help='choose a model: t5, GPT-3')
args = parser.parse_args()

if __name__ == '__main__':
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
    train_data, dev_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 开始训练
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter)
