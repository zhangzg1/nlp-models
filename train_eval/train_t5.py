import torch
import torch.nn as nn
import time
from datetime import timedelta
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

PAD = '[PAD]'  # padding符号


# 获取已使用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=config.tokenizer.convert_tokens_to_ids(PAD))

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False  # 记录是否很久没有效果提升

    # 进行多个epoch的训练
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for question, answer in train_iter:
            # predict = [batch_size, trg_seq_len-1, output_dim]
            predict_out = model(question, answer)
            model.zero_grad()
            # [batch_size, trg_seq_len-1] ==> [(trg_seq_len-1) * batch_size]
            trg_token_ids = answer[0][:, 1:].contiguous().view(-1)
            # [batch_size, trg_seq_len-1, output_dim] ==> [(trg_seq_len-1) * batch_size, output_dim]
            predict = predict_out.contiguous().view(-1, predict_out.size(-1))
            loss = criterion(predict, trg_token_ids)
            loss.backward()
            optimizer.step()

            # 每100个批次就输出在训练集和验证集上的损失和BLEU指标
            if total_batch % 100 == 0 and total_batch > 0:
                # predict_out[batch_size, trg_seq_len-1, output_dim] ==> predict_ids[batch_size, trg_seq_len-1]
                predict_ids = predict_out.argmax(dim=-1)
                predict_text = [config.tokenizer.convert_ids_to_tokens(sentence) for sentence in predict_ids]
                # trg_ids[batch_size, trg_seq_len-1]
                trg_ids = answer[0][:, 1:]
                trg_text = [config.tokenizer.convert_ids_to_tokens(sentence) for sentence in trg_ids]
                # 使用 SmoothingFunction 来避免在没有匹配时得到零分
                smooth = SmoothingFunction().method1
                bleu_1_scores = []
                # 计算每对句子的 BLEU 分数
                for trg, pre in zip(trg_text, predict_text):
                    bleu_1 = sentence_bleu([trg], pre, weights=(1, 0, 0, 0), smoothing_function=smooth)
                    bleu_1_scores.append(bleu_1)
                # 计算平均 BLEU 分数
                train_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)

                # 计算验证集上所有数据损失和BLEU分数的平均值
                dev_loss, dev_blue_1, dev_blue_2, dev_blue_3, dev_blue_4 = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 保存验证集上损失最小的模型
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Val Loss: {2:>5.2}, Train BLUE_1: {3:>5.2}, Val BLUE_1: {4:>5.2}, Time: {5} {6}'
                print(
                    msg.format(total_batch, loss.item(), dev_loss, train_bleu_1, dev_blue_1, time_dif, improve))
                model.train()
            total_batch += 1

            # 判断是否需要提前结束训练
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def evaluate(config, model, data_iter):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=config.tokenizer.convert_tokens_to_ids(PAD))
    loss_total, bleu_1_scores = 0, 0

    with torch.no_grad():
        for question, answer in data_iter:
            predict_out = model(question, answer)
            model.zero_grad()
            trg_token_ids = answer[0][:, 1:].contiguous().view(-1)
            predict = predict_out.contiguous().view(-1, predict_out.size(-1))
            loss = criterion(predict, trg_token_ids)
            loss_total += loss

            # 计算一个batch数据的BLEU分数
            predict_ids = predict_out.argmax(dim=-1)
            predict_text = [config.tokenizer.convert_ids_to_tokens(sentence) for sentence in predict_ids]
            trg_ids = answer[0][:, 1:]
            trg_text = [config.tokenizer.convert_ids_to_tokens(sentence) for sentence in trg_ids]
            smooth = SmoothingFunction().method1
            bleu_1_scores = []
            for trg, pre in zip(trg_text, predict_text):
                bleu_1 = sentence_bleu([trg], pre, weights=(1, 0, 0, 0), smoothing_function=smooth)
                bleu_1_scores.append(bleu_1)

    return loss_total / len(data_iter), sum(bleu_1_scores) / len(bleu_1_scores)
