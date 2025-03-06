import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from datetime import timedelta


# 获取已使用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 绘图函数
def plot_and_save(label1, values1, label2, values2, title, filename):
    plt.figure(figsize=(14, 5))
    plt.plot(values1, label=label1)
    plt.plot(values2, label=label2)
    plt.title(title)
    plt.xlabel('Batch')
    plt.ylabel(label1)
    plt.legend()
    plt.savefig(f'./figure/{filename}.png')


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    # 记录是否很久没有效果提升
    flag = False
    loss_train_batch = []
    acc_topic_train_batch = []
    acc_sentiment_train_batch = []

    # 进行多个epoch的训练
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for trains, labels in train_iter:
            topic_out, sentiment_out = model(trains)
            model.zero_grad()
            criterion = nn.BCEWithLogitsLoss()
            topic_loss = criterion(topic_out, labels[0])
            # cross_entropy()中的预测数据需要是浮点数类型，但标签数据需要是整数类型
            sentiment_loss = F.cross_entropy(sentiment_out, labels[1])
            loss = topic_loss + (1 - config.alpha) * sentiment_loss
            loss_train_batch.append(loss.item())
            loss.backward()
            optimizer.step()

            # 每100个批次就输出在训练集和验证集上的效果
            if total_batch % 100 == 0:
                # GPU上的数据不能直接转成numpy数组，需要先放在CPU上
                topic_true = labels[0].data.cpu()
                topic_num_classes = topic_true.sum(dim=1).long()
                top_indices = torch.argsort(topic_out, dim=1, descending=True)
                topic_predict = torch.stack(
                    [torch.zeros_like(row, dtype=torch.int).scatter(0, top[:n], 1) for row, top, n in
                     zip(topic_out, top_indices, topic_num_classes)])
                topic_predict = topic_predict.cpu()
                # 计算主题分类准确率
                train_topic_acc = metrics.accuracy_score(topic_true, topic_predict)
                acc_topic_train_batch.append(train_topic_acc)

                # 计算情感分类准确率
                sentiment_true = labels[1].data.cpu()
                sentiment_predict = torch.max(sentiment_out.data, 1)[1].cpu()
                train_sentiment_acc = metrics.accuracy_score(sentiment_true, sentiment_predict)
                acc_sentiment_train_batch.append(train_sentiment_acc)

                # 计算验证集上所有数据的准确率和损失
                dev_topic_acc, dev_sentiment_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 保存验证集上损失最小的模型
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {Z:>5.2}, Train Topic Acc: {2:>6.2%}, Train Sentiment Acc: {3:>6.2%}, Val Loss: {4:>5.2}, Val Topic Acc: {5:>6.2%}, Val Sentiment Acc: {6:>6.2%}, Time: {7} {8}'
                print(
                    msg.format(total_batch, loss.item(), train_topic_acc, train_sentiment_acc, dev_loss, dev_topic_acc,
                               dev_sentiment_acc, time_dif, improve))
                model.train()
            total_batch += 1

            # 判断是否需要提前结束训练
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    # 绘制训练集中的主题分类准确率和情感分类准确率曲线
    plot_and_save('Topic Accuracy', acc_topic_train_batch, 'Sentiment Accuracy', acc_sentiment_train_batch,
                  'Train Accuracy Over Batch(100)', 'train_acc_curve')

    test(config, model, test_iter, loss_train_batch)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    loss_test_batch = []
    acc_topic_test_batch = []
    acc_sentiment_test_batch = []

    topic_predict_all = np.array([], dtype=int)
    topic_labels_all = np.array([], dtype=int)
    sentiment_predict_all = np.array([], dtype=int)
    sentiment_labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            topic_out, sentiment_out = model(texts)
            criterion = nn.BCEWithLogitsLoss()
            topic_loss = criterion(topic_out, labels[0])
            sentiment_loss = F.cross_entropy(sentiment_out, labels[1])
            loss = topic_loss + (1 - config.alpha) * sentiment_loss
            loss_test_batch.append(loss.item())
            loss_total += loss

            # 计算主题分类的准确率
            topic_true = labels[0].data.cpu().numpy()
            topic_num_classes = np.sum(topic_true, axis=1).astype(int)
            top_indices = torch.argsort(topic_out, dim=1, descending=True)
            topic_predict = torch.stack(
                [torch.zeros_like(row, dtype=torch.int).scatter(0, top[:n], 1) for row, top, n in
                 zip(topic_out, top_indices, topic_num_classes)])
            topic_predict = topic_predict.cpu().numpy()
            topic_labels_all = np.append(topic_labels_all, topic_true)
            topic_predict_all = np.append(topic_predict_all, topic_predict)
            # 计算每个batch的主题分类准确率用于绘图
            acc_topic_test_batch.append(metrics.accuracy_score(topic_true, topic_predict))

            # 计算情感分类的准确率
            sentiment_true = labels[1].data.cpu().numpy()
            sentiment_predict = torch.max(sentiment_out.data, 1)[1].cpu().numpy()
            sentiment_labels_all = np.append(sentiment_labels_all, sentiment_true)
            sentiment_predict_all = np.append(sentiment_predict_all, sentiment_predict)
            # 计算每个batch的情感分类准确率用于绘图
            acc_sentiment_test_batch.append(metrics.accuracy_score(sentiment_true, sentiment_predict))

    dev_topic_acc = metrics.accuracy_score(topic_labels_all, topic_predict_all)
    dev_sentiment_acc = metrics.accuracy_score(sentiment_labels_all, sentiment_predict_all)

    if test:
        sentiment_report = metrics.classification_report(sentiment_labels_all, sentiment_predict_all,
                                                         target_names=['负向', '中立', '正向'], digits=4)
        topic_confusion = metrics.multilabel_confusion_matrix(topic_labels_all, topic_predict_all)
        sentiment_confusion = metrics.confusion_matrix(sentiment_labels_all, sentiment_predict_all)
        # 绘制训练集中的主题分类准确率和情感分类准确率曲线
        plot_and_save('Topic Accuracy', acc_topic_test_batch, 'Sentiment Accuracy', acc_sentiment_test_batch,
                      'Test Accuracy Over Batch', 'test_acc_curve')
        return dev_topic_acc, dev_sentiment_acc, loss_total / len(
            data_iter), sentiment_report, topic_confusion, sentiment_confusion, loss_test_batch

    return dev_topic_acc, dev_sentiment_acc, loss_total / len(data_iter)


def test(config, model, test_iter, loss_train_batch):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    dev_topic_acc, dev_sentiment_acc, test_loss, sentiment_report, topic_confusion, sentiment_confusion, loss_test_batch = evaluate(
        config, model, test_iter, test=True)
    print('=' * 100)
    msg = 'Test Loss: {0:>5.2}, Test Topic Acc: {Z:>6.2%}, Test Sentiment Acc: {2:>6.2%}'
    print(msg.format(test_loss, dev_topic_acc, dev_sentiment_acc))
    print("Sentiment Precision, Recall and F1-Score...")
    print(sentiment_report)
    print("Topic Confusion Matrix...")
    print(topic_confusion)
    print("Sentiment Confusion Matrix...")
    print(sentiment_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 绘制训练集和测试集中的loss曲线
    plot_and_save('Training Loss', loss_train_batch, 'Testing Loss', loss_test_batch, 'Loss Over Batch', 'loss_curve')
