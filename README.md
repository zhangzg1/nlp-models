# 基于Bert的中文文本分类模型，基于T5的问答模型，pytorch框架

## 1、介绍

这个项目完成了两种NLP常见任务：基于Bert模型的文本分类任务，基于T5模型的问答任务。

关于文本分类任务，这里是以Bert模型作为基座模型，使用汽车行业用户观点数据集对Bert进行微调，同时完成两种文本分类任务，情感识别和主题识别。情感识别是**单分类任务**，主要有3个类别：中立，正向、负向；主题识别是**多分类任务**，主要有10个类别：动力、价格、内饰、配置、安全性、外观、操控、油耗、空间、舒适性。

关于问答任务，这里是以T5模型作为基础模型，使用问答数据集对T5模型进行微调，完成**生成式**的问答任务，通过输入一个问题和一段文章，微调后的T5模型会输出问题的答案。

## 2、显卡要求

24G显存，一块3090或4090。

## 3、下载源码

```
git clone https://github.com/zhangzg1/nlp-models.git
cd nlp-models
```

## 4、安装依赖环境（Linux）

```
# 创建虚拟环境
conda create -n nlp_models python=3.9
conda activate nlp_models
# 安装其他依赖包
pip install -r requirements.txt
```

## 5、代码结构

```text
.
├── dataset                           
   ├── CARUSERdata	            # bert文本分类数据集
   ├── DuReaderQG                    # t5问答数据集
├── examples
   ├── bert_predict.py               # 分类任务测试样例
   ├── t5_predict.py		    # 问答任务测试样例
├── models
	 ├── bert_base_chinese       # bert预训练模型
	 ├── t5_base                 # t5预训练模型
├── saves
	 ├── bert_saved_dict         # bert模型微调后的权重参数
	 ├── t5_saved_dict           # t5模型微调后的权重参数
├── train_eval
   ├── train_bert.py                 # bert模型的训练流程
   ├── train_t5.py                   # t5模型的训练流程
├── utils
   ├── bert_util.py                  # bert模型数据处理
   ├── t5_util.py                    # t5模型数据处理
├── run_bert.py                      # 微调训练bert模型
├── run_t5.py                        # 微调训练t5模型
├── requirements.txt                 # 第三方依赖库
├── README.md                        # 说明文档             
```

## 6、中文数据集

关于中文文本分类的数据集dataset/CARUSERdata，主要来自真实的汽车行业用户观点数据，任务是识别文本的主题及情感。其中数据集中的每段文本都有两类标签有，分别是主题和情感。数据集中主题被分为10类，包括：动力、价格、内饰、配置、安全性、外观、操控、油耗、空间、舒适性。情感分为3类，分别用数字0、1、-1表示中立、正向、负向。由于数据集中每个文本的主题标签都伴随着情感类别，所以当主题有多个标签时，情感采用数字加和的方式，sum大于0为正向，小于0为负向，0为中性。

关于问答任务的数据集data/DuReaderQG，任务是根据参考文本和用户的问题，模型输出问题的答案。数据集中的每一行为一个数据样本，主要是`json` 格式。其中，`context` 代表参考文章，`question` 代表问题，`answer` 代表问题答案。

## 7、模型训练

当下载好预训练的基座模型后，就可以选择模型进行微调训练了。

```
# 基于Bert的中文文本分类模型训练并测试
python run_bert.py

# 基于T5的问答模型训练并测试
python run_t5.py
```

文本分类模型的测试指标主要有：accuracy，precision，recall，F1，损失曲线等等。

问答模型的测试指标主要有：BLEU-1、验证集损失等等。

## 8、样例测试

在模型微调训练完成后，这里还提供了微调后的文本分类模型和问答模型的样例测试，模型会基于用户的问题输出预测结果。

```
# Bert分类模型的样例测试
python examples/bert_predict.py

# T5问答模型的样例测试
python examples/t5_predict.py
```