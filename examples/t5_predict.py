import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
PAD, UNK, S, E = '[PAD]', '[UNK]', '[CLS]', '[SEP]'           # padding符号，UNK未知符号，S句子起始符号，E句子终止符号
question_max_len = 512                                        # 输入问题的最大长度
answer_max_len = 32                                           # 生成答案的最大长度
t5_path = '../pre_train_model/t5_base'  # bert模型路径
checkpoint_path = '../saves/t5_saved_dict/t5.ckpt'  # 模型微调后的权重参数路径


def question_answer(context, question):
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(t5_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    # 权重模型checkpoint keys中的键前缀包含了't5.'，而模型的state_dict键不包含这个前缀，这会出现不匹配的情况，所以需要去掉前缀
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith('t5.'):
            new_key = k[len('t5.'):]
        else:
            new_key = k
        new_checkpoint[new_key] = v
    # 此时微调得到的权重参数的字典中的键与原模型的state_dict中的键是一一对应的，可以直接加载权重参数
    model.load_state_dict(new_checkpoint)
    model.to(device)
    model.eval()

    src = "question:" + question + " context:" + context
    src_token = tokenizer.tokenize(src)
    src_token = [S] + src_token + [E]
    src_len = len(src_token)
    src_token_ids = tokenizer.convert_tokens_to_ids(src_token)

    src_mask = []
    if question_max_len:
        if src_len < question_max_len:
            src_mask = [1] * len(src_token_ids) + [0] * (question_max_len - src_len)
            src_token_ids += ([0] * (question_max_len - src_len))
        else:
            src_mask = [1] * question_max_len
            src_token_ids = src_token_ids[:question_max_len - 1] + [src_token_ids[-1]]
    src_token_ids = torch.tensor([src_token_ids]).long().to(device)
    src_mask = torch.tensor([src_mask]).long().to(device)

    # 初始化一个空列表 generated，用于存储生成摘要的 token ID，初始化时只包含一个起始符号 [S]
    generated = tokenizer.convert_tokens_to_ids([S])

    for _ in range(answer_max_len):
        curr_trg_ids = torch.tensor([generated]).long().to(device)
        predict = model(input_ids=src_token_ids, attention_mask=src_mask, decoder_input_ids=curr_trg_ids)
        # 获取模型预测的下一个 token 的向量分布
        next_token_logits = predict.logits[0][-1]
        # 避免生成与上一个 token 重复的 token
        next_token_logits[generated[-1]] = -float('Inf')
        # 对于[UNK]的概率设为无穷小
        next_token_logits[tokenizer.convert_tokens_to_ids(UNK)] = -float('Inf')
        # 进行softmax操作，得到下一个 token 的概率分布，并获取概率最大的 token ID
        predict_ids = next_token_logits.argmax(dim=-1)
        # 若预测的结果为终止符号，则停止生成
        if predict_ids.item() == tokenizer.convert_tokens_to_ids(E):
            generated.append(predict_ids.item())
            break
        # 将生成的 token ID 加入到列表 generated 中
        generated.append(predict_ids.item())

    # 将生成的token ID转换成对应的文本内容
    answer = tokenizer.convert_ids_to_tokens(generated)
    print("answer:" + "".join(answer))


if __name__ == '__main__':
    context = "美赞臣三阶段是180毫升四勺奶粉,勺子比较大,一二阶段的是小勺要小一倍呢,三阶段是一二阶段的两倍,但180除以4的话等于45一勺奶粉。三阶段价格虽然便宜，但奶粉费了，消耗起来快。一二阶段美赞臣小勺是一勺30毫升的水。三阶段不一样的。我家就吃这个。"
    question = "美赞臣3段用量"
    answer = question_answer(context, question)
