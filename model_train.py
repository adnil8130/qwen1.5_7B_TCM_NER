# Step1 相关包导入以及环境设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 代理下模型快

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

data_files = '/root/sft_qwen1p5_7B/datasets/'
use_auth_token = False
qwen_model_path = "/root/autodl-tmp/models--Qwen--Qwen1.5-7B-Chat/snapshots/5f4f5e69ac7f1d508f8369e977de208b4803444b"
do_train = True
do_valid = True
do_test = False
device = "cuda"


# Step2 加载数据集
raw_dataset = load_dataset('json', data_files=data_files+'/medical.train.json')
raw_dataset_dev = load_dataset('json', data_files=data_files+'/medical.dev.json')
raw_dataset_test = load_dataset('json', data_files=data_files+'/medical.test.json')
raw_dataset['validation'] = raw_dataset_dev['train']
raw_dataset['test'] = raw_dataset_test['train']
columns = raw_dataset['train'].column_names
print("raw_dataset: ", raw_dataset)



# Step3 数据格式预处理
tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
def tranfer_old_to_new(example):
    """
    用数据集拼system prompt和输入输出
    example:
    {'text': '目的观察复方丁香开胃贴外敷神阙穴治疗慢性心功能不全伴功能性消化不良的临床疗效', 'entities': [{'end_idx': 10, 'entity_label': '中医治疗', 'entity_text': '复方丁香开胃贴', 'start_idx': 4}, {'end_idx': 32, 'entity_label': '西医诊断', 'entity_text': '心功能不全伴功能性消化不良', 'start_idx': 20}]}
    
    res:
    {'instruction': '\n    你是一个文本实体识别领域的专家，你需要从给定的句子中提取\n    - 中医治则\n    - 中医治疗\n    - 中医证候\n    - 中医诊断\n    - 中药\n    - 临床表现\n    - 其他治疗\n    - 方剂\n    - 西医治疗\n    - 西医诊断\n    这些实体. 以 json 格式输出, 如 {"entity_text": "丹参", "entity_label": "中药"} , {"entity_text": "黄疸", "entity_label": "中医诊断"} \n    注意: \n    1. 输出的每一行都必须是正确的 json 字符串. \n    2. 找不到任何实体时, 输出"没有找到任何实体". \n    \n    ', 'input': '现头昏口苦', 'output': '{"entity_text": "口苦", "entity_label": "临床表现"}'}
    """
    input_text = example["text"]
    entities = example["entities"]
        
    entity_sentence = ""
    for entity in entities:
        entity_json = dict(entity)
        entity_text = entity_json["entity_text"]
        entity_label = entity_json["entity_label"]

        entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""

    if entity_sentence == "":
        entity_sentence = "没有找到任何实体"
    
    res = dict()
    res["instruction"] = """
    你是一个文本实体识别领域的专家，你需要从给定的句子中提取
    - 中医治则
    - 中医治疗
    - 中医证候
    - 中医诊断
    - 中药
    - 临床表现
    - 其他治疗
    - 方剂
    - 西医治疗
    - 西医诊断
    这些实体. 以 json 格式输出, 如 {"entity_text": "丹参", "entity_label": "中药"} , {"entity_text": "黄疸", "entity_label": "中医诊断"} 
    注意: 
    1. 输出的每一行都必须是正确的 json 字符串. 
    2. 找不到任何实体时, 输出"没有找到任何实体". 
    
    """
    res["input"] = f"{input_text}"
    res["output"] = entity_sentence
    return res

def process_func(old_example):
    example = tranfer_old_to_new(old_example)
    MAX_LENGTH = 784
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(        
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        print("trunct!", len(input_ids))
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def print_dataset_example(example):
    print("input_ids: ",example["input_ids"])
    print("inputs: ", tokenizer.decode(example["input_ids"]))
    print("label_ids: ", example["labels"])
    print("labels: ", tokenizer.decode(list(map(lambda x: x if x != -100 else 12, example["labels"]))))

train_dataset = raw_dataset['train'].map(
                process_func,
                remove_columns=columns,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
print_dataset_example(train_dataset[0])
print_dataset_example(train_dataset[1])

valid_dataset = raw_dataset['validation'].map(
                process_func,
                remove_columns=columns,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
print_dataset_example(valid_dataset[0])
print_dataset_example(valid_dataset[1])


# Step4 模型创建
# 原始模型
model = AutoModelForCausalLM.from_pretrained(qwen_model_path).half().to(device)
print("======== source model parameters name begins=================")
# for name, parameter in model.named_parameters():
#     print(name)
print("======== source model parameters name ends=================")
# Lora模型
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=8,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)
print("======== lora config begins=================")
print("config:", config)
print("======== lora config ends=================")
model = get_peft_model(model, config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads() 
print("======== source model parameters name begins=================")
model.print_trainable_parameters()
print("======== source model parameters name ends=================")


# Step5 训练参数配置
args = TrainingArguments(
    output_dir="./tcm_ner",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1
)


# Step6 使用训练器进行训练
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        hypothesis = ' '.join(hypothesis)
        if not hypothesis:
            hypothesis = "-"
        scores = rouge.get_scores(hypothesis, ' '.join(reference))
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)


# Step7 模型推理验证、测试
train_result = trainer.train()
metrics = train_result.metrics
print("train result: ", train_result)
print("train metrics: ", metrics)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()