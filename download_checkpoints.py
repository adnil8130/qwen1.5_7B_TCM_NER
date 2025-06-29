import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 代理下载快
from huggingface_hub import snapshot_download

# 缓存模型到本地
cache_dir = "/root/autodl-tmp"

repo_id = "Qwen/Qwen1.5-7B-Chat"
downloaded = snapshot_download(
    repo_id,
    cache_dir=cache_dir,
)

from transformers import Qwen2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

qwen_model_path = "/root/autodl-tmp/models--Qwen--Qwen1.5-7B-Chat/snapshots/5f4f5e69ac7f1d508f8369e977de208b4803444b"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

model = AutoModelForCausalLM.from_pretrained(qwen_model_path).half().to(device)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print("model_inputs: ", model_inputs)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
print("generated_ids: ", generated_ids)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print("generated_ids: ", generated_ids)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("response: ", response)