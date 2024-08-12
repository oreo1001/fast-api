from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
base_model = 'google/gemma-2-2b-it'
token = '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'
import huggingface_hub
huggingface_hub.login(token=token)

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)      #base모델의 토크나이저 불러오기
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer                                          #패딩토큰 = 종료토큰

def get_model_from_huggingface():       #llm model 
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    return model

# 추론
def generate_org(input_text):
    tokenizer = get_tokenizer()
    model = get_model_from_huggingface()
    messages = [
        {"role": "user", "content": input_text},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cpu")
    outputs = model.generate(**input_ids,max_new_tokens=256)
    return tokenizer.decode(outputs[0])

# generate_org("머신러닝에 대한 시를 써줄래?")

print(generate_org("머신러닝에 대한 시를 써줄래?"))