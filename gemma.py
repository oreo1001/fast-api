import pandas as pd
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
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = get_tokenizer()
    model.resize_token_embeddings(len(tokenizer))          #토크나이저에 맞춰서 토큰 resize 
    # EOS_TOKEN = tokenizer.eos_token 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model,tokenizer,device

# 추론
def generate_org(prompt_text):
    max_new_tokens=512
    # max_length = 200  # 출력 길이 제한 (원하는 경우 설정)
    num_return_sequences = 1  # 반환할 시퀀스 수 (원하는 경우 설정)
    model, tokenizer, device = get_model_from_huggingface()

    # prompt_text = prompt_eos_str(prompt_text)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, 
        num_return_sequences=num_return_sequences, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True) 
    return decoded_output[0]

# input_text = '다음 문장을 읽고, 이어질 알맞은 문장을 제시된 보기 중에서 고르세요.\n\n문장: 부츠를 대체할 수 있는 것이 바로 아쿠아슈즈다.\n\nA. 글쓰기가 시작된 곳은 기원전 3000년께 메소포타미아 티그리스강과 유프라테스강 유역 비옥한 농경지대다.\nB. 인류 문명의 시작과 함께 도서관은 탄생했다.\nC. 최초 도서관도 이곳에서 등장했다.\nD. 아쿠아 슈즈는 물놀이뿐 아니라 하이킹 및 일상생활에서도 신을 수 있어 활용도가 좋다.'  
input_text= "지금 현재 대한민국의 대통령에 대해서 알려줘 "  
text = '한국어로 답해'
prompt_text = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
{text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
"""

print(generate_org(prompt_text))