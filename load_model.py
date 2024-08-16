from transformers import AutoTokenizer, pipeline
import torch
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline

llm = None

def load_model_func():
    global llm
    base_model = 'google/gemma-2-2b-it'
    token = "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" 
    login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    hf_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("모델이 성공적으로 로드되었습니다.")