import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

import torch
base_model = '4yo1/sapie'
token = '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'
import huggingface_hub
huggingface_hub.login(token=token)

# def get_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)      #base모델의 토크나이저 불러오기
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer                                          
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# hf_pipeline = pipeline(           #파이프라인을 직접 만드는 방법은 호환이 안되는 듯 하다.
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1,  # GPU를 사용할 경우 0, CPU는 -1
#     max_new_tokens=512,
# )
# llm = HuggingFacePipeline(pipeline=hf_pipeline)
llm = HuggingFacePipeline.from_model_id(
    model_id=base_model,  # 사용할 모델의 ID를 지정합니다.
    task="text-generation",  # 수행할 작업을 지정합니다. 여기서는 텍스트 생성입니다.
    device=0 if torch.cuda.is_available() else -1,  # GPU를 사용할 경우 0, CPU는 -1
    pipeline_kwargs={"max_new_tokens": 512, "pad_token_id": tokenizer.eos_token_id},      #tokenizer.pad_token = tokenizer.eos_token
)

template = """Answer the following question in Korean.
#Question: 
{question}

#Answer: """  # 질문과 답변 형식을 정의하는 템플릿
prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성
chain = prompt | llm | StrOutputParser()

# answer=chain.stream({"topic":"deep learnging"})
# stream_response(answer)
print( chain.invoke({"question": "현재 대한민국의 대통령은 누구야?"}))

# llm = create_huggingface_llm()
# chat_model = ChatHuggingFace(llm=llm)
# ai_msg = chat_model.invoke(messages)
# print(ai_msg.content)
