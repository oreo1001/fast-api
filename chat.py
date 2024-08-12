from langserve import RemoteRunnable
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
#from sentence_transformers import SentenceTransformer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.vectorstores import FAISS
import boto3
from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import torch
base_model = 'google/gemma-2-2b-it'
token = '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'
import huggingface_hub
huggingface_hub.login(token=token)

# def get_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)      #base모델의 토크나이저 불러오기
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer  
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)                                     
tokenizer = AutoTokenizer.from_pretrained(base_model)

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",  # 사용할 모델의 ID를 지정합니다.
    task="text-generation",  # 수행할 작업을 지정합니다. 여기서는 텍스트 생성입니다.
    device=0 if torch.cuda.is_available() else -1,  # GPU를 사용할 경우 0, CPU는 -1
    pipeline_kwargs={"max_new_tokens": 512, "pad_token_id": tokenizer.eos_token_id},      #tokenizer.pad_token = tokenizer.eos_token
)
from langchain_core.output_parsers import StrOutputParser

# chat = [
#     { "role": "user", "content": "Write a hello world program" },
# ]


template = """Answer the following question in Korean.
#Question: 
{messages}

#Answer: """  # 질문과 답변 형식을 정의하는 템플릿
prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성
# DEFAULT_SYSTEM_PROMPT = """"You are helpful AI."""
# def prompt_template(sys_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
#     """Template for the prompt to be used in the model.
    
#     Args:
#         sys_prompt (str, optional): System's prompt. Defaults to DEFAULT_SYSTEM_PROMPT.
    
#     Returns:
#         str: Prompt template.
#     """
#     context = "{messages}"
#     template = f"""### System:
#     {sys_prompt}
#     ### User:
#     {context}
#     ### Assistant:
#     """
#     return template
# template = prompt_template()
# prompt = PromptTemplate(template=template, input_variables=["messages"])
chain = prompt | llm
# prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# chain = prompt | llm | StrOutputParser()


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful and kind cafe order Assistant. You must answer in Korean.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )