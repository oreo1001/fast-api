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
from load_model import llm
from langchain_core.output_parsers import StrOutputParser

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