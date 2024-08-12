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
# template = """Answer the following question in Korean.
# #Question: 
# {question}

# #Answer: """  # 질문과 답변 형식을 정의하는 템플릿
# prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성
# chain = prompt | llm | StrOutputParser()
# print( chain.invoke({"question": "현재 대한민국의 대통령은 누구야?"}))

embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)
vectorstore = FAISS.load_local('./sllm_db/faiss', embeddings_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr")

###############################################################################
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved html formed table context to answer the question. \
If you don't know the answer, just say "문서에 없는 내용입니다. 다시 질문해주세요." \
Answer correctly using given context.
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

#서브체인 (takes historical messages and the latest user question)
history_aware_retriever = create_history_aware_retriever(     #conversation history를 받고 document를 내보낸다.
    llm, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
gemma_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    gemma_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
