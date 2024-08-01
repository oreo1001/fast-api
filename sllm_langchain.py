from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import boto3
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
load_dotenv()

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model_kwargs = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    streaming=True,    #for stream responses
    callbacks=[StreamingStdOutCallbackHandler()]    #스트리밍으로 답변을 받기 위한 콜백
)

embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)
vectorstore = FAISS.load_local('./sllm_db/faiss', embeddings_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr")

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
If you don't know the answer, just say "입학전형 문서에 없는 내용입니다. 다시 질문해주세요." \
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
s_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)