import asyncio
import json
import time
from fastapi import APIRouter, HTTPException, Request, logger
from fastapi.responses import JSONResponse, StreamingResponse
import pytz
from custom_mongo_chat import CustomMongoDBChatHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sllm_langchain import s_rag_chain

router = APIRouter(
    prefix="/sllm",
    tags=["sllm"],
    responses={404: {"description": "Not found"}}
)
korea_tz = pytz.timezone("Asia/Seoul")
from pymongo import MongoClient

#connectionString = "mongodb://dba:20240731@localhost:11084/"
connectionString = "mongodb://localhost:27017/"
dbClient = MongoClient(connectionString)
db = dbClient['sllm']
chatCollection = db["chat"]
historyCollection = db["chat_histories"]

headers = {
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Type': 'text/event-stream'
}

@router.post("/messages")
async def post_message(request: Request):
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id")
    if question == '':
        raise HTTPException(status_code=400, detail="No input data")

    myChat = chatCollection.find_one({"session_id": session_id})
    if myChat:
        chatCollection.update_one(
            {"session_id": session_id},
            {"$set": {"temp_question": question}}
        )
        message = "세션에 question 추가함!"
    else:
        chatCollection.insert_one(
            {
                "session_id": session_id,
                "temp_question": question
            }
        )
        message = "새로운 세션 추가"
    return JSONResponse(content={"message": message})

@router.get("/messages")
async def get_message(session_id: str):
    if not session_id:
        return StreamingResponse(content='message: No session_id provided', headers=headers, media_type='text/event-stream')

    myChat = chatCollection.find_one({"session_id": session_id})
    if not myChat:
        return StreamingResponse(content='Session not found', headers=headers, media_type='text/event-stream')

    question = myChat['temp_question']
    return run_langchain_stream(question=question, session_id=session_id)

@router.delete("/messages")
async def delete_message(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    myChat = chatCollection.find_one({"session_id": session_id})
    historyCollection.delete_many({"SessionId": session_id})

    if myChat:
        message = "Session store and question list initialized"
    else:
        message = "Session not found"
    return JSONResponse(content={"message": message})

@router.post("/messageList")
async def post_message_list(request: Request):
    data = await request.json()
    sessionId = data.get("session_id")
    chatHistories = list(historyCollection.find({"SessionId": sessionId}))
    messageList = []
    for chatHistory in chatHistories:
        history = json.loads(chatHistory['History'])
        speaker = history['type']
        content = history['data']['content']
        message = {
            "speaker": speaker,
            "content": content
        }
        messageList.append(message)
    return JSONResponse(content={"msList": messageList})

def run_langchain_stream(question, session_id):
    conversational_rag_chain = RunnableWithMessageHistory(
        s_rag_chain,
        lambda session_id: CustomMongoDBChatHistory(
            session_id=session_id,
            connection_string=connectionString,
            database_name="sllm",
            collection_name="chat_histories",
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    async def generate():
        try:
            config = {"configurable": {"session_id": session_id}}
            response_stream = get_response(conversational_rag_chain, question, config)

            for response in response_stream:
                response_replaced = response.replace('\n', '🖐️')
                yield f"data: {response_replaced}\n\n"
                await asyncio.sleep(0.01)  # Simulate delay
            yield 'data: \u200C\n\n'
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return  StreamingResponse(generate(), headers=headers, media_type='text/event-stream')

def get_response(chain, prompt, config):
    return (
        val for chunk in chain.stream({"input": prompt}, config)     #input과 config를 넣고 chunk를 스트림으로 받는다.
        for key, val in chunk.items() if key == 'answer'
    )

##############################################################################################################################
#API 테스트를 위한 simplechat api

@router.post("/simplechat")
async def get_simple(request: Request):
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id")
    if question == '':
        raise HTTPException(status_code=400, detail="No input data")
    if not session_id:
        return StreamingResponse(content='No Session Id Provided', headers=headers, media_type='text/event-stream')

    answer = run_langchain(question,session_id)
    return JSONResponse(content={"answer": answer})

def run_langchain(question, session_id):
    conversational_rag_chain = RunnableWithMessageHistory(
        s_rag_chain,
        lambda session_id: CustomMongoDBChatHistory(
            session_id=session_id,
            connection_string=connectionString,
            database_name="sllm",
            collection_name="chat_histories",
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    config = {"configurable": {"session_id": session_id}}
    result = conversational_rag_chain.invoke({"input": question}, config )["answer"]
    return result
