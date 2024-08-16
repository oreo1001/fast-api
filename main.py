import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from langserve.pydantic_v1 import BaseModel,Field
from load_model import llm, load_model_func
from logs import router as logs_router
from test import router as test_router
from sllm import router as sllm_router
from starlette.concurrency import iterate_in_threadpool

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Union
from langserve import add_routes
from chat import chain as chat_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_func()
    # 컨텍스트 진입: 초기화 작업이 완료된 상태에서 애플리케이션이 실행됩니다.
    yield
    # 애플리케이션 종료 시 실행할 정리 작업 (필요할 경우)
    print("애플리케이션이 종료되었습니다.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def redirect_root_to_docs():
    print("get res")
    return RedirectResponse("/chat/playground")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


#로깅 설정
logging.basicConfig(filename='fast_app.log', level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# Request 로깅 미들웨어
@app.middleware("http")
async def log_request(request: Request, call_next):
    logging.info(f"Request: {request.method} {request.url}")
    body = await request.body()
    logging.info(f"Request Body: {body.decode()}")
    response = await call_next(request)
    return response

# Response 로깅 미들웨어
@app.middleware("http")
async def log_response(request: Request, call_next):
    response = await call_next(request)
    response_body = [chunk async for chunk in response.body_iterator]
    if response_body:
        logging.info(f"Response Body: {response_body[0].decode()}")
    # 바디를 다시 설정
    response.body_iterator = iterate_in_threadpool(iter(response_body))
    
    logging.info(f"Response: {response.status_code}")
    return response

def read_logs(log_file='fast_app.log'):
    with open(log_file, 'r') as file:
        logs = file.readlines()
    return logs

templates = Jinja2Templates(directory="templates")
# 로그 뷰 엔드포인트
@app.get("/logs", response_class=HTMLResponse, include_in_schema=False)
@app.get("/logs/", response_class=HTMLResponse, include_in_schema=False)
async def get_logs(request: Request):
    logs = read_logs()
    return templates.TemplateResponse("log_viewer.html", {"request": request, "logs": logs})


# .env 파일 로드
load_dotenv('.env')
app.include_router(test_router)
# app.include_router(sllm_router)

add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

# add_routes(
#     app,
#     chat_chain.with_types(input_type=InputChat),
#     path="/ollama",
#     enable_feedback_endpoint=True,
#     enable_public_trace_link_endpoint=True,
#     playground_type="ollama",
# )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)