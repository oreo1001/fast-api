import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from logs import router as logs_router
from test import router as test_router
from sllm import router as sllm_router
from starlette.concurrency import iterate_in_threadpool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    logging.info(f"Response: {response.status_code}")
    # 응답 바디 로깅
    response_body = [chunk async for chunk in response.body_iterator]
    response.body_iterator = iterate_in_threadpool(iter(response_body))
    logging.info(f"Response Body: {response_body[0].decode()}")
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
app.include_router(sllm_router)