from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(
    prefix="/logs",
    tags=["logs"],
    responses={404: {"description": "Not found"}}
)

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 로그 파일 읽기 함수
def read_logs(log_file='fast_app.log'):
    with open(log_file, 'r') as file:
        logs = file.readlines()
    return logs

# 로그 뷰 엔드포인트
@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def get_logs(request: Request):
    logs = read_logs()
    return templates.TemplateResponse("log_viewer.html", {"request": request, "logs": logs})