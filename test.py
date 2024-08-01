from fastapi import APIRouter, logger
import pytz

router = APIRouter(
    prefix="/test",
    tags=["test"],
    responses={404: {"description": "Not found"}}
)

korea_tz = pytz.timezone("Asia/Seoul")

@router.get("/")
@router.get("")
def root():
    return {"message": "Hello World"}

@router.get("/home")
def home():
    return {"message": "home"}