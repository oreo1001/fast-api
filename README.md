# Sllm fast-api

#env aws cofiguration code
cp .env.example .env

#가상환경 설치후 다운로드

python -m venv venv <br>
source ./venv/Scripts/activate (윈도우) <br>
pip install -r requirements.txt <br>

#서버 실행

uvicorn main:app --reload

fast-api와 asgi인 uvicorn을 사용해서 웹 애플리케이션 제공

#API Docs (로컬)

localhost:8000/docs

#로그 확인 (개선 필요)

localhost:8000/logs

#SimpleChat API 테스트

localhost:8000/sllm/simplechat

포스트맨에서 테스트
