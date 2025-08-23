# 1. 베이스 이미지 선택 (Python 3.9)
# 운영체제와 파이썬이 미리 설치된 공식 이미지를 사용합니다.
FROM python:3.9-slim

# 2. 작업 디렉터리 설정
# 컨테이너 내에서 코드가 위치할 폴더를 만듭니다.
WORKDIR /app

# 3. 필요한 파일 복사
# 현재 폴더의 모든 파일을 컨테이너의 /app 폴더로 복사합니다.
# (.dockerignore에 명시된 파일들은 제외됩니다.)
COPY . .

# 4. 시스템 라이브러리 설치 (OpenCV 의존성)
# opencv-python이 동영상을 처리하는 데 필요한 시스템 라이브러리를 설치합니다.
# libgl1-mesa-glx를 libgl1으로 변경
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Python 라이브러리 설치
# requirements.txt 파일을 사용하여 모든 파이썬 라이브러리를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 6. 서버 실행 포트 노출
# 컨테이너의 8000번 포트를 외부에서 접근할 수 있도록 열어줍니다.
EXPOSE 8000

# 7. 서버 실행 명령어
# 컨테이너가 시작될 때 실행될 명령어를 지정합니다.
# Gunicorn은 여러 개의 워커를 사용하여 uvicorn을 실행하는 운영 환경용 서버입니다.
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "predict:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]