import os
import cv2
import torch
import numpy as np
import boto3  # AWS S3 연동을 위한 라이브러리
import tempfile
import logging
from urllib.parse import urlparse # URL 파싱을 위함
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import Counter
from inference import EmbeddingClassifier, PredictionResult

# keyword_to_korean 딕셔너리
keyword_to_korean = {
    "Salmo": "송어",
    "Oncorhynchus": "연어",
    "Salvelinus": "산천어",

    "Scomber": "고등어",
    "Scomberomorus": "삼치",
    "Thunnus": "참치",
    "Katsuwonus": "가다랑어",
    "Euthynnus": "참치",
    "Auxis": "참치",
    
    "Pagrus": "돔",
    "Sparus": "돔",
    "Dentex": "돔",

    "Paralichthys": "광어",
    "Pleuronectes": "도다리",
    "Platichthys": "가자미",

    "Morone": "농어",
    "Lateolabrax": "농어",
    "Epinephelus": "돔",
    "Mycteroperca": "돔",
    "Cephalopholis": "돔",
    "Serranus": "돔",
    "Pomadasys": "돔",

    "Sebastes": "우럭",

    "Ictalurus": "메기",
    "Ameiurus": "메기",
    "Clarias": "메기",

    "Cyprinus": "잉어",
    "Carassius": "붕어",
    "Barbus": "잉어",
    "Leuciscus": "잉어",
    "Rutilus": "잉어",
    "Gobio": "돌고기",
    "Pimephales": "잉어",

    "Clupea": "청어",
    "Sprattus": "정어리",
    "Alosa": "청어",
    "Brevoortia": "청어",

    "Anguilla": "뱀장어",

    "Carcharhinus": "상어",
    "Sphyrna": "상어",
    "Galeocerdo": "상어",
    "Prionace": "상어",
    "Rhincodon": "상어",
    "Dasyatis": "가오리",
    "Myliobatis": "가오리",
    "Aetobatus": "가오리",

    "Lagocephalus": "복어",
    "Takifugu": "복어",
    "Tetraodon": "복어",

    "Menidia": "실버사이드",
    "Gambusia": "구피",
    "Poecilia": "구피",

    "Oreochromis": "틸라피아",
    "Pelmatolapia": "틸라피아",
    "Cichla": "시클리드",
    "Astronotus": "오스카",
    "Heros": "시클리드",
    "Amphilophus": "시클리드",

    "Stegastes": "돔",
    "Halichoeres": "놀래기",
    "Thalassoma": "놀래기",
    "Monodactylus": "모노닥",
    "Rypticus": "비누고기",
    "Parupeneus": "촉수과 어류",
    "Atule": "전갱이",
    "Platax": "제비활치",
    "Seriola": "방어",
    "Ocyurus": "돔",
    "Boops": "돔",
    "Platycephalus": "양태",
    "Scarus": "앵무고기",
    "Sparisoma": "앵무고기",
    "Holocentrus": "청줄놀래기",
    "Myripristis": "병정고기",
    "Oligoplites": "쥐치",
    "Acanthurus": "외과의사어",
    "Caranx": "전갱이",
    "Lepisosteus": "가아",
    "Gasterosteus": "가시고기",
    "Trachinotus falcatus": "은상어",

    "Lutjanus gibbus": "돔",
    "Lutjanus fulvus": "검은줄바리",
    "Lutjanus sebae": "돔",
    "Lutjanus argentiventris": "노랑줄바리",
    "Lutjanus apodus": "황줄바리",
    "Lutjanus jocu": "큰입바리",

    "Snapper": "돔",
    "Squirrelfish": "청줄놀래기",
    "Soldierfish": "병정고기",
    "Leatherjacket": "쥐치",
    "Surgeonfish": "외과의사어",
    "Parrotfish": "앵무고기",
    
    "Lutjanus fulvus": "돔",
    "Lutjanus sebae": "돔",
    "Lutjanus argentiventris": "돔",
    "Lutjanus apodus": "돔",
    "Lutjanus jocu": "돔",
    "Trachinotus falcatus": "",
    "Bagre marinus": "메기",
    "Mustelus canis" : "상어",
    "Carcharodon carcharias": "백상아리",
    "Chrysoblephus laticeps": "돔"
}


def get_korean_name(scientific_name: str, mapping: dict) -> str:
    if scientific_name in mapping:
        return mapping[scientific_name]
    
    
    genus = scientific_name.split()[0]
    return mapping.get(genus, scientific_name)

try:
    # 1. 어종 분류기(EmbeddingClassifier) 초기화
    config = {
        "dataset": {"path": os.path.join(os.path.dirname(__file__), "database.pt")},
        "model": {"path": os.path.join(os.path.dirname(__file__), "model.ckpt"), "device": "cpu"}
    }
    classifier = EmbeddingClassifier(config)
    print(">>> 어종 분류기(EmbeddingClassifier) 로드 완료.")

    # 2. 객체 탐지기(YOLOv5) 초기화
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    print(">>> 객체 탐지기(YOLOv5) 로드 완료.")
except Exception as e:
    print(f"[오류] 모델 로딩에 실패했습니다: {e}")
    exit()

# --- FastAPI 앱 설정 ---
app = FastAPI()

# S3 URL을 받기 위한 요청 모델 정의
class VideoRequest(BaseModel):
    s3_url: str

def download_video_from_s3(s3_url: str) -> str:
    """S3 URL에서 영상을 다운로드하고, 로컬 파일 경로를 반환합니다."""
    try:
        s3_client = boto3.client('s3')
        
        parsed_url = urlparse(s3_url)
        
        if parsed_url.scheme == 'https' and parsed_url.netloc.endswith('.amazonaws.com'):
            bucket_name = parsed_url.netloc.split('.')[0]
        else:
            bucket_name = parsed_url.netloc

        object_key = parsed_url.path.lstrip('/')
        
        temp_dir = tempfile.gettempdir() 
        local_filename = os.path.join(temp_dir, os.path.basename(object_key))
        
        print(f">>> S3에서 영상 다운로드 시작: {bucket_name}/{object_key}")
        s3_client.download_file(bucket_name, object_key, local_filename)
        print(f">>> 영상 다운로드 완료: {local_filename}")
        
        return local_filename
    except Exception as e:
        logging.error(f"S3 파일 다운로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"S3 파일 다운로드 오류: {e}")

# --- 핵심 분석 함수 ---
def analyze_video(video_path: str):
    """주어진 영상 경로를 분석하고 탐지된 어종 리스트를 반환합니다."""
    if not os.path.exists(video_path):
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> 영상 처리 시작: {os.path.basename(video_path)} (총 {total_frames} 프레임)")

    all_detected_species = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # 10프레임마다 한 번씩만 처리 (기존 로직 유지)
        if frame_count % 10 != 0:
            continue
        
        print(f"--- 프레임 {frame_count}/{total_frames} 처리 중 ---")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = yolo_model(frame_rgb)
        
        for *box, conf, cls in detections.xyxy[0]:
            if conf > 0.4:
                x1, y1, x2, y2 = map(int, box)
                fish_crop_img = frame_rgb[y1:y2, x1:x2]

                if fish_crop_img.size == 0:
                    continue

                try:
                    results: list[PredictionResult] = classifier(fish_crop_img)
                    if results:
                        best_fish = max(results, key=lambda x: x.accuracy)
                        korean_name = get_korean_name(best_fish.name, keyword_to_korean)
                        all_detected_species.append(korean_name)
                        print(f"분류 결과: 이름={results[0].name}, 정확도={results[0].accuracy:.4f}")
                except Exception as e:
                    print(f"   [경고] 어종 분류 중 오류 발생: {e}")
    
    cap.release()
    return all_detected_species

# --- API 엔드포인트 정의 ---
@app.post("/analyze_video")
async def analyze_video_endpoint(request: VideoRequest):
    """S3 URL을 받아 영상을 분석하고 어종과 횟수를 반환합니다."""
    local_video_path = None
    try:

        local_video_path = download_video_from_s3(request.s3_url)
        
        detected_species_list = analyze_video(local_video_path)
        
        if not detected_species_list:
            return {"message": "영상에서 어종을 탐지하지 못했습니다.", "analysis_result": []}
            
        summary = Counter(detected_species_list)
        
        # 4. 최종 결과 JSON 형식으로 변환
        final_result = {
            species: count 
            for species, count in summary.most_common()
        }
        
        print("영상 처리 완료! 최종 결과를 반환합니다.")
        return {"analysisResult": final_result}

    finally:
        if local_video_path and os.path.exists(local_video_path):
            os.remove(local_video_path)
            print(f">>> 임시 파일 삭제 완료: {local_video_path}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)