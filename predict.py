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
import argparse

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
    "Trinectes maculatus": "가자미",
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
    "Chrysoblephus laticeps": "돔",
    "Trichiurus lepturus": "갈치",
    
    "Ariopsis felis": "메기",
    "Hypsypops rubicundus": "자리돔",
    "Chaetodon ephippium": "나비고기",
    "Silurus glanis": "메기",
    "Diodon holocanthus": "가시복",
    "Achoerodus viridis": "다금바리",
    "Amphiprion percula": "흰동가리",
    "Sphyraena barracuda": "창꼬치",
    "Rhizoprionodon terraenovae": "상어",
    "Archosargus probatocephalus": "돔",
    "Chaetodipterus faber": "제비활치",
    "Opsanus tau": "상어",
    "Nocomis micropogon": "잉어",
    "Galeorhinus galeus": "상어",
    "Micropterus nigricans": "우럭",
    "Pomatomus saltatrix": "농어",
    "Pachymetopon blochii": "돔",
    "Pylodictis olivaris": "메기",
    "Epinephelus morio": "돔",
    "Moxostoma erythrurum": "잉어",
    "Rachycentron canadum": "고등어",
    "Alectis ciliaris": "전갱이"
}

def imread_unicode(path: str):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def calculate_iou(box1, box2):
    """두 박스의 IoU(Intersection over Union)를 계산합니다."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교집합 영역 계산
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def filter_overlapping_detections(detections, iou_threshold=0.5):
    """겹치는 탐지 결과를 필터링합니다."""
    if len(detections) <= 1:
        return detections
    
    # 신뢰도 순으로 정렬
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    filtered = []
    for detection in sorted_detections:
        x1, y1, x2, y2, conf, cls = detection
        box_area = (x2 - x1) * (y2 - y1)
        
        # 너무 작은 박스는 제외 (노이즈 제거)
        if box_area < 1000:  # 최소 1000 픽셀 이상
            continue
            
        is_duplicate = False
        for existing in filtered:
            iou = calculate_iou(detection[:4], existing[:4])
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(detection)
    
    return filtered

def deduplicate_species_results(species_list, max_per_species=5):
    """같은 어종의 중복 결과를 제한합니다. (최대 개수를 5로 증가)"""
    if len(species_list) <= 1:
        return species_list
    
    # 어종별로 그룹화
    species_groups = {}
    for species in species_list:
        if species not in species_groups:
            species_groups[species] = 0
        species_groups[species] += 1
    
    # 각 그룹에서 최대 max_per_species개까지만 선택
    deduplicated = []
    for species, count in species_groups.items():
        # 최대 개수만큼만 추가
        add_count = min(count, max_per_species)
        deduplicated.extend([species] * add_count)
    
    return deduplicated


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

# --- FastAPI 앱 설정 --- // for server
app = FastAPI()

# S3 URL을 받기 위한 요청 모델 정의
class MediaRequest(BaseModel):
    s3_url: str
    file_type: str = "video"  # "image" 또는 "video", 기본값은 video

# 기존 호환성을 위한 별도 모델
class VideoRequest(BaseModel):
    s3_url: str

def download_media_from_s3(s3_url: str) -> str:
    """S3 URL에서 미디어 파일을 다운로드하고, 로컬 파일 경로를 반환합니다."""
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
        
        print(f">>> S3에서 미디어 파일 다운로드 시작: {bucket_name}/{object_key}")
        s3_client.download_file(bucket_name, object_key, local_filename)
        print(f">>> 미디어 파일 다운로드 완료: {local_filename}")
        
        return local_filename
    except Exception as e:
        logging.error(f"S3 파일 다운로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"S3 파일 다운로드 오류: {e}")

# --- 핵심 분석 함수 ---
def analyze_media_server(media_path: str, file_type: str, conf_threshold: float = 0.1):
    """주어진 미디어 경로를 분석하고 탐지된 어종 리스트를 반환합니다."""
    if not os.path.exists(media_path):
        print(f"[오류] 미디어 파일을 찾을 수 없습니다: {media_path}")
        return []

    # 결과 저장 디렉토리 생성
    save_dir = "detection_results"
    os.makedirs(save_dir, exist_ok=True)

    if file_type == "image":
        return analyze_image(media_path, save_dir, conf_threshold)
    elif file_type == "video":
        return analyze_video(media_path, save_dir, conf_threshold)
    else:
        print(f"[오류] 지원하지 않는 파일 타입입니다: {file_type}")
        return []

# --- API 엔드포인트 정의 ---
@app.post("/analyze_video")
async def analyze_video_endpoint(request: VideoRequest):
    """S3 URL을 받아 미디어를 분석하고 어종과 횟수를 반환합니다. (자동 파일 타입 감지)"""
    local_media_path = None
    try:
        local_media_path = download_media_from_s3(request.s3_url)
        
        # 파일 확장자로 자동 감지
        file_extension = os.path.splitext(local_media_path)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        if file_extension in image_extensions:
            file_type = "image"
            print(f">>> 이미지 파일로 감지됨: {file_extension}")
        elif file_extension in video_extensions:
            file_type = "video"
            print(f">>> 비디오 파일로 감지됨: {file_extension}")
        else:
            # 기본값은 video로 설정하되 경고 메시지 출력
            file_type = "video"
            print(f">>> 알 수 없는 파일 형식, 비디오로 처리: {file_extension}")
        
        detected_species_list = analyze_media_server(local_media_path, file_type)
        
        if not detected_species_list:
            return {"message": f"{file_type}에서 어종을 탐지하지 못했습니다.", "analysis_result": []}
            
        summary = Counter(detected_species_list)
        
        # 최종 결과 JSON 형식으로 변환
        final_result = {
            species: count 
            for species, count in summary.most_common()
        }
        
        print(f"{file_type} 처리 완료! 최종 결과를 반환합니다.")
        return {"analysisResult": final_result}

    finally:
        if local_media_path and os.path.exists(local_media_path):
            os.remove(local_media_path)
            print(f">>> 임시 파일 삭제 완료: {local_media_path}")

@app.post("/analyze_media")
async def analyze_media_endpoint(request: MediaRequest):
    """S3 URL을 받아 이미지 또는 영상을 분석하고 어종과 횟수를 반환합니다."""
    local_media_path = None
    try:
        local_media_path = download_media_from_s3(request.s3_url)
        
        detected_species_list = analyze_media_server(local_media_path, request.file_type)
        
        if not detected_species_list:
            return {"message": f"{request.file_type}에서 어종을 탐지하지 못했습니다.", "analysis_result": []}
            
        summary = Counter(detected_species_list)
        
        # 최종 결과 JSON 형식으로 변환
        final_result = {
            species: count 
            for species, count in summary.most_common()
        }
        
        print(f"{request.file_type} 처리 완료! 최종 결과를 반환합니다.")
        return {"analysisResult": final_result}

    finally:
        if local_media_path and os.path.exists(local_media_path):
            os.remove(local_media_path)
            print(f">>> 임시 파일 삭제 완료: {local_media_path}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
     
def get_korean_name(scientific_name: str, mapping: dict) -> str:
    """과학적 이름을 한글 이름으로 변환합니다."""
    if scientific_name in mapping:
        return mapping[scientific_name]
    genus = scientific_name.split()[0]
    return mapping.get(genus, scientific_name)

# --- 모델 로딩 ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "dataset": {"path": os.path.join(base_dir, "database.pt")},
        "model": {"path": os.path.join(base_dir, "model.ckpt"), "device": "cpu"}
    }
    classifier = EmbeddingClassifier(config)
    print(">>> 어종 분류기(EmbeddingClassifier) 로드 완료.")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    print(">>> 객체 탐지기(YOLOv5) 로드 완료.")
except Exception as e:
    print(f"[오류] 모델 로딩에 실패했습니다: {e}")
    exit()

# --- 핵심 분석 함수 ---

def process_frame(frame: np.ndarray, save_dir: str, conf_threshold: float, frame_id: str) -> list:
    """단일 프레임을 받아 어종을 탐지하고, 결과를 반환하며, 디버그 이미지를 저장합니다."""
    detected_species = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # YOLO 모델의 신뢰도 임계값을 낮춤
    yolo_model.conf = 0.01  # 매우 낮은 임계값
    yolo_model.iou = 0.7  # NMS IoU 임계값을 높여서 중복 제거 강화
    detections = yolo_model(frame_rgb)
    
    # 중복 탐지 필터링 적용
    raw_detections = detections.xyxy[0].cpu().numpy()
    filtered_detections = filter_overlapping_detections(raw_detections, iou_threshold=0.5)
    
    # 간단한 탐지 결과 출력
    print(f"  [정보] 탐지된 객체 수: {len(filtered_detections)}")
    
    frame_with_boxes = frame.copy()
    detection_count = 0
    classification_attempts = 0
    successful_classifications = 0

    for i, det in enumerate(filtered_detections):
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 신뢰도에 따라 박스 색상 변경 (초록: 통과, 빨강: 미달)
        box_color = (0, 255, 0) if conf >= conf_threshold else (0, 0, 255)
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
        label = f"Conf: {conf:.2f}"
        cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        detection_count += 1
        
        # 설정된 신뢰도 임계값을 넘는 경우에만 분류 진행
        if conf >= conf_threshold:
            fish_crop_img = frame_rgb[y1:y2, x1:x2]
            if fish_crop_img.size == 0:
                continue

            try:
                classification_attempts += 1
                results: list[PredictionResult] = classifier(fish_crop_img)
                if results:
                    best_fish = max(results, key=lambda x: x.accuracy)
                    korean_name = get_korean_name(best_fish.name, keyword_to_korean)
                    
                    # 어종 분류 정확도 임계값을 더 낮춤 (0.25 이상으로 변경)
                    if best_fish.accuracy >= 0.25:
                        successful_classifications += 1
                        if korean_name:
                            detected_species.append(korean_name)
                            print(f"  [탐지 성공] 이름: {korean_name}, 정확도: {best_fish.accuracy:.2f}")
                            # 분류된 이름도 이미지에 추가
                            cv2.putText(frame_with_boxes, korean_name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                        else:
                            # 한글 이름이 없는 경우 과학적 이름 사용
                            detected_species.append(best_fish.name)
                            print(f"  [탐지 성공] 이름: {best_fish.name}, 정확도: {best_fish.accuracy:.2f}")
                            cv2.putText(frame_with_boxes, best_fish.name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    else:
                        print(f"  [탐지 실패] 정확도 부족: {best_fish.name}, 정확도: {best_fish.accuracy:.2f}")
                else:
                    print(f"  [탐지 실패] 분류 결과 없음")

            except Exception as e:
                print(f"  [경고] 어종 분류 중 오류 발생: {e}")

    # 탐지된 객체가 하나라도 있으면 디버그 이미지 저장
    if detection_count > 0 and save_dir:
        save_path = os.path.join(save_dir, f"{frame_id}.jpg")
        cv2.imwrite(save_path, frame_with_boxes)
        print(f"  [정보] 탐지 결과 이미지를 저장했습니다: {save_path}")

    # 통계 정보 출력
    print(f"  [통계] 분류 시도: {classification_attempts}회, 성공: {successful_classifications}회")
    
    # 어종 중복 제거 적용
    detected_species = deduplicate_species_results(detected_species)
    print(f"  [최종] 탐지된 어종 수: {len(detected_species)}마리")

    return detected_species

def analyze_image(image_path: str, save_dir: str, conf_threshold: float) -> list:
    """주어진 이미지 경로를 분석하고 탐지된 어종 리스트를 반환합니다."""
    print(f">>> 이미지 처리 시작: {os.path.basename(image_path)}")
    frame = imread_unicode(image_path)
    if frame is None:
        print(f"[오류] 이미지를 열 수 없습니다: {image_path}")
        return []
    
    
    frame_id = os.path.splitext(os.path.basename(image_path))[0]
    return process_frame(frame, save_dir, conf_threshold, frame_id)

def analyze_video(video_path: str, save_dir: str, conf_threshold: float) -> list:
    """주어진 영상 경로를 분석하고 탐지된 어종 리스트를 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> 영상 처리 시작: {os.path.basename(video_path)} (총 {total_frames} 프레임)")

    all_detected_species = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # 10프레임마다 한 번씩만 처리
        if frame_count % 10 != 0: continue
        
        print(f"--- 프레임 {frame_count}/{total_frames} 처리 중 ---")
        frame_id = f"frame_{frame_count:05d}"
        detected_in_frame = process_frame(frame, save_dir, conf_threshold, frame_id)
        all_detected_species.extend(detected_in_frame)
    
    cap.release()
    return all_detected_species

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="로컬 비디오 또는 이미지 파일에서 어종을 분석합니다.")
    parser.add_argument("input_path", type=str, help="분석할 비디오 또는 이미지 파일의 경로")
    parser.add_argument("--conf", type=float, default=0.1, help="객체 탐지를 위한 최소 신뢰도 (기본값: 0.1)")
    parser.add_argument("--save_dir", type=str, default="detection_results", help="탐지 결과 이미지를 저장할 디렉토리 (기본값: detection_results)")
    args = parser.parse_args()

    # 결과 저장 디렉토리 생성
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f">>> 탐지 결과는 '{args.save_dir}' 폴더에 저장됩니다.")

    # 입력 파일의 확장자 확인 및 존재 여부 검사
    if not os.path.exists(args.input_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {args.input_path}")
        exit()

    file_extension = os.path.splitext(args.input_path)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    detected_species_list = []

    # 확장자에 따라 다른 함수 호출
    if file_extension in image_extensions:
        detected_species_list = analyze_image(args.input_path, args.save_dir, args.conf)
    elif file_extension in video_extensions:
        detected_species_list = analyze_video(args.input_path, args.save_dir, args.conf)
    else:
        print(f"[오류] 지원하지 않는 파일 형식입니다: {file_extension}")
        exit()
    
    # 결과 집계 및 출력
    if not detected_species_list:
        print("\n>>> 최종 결과: 파일에서 어종을 탐지하지 못했습니다.")
    else:
        summary = Counter(detected_species_list)
        print("\n--- 최종 분석 결과 ---")
        for species, count in summary.most_common():
            print(f"- {species}: {count}회 탐지")
        print("--------------------")

    print("분석이 완료되었습니다.")
