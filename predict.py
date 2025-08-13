import os
import cv2
import torch
import numpy as np
from PIL import Image
from collections import Counter

from inference import EmbeddingClassifier, PredictionResult

keyword_to_korean = {
    # 연어과 (Salmonidae)
    "Salmo": "송어", "Oncorhynchus": "연어", "Salvelinus": "송어",
    # 고등어과 (Scombridae)
    "Scomber": "고등어", "Scomberomorus": "고등어", "Thunnus": "참치", "Katsuwonus": "가다랑어", "Euthynnus": "가다랑어", "Auxis": "가다랑어",
    # 도미과 (Sparidae)
    "Pagrus": "도미", "Sparus": "도미", "Dentex": "도미",
    # 넙치과 (Pleuronectidae)
    "Paralichthys": "광어", "Pleuronectes": "넙치", "Platichthys": "넙치",
    # 농어과 (Moronidae)
    "Morone": "농어",
    # 메기과 (Ictaluridae)
    "Ictalurus": "메기", "Ameiurus": "메기",
    # 메기목 (Siluriformes)
    "Clarias": "메기",
    # 잉어과 (Cyprinidae)
    "Cyprinus": "잉어", "Carassius": "잉어", "Barbus": "잉어", "Leuciscus": "잉어", "Rutilus": "잉어", "Gobio": "잉어", "Pimephales": "잉어",
    # 청어과 (Clupeidae)
    "Clupea": "청어", "Sprattus": "청어", "Alosa": "청어", "Brevoortia": "청어",
    # 농어목 (Perciformes)
    "Epinephelus": "농어", "Mycteroperca": "농어", "Cephalopholis": "농어", "Serranus": "농어",
    # 쏨뱅이과 (Scorpaenidae)
    "Sebastes": "쏨뱅이",
    # 가자미과 (Paralichthyidae)
    "Paralichthys": "가자미",
    # 뱀장어과 (Anguillidae)
    "Anguilla": "뱀장어",
    # 상어류 (Selachimorpha)
    "Carcharhinus": "상어", "Sphyrna": "상어", "Galeocerdo": "상어", "Prionace": "상어", "Rhincodon": "상어",
    # 가오리과 (Dasyatidae)
    "Dasyatis": "가오리", "Myliobatis": "가오리", "Aetobatus": "가오리",
    # 복어과 (Tetraodontidae)
    "Lagocephalus": "복어", "Takifugu": "복어", "Tetraodon": "복어",
    # 송사리과 (Atherinopsidae)
    "Menidia": "송사리",
    # 열대어류 (Cichlidae)
    "Oreochromis": "틸라피아", "Cichla": "피라니아", "Astronotus": "피라니아", "Pelmatolapia": "틸라피아", "Heros": "피라니아",
    # 기타
    "Lutjanus": "돔류", "Pomadasys": "농어", "Trachinotus": "농어", "Caranx": "망치", "Lepisosteus": "가시고기", "Micropterus": "배스", "Gambusia": "구피", "Poecilia": "구피",
}

def get_korean_name(scientific_name: str, mapping: dict) -> str:
    """학명에서 키워드를 찾아 한글명으로 변환합니다."""
    # 학명은 보통 'Genus species' 형식이므로 첫 단어(속명)를 키워드로 사용
    genus = scientific_name.split()[0]
    if genus in mapping:
        return mapping[genus]
    return scientific_name  # 매칭되는 키워드가 없으면 학명 그대로 반환

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    print(">>> 초기 설정 시작...")

    # 1. 어종 분류기(EmbeddingClassifier) 초기화
    config = {
        "dataset": {"path": os.path.join(os.path.dirname(__file__), "database.pt")},
        "model": {"path": os.path.join(os.path.dirname(__file__), "model.ckpt"), "device": "cpu"}
    }
    try:
        classifier = EmbeddingClassifier(config)
        print(">>> 어종 분류기(EmbeddingClassifier) 로드 완료.")
    except Exception as e:
        print(f"[오류] 어종 분류기 로딩에 실패했습니다: {e}")
        exit()

    # 2. 객체 탐지기(YOLOv5) 초기화
    try:
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        print(">>> 객체 탐지기(YOLOv5) 로드 완료.")
    except Exception as e:
        print(f"[오류] YOLOv5 모델 로딩에 실패했습니다. 인터넷 연결을 확인해주세요: {e}")
        exit()

    # 3. 처리할 영상 파일 설정
    video_filename = "bycatch_video.mp4" # <<< 여기에 분석하고 싶은 영상 파일 이름을 넣으세요.
    video_path = os.path.join(os.path.dirname(__file__), video_filename)
    if not os.path.exists(video_path):
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {video_path}")
        exit()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> 영상 처리 시작: {video_filename} (총 {total_frames} 프레임)")

    all_detected_species = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # 10프레임마다 한 번씩만 처리하여 속도 향상
        if frame_count % 10 != 0:
            continue
        
        print(f"--- 프레임 {frame_count}/{total_frames} 처리 중 ---")

        # OpenCV의 BGR을 YOLO가 기대하는 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 객체 탐지 수행
        detections = yolo_model(frame_rgb)
        
        # 탐지된 객체들의 정보를 순회
        # detections.xyxy[0] -> [x1, y1, x2, y2, confidence, class_id]
        for *box, conf, cls in detections.xyxy[0]:
            # confidence가 0.4 이상인 객체만 처리
            if conf > 0.4:
                x1, y1, x2, y2 = map(int, box)

                # 탐지된 객체(물고기)의 이미지만 잘라내기
                fish_crop_img = frame_rgb[y1:y2, x1:x2]

                if fish_crop_img.size == 0:
                    continue

                # 어종 분류기(EmbeddingClassifier)로 예측 수행
                try:
                    results: list[PredictionResult] = classifier(fish_crop_img)
                    
                    if results:
                        # 여러 예측 결과 중 가장 정확도가 높은 것을 선택
                        best_fish = max(results, key=lambda x: x.accuracy)
                        korean_name = get_korean_name(best_fish.name, keyword_to_korean)
                        
                        print(f"  [탐지 완료] 어종: {korean_name} (정확도: {best_fish.accuracy:.2f})")
                        all_detected_species.append(korean_name)

                except Exception as e:
                    print(f"  [경고] 어종 분류 중 오류 발생: {e}")

    # --- 최종 결과 집계 ---
    cap.release()
    print("\n\n" + "="*50)
    print("✅ 영상 처리 완료! 최종 결과를 요약합니다.")
    print("="*50)

    if not all_detected_species:
        print("영상에서 어종을 탐지하지 못했습니다.")
    else:
        summary = Counter(all_detected_species)
        print("【 혼획물 영상 어종 분류 최종 요약 】")
        for species, count in summary.most_common():
            print(f"  ▶ {species}: {count} 회 탐지됨")