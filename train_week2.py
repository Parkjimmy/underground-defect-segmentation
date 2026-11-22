from pathlib import Path
from ultralytics import YOLO

# 1. 경로 설정
BASE_DIR = Path(r"C:\Users\pjm73\Desktop\3-3실습")
DATA_DIR = BASE_DIR / "사용한 데이터"
DATA_YAML = BASE_DIR / "data.yaml"

# 2. data.yaml 자동 생성 (없을 경우)
if not DATA_YAML.exists():
    yaml_content = f"""path: {DATA_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: crack
  1: efflorescence
  2: spall_detach
"""
    DATA_YAML.write_text(yaml_content, encoding="utf-8")
    print(f"[생성 완료] {DATA_YAML}")

# 3. 사전학습 모델 로드 (최초 실행 시 yolov8s-seg.pt 자동 다운로드)
model = YOLO("yolov8s-seg.pt")

# 4. 논문 Table 1 설정 그대로 미세조정(Fine-tuning) 학습 시작
if __name__ == "__main__":
    results = model.train(
        data=str(DATA_YAML),
        epochs=60,              # 논문 설정: 60 에폭
        imgsz=960,              # 논문 설정: 입력 크기 960 (letterbox 자동 적용)
        batch=4,                # 논문 설정: 배치 크기 4 (GPU 메모리 부족 시 2로 낮춤)
        optimizer="AdamW",      # 논문 설정: AdamW
        lr0=0.00015,            # 논문 설정: 초기 학습률 0.00015
        mosaic=0.0,             # 논문 전처리 원칙: 공간 맥락 왜곡 방지를 위해 비활성화
        mixup=0.0,              # 논문 전처리 원칙: 결함 경계 훼손 방지를 위해 비활성화
        seed=42,                # 재현성 확보용 시드 고정
        project=str(BASE_DIR / "runs"),
        name="step3",           # 저장 폴더: runs/step3/weights/best.pt
        exist_ok=True
    )
    print("\n[학습 완료] best.pt 저장 위치:", BASE_DIR / "runs" / "step3" / "weights" / "best.pt")