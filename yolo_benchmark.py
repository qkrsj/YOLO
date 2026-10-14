"""
YOLO 버전별 웹캠 성능 비교 벤치마크
=====================================
측정 지표:
  1) 속도  : 프레임당 추론 시간(ms), FPS
  2) 정확도(proxy): 평균 탐지 신뢰도(confidence), 평균 탐지 개수
     * 웹캠 실시간 테스트에는 정답(ground truth) 라벨이 없기 때문에
       엄밀한 mAP 계산은 불가능합니다. 대신 "얼마나 자신있게, 얼마나
       빠짐없이 탐지하는가"를 정확도의 대리 지표로 사용합니다.
     * 더 엄밀한 정확도가 필요하면 스크립트 하단의 '참고' 항목을 확인하세요.

사용 전 설치 (Windows cmd 또는 PowerShell):
    python -m venv yolo_test
    yolo_test\\Scripts\\activate
    pip install ultralytics opencv-python

실행:
    python yolo_benchmark.py

주의:
  - 각 모델은 최초 실행 시 자동으로 가중치(.pt)를 다운로드합니다. 인터넷 연결 필요.
  - 테스트 중 'q'를 누르면 해당 모델 테스트를 조기 종료하고 다음 모델로 넘어갑니다.
"""

import cv2
import time
import csv
from datetime import datetime

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 비교할 모델 목록 (CPU에서도 그나마 돌아가는 경량 모델 위주로 구성)
# 필요하면 이 딕셔너리만 수정해서 원하는 조합으로 바꿀 수 있습니다.
# ---------------------------------------------------------------------------
MODELS = {
    "YOLOv5n":  "yolov5nu.pt",
    "YOLOv8n":  "yolov8n.pt",
    "YOLOv9t":  "yolov9t.pt",
    "YOLOv10n": "yolov10n.pt",
    "YOLO11n":  "yolo11n.pt",
}

FRAMES_PER_MODEL = 100   # 모델당 측정할 프레임 수 (늘릴수록 결과가 안정적이지만 시간이 오래 걸림)
CONF_THRESHOLD = 0.4     # 이 신뢰도 이상만 탐지로 인정


def benchmark_model(model_name: str, model_path: str, cap: cv2.VideoCapture) -> dict:
    print(f"\n=== {model_name} 테스트 시작 ===")
    model = YOLO(model_path)  # 최초 실행 시 자동 다운로드

    # 워밍업 (첫 추론은 그래프 컴파일 등으로 느리므로 측정에서 제외)
    ret, frame = cap.read()
    if ret:
        model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    times, confidences, detection_counts = [], [], []
    frame_count = 0

    while frame_count < FRAMES_PER_MODEL:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.perf_counter()
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        elapsed = time.perf_counter() - start

        r = results[0]
        boxes = r.boxes
        n_det = len(boxes)
        avg_conf = float(boxes.conf.mean()) if n_det > 0 else 0.0

        times.append(elapsed)
        confidences.append(avg_conf)
        detection_counts.append(n_det)

        annotated = r.plot()
        cv2.putText(
            annotated,
            f"{model_name} | {1/elapsed:.1f} FPS | frame {frame_count+1}/{FRAMES_PER_MODEL}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("YOLO Benchmark (press q to skip to next model)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    avg_time = sum(times) / len(times)
    return {
        "model": model_name,
        "avg_inference_ms": round(avg_time * 1000, 2),
        "avg_fps": round(1 / avg_time, 2),
        "avg_confidence": round(sum(confidences) / len(confidences), 3),
        "avg_detections_per_frame": round(sum(detection_counts) / len(detection_counts), 2),
        "frames_tested": len(times),
    }


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 연결/권한을 확인하세요.")
        return

    results_summary = []

    for name, path in MODELS.items():
        input(f"\n[{name}] 테스트를 시작하려면 Enter를 누르세요. "
              f"(카메라 앞에 테스트할 물체를 준비해주세요. 매 모델마다 같은 물체/구도를 유지하면 비교가 공정해집니다)")
        try:
            result = benchmark_model(name, path, cap)
            results_summary.append(result)
            print(f"{name} 결과: {result}")
        except Exception as e:
            print(f"{name} 테스트 중 오류 발생, 건너뜁니다: {e}")

    cap.release()
    cv2.destroyAllWindows()

    if not results_summary:
        print("측정된 결과가 없습니다.")
        return

    filename = f"yolo_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
        writer.writeheader()
        writer.writerows(results_summary)
    print(f"\n결과가 '{filename}' 에 저장되었습니다.")

    print("\n=== 최종 비교 (FPS 높은 순) ===")
    for r in sorted(results_summary, key=lambda x: x["avg_fps"], reverse=True):
        print(
            f"{r['model']:10s} | FPS: {r['avg_fps']:.2f} | 추론시간: {r['avg_inference_ms']:.1f}ms | "
            f"평균신뢰도: {r['avg_confidence']:.3f} | 평균탐지수: {r['avg_detections_per_frame']:.2f}"
        )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 참고: 더 엄밀한 정확도(mAP) 비교가 필요하다면
# ---------------------------------------------------------------------------
# 실시간 웹캠 테스트는 정답 라벨이 없어 "진짜 정확도"를 계산할 수 없습니다.
# 대신 다음 방법을 고려할 수 있습니다:
#   1) 고정된 테스트 이미지 세트(예: 특정 물체 10장)를 만들고, 각 모델의
#      탐지 결과를 사람이 직접 맞음/틀림으로 채점 (정성적이지만 실용적)
#   2) COCO128 같은 공개 라벨링 데이터셋으로 model.val()을 돌려 mAP50-95를
#      비교 (다만 이건 '내 방 물체'가 아닌 COCO 클래스 기준 정확도)
# ---------------------------------------------------------------------------