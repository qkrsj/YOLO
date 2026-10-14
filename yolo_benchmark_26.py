"""
YOLO 버전별 웹캠 성능 비교 벤치마크 (v2)
=====================================
측정 지표:
  1) 인식 시간 : 물체를 카메라에 보여준 순간부터 모델이 처음 인식(신뢰도
                threshold 통과)할 때까지 걸린 시간(초)
  2) 정확도    : 여러 번 시도 중 실제로 인식에 성공한 비율(성공률) +
                인식했을 때의 평균 신뢰도(confidence)

동작 방식:
  - 모델마다 TRIALS_PER_MODEL 회(기본 5회) 반복 시도합니다.
  - 매 시도마다: 물체를 카메라 밖에 숨기고 Enter를 누른 뒤, 곧바로
    물체를 카메라 앞에 보여주세요. 그 순간부터 시간을 잽니다.
  - TIMEOUT_SECONDS(기본 5초) 안에 인식하지 못하면 실패로 기록합니다.

사용 전 설치 (Windows cmd 또는 PowerShell):
    python -m venv yolo_test
    yolo_test\\Scripts\\activate
    pip install ultralytics opencv-python

실행:
    python yolo_benchmark_v2.py
"""

import cv2
import time
import csv
from datetime import datetime

from ultralytics import YOLO

# ---------------------------------------------------------------------------
MODELS = {
    "YOLO26n": "yolo26n.pt",
    "YOLO26s": "yolo26s.pt",
    "YOLO26m": "yolo26m.pt",
    "YOLO26l": "yolo26l.pt",
    "YOLO26x": "yolo26x.pt",
}

TRIALS_PER_MODEL = 15    # 모델당 시도 횟수 (많을수록 결과가 안정적)
TIMEOUT_SECONDS = 8.0    # 이 시간 안에 인식 못 하면 실패 처리
CONF_THRESHOLD = 0.5     # 이 신뢰도 이상만 "인식 성공"으로 인정
# ---------------------------------------------------------------------------


def run_single_trial(model: YOLO, cap: cv2.VideoCapture) -> dict:
    """물체가 보여진 순간부터 인식될 때까지의 시간을 측정하는 단일 시도"""
    start = time.perf_counter()

    while time.perf_counter() - start < TIMEOUT_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        boxes = r.boxes

        # 화면에 실시간으로 보여주기 (진행 상황 확인용)
        annotated = r.plot()
        elapsed_live = time.perf_counter() - start
        cv2.putText(annotated, f"searching... {elapsed_live:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO Recognition Test", annotated)
        cv2.waitKey(1)

        if len(boxes) > 0:
            elapsed = time.perf_counter() - start
            best_idx = int(boxes.conf.argmax())
            confidence = float(boxes.conf[best_idx])
            cls_id = int(boxes.cls[best_idx])
            class_name = model.names[cls_id]

            # 인식 성공 프레임을 잠깐 더 보여줌
            cv2.putText(annotated, f"DETECTED: {class_name} ({confidence:.2f}) in {elapsed:.2f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("YOLO Recognition Test", annotated)
            cv2.waitKey(600)

            return {"success": True, "time_sec": round(elapsed, 3),
                    "confidence": round(confidence, 3), "class": class_name}

    return {"success": False, "time_sec": None, "confidence": None, "class": None}


def benchmark_model(model_name: str, model_path: str, cap: cv2.VideoCapture) -> dict:
    print(f"\n=== {model_name} 테스트 시작 ===")
    model = YOLO(model_path)  # 최초 실행 시 자동 다운로드

    # 워밍업
    ret, frame = cap.read()
    if ret:
        model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    trial_results = []
    for t in range(1, TRIALS_PER_MODEL + 1):
        input(f"\n  [{model_name}] 시도 {t}/{TRIALS_PER_MODEL} - "
              f"물체를 카메라 밖에 숨기고 준비되면 Enter, 그 즉시 물체를 카메라에 보여주세요")
        result = run_single_trial(model, cap)
        trial_results.append(result)
        if result["success"]:
            print(f"    -> 성공: {result['class']} 인식, {result['time_sec']}초, "
                  f"신뢰도 {result['confidence']}")
        else:
            print(f"    -> 실패: {TIMEOUT_SECONDS}초 안에 인식 못 함")

    successes = [r for r in trial_results if r["success"]]
    success_rate = len(successes) / len(trial_results)
    avg_time = (sum(r["time_sec"] for r in successes) / len(successes)) if successes else None
    avg_conf = (sum(r["confidence"] for r in successes) / len(successes)) if successes else None

    return {
        "model": model_name,
        "success_rate": round(success_rate, 2),
        "avg_recognition_time_sec": round(avg_time, 3) if avg_time else None,
        "avg_confidence": round(avg_conf, 3) if avg_conf else None,
        "trials": TRIALS_PER_MODEL,
    }


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 연결/권한을 확인하세요.")
        return

    results_summary = []

    for name, path in MODELS.items():
        input(f"\n[{name}] 모델 테스트를 시작하려면 Enter를 누르세요.")
        try:
            result = benchmark_model(name, path, cap)
            results_summary.append(result)
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

    print("\n=== 최종 비교 (인식 시간 빠른 순) ===")
    ranked = sorted(
        [r for r in results_summary if r["avg_recognition_time_sec"] is not None],
        key=lambda x: x["avg_recognition_time_sec"],
    )
    for r in ranked:
        print(f"{r['model']:10s} | 성공률: {r['success_rate']*100:.0f}% | "
              f"평균 인식시간: {r['avg_recognition_time_sec']}초 | "
              f"평균 신뢰도: {r['avg_confidence']}")


if __name__ == "__main__":
    main()