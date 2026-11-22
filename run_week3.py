from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE_DIR = Path(r"C:\Users\pjm73\Desktop\3-3실습")
DATA_DIR = BASE_DIR / "사용한 데이터"
DATA_YAML = BASE_DIR / "data.yaml"
BEST_PT = BASE_DIR / "runs" / "step3" / "weights" / "best.pt"

OUTPUT_DIR = BASE_DIR / "week3_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["crack", "efflorescence", "spall_detach"]
THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]

def load_split_data(split_name):
    img_dir = DATA_DIR / "images" / split_name
    lbl_dir = DATA_DIR / "labels" / split_name
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    gts = []
    for img_p in img_paths:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        cls_set = set()
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if parts:
                    cls_set.add(int(parts[0]))
        gts.append(sorted(list(cls_set)))
    return [str(p) for p in img_paths], gts

def run_inference_cache(model, paths):
    preds = []
    for p in paths:
        r = model.predict(p, imgsz=960, conf=0.0005, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            conf = r.boxes.conf.cpu().numpy()
            cid = r.boxes.cls.cpu().numpy().astype(int)
        else:
            conf = np.array([])
            cid = np.array([], dtype=int)
        preds.append((cid, conf))
    return preds

def eval_onoff(preds, gts, thr, cid_target):
    tp = fp = fn = tn = 0
    for (cid, conf), gt in zip(preds, gts):
        pred_on = bool(((cid == cid_target) & (conf >= thr)).any())
        gt_on = cid_target in gt
        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1

    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    model = YOLO(str(BEST_PT))
    val_imgs, val_gts = load_split_data("val")
    test_imgs, test_gts = load_split_data("test")

    # 1. 테스트셋 세그멘테이션 성능 평가 및 Confusion Matrix 생성
    print("=" * 65)
    print("[1] 테스트셋 세그멘테이션 성능 평가 (mask mAP & Confusion Matrix)")
    print("=" * 65)
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=960,
        plots=True,
        project=str(OUTPUT_DIR),
        name="seg_test_eval",
        exist_ok=True
    )
    print(f"전체 mask mAP50    : {results.seg.map50:.4f}")
    print(f"전체 mask mAP50-95 : {results.seg.map:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f" - {name:14s} | mask mAP50: {results.seg.ap50[i]:.4f} | mask mAP50-95: {results.seg.maps[i]:.4f}")
    print()

    # 2. 검증셋(val) 기반 클래스별 최적 임계값 탐색
    print("=" * 65)
    print("[2] 검증셋(val) 기반 클래스별 최적 임계값 탐색")
    print("=" * 65)
    val_preds = run_inference_cache(model, val_imgs)
    best_thrs = {}
    all_scores = {}

    for cid, name in enumerate(CLASS_NAMES):
        scores = [eval_onoff(val_preds, val_gts, t, cid)["f1"] for t in THRS]
        all_scores[name] = scores
        max_f1 = max(scores)
        best_idx = next(i for i, s in enumerate(scores) if s == max_f1)
        best_thrs[name] = THRS[best_idx]
        print(f">> [{name:14s}] 선택 임계값: {best_thrs[name]} (Val F1={max_f1:.3f})")

    plt.figure(figsize=(8, 5))
    for name in CLASS_NAMES:
        plt.plot(THRS, all_scores[name], marker="o", label=f"{name} (best={best_thrs[name]})")
    plt.xscale("log")
    plt.xticks(THRS, [str(t) for t in THRS])
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Validation F1-score")
    plt.title("Validation ON/OFF F1 by Threshold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "val_threshold_f1.png", dpi=150)
    plt.close()

    # 3. 테스트셋 성능 비교 (기본 단일 임계값 vs 검증셋 기반 클래스별 임계값)
    print("\n" + "=" * 65)
    print("[3] 테스트셋(test) 성능 비교 (README 2번 항목용)")
    print("=" * 65)
    test_preds = run_inference_cache(model, test_imgs)

    def summarize_strategy(label, thr_dict):
        rows = []
        for cid, name in enumerate(CLASS_NAMES):
            r = eval_onoff(test_preds, test_gts, thr_dict[name], cid)
            rows.append([name, thr_dict[name], r["tp"], r["fp"], r["fn"], r["tn"], r["accuracy"], r["precision"], r["recall"], r["f1"]])
        df = pd.DataFrame(rows, columns=["class", "thr", "TP", "FP", "FN", "TN", "Accuracy", "Precision", "Recall", "F1"])
        print(f"\n--- {label} ---")
        print(df.round(3).to_string(index=False))
        print(f"Macro 평균 -> Accuracy: {df['Accuracy'].mean():.3f} | Precision: {df['Precision'].mean():.3f} | Recall: {df['Recall'].mean():.3f} | F1-score: {df['F1'].mean():.3f}")
        return df

    df_base25 = summarize_strategy("전략 A: 기본 단일 임계값 (conf=0.25)", {n: 0.25 for n in CLASS_NAMES})
    df_base10 = summarize_strategy("전략 B: 낮은 단일 임계값 (conf=0.10)", {n: 0.10 for n in CLASS_NAMES})
    df_opt    = summarize_strategy("전략 C: 검증셋 기반 클래스별 최적 임계값 (제안 방식)", best_thrs)
    df_opt.round(4).to_csv(OUTPUT_DIR / "test_onoff_results.csv", index=False, encoding="utf-8-sig")

    # 4. 클래스별 대표 FP / FN 오류 사례 추출
    error_dir = OUTPUT_DIR / "error_cases"
    error_dir.mkdir(exist_ok=True)
    for cid, name in enumerate(CLASS_NAMES):
        thr = best_thrs[name]
        fp_imgs = [p for p, gt, (p_cls, conf) in zip(test_imgs, test_gts, test_preds) if ((p_cls == cid) & (conf >= thr)).any() and (cid not in gt)]
        fn_imgs = [p for p, gt, (p_cls, conf) in zip(test_imgs, test_gts, test_preds) if not ((p_cls == cid) & (conf >= thr)).any() and (cid in gt)]
        cls_dir = error_dir / name
        cls_dir.mkdir(exist_ok=True)
        for err_type, paths in [("FP", fp_imgs), ("FN", fn_imgs)]:
            for i, p in enumerate(paths[:2]):
                r = model.predict(p, imgsz=960, conf=thr, verbose=False)[0]
                out_path = cls_dir / f"{err_type}_{i+1}_{Path(p).name}"
                ok, buf = cv2.imencode(out_path.suffix, r.plot())
                if ok:
                    buf.tofile(str(out_path))

    print(f"\n[완료] README 작성에 필요한 모든 결과와 Confusion Matrix가 '{OUTPUT_DIR}'에 저장되었습니다.")