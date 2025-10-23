import argparse, json, time
from pathlib import Path
import numpy as np, cv2
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import detect_faces, align_and_crop
from pipeline.embedder import embed

def imread_bgr(p: Path):
    arr = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Folder with images")
    ap.add_argument("--limit", type=int, default=0, help="Max images (0=all)")
    args = ap.parse_args()

    folder = Path(args.path)
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
    if args.limit > 0:
        imgs = imgs[:args.limit]

    totals = {
        "images": 0,
        "images_failed_decode": 0,
        "faces_detected": 0,
        "faces_embedded": 0,
        "timings_ms": {"decode":0.0,"detect":0.0,"align":0.0,"embed":0.0},
        "per_image": []
    }

    t0_all = time.perf_counter()
    for p in imgs:
        t0 = time.perf_counter()
        img = imread_bgr(p)
        t1 = time.perf_counter()
        if img is None:
            totals["images_failed_decode"] += 1
            continue
        totals["images"] += 1
        decode_ms = (t1 - t0)*1000

        t2 = time.perf_counter()
        faces = detect_faces(img)
        t3 = time.perf_counter()

        det_ms = (t3 - t2)*1000
        totals["faces_detected"] += len(faces)

        align_ms_acc = 0.0
        embed_ms_acc = 0.0
        embedded_count = 0
        for f in faces:
            lmk = f.get("landmarks") or []
            if len(lmk) < 5: 
                continue
            t4 = time.perf_counter()
            crop = align_and_crop(img, lmk)
            t5 = time.perf_counter()
            feat = embed(crop)
            t6 = time.perf_counter()
            align_ms_acc += (t5 - t4)*1000
            embed_ms_acc += (t6 - t5)*1000
            embedded_count += 1

        totals["faces_embedded"] += embedded_count
        totals["timings_ms"]["decode"] += decode_ms
        totals["timings_ms"]["detect"] += det_ms
        totals["timings_ms"]["align"]  += align_ms_acc
        totals["timings_ms"]["embed"]  += embed_ms_acc

        totals["per_image"].append({
            "file": p.name,
            "faces": len(faces),
            "embedded": embedded_count,
            "decode_ms": round(decode_ms,2),
            "detect_ms": round(det_ms,2),
            "align_ms": round(align_ms_acc,2),
            "embed_ms": round(embed_ms_acc,2),
        })

    elapsed_ms = (time.perf_counter()-t0_all)*1000
    n = max(totals["images"], 1)
    report = {
        "summary": {
            "images": totals["images"],
            "failed_decode": totals["images_failed_decode"],
            "faces_detected": totals["faces_detected"],
            "faces_embedded": totals["faces_embedded"],
            "avg_ms_per_image": {
                "decode": round(totals["timings_ms"]["decode"]/n,2),
                "detect": round(totals["timings_ms"]["detect"]/n,2),
                "align":  round(totals["timings_ms"]["align"]/n,2),
                "embed":  round(totals["timings_ms"]["embed"]/n,2),
            },
            "total_elapsed_ms": round(elapsed_ms,2)
        },
        "details": totals["per_image"]
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
