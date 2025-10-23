import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.processor import process_image
from pipeline.storage import ensure_buckets, put_bytes
from config.settings import settings

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def imread_bytes(path: Path) -> bytes:
    # read raw file bytes (works for any image)
    return path.read_bytes()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--tenant", default="demo-tenant")
    ap.add_argument("--site", default="local-folder")
    args = ap.parse_args()

    ensure_buckets()

    folder = Path(args.path)
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])

    results = []
    for p in tqdm(imgs, desc="processing"):
        raw = imread_bytes(p)
        sha = sha256_bytes(raw)

        # Optionally stage the raw image in MinIO to mimic crawler handoff
        raw_key = f"{args.tenant}/{sha}.jpg"
        # decode once to standardize to jpg
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            put_bytes(settings.MINIO_BUCKET_RAW, raw_key, enc.tobytes(), "image/jpeg")
        else:
            continue

        message = {
            "image_sha256": sha,
            "bucket": settings.MINIO_BUCKET_RAW,
            "key": raw_key,
            "tenant_id": args.tenant,
            "site": args.site,
            "url": f"https://example.com/{p.name}",
            "image_phash": "0"*16,
            "face_hints": None,
        }

        out = process_image(message)  # expects to load from MinIO and do full pipeline
        results.append(out)

    print(json.dumps({"count": len(results), "items": results[:5]}, indent=2))

if __name__ == "__main__":
    main()
