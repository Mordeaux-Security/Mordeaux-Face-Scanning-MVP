import argparse
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.embedder import embed
from pipeline.detector import detect_faces, align_and_crop
from pipeline.indexer import search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tenant", default="demo-tenant")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    arr = np.fromfile(str(Path(args.image)), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    faces = detect_faces(img)
    if not faces:
        print("No faces on query image.")
        return
    crop = align_and_crop(img, faces[0]["landmarks"])
    vec = embed(crop).astype(float).tolist()
    hits = search(vec, top_k=args.topk, tenant_id=args.tenant)
    print(f"hits={len(hits)}")
    for h in hits:
        print(f"{h.id} score={h.score:.3f} payload_site={h.payload.get('site')} url={h.payload.get('url')}")

if __name__ == "__main__":
    main()
