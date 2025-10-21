import sys
import numpy as np
import cv2
from pathlib import Path
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import detect_faces, align_and_crop
from pipeline.embedder import embed

def main(img_path: str):
    p = Path(img_path)
    if not p.exists():
        print(f"File not found: {img_path}")
        sys.exit(2)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        print("cv2.imdecode failed")
        sys.exit(3)

    faces = detect_faces(img)
    print(f"Detected faces: {len(faces)}")
    for i, f in enumerate(faces):
        crop = align_and_crop(img, f["landmarks"])
        vec = embed(crop)
        print(f"[{i}] crop {crop.shape} embed {vec.shape} || norm={np.linalg.norm(vec):.4f} score={f['confidence']:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_models.py /path/to/face.jpg")
        sys.exit(1)
    main(sys.argv[1])
