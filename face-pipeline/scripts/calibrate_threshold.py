#!/usr/bin/env python3
"""
Threshold Calibration Script

Computes optimal similarity threshold by analyzing positive and negative face pairs
from sample images. Uses naming convention to identify pairs automatically.

Usage:
    python calibrate_threshold.py --samples-dir samples/
    python calibrate_threshold.py --samples-dir /path/to/samples --help

Sample Naming Convention:
  - Positive pairs (same person): person1_a.jpg, person1_b.jpg
  - Negative pairs (different people): person1_a.jpg, person2_a.jpg
  - Extract prefix before underscore/dot to determine pairs

Output: JSON report with similarity statistics and suggested thresholds.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import detect_faces, align_and_crop
from pipeline.embedder import embed
from config.settings import settings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized vectors.
    
    Args:
        vec1: First embedding vector (512-dim, L2-normalized)
        vec2: Second embedding vector (512-dim, L2-normalized)
        
    Returns:
        Cosine similarity score (0-1, higher = more similar)
    """
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    return float(np.dot(vec1, vec2))


def extract_prefix(filename: str) -> str:
    """
    Extract prefix from filename for grouping pairs.
    
    Examples:
        "person1_a.jpg" -> "person1"
        "person2_b.png" -> "person2"
        "face_test.jpg" -> "face_test"
    """
    # Remove extension
    name = Path(filename).stem
    
    # Split by underscore and take the part before the last underscore
    # This handles cases like "person1_a", "person2_b", etc.
    parts = name.split('_')
    if len(parts) > 1:
        # Join all parts except the last one
        return '_'.join(parts[:-1])
    else:
        # No underscore, return the whole name
        return name


def load_and_embed_image(image_path: Path) -> Tuple[str, np.ndarray]:
    """
    Load image, detect faces, align, and embed the first face.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (filename, embedding_vector) or (filename, None) if no face found
    """
    try:
        # Load image as BGR
        import cv2
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"Warning: Could not load image {image_path}", file=sys.stderr)
            return image_path.name, None
        
        # Detect faces
        faces = detect_faces(img_bgr)
        if not faces:
            print(f"Warning: No faces detected in {image_path}", file=sys.stderr)
            return image_path.name, None
        
        # Align and crop first face
        face_data = faces[0]
        landmarks = face_data.get("landmarks", [])
        if len(landmarks) < 5:
            print(f"Warning: Insufficient landmarks in {image_path}", file=sys.stderr)
            return image_path.name, None
        
        crop_bgr = align_and_crop(img_bgr, landmarks, image_size=settings.IMAGE_SIZE)
        
        # Generate embedding
        embedding = embed(crop_bgr)
        
        return image_path.name, embedding
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return image_path.name, None


def compute_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of similarity scores."""
    if not scores:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    scores_array = np.array(scores)
    return {
        "count": len(scores),
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array))
    }


def main():
    """Main entry point for threshold calibration script."""
    parser = argparse.ArgumentParser(
        description="Calibrate similarity threshold using sample face pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sample Naming Convention:
  Positive pairs (same person): person1_a.jpg, person1_b.jpg
  Negative pairs (different people): person1_a.jpg, person2_a.jpg
  
Examples:
  python calibrate_threshold.py --samples-dir samples/
  python calibrate_threshold.py --samples-dir /path/to/samples
        """
    )
    
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="samples/",
        help="Directory containing sample images (default: samples/)"
    )
    
    args = parser.parse_args()
    
    # Validate samples directory
    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory {samples_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not samples_dir.is_dir():
        print(f"Error: {samples_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [
        f for f in samples_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"Error: No image files found in {samples_dir}", file=sys.stderr)
        print(f"Supported extensions: {', '.join(image_extensions)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files in {samples_dir}", file=sys.stderr)
    
    # Process all images and extract embeddings
    embeddings = {}
    faces_embedded = 0
    
    for image_file in image_files:
        filename, embedding = load_and_embed_image(image_file)
        if embedding is not None:
            embeddings[filename] = embedding
            faces_embedded += 1
        else:
            print(f"Skipping {filename} (no face detected)", file=sys.stderr)
    
    if faces_embedded < 2:
        print("Error: Need at least 2 faces to compute similarities", file=sys.stderr)
        sys.exit(1)
    
    print(f"Successfully embedded {faces_embedded} faces", file=sys.stderr)
    
    # Group images by prefix
    prefix_groups = defaultdict(list)
    for filename in embeddings.keys():
        prefix = extract_prefix(filename)
        prefix_groups[prefix].append(filename)
    
    print(f"Found {len(prefix_groups)} person groups", file=sys.stderr)
    
    # Compute positive pairs (same prefix)
    positive_scores = []
    for prefix, filenames in prefix_groups.items():
        if len(filenames) >= 2:
            # Compute all pairwise similarities within this group
            for i in range(len(filenames)):
                for j in range(i + 1, len(filenames)):
                    vec1 = embeddings[filenames[i]]
                    vec2 = embeddings[filenames[j]]
                    similarity = cosine_similarity(vec1, vec2)
                    positive_scores.append(similarity)
    
    # Compute negative pairs (different prefixes)
    negative_scores = []
    prefixes = list(prefix_groups.keys())
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            # Compare all images from group i with all images from group j
            for filename1 in prefix_groups[prefixes[i]]:
                for filename2 in prefix_groups[prefixes[j]]:
                    vec1 = embeddings[filename1]
                    vec2 = embeddings[filename2]
                    similarity = cosine_similarity(vec1, vec2)
                    negative_scores.append(similarity)
    
    # Compute statistics
    positive_stats = compute_statistics(positive_scores)
    negative_stats = compute_statistics(negative_scores)
    
    # Suggest thresholds
    pos_mean = positive_stats["mean"]
    pos_std = positive_stats["std"]
    neg_mean = negative_stats["mean"]
    neg_std = negative_stats["std"]
    
    suggested_thresholds = {
        "conservative": max(0.0, pos_mean - 2 * pos_std),
        "balanced": (pos_mean + neg_mean) / 2,
        "aggressive": min(1.0, neg_mean + 2 * neg_std)
    }
    
    # Choose recommendation
    if positive_stats["count"] > 0 and negative_stats["count"] > 0:
        if pos_mean > neg_mean + 2 * (pos_std + neg_std):
            # Good separation, use balanced threshold
            recommended_threshold = suggested_thresholds["balanced"]
            recommendation = f"Use balanced threshold of {recommended_threshold:.3f} for general use"
        else:
            # Poor separation, be more conservative
            recommended_threshold = suggested_thresholds["conservative"]
            recommendation = f"Use conservative threshold of {recommended_threshold:.3f} due to poor separation"
    else:
        recommended_threshold = 0.5
        recommendation = "Insufficient data for reliable threshold recommendation"
    
    # Build final report
    report = {
        "samples_analyzed": len(image_files),
        "faces_embedded": faces_embedded,
        "person_groups": len(prefix_groups),
        "positive_pairs": positive_stats,
        "negative_pairs": negative_stats,
        "suggested_thresholds": suggested_thresholds,
        "recommendation": recommendation
    }
    
    # Output JSON report
    print(json.dumps(report, indent=2))
    
    # Print summary to stderr for visibility
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Samples analyzed: {len(image_files)}", file=sys.stderr)
    print(f"  Faces embedded: {faces_embedded}", file=sys.stderr)
    print(f"  Person groups: {len(prefix_groups)}", file=sys.stderr)
    print(f"  Positive pairs: {positive_stats['count']}", file=sys.stderr)
    print(f"  Negative pairs: {negative_stats['count']}", file=sys.stderr)
    if positive_stats['count'] > 0:
        print(f"  Positive mean: {positive_stats['mean']:.3f}", file=sys.stderr)
    if negative_stats['count'] > 0:
        print(f"  Negative mean: {negative_stats['mean']:.3f}", file=sys.stderr)
    print(f"  Recommendation: {recommendation}", file=sys.stderr)


if __name__ == "__main__":
    main()
