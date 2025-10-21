#!/usr/bin/env python3
"""
Model Warm-Cache Script

Pre-downloads and caches InsightFace models to /models/insightface
for faster subsequent runs in Docker containers.

Usage:
    python scripts/warm_models.py
"""
import sys
import os
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def warm_models():
    """Warm the model cache by loading detector and embedder."""
    print("🔥 Warming model cache...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Import and load models
        from pipeline.detector import load_detector
        from pipeline.embedder import load_model
        
        print("📥 Loading detector...")
        detector = load_detector()
        print("✅ Detector loaded successfully")
        
        print("📥 Loading embedder...")
        embedder = load_model()
        print("✅ Embedder loaded successfully")
        
        # Check model cache directory
        cache_dir = Path.home() / ".insightface" / "models"
        if cache_dir.exists():
            print(f"📁 Model cache directory: {cache_dir}")
            model_files = list(cache_dir.rglob("*.onnx"))
            print(f"📦 Cached models: {len(model_files)} files")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"   - {model_file.name}: {size_mb:.1f} MB")
        
        elapsed = time.time() - start_time
        print(f"\n🎉 Models warmed successfully in {elapsed:.2f}s")
        print("🚀 Subsequent runs will be instant!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error warming models: {e}")
        return False

if __name__ == "__main__":
    success = warm_models()
    sys.exit(0 if success else 1)
