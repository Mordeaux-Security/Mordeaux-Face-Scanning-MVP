#!/usr/bin/env python3
"""
Model warming script for Docker build
Pre-downloads InsightFace models during container build
"""
import os
import sys

# Set environment variables
os.environ['ONNX_PROVIDERS_CSV'] = 'CPUExecutionProvider'
os.environ['DET_SCORE_THRESH'] = '0.20'
os.environ['DET_SIZE'] = '1280,1280'

try:
    print('Loading models...')
    from pipeline.detector import load_detector
    from pipeline.embedder import load_model
    
    load_detector()
    load_model()
    print('✅ Models warmed successfully')
except Exception as e:
    print(f'⚠️  Model warming failed: {e}')
    print('Models will be downloaded on first run')
    sys.exit(0)  # Don't fail the build