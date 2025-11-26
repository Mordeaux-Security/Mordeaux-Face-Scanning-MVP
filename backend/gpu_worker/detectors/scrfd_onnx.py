"""
SCRFD (Scalable CNN-based Real-time Face Detector) ONNX Implementation

Provides batched face detection using ONNX Runtime with DirectML support.
"""

import os
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

from .common import letterbox, LetterboxInfo, nms, DetectionResult

logger = logging.getLogger(__name__)


class SCRFDOnnx:
    """SCRFD face detector with batched ONNX inference."""
    
    def __init__(self, face_app: Optional[FaceAnalysis] = None, 
                 onnx_path: Optional[str] = None,
                 provider: str = "DmlExecutionProvider",
                 device_id: int = 0,
                 input_size: int = 1024,
                 score_thr: float = 0.65,
                 nms_iou: float = 0.4):
        """
        Initialize SCRFD detector.
        
        Args:
            face_app: InsightFace FaceAnalysis instance (used to access detection model)
            onnx_path: Optional path to ONNX model file (overrides face_app model)
            provider: Execution provider (DmlExecutionProvider or CPUExecutionProvider)
            device_id: Device ID for execution provider
            input_size: Input size for detection (default 1024 for better small face detection)
            score_thr: Score threshold for filtering detections (default 0.65 for higher precision)
            nms_iou: IoU threshold for NMS
        """
        self.input_size = input_size
        self.score_thr = score_thr
        self.nms_iou = nms_iou
        self.provider = provider
        
        # Get ONNX session from InsightFace model or load from path
        if onnx_path and os.path.exists(onnx_path):
            self._load_from_path(onnx_path, provider, device_id)
        elif face_app is not None:
            self._load_from_face_app(face_app, provider, device_id)
        else:
            raise ValueError("Either face_app or onnx_path must be provided")
        
        # Verify batch dimension is dynamic
        self._verify_dynamic_batch()
        
        # Precompute anchors for SCRFD decode
        self._precompute_anchors()
    
    def _load_from_path(self, onnx_path: str, provider: str, device_id: int):
        """Load ONNX model from file path."""
        logger.info(f"Loading SCRFD model from path: {onnx_path}")
        
        # Get available providers
        available_providers = ort.get_available_providers()
        if provider not in available_providers:
            logger.warning(f"Provider {provider} not available, using CPU")
            provider = "CPUExecutionProvider"
        
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        providers = [provider]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info(f"SCRFD model loaded with providers: {self.session.get_providers()}")
    
    def _load_from_face_app(self, face_app: FaceAnalysis, provider: str, device_id: int):
        """Load ONNX session from InsightFace FaceAnalysis model."""
        logger.info("Accessing SCRFD model from InsightFace FaceAnalysis")
        
        # Access detection model from FaceAnalysis
        if not hasattr(face_app, 'models') or 'detection' not in face_app.models:
            raise ValueError("FaceAnalysis does not have 'detection' model")
        
        detector_model = face_app.models['detection']
        
        if not hasattr(detector_model, 'session') or detector_model.session is None:
            raise ValueError("Detection model does not have ONNX session")
        
        # Use the existing session (already configured with DirectML if available)
        self.session = detector_model.session
        
        logger.info(f"SCRFD model accessed from FaceAnalysis with providers: {self.session.get_providers()}")
    
    def _verify_dynamic_batch(self):
        """Verify that model input supports dynamic batch dimension."""
        inputs = self.session.get_inputs()
        if len(inputs) == 0:
            raise ValueError("Model has no inputs")
        
        input_shape = inputs[0].shape
        if len(input_shape) < 4:
            raise ValueError(f"Expected input shape (N,3,H,W), got {input_shape}")
        
        # First dimension should be dynamic (-1 or 'batch' or None)
        batch_dim = input_shape[0]
        if batch_dim not in (-1, 'batch', None, 'N'):
            if isinstance(batch_dim, int) and batch_dim == 1:
                logger.warning(f"Model has fixed batch=1 dimension. Batching will not work correctly.")
                logger.warning("Consider exporting model with dynamic batch dimension.")
            else:
                logger.warning(f"Unexpected batch dimension: {batch_dim}. May cause issues with batching.")
    
    def _precompute_anchors(self):
        """Precompute anchor grids for SCRFD decode."""
        # SCRFD uses multi-scale feature maps
        # Common scales: [8, 16, 32] for input_size=640
        # Each scale has different anchor configurations
        
        # For SCRFD, we'll compute anchors on-the-fly during decode
        # Store stride values and base anchor sizes
        self.strides = [8, 16, 32]  # Common strides for SCRFD-10G
        self.base_sizes = [16, 32, 64]  # Base anchor sizes
        
        logger.debug(f"SCRFD anchors precomputed: strides={self.strides}, base_sizes={self.base_sizes}")
    
    def preprocess(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[LetterboxInfo]]:
        """
        Preprocess images into batched tensor.
        
        Args:
            images: List of images (H, W, 3) in BGR format
        
        Returns:
            Tuple of (batch_tensor (N,3,H,W), letterbox_infos)
        """
        batch_tensors = []
        letterbox_infos = []
        
        for image in images:
            # Letterbox to input_size
            letterboxed, info = letterbox(image, target_size=self.input_size)
            
            # Convert BGR to RGB and normalize
            rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] and convert to float32
            normalized = rgb.astype(np.float32) / 255.0
            
            # Convert HWC to CHW
            chw = np.transpose(normalized, (2, 0, 1))
            
            batch_tensors.append(chw)
            letterbox_infos.append(info)
        
        # Stack into batch (N, 3, H, W)
        batch = np.stack(batch_tensors, axis=0)
        
        return batch.astype(np.float32), letterbox_infos
    
    def infer(self, batch_nchw: np.ndarray) -> List[np.ndarray]:
        """
        Run batched forward pass.
        
        Args:
            batch_nchw: Batched images (N, 3, H, W) in float32
        
        Returns:
            List of output arrays from ONNX model (one per output)
        """
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: batch_nchw})
        
        return outputs
    
    def decode(self, outputs: List[np.ndarray], batch_size: int) -> List[DetectionResult]:
        """
        Decode SCRFD outputs to bounding boxes, scores, and keypoints.
        
        Args:
            outputs: Model outputs (varies by SCRFD variant)
            batch_size: Number of images in batch
        
        Returns:
            List of DetectionResult (one per image)
        """
        # SCRFD output format varies by model variant
        # Common format: [bbox_pred, kps_pred, score_pred] or combined outputs
        # For buffalo_l SCRFD, we need to handle multi-stride feature maps
        
        # Try to infer output format
        if len(outputs) >= 3:
            # Likely separate outputs for bbox, kps, score
            bbox_outputs = outputs[0]  # (N, H, W, 4) or (N, H*W, 4)
            score_outputs = outputs[1]  # (N, H, W, 1) or (N, H*W, 1)
            kps_outputs = outputs[2] if len(outputs) > 2 else None  # (N, H, W, 10) or (N, H*W, 10)
        else:
            # Combined output or different format
            # For now, fallback: try to use InsightFace's built-in decode
            logger.warning("Unrecognized SCRFD output format, using fallback decode")
            return self._decode_fallback(outputs, batch_size)
        
        results = []
        for i in range(batch_size):
            boxes, scores, kps = self._decode_single_image(
                bbox_outputs[i] if len(bbox_outputs.shape) > 2 else bbox_outputs,
                score_outputs[i] if len(score_outputs.shape) > 2 else score_outputs,
                kps_outputs[i] if kps_outputs is not None and len(kps_outputs.shape) > 2 else kps_outputs
            )
            
            # Apply score threshold
            mask = scores >= self.score_thr
            boxes = boxes[mask]
            scores = scores[mask]
            kps = kps[mask] if kps is not None else None
            
            # Apply NMS
            if len(boxes) > 0:
                keep = nms(boxes, scores, self.nms_iou)
                boxes = boxes[keep]
                scores = scores[keep]
                kps = kps[keep] if kps is not None else None
            
            results.append(DetectionResult(boxes, scores, kps))
        
        return results
    
    def _decode_single_image(self, bbox_output: np.ndarray, 
                            score_output: np.ndarray,
                            kps_output: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Decode outputs for a single image.
        
        This is a simplified decode - SCRFD actual decode is more complex with
        multi-stride feature maps and anchor generation.
        """
        # Flatten spatial dimensions if needed
        if len(bbox_output.shape) == 3:  # (H, W, 4)
            h, w, _ = bbox_output.shape
            bbox_output = bbox_output.reshape(-1, 4)
            score_output = score_output.reshape(-1)
            if kps_output is not None:
                kps_output = kps_output.reshape(-1, 10) if len(kps_output.shape) == 3 else kps_output
        
        # Extract boxes and scores
        boxes = bbox_output.copy()
        scores = score_output.copy()
        
        # Convert boxes from center+size to xyxy if needed
        # SCRFD typically outputs (cx, cy, w, h) - convert to (x1, y1, x2, y2)
        if boxes.shape[1] == 4:
            # Assume format is (cx, cy, w, h) and convert
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Reshape keypoints if present
        kps = None
        if kps_output is not None:
            # Keypoints are typically (x1, y1, x2, y2, ..., x5, y5) = 10 values
            kps = kps_output.reshape(-1, 5, 2)
        
        return boxes, scores, kps
    
    def _decode_fallback(self, outputs: List[np.ndarray], batch_size: int) -> List[DetectionResult]:
        """
        Fallback decode method when output format is unclear.
        
        Returns empty results - caller should fallback to single-image processing.
        """
        logger.warning("Using fallback decode (returning empty results)")
        return [DetectionResult(
            boxes=np.array([], dtype=np.float32).reshape(0, 4),
            scores=np.array([], dtype=np.float32),
            kps=None
        ) for _ in range(batch_size)]
    
    def process_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        Process a batch of images end-to-end.
        
        Args:
            images: List of images (H, W, 3) in BGR format
        
        Returns:
            List of DetectionResult (one per image)
        """
        import time
        
        if len(images) == 0:
            return []
        
        total_start = time.perf_counter()
        
        # Preprocess (CPU: letterbox, normalization, stacking)
        t_prep_start = time.perf_counter()
        batch_nchw, letterbox_infos = self.preprocess(images)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000
        
        # GPU Inference (ONNX forward pass)
        t_infer_start = time.perf_counter()
        outputs = self.infer(batch_nchw)
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000
        
        # Decode (CPU: decode outputs, NMS)
        t_decode_start = time.perf_counter()
        results = self.decode(outputs, len(images))
        t_decode_ms = (time.perf_counter() - t_decode_start) * 1000
        
        # Map boxes back to original coordinates (CPU: coordinate transformation)
        t_map_start = time.perf_counter()
        for i, (result, info) in enumerate(zip(results, letterbox_infos)):
            if len(result.boxes) > 0:
                # Map boxes from letterbox to original
                mapped_boxes = []
                for box in result.boxes:
                    mapped_box = info.map_box_to_original(box)
                    mapped_boxes.append(mapped_box)
                result.boxes = np.array(mapped_boxes, dtype=np.float32)
                
                # Map keypoints if present
                if result.kps is not None:
                    result.kps = info.map_kps_to_original(result.kps)
        t_map_ms = (time.perf_counter() - t_map_start) * 1000
        
        total_ms = (time.perf_counter() - total_start) * 1000
        
        # Log detailed timing breakdown (INFO level for visibility)
        pct_prep = (t_prep_ms / total_ms * 100) if total_ms > 0 else 0
        pct_gpu = (t_infer_ms / total_ms * 100) if total_ms > 0 else 0
        pct_decode = (t_decode_ms / total_ms * 100) if total_ms > 0 else 0
        pct_map = (t_map_ms / total_ms * 100) if total_ms > 0 else 0
        
        logger.info(f"[SCRFD-TIMING] batch_size={len(images)}, "
                   f"total={total_ms:.2f}ms (100%) | "
                   f"preprocess={t_prep_ms:.2f}ms ({pct_prep:.1f}%) | "
                   f"gpu_infer={t_infer_ms:.2f}ms ({pct_gpu:.1f}%) | "
                   f"decode={t_decode_ms:.2f}ms ({pct_decode:.1f}%) | "
                   f"coordinate_map={t_map_ms:.2f}ms ({pct_map:.1f}%) | "
                   f"cpu_total={t_prep_ms+t_decode_ms+t_map_ms:.2f}ms | "
                   f"gpu_vs_cpu={t_infer_ms/(t_prep_ms+t_decode_ms+t_map_ms+1e-6)*100:.1f}%")
        
        return results

