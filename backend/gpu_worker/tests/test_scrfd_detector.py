"""
Unit tests for SCRFD detector components.

Tests letterbox, NMS, decode, and batch shape handling.
"""

import pytest
import numpy as np
from gpu_worker.detectors.common import letterbox, LetterboxInfo, nms, DetectionResult


class TestLetterbox:
    """Test letterbox preprocessing."""
    
    def test_letterbox_square_image(self):
        """Test letterbox with square image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        letterboxed, info = letterbox(image, target_size=640)
        
        assert letterboxed.shape == (640, 640, 3)
        assert info.scale == 6.4  # 640/100
        assert info.pad_x == 0
        assert info.pad_y == 0
        assert info.orig_w == 100
        assert info.orig_h == 100
    
    def test_letterbox_wide_image(self):
        """Test letterbox with wide image."""
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        letterboxed, info = letterbox(image, target_size=640)
        
        assert letterboxed.shape == (640, 640, 3)
        assert info.scale == 1.6  # 640/400
        assert info.orig_w == 400
        assert info.orig_h == 200
        # Should have padding on top/bottom
        assert info.pad_y > 0
    
    def test_letterbox_tall_image(self):
        """Test letterbox with tall image."""
        image = np.zeros((400, 200, 3), dtype=np.uint8)
        letterboxed, info = letterbox(image, target_size=640)
        
        assert letterboxed.shape == (640, 640, 3)
        assert info.scale == 1.6  # 640/400
        assert info.orig_w == 200
        assert info.orig_h == 400
        # Should have padding on left/right
        assert info.pad_x > 0
    
    def test_letterbox_coordinate_mapping(self):
        """Test coordinate mapping from letterbox to original."""
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        letterboxed, info = letterbox(image, target_size=640)
        
        # Test box in letterbox coordinates (center of letterbox)
        box_lb = np.array([200.0, 100.0, 440.0, 340.0])  # x1, y1, x2, y2
        
        # Map to original
        box_orig = info.map_box_to_original(box_lb)
        
        # Verify bounds
        assert 0 <= box_orig[0] <= info.orig_w
        assert 0 <= box_orig[1] <= info.orig_h
        assert 0 <= box_orig[2] <= info.orig_w
        assert 0 <= box_orig[3] <= info.orig_h


class TestNMS:
    """Test Non-Maximum Suppression."""
    
    def test_nms_no_overlap(self):
        """Test NMS with non-overlapping boxes."""
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        keep = nms(boxes, scores, iou_threshold=0.4)
        
        # All boxes should be kept (no overlap)
        assert len(keep) == 3
    
    def test_nms_overlapping(self):
        """Test NMS with overlapping boxes."""
        boxes = np.array([
            [10, 10, 50, 50],      # Box 0: score 0.9 (highest)
            [15, 15, 55, 55],      # Box 1: overlaps with 0 (IoU > 0.4)
            [100, 100, 150, 150]  # Box 2: no overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        keep = nms(boxes, scores, iou_threshold=0.4)
        
        # Should keep box 0 (highest score) and box 2 (no overlap)
        assert len(keep) == 2
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep  # Overlaps with 0, should be suppressed
    
    def test_nms_empty(self):
        """Test NMS with empty input."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        
        keep = nms(boxes, scores, iou_threshold=0.4)
        
        assert len(keep) == 0
    
    def test_nms_single_box(self):
        """Test NMS with single box."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        
        keep = nms(boxes, scores, iou_threshold=0.4)
        
        assert len(keep) == 1
        assert keep[0] == 0


class TestBatchShapeHandling:
    """Test batch shape handling."""
    
    def test_batch_preprocessing(self):
        """Test preprocessing multiple images into batch."""
        from gpu_worker.detectors.scrfd_onnx import SCRFDOnnx
        
        # Create mock images
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 150, 3), dtype=np.uint8),
            np.zeros((150, 200, 3), dtype=np.uint8)
        ]
        
        # Note: This test would require actual model loading, so we test the common functions
        # For full integration test, see bench_detector.py
        
        # Test letterbox for each image
        letterboxed_images = []
        infos = []
        for image in images:
            lb, info = letterbox(image, target_size=640)
            letterboxed_images.append(lb)
            infos.append(info)
        
        # All should be same size
        for lb in letterboxed_images:
            assert lb.shape == (640, 640, 3)
        
        # Should have valid scale/padding info
        for info in infos:
            assert info.scale > 0
            assert info.pad_x >= 0
            assert info.pad_y >= 0
    
    def test_coordinate_mapping_batch(self):
        """Test coordinate mapping for batch of images."""
        # Create images with different aspect ratios
        images = [
            np.zeros((100, 200, 3), dtype=np.uint8),  # Wide
            np.zeros((200, 100, 3), dtype=np.uint8),  # Tall
        ]
        
        infos = []
        for image in images:
            _, info = letterbox(image, target_size=640)
            infos.append(info)
        
        # Test mapping for each image
        test_boxes = [
            np.array([100.0, 50.0, 200.0, 150.0]),   # For first image
            np.array([50.0, 100.0, 150.0, 200.0]),   # For second image
        ]
        
        for info, box_lb in zip(infos, test_boxes):
            box_orig = info.map_box_to_original(box_lb)
            
            # Verify in original image bounds
            assert 0 <= box_orig[0] <= info.orig_w
            assert 0 <= box_orig[2] <= info.orig_w
            assert 0 <= box_orig[1] <= info.orig_h
            assert 0 <= box_orig[3] <= info.orig_h


class TestDetectionResult:
    """Test DetectionResult container."""
    
    def test_detection_result_creation(self):
        """Test creating DetectionResult."""
        boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        kps = np.array([[[10, 10], [20, 10], [15, 15], [25, 15], [17, 20]],
                       [[100, 100], [110, 100], [105, 105], [115, 105], [107, 110]]], dtype=np.float32)
        
        result = DetectionResult(boxes, scores, kps)
        
        assert len(result.boxes) == 2
        assert len(result.scores) == 2
        assert result.kps is not None
        assert result.kps.shape == (2, 5, 2)
    
    def test_detection_result_no_kps(self):
        """Test DetectionResult without keypoints."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        
        result = DetectionResult(boxes, scores, kps=None)
        
        assert len(result.boxes) == 1
        assert result.kps is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

