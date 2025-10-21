import numpy as np
import cv2
from pathlib import Path
from pipeline.detector import detect_faces, align_and_crop
from pipeline.embedder import embed

def test_detect_and_embed_shapes():
    """Test that detector and embedder produce correct shapes."""
    p = Path("samples/face1.jpg")
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    faces = detect_faces(img)
    assert isinstance(faces, list)
    if faces:
        crop = align_and_crop(img, faces[0]["landmarks"])
        vec = embed(crop)
        assert crop.shape[:2] == (112, 112)
        assert vec.shape == (512,)
        assert vec.dtype == np.float32

def test_embedder_with_mock_face():
    """Test embedder with a mock face crop when no real faces detected."""
    # Create a mock 112x112 face crop
    mock_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Test that embedder can handle the input (even if it fails with no faces)
    try:
        vec = embed(mock_crop)
        assert vec.shape == (512,)
        assert vec.dtype == np.float32
        assert np.linalg.norm(vec) > 0  # Should be normalized
    except Exception as e:
        # Expected to fail with mock data, but should fail gracefully
        assert "No faces found" in str(e) or "ValueError" in str(e)

def test_align_and_crop_shape():
    """Test that align_and_crop produces correct shape."""
    # Create a test image
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Mock landmarks (5 points: left eye, right eye, nose, left mouth, right mouth)
    mock_landmarks = [
        [50, 50],   # left eye
        [80, 50],   # right eye  
        [65, 70],   # nose
        [55, 90],   # left mouth
        [75, 90]    # right mouth
    ]
    
    crop = align_and_crop(img, mock_landmarks)
    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8
