import numpy as np
import cv2

# Create a simple test image with a face-like pattern
img = np.ones((300, 300, 3), dtype=np.uint8) * 128  # Gray background

# Draw a simple face-like pattern
# Face outline
cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (200, 180, 160), -1)

# Eyes
cv2.circle(img, (130, 130), 8, (0, 0, 0), -1)
cv2.circle(img, (170, 130), 8, (0, 0, 0), -1)

# Nose
cv2.ellipse(img, (150, 150), (3, 8), 0, 0, 360, (180, 160, 140), -1)

# Mouth
cv2.ellipse(img, (150, 180), (15, 8), 0, 0, 180, (0, 0, 0), 2)

# Save the test image
cv2.imwrite('samples/face1.jpg', img)
print("Created test image: samples/face1.jpg")
