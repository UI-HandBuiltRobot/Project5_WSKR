#!/home/ros_setup/ros2_ws/venv/bin/python
"""OpenCV image orientation testing utility (work in progress).

Utility for testing orientation detection algorithms on images.
Intended for future integration with reinforcement learning.
"""
# LEAVING FOR LATER WILL FIRST INTEGRATE Q LEARNING INTO ROS THEN CHECK THIS


import cv2
import numpy as np

# Load image
img = cv2.imread("test.png")

# ---- 1. Define ROI (x, y, width, height) ----
# Example: object is around this region
x, y, w, h = 100, 150, 200, 200
roi = img[y:y+h, x:x+w]

# ---- 2. Preprocess ROI ----
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Blur helps reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold (adjust if needed)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

# Optional: invert if object is dark
# thresh = cv2.bitwise_not(thresh)

# ---- 3. Find contours in ROI ----
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("No object found in ROI")
    exit()

# Take largest contour
cnt = max(contours, key=cv2.contourArea)

# ---- 4. Compute minAreaRect ----
rect = cv2.minAreaRect(cnt)
(center, (rw, rh), angle) = rect

# Normalize angle
if rw < rh:
    orientation = angle
else:
    orientation = angle + 90

print("Orientation (degrees):", orientation)

# ---- 5. Draw result (on original image) ----
box = cv2.boxPoints(rect)
box = box.astype(int)

# Shift box coordinates back to original image space
box[:, 0] += x
box[:, 1] += y

# Draw bounding box
cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

# Draw ROI rectangle
cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# ---- 6. Show result ----
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()