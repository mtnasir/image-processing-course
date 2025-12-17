import cv2
import numpy as np

# ========== 1. Acquire Image ==========

# OPTION A: From webcam (single frame)
cap = cv2.VideoCapture(0)   # 0 = default camera
ret, frame_bgr = cap.read()
cap.release()

if not ret:
    print("Failed to capture image from webcam.")
    raise SystemExit

# OPTION B: From image file
# frame_bgr = cv2.imread('your_image.jpg')

# Show original (BGR, as OpenCV expects)
cv2.imshow("Original Image", frame_bgr)

# ========== 2. Convert BGR (OpenCV default) to RGB ==========
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# Split into R, G, B channels
R = frame_rgb[:, :, 0]
G = frame_rgb[:, :, 1]
B = frame_rgb[:, :, 2]

# ========== 3. Threshold for RED in RGB ==========
# Condition: R is high, G and B are relatively low.
# You can adjust these thresholds based on your image.
red_mask = (R > 100) & (G < 60) & (B < 60)

# Convert boolean mask to uint8 (0 or 255)
red_mask = red_mask.astype(np.uint8) * 255

# Optional: clean noise with morphology
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=1)

cv2.imshow("Red Mask (Binary)", red_mask)

# ========== 4. Apply mask to original image ==========
# Use the mask on the original BGR frame (for correct display colors in OpenCV)
red_segment_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=red_mask)

cv2.imshow("Segmented Red Regions", red_segment_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()