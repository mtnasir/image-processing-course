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

cv2.imshow("Original Image", frame_bgr)

# ========== 2. Convert BGR (OpenCV default) to RGB ==========
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
img = frame_rgb.astype(np.float32)
h, w, _ = img.shape

# Reshape image to N x 3 (each row is z = [R,G,B])
Z = img.reshape(-1, 3)   # shape: (N, 3)

# ========== 3. Define target color m and C^{-1} ==========
# Mean red color (you MUST tune this for your data)
m = np.array([200.0, 40.0, 40.0], dtype=np.float32)  # example "red" in RGB

# For Euclidean distance: C = I  =>  C^{-1} = I
C_inv = np.eye(3, dtype=np.float32)

# ========== 4. Compute distance D(z, m) ==========
# D(z, m) = sqrt( (z - m)^T C^{-1} (z - m) )
Z_minus_m = Z - m            # shape: (N, 3)

# (z - m)^T C^{-1} (z - m) implemented for all pixels:
# First: (z - m) * C^{-1}  -> (N x 3)
temp = Z_minus_m @ C_inv

# Then elementwise multiply with (z - m) and sum over channels
dist_sq = np.sum(temp * Z_minus_m, axis=1)   # shape: (N,)
D = np.sqrt(dist_sq)                         # Euclidean distance because C=I

# Reshape back to image
D_img = D.reshape(h, w)

# ========== 5. Threshold distance to create red mask ==========
# Small distance -> color close to m (red); larger distance -> not red.
distance_threshold = 80.0   # you must adjust this
red_mask = (D_img < distance_threshold).astype(np.uint8) * 255

# Optional: clean noise with morphology
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=1)

cv2.imshow("Red Mask (Euclidean distance)", red_mask)

# ========== 6. Apply mask to original image ==========
red_segment_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=red_mask)
cv2.imshow("Segmented Red Regions", red_segment_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()