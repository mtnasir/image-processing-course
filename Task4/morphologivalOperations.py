import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image (using OpenCV's built-in sample)
# If you don't have this image, you can download it or use another image
image_path = cv2.data.haarcascades.replace('haarcascade_frontalface_default.xml', '') + '../samples/data/lena.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# If the above doesn't work, create a sample image
if img is None:
    # Create a sample binary image with shapes
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (100, 100), 255, -1)  # White rectangle
    cv2.circle(img, (200, 150), 30, 255, -1)           # White circle
    cv2.rectangle(img, (150, 200), (250, 250), 255, 2) # Rectangle outline

# Binarize the image
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Define structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# 1. Erosion
erosion = cv2.erode(binary_img, kernel, iterations=1)

# 2. Dilation
dilation = cv2.dilate(binary_img, kernel, iterations=1)

# 3. Opening
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

# 4. Closing
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# 5. Hit-or-Miss Transform
# Define kernels for hit-or-miss
kernel_h = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.int8)
kernel_m = np.array([[1, 0, 1],
                     [0, 0, 0],
                     [1, 0, 1]], dtype=np.int8)

# Apply hit-or-miss transform
hit_miss = cv2.morphologyEx(binary_img, cv2.MORPH_HITMISS, kernel_h)

# 6. Boundary Extraction
# Erode the image and subtract from original
eroded_for_boundary = cv2.erode(binary_img, kernel, iterations=1)
boundary = cv2.subtract(binary_img, eroded_for_boundary)

# 7. Hole Filling
# Invert the image
inverted_img = cv2.bitwise_not(binary_img)
# Create a mask for flood fill (must be larger than source image)
h, w = inverted_img.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
# Flood fill from point (0, 0)
flood_filled = inverted_img.copy()
cv2.floodFill(flood_filled, mask, (0,0), 255)
# Invert again to get filled holes
holes_filled = cv2.bitwise_not(flood_filled)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.imshow(binary_img, cmap='gray')
plt.title('Original Binary Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(hit_miss, cmap='gray')
plt.title('Hit-or-Miss')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(boundary, cmap='gray')
plt.title('Boundary Extraction')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(holes_filled, cmap='gray')
plt.title('Hole Filling')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print kernel information
print("Structuring Element (Kernel):")
print(kernel)