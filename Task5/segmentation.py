import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Global Thresholding ----------
def global_thresholding(image, method='otsu'):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    m = method.lower()
    if m == 'otsu':
        thresh_val, binary = cv2.threshold(image, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, thresh_val
    if m == 'binary':
        thresh_val = 127
        _, binary = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
        return binary, thresh_val
    if m == 'adaptive_mean':
        binary = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return binary, "Adaptive Mean"
    if m == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return binary, "Adaptive Gaussian"
    raise ValueError("Method must be 'otsu', 'binary', 'adaptive_mean', or 'adaptive_gaussian'")

# ---------- Multi-level Thresholding ----------
def multi_level_thresholding(image, num_levels=3):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()

    peaks = [i for i in range(1, 255) if hist_norm[i-1] < hist_norm[i] > hist_norm[i+1]]
    peak_heights = sorted([(p, hist_norm[p]) for p in peaks],
                          key=lambda x: x[1], reverse=True)
    selected_peaks = sorted([p for p, _ in peak_heights[:num_levels-1]])

    segmented = np.zeros_like(image)
    thresholds = [0] + selected_peaks + [255]
    for i in range(len(thresholds) - 1):
        mask = (image >= thresholds[i]) & (image < thresholds[i+1])
        segmented[mask] = int(255 * i / (len(thresholds) - 1))
    return segmented, selected_peaks

# ---------- Region Growing ----------
def region_growing(image, seed_points, threshold=10):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)

    def neighbors(x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    yield nx, ny

    region_id = 1
    for seed in seed_points:
        if not (isinstance(seed, tuple) and len(seed) == 2):
            continue
        x, y = seed
        if not (0 <= x < w and 0 <= y < h):
            continue
        stack = [(x, y)]
        seed_val = image[y, x]
        while stack:
            cx, cy = stack.pop()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            if abs(int(image[cy, cx]) - int(seed_val)) <= threshold:
                segmented[cy, cx] = region_id * 50
                for nx, ny in neighbors(cx, cy):
                    if not visited[ny, nx]:
                        stack.append((nx, ny))
        region_id += 1
    return segmented

# ---------- Watershed ----------
def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if image.ndim == 3:
        markers = cv2.watershed(image, markers)
    else:
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    return markers

# ---------- K-means ----------
def kmeans_segmentation(image, k=3):
    Z = image.reshape((-1, 3)) if image.ndim == 3 else image.reshape((-1, 1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    if image.ndim == 3:
        segmented = segmented.reshape(image.shape)
        labels_img = labels.reshape((image.shape[0], image.shape[1]))
    else:
        segmented = segmented.reshape((image.shape[0], image.shape[1]))
        labels_img = labels.reshape((image.shape[0], image.shape[1]))
    return segmented, labels_img

# ---------- Edges ----------
def sobel_edge_detection(image, threshold=None):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    direction = np.arctan2(sy, sx)
    mag_uint8 = np.uint8(255 * mag / np.max(mag))
    if threshold is not None:
        _, edges = cv2.threshold(mag_uint8, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, edges = cv2.threshold(mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges, mag_uint8, direction

def canny_edge_detection(image, low_threshold=None, high_threshold=None):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if low_threshold is None or high_threshold is None:
        median = np.median(image)
        low_threshold = max(0, (1.0 - 0.33) * median)
        high_threshold = min(255, (1.0 + 0.33) * median)
    edges = cv2.Canny(image, int(low_threshold), int(high_threshold))
    return edges, (low_threshold, high_threshold)

def laplacian_edge_detection(image, threshold=None):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap_abs = np.uint8(np.absolute(lap))
    if threshold is not None:
        _, edges = cv2.threshold(lap_abs, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, edges = cv2.threshold(lap_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges, lap_abs

# ---------- Hough Lines ----------
def hough_line_detection(edge_image, min_line_length=50, max_line_gap=10):
    if edge_image.ndim == 3:
        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(edge_image, 1, np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def detect_lines_with_hough(image, edge_method='canny', min_line_length=50):
    if edge_method.lower() == 'canny':
        edges, _ = canny_edge_detection(image)
    elif edge_method.lower() == 'sobel':
        edges, _, _ = sobel_edge_detection(image)
    elif edge_method.lower() == 'laplacian':
        edges, _ = laplacian_edge_detection(image)
    else:
        raise ValueError("edge_method must be 'canny', 'sobel', or 'laplacian'")

    lines = hough_line_detection(edges, min_line_length=min_line_length)
    line_params = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                angle = np.arctan(slope) * 180 / np.pi
            else:
                slope = float('inf')
                intercept = x1
                angle = 90
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_params.append({
                'points': (x1, y1, x2, y2),
                'slope': slope,
                'intercept': intercept,
                'angle': angle,
                'length': length
            })
    return edges, lines, line_params

def visualize_lines(image, lines, line_params=None):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)
            if line_params and i < len(line_params):
                angle = line_params[i]['angle']
                cv2.putText(vis, f"{angle:.1f}°", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return vis

# ---------- Test Image ----------
def create_test_image():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (200, 100), 40, (128, 128, 128), -1)
    cv2.line(img, (0, 0), (300, 300), (255, 255, 255), 3)
    cv2.line(img, (300, 0), (0, 300), (255, 255, 255), 3)
    return img

# ---------- Example Usage ----------
if __name__ == "__main__":
    img = create_test_image()

    print("Global thresholding (Otsu):")
    binary_otsu, t = global_thresholding(img, 'otsu')
    print("Threshold:", t)

    print("Canny edges:")
    canny_edges, th = canny_edge_detection(img, 50, 150)
    print("Thresholds:", th)

    print("Sobel edges:")
    sobel_edges, mag, _ = sobel_edge_detection(img)

    print("Laplacian edges:")
    lap_edges, lap_abs = laplacian_edge_detection(img)

    print("Hough lines:")
    edges, lines, params = detect_lines_with_hough(img, 'canny')
    print("Lines found:", 0 if lines is None else len(lines))

    print("Region growing:")
    seeds = [(75, 75), (200, 110)]
    reg = region_growing(img, seeds, threshold=30)

    print("Watershed:")
    ws = watershed_segmentation(img)
    print("Segments:", len(np.unique(ws)))

    print("K-means:")
    kseg, labels = kmeans_segmentation(img, k=3)
    print("Clusters:", len(np.unique(labels)))

    # Quick visualization examples
    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0,0].set_title("Original"); ax[0,0].axis('off')
    ax[0,1].imshow(binary_otsu, cmap='gray'); ax[0,1].set_title("Otsu"); ax[0,1].axis('off')
    ax[0,2].imshow(canny_edges, cmap='gray'); ax[0,2].set_title("Canny"); ax[0,2].axis('off')
    ax[1,0].imshow(sobel_edges, cmap='gray'); ax[1,0].set_title("Sobel"); ax[1,0].axis('off')
    ax[1,1].imshow(lap_edges, cmap='gray'); ax[1,1].set_title("Laplacian"); ax[1,1].axis('off')
    ax[1,2].imshow(kseg if kseg.ndim == 2 else cv2.cvtColor(kseg, cv2.COLOR_BGR2RGB))
    ax[1,2].set_title("K-means"); ax[1,2].axis('off')
    plt.tight_layout()
    plt.show()