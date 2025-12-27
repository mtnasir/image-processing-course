import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageSegmentationOpenCV:
    """
    Image Segmentation and Line Detection using OpenCV only
    No scikit-image dependency required
    """
    def __init__(self):
        pass
    
    # ==================== GLOBAL SEGMENTATION METHODS ====================
    
    def global_thresholding(self, image, method='otsu'):
        """
        Global thresholding methods using OpenCV
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method.lower() == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        elif method.lower() == 'binary':
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            threshold = 127
        elif method.lower() == 'adaptive_mean':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            threshold = "Adaptive Mean"
        elif method.lower() == 'adaptive_gaussian':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            threshold = "Adaptive Gaussian"
        else:
            raise ValueError("Method must be 'otsu', 'binary', 'adaptive_mean', or 'adaptive_gaussian'")
        
        return binary, threshold
    
    def multi_level_thresholding(self, image, num_levels=3):
        """
        Multi-level thresholding using OpenCV only
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.max()
        
        # Find peaks and valleys
        peaks = []
        for i in range(1, 255):
            if hist_norm[i-1] < hist_norm[i] > hist_norm[i+1]:
                peaks.append(i)
        
        # Sort peaks by height and select top num_levels-1
        peak_heights = [(peak, hist_norm[peak]) for peak in peaks]
        peak_heights.sort(key=lambda x: x[1], reverse=True)
        selected_peaks = [peak for peak, _ in peak_heights[:num_levels-1]]
        selected_peaks.sort()
        
        # Create segmented image
        segmented = np.zeros_like(image)
        thresholds = [0] + selected_peaks + [255]
        
        for i in range(len(thresholds)-1):
            mask = (image >= thresholds[i]) & (image < thresholds[i+1])
            segmented[mask] = int(255 * i / (len(thresholds)-1))
        
        return segmented, selected_peaks
    
    # ==================== LOCAL SEGMENTATION METHODS ====================
    
    def region_growing(self, image, seed_points, threshold=10):
        """
        Region growing segmentation using OpenCV only
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        height, width = image.shape
        segmented = np.zeros_like(image)
        visited = np.zeros_like(image, dtype=bool)
        
        def get_neighbors(x, y):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors.append((nx, ny))
            return neighbors
        
        region_id = 1
        for seed in seed_points:
            if isinstance(seed, tuple) and len(seed) == 2:
                x, y = seed
                if 0 <= x < width and 0 <= y < height:
                    stack = [(x, y)]
                    seed_value = image[y, x]
                    
                    while stack:
                        cx, cy = stack.pop()
                        if visited[cy, cx]:
                            continue
                        
                        visited[cy, cx] = True
                        if abs(int(image[cy, cx]) - int(seed_value)) <= threshold:
                            segmented[cy, cx] = region_id * 50
                            
                            for nx, ny in get_neighbors(cx, cy):
                                if not visited[ny, nx]:
                                    stack.append((nx, ny))
                
                region_id += 1
        
        return segmented
    
    def watershed_segmentation(self, image):
        """
        Watershed segmentation using OpenCV only
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            # Convert grayscale to BGR for watershed
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(image_bgr, markers)
        
        return markers
    
    def kmeans_segmentation(self, image, k=3):
        
        # Reshape image to be a list of pixels
        if len(image.shape) == 3:
            Z = image.reshape((-1, 3))
        else:
            Z = image.reshape((-1, 1))
        
        # Convert to np.float32
        Z = np.float32(Z)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        
        # Reshape back to the original image
        if len(image.shape) == 3:
            segmented = segmented.reshape((image.shape))
        else:
            segmented = segmented.reshape((image.shape[0], image.shape[1]))
        
        return segmented, labels.reshape((image.shape[0], image.shape[1]))
    
    # ==================== LINE SEGMENTATION METHODS ====================
    
    def sobel_edge_detection(self, image, threshold=None):
        """
        Sobel edge detection using OpenCV
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sobel operators
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        # Normalize to 0-255 range
        gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
        
        # Apply threshold if provided
        if threshold is not None:
            _, edges = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
        else:
            # Automatic threshold using Otsu's method
            _, edges = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return edges, gradient_magnitude, gradient_direction
    
    def canny_edge_detection(self, image, low_threshold=None, high_threshold=None):
        """
        Canny edge detection using OpenCV
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Automatic threshold calculation if not provided
        if low_threshold is None or high_threshold is None:
            # Calculate median and use it for automatic thresholding
            median = np.median(image)
            low_threshold = max(0, (1.0 - 0.33) * median)
            high_threshold = min(255, (1.0 + 0.33) * median)
        
        edges = cv2.Canny(image, int(low_threshold), int(high_threshold))
        return edges, (low_threshold, high_threshold)
    
    def laplacian_edge_detection(self, image, threshold=None):
        """
        Laplacian edge detection (alternative to advanced Sobel)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Convert back to uint8
        laplacian_abs = np.uint8(np.absolute(laplacian))
        
        # Apply threshold if provided
        if threshold is not None:
            _, edges = cv2.threshold(laplacian_abs, threshold, 255, cv2.THRESH_BINARY)
        else:
            # Automatic threshold using Otsu's method
            _, edges = cv2.threshold(laplacian_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return edges, laplacian_abs
    
    # ==================== HOUGH TRANSFORM FOR LINE DETECTION ====================
    
    def hough_line_detection(self, edge_image, min_line_length=50, max_line_gap=10):
        """
        Hough Transform for line detection using OpenCV
        """
        if len(edge_image.shape) == 3:
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
        
        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edge_image, 1, np.pi/180, threshold=50, 
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        return lines
    
    def detect_lines_with_hough(self, image, edge_method='canny', min_line_length=50):
        """
        Complete line detection pipeline using OpenCV only
        """
        # Step 1: Edge detection
        if edge_method.lower() == 'canny':
            edges, _ = self.canny_edge_detection(image)
        elif edge_method.lower() == 'sobel':
            edges, _, _ = self.sobel_edge_detection(image)
        elif edge_method.lower() == 'laplacian':
            edges, _ = self.laplacian_edge_detection(image)
        else:
            raise ValueError("edge_method must be 'canny', 'sobel', or 'laplacian'")
        
        # Step 2: Hough Transform
        lines = self.hough_line_detection(edges, min_line_length=min_line_length)
        
        # Step 3: Extract line parameters
        line_params = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line parameters
                if x2 - x1 != 0:
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
    
    def visualize_lines(self, image, lines, line_params=None):
        """
        Visualize detected lines on the original image
        """
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
                cv2.line(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add line information if available
                if line_params and i < len(line_params):
                    angle = line_params[i]['angle']
                    cv2.putText(vis_image, f"{angle:.1f}°", (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return vis_image


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Create sample image for testing
    def create_test_image():
        # Create a simple geometric test image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img, (200, 100), 40, (128, 128, 128), -1)
        cv2.line(img, (0, 0), (300, 300), (255, 255, 255), 3)
        cv2.line(img, (300, 0), (0, 300), (255, 255, 255), 3)
        return img
    
    # Initialize segmentation class
    seg = ImageSegmentationOpenCV()
    
    # Create test image
    test_image = create_test_image()
    imshow = lambda title, img: cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    print("=== TESTING OPENCV-ONLY SEGMENTATION METHODS ===")
    
    # Test global thresholding
    print("\n1. Global Thresholding (Otsu):")
    binary_otsu, threshold = seg.global_thresholding(test_image, 'otsu')
    print(f"   Threshold value: {threshold}")
    figure, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(binary_otsu, cmap='gray')
    ax[1].set_title('Otsu Thresholding')
    ax[1].axis('off')
    plt.show()

    
    # Test Canny edge detection
    print("\n2. Canny Edge Detection:")
    canny_edges, thresholds = seg.canny_edge_detection(test_image, 50, 150)
    print(f"   Thresholds: {thresholds}")
    figure, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')   
    ax[0].axis('off')
    ax[1].imshow(canny_edges, cmap='gray')
    ax[1].set_title('Canny Edges')
    ax[1].axis('off')
    plt.show()  

    # Test Sobel edge detection
    print("\n3. Sobel Edge Detection:")
    sobel_edges, magnitude, direction = seg.sobel_edge_detection(test_image)
    print(f"   Gradient magnitude range: {magnitude.min()}-{magnitude.max()}")
    figure, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')   
    ax[0].axis('off')
    ax[1].imshow(sobel_edges, cmap='gray')
    ax[1].set_title('Sobel Edges')
    ax[1].axis('off')
    plt.show() 
    # Test Laplacian edge detection
    print("\n4. Laplacian Edge Detection:")
    laplacian_edges, laplacian_abs = seg.laplacian_edge_detection(test_image)
    print(f"   Laplacian range: {laplacian_abs.min()}-{laplacian_abs.max()}")
    figure, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')   
    ax[0].axis('off')
    ax[1].imshow(laplacian_edges, cmap='gray')
    ax[1].set_title('Laplacian Edges')
    ax[1].axis('off')
    plt.show() 
    # Test Hough line detection
    print("\n5. Hough Transform Line Detection:")
    edges, lines, line_params = seg.detect_lines_with_hough(test_image, 'canny')
    if lines is not None:
        print(f"   Number of lines detected: {len(lines)}")
        for i, param in enumerate(line_params[:3]):
            print(f"   Line {i+1}: angle={param['angle']:.1f}°, length={param['length']:.1f}")
    else:
        print("   No lines detected")
    visualize_lines = seg.visualize_lines(test_image, lines, line_params)
    figure, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image') 
    ax[0].axis('off')
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Hough Lines')
    ax[1].axis('off')
    ax[2].imshow(visualize_lines)
    ax[2].set_title('Detected Lines Visualization')
    ax[2].axis('off')
    plt.show()
    # Test region growing
    print("\n6. Region Growing:")
    seed_points = [(75, 75), (200, 110)]
    region_grown = seg.region_growing(test_image, seed_points, threshold=30)
    print(f"   Used seed points: {seed_points}")
    seed_points_img = np.zeros_like(test_image[:,:,0])
    for x, y in seed_points:
        cv2.circle(seed_points_img, (x, y), 3, 255, -1)

    figure, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')  
    ax[0].axis('off')
    ax[1].imshow(region_grown, cmap='gray')
    ax[1].set_title('Region Grown')
    ax[1].axis('off')
    ax[2].imshow(seed_points_img, cmap='gray')
    ax[2].set_title('Seed Points')
    ax[2].axis('off')
    plt.show() 
    # Test watershed
    print("\n7. Watershed Segmentation:")
    watershed_result = seg.watershed_segmentation(test_image)
    print(f"   Number of segments: {len(np.unique(watershed_result))}")
    figure, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')   
    ax[0].axis('off')
    ax[1].imshow(watershed_result, cmap='gray')
    ax[1].set_title('Watershed Segmentation')
    ax[1].axis('off')
    plt.show() 
    # Test K-means segmentation
    print("\n8. K-means Segmentation:")
    kmeans_result, labels = seg.kmeans_segmentation(test_image, k=3)
    print(f"   Number of clusters: {len(np.unique(labels))}")
    figure, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')   
    ax[0].axis('off')
    ax[1].imshow(kmeans_result, cmap='gray')
    ax[1].set_title('K-means Segmentation')
    ax[1].axis('off')
    ax[2].imshow(labels, cmap='gray')
    ax[2].set_title('K-means Labels')
    ax[2].axis('off')
    plt.show()
    print("\n=== ALL OPENCV-ONLY METHODS TESTED SUCCESSFULLY ===")
    print("You can now use this class without scikit-image dependency!")
    
    # Save example results
    cv2.imwrite('Task5/test_image_opencv.jpg', test_image)
    cv2.imwrite('Task5/canny_edges_opencv.jpg', canny_edges)
    cv2.imwrite('Task5/sobel_edges_opencv.jpg', sobel_edges)
    cv2.imwrite('Task5/laplacian_edges_opencv.jpg', laplacian_edges)
    cv2.imwrite('Task5/binary_otsu_opencv.jpg', binary_otsu)
    cv2.imwrite('Task5/seed_points_opencv.jpg', seed_points_img)
    cv2.imwrite('Task5/region_grown_opencv.jpg', region_grown)
    cv2.imwrite('Task5/watershed_segmentation_opencv.jpg', watershed_result.astype(np.uint8))
    cv2.imwrite('Task5/kmeans_segmentation_opencv.jpg', kmeans_result) 
    if lines is not None:
        line_vis = seg.visualize_lines(test_image, lines, line_params)
        cv2.imwrite('Task5/detected_lines_opencv.jpg', line_vis)

    print("\nExample images saved:")
    print("- test_image_opencv.jpg (original)")
    print("- canny_edges_opencv.jpg")
    print("- sobel_edges_opencv.jpg")
    print("- laplacian_edges_opencv.jpg")
    print("- detected_lines_opencv.jpg (if lines were found)")