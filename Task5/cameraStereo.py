import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ============================================
# ELP-USB960P2CAM-V90 Camera Parameters
# ============================================
BASELINE = 60.0  # mm (distance between two camera lenses)
FOCAL_LENGTH = 850.0  # pixels (estimated for 1.3MP sensor)

# Camera resolution (each frame in side-by-side)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

# ============================================
# AI Model Selection
# ============================================
MODEL_TYPE = "raft-stereo"  # Options: "raft-stereo", "crestereo", "sgbm"

# ============================================
# RAFT-Stereo Deep Learning Model
# ============================================
class RAFTStereoDepth:
    """
    RAFT-Stereo: State-of-the-art deep learning stereo matching
    Paper: https://arxiv.org/abs/2109.07547
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        try:
            # Load pretrained RAFT-Stereo model from torch hub
            print("Loading RAFT-Stereo model... (this may take a minute)")
            self.model = torch.hub.load('princeton-vl/RAFT-Stereo', 'raftstereo_middlebury', 
                                       pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ RAFT-Stereo model loaded successfully!")
            self.available = True
        except Exception as e:
            print(f"⚠ Could not load RAFT-Stereo: {e}")
            print("Falling back to traditional SGBM method")
            self.available = False

    def preprocess(self, left, right):
        """Preprocess images for RAFT-Stereo"""
        # Convert to RGB and normalize
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize to [0, 1]
        left_tensor = torch.from_numpy(left_rgb).permute(2, 0, 1).float() / 255.0
        right_tensor = torch.from_numpy(right_rgb).permute(2, 0, 1).float()  / 255.0

        # Add batch dimension
        left_tensor = left_tensor.unsqueeze(0).to(self.device)
        right_tensor = right_tensor.unsqueeze(0).to(self.device)

        return left_tensor, right_tensor

    def predict(self, left, right):
        """Predict disparity using RAFT-Stereo"""
        if not self.available:
            return None

        with torch.no_grad():
            # Preprocess
            left_tensor, right_tensor = self.preprocess(left, right)

            # Pad to multiple of 8
            h, w = left_tensor.shape[2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            if pad_h > 0 or pad_w > 0:
                left_tensor = F.pad(left_tensor, (0, pad_w, 0, pad_h), mode='replicate')
                right_tensor = F.pad(right_tensor, (0, pad_w, 0, pad_h), mode='replicate')

            # Predict disparity
            _, disparity = self.model(left_tensor, right_tensor, iters=12, test_mode=True)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                disparity = disparity[:, :, :h, :w]

            # Convert to numpy
            disparity = disparity.squeeze().cpu().numpy()

            return disparity

# ============================================
# CREStereo Deep Learning Model (Alternative)
# ============================================
class CREStereoDepth:
    """
    CREStereo: Practical Stereo Matching Network
    Faster than RAFT-Stereo, good for real-time
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        try:
            print("Loading CREStereo model...")
            # Try to load from local or download
            import sys
            sys.path.append('.')

            # This is a placeholder - you'd need to download CREStereo separately
            # For now, we'll use RAFT-Stereo or SGBM
            self.available = False
            print("⚠ CREStereo not available, use RAFT-Stereo or SGBM")
        except Exception as e:
            print(f"⚠ Could not load CREStereo: {e}")
            self.available = False

# ============================================
# Traditional SGBM (Fallback)
# ============================================
class SGBMDepth:
    """Traditional Semi-Global Block Matching"""
    def __init__(self):
        self.min_disp = 0
        self.num_disp = 128
        self.block_size = 11

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.available = True
        print("✓ SGBM stereo matcher initialized")

    def predict(self, left, right):
        """Predict disparity using SGBM"""
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity

    def adjust_num_disp(self, delta):
        """Adjust number of disparities"""
        self.num_disp = max(16, min(256, self.num_disp + delta))
        self.stereo.setNumDisparities(self.num_disp)
        return self.num_disp

# ============================================
# Distance Calculation Function
# ============================================
def calculate_distance(disparity, baseline=BASELINE, focal_length=FOCAL_LENGTH):
    """
    Calculate distance in mm from disparity
    Z = (f * B) / d
    """
    # Avoid division by zero
    disparity_safe = disparity.copy()
    disparity_safe[disparity_safe <= 0] = 0.1
    distance = (focal_length * baseline) / disparity_safe
    return distance

# ============================================
# Mouse Callback for Distance Measurement
# ============================================
selected_point = None
def mouse_callback(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)

# ============================================
# Main Loop
# ============================================
def main():
    global selected_point

    # Initialize depth estimation model
    print("=" * 60)
    print("AI-Powered Stereo Vision Distance Measurement")
    print("=" * 60)

    # Try to load AI models in order of preference
    depth_model = None
    model_name = "Unknown"

    if MODEL_TYPE == "raft-stereo":
        raft = RAFTStereoDepth()
        if raft.available:
            depth_model = raft
            model_name = "RAFT-Stereo (Deep Learning)"

    if depth_model is None and MODEL_TYPE == "crestereo":
        crestereo = CREStereoDepth()
        if crestereo.available:
            depth_model = crestereo
            model_name = "CREStereo (Deep Learning)"

    # Fallback to SGBM
    if depth_model is None:
        depth_model = SGBMDepth()
        model_name = "SGBM (Traditional)"

    print(f"Using model: {model_name}")
    print("=" * 60)

    # Open the stereo camera
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        print("Make sure your ELP camera is connected")
        return

    # Create windows
    cv2.namedWindow('Stereo View')
    cv2.namedWindow('Disparity Map')
    cv2.namedWindow('Depth Map')
    cv2.setMouseCallback('Depth Map', mouse_callback)

    print("Controls:")
    print("  - Click on 'Depth Map' window to measure distance")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'f' to toggle filtering")
    if isinstance(depth_model, SGBMDepth):
        print("  - Press '+' to increase disparity range")
        print("  - Press '-' to decrease disparity range")
    print("=" * 60)

    frame_count = 0
    use_filter = True
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        frame_count += 1
        start_time = cv2.getTickCount()

        # Get actual frame dimensions
        h, w = frame.shape[:2]

        # Split the side-by-side frame into left and right
        mid = w // 2
        left_frame = frame[:, :mid]
        right_frame = frame[:, mid:]

        # Compute disparity using selected model
        disparity = depth_model.predict(left_frame, right_frame)

        if disparity is None:
            print("Error: Could not compute disparity")
            continue

        # Apply filtering to reduce noise (optional)
        if use_filter:
            # Weighted Least Squares filter for edge-preserving smoothing
            disparity_filtered = cv2.ximgproc.weightedMedianFilter(
                left_frame.astype(np.uint8), 
                disparity.astype(np.float32), 
                r=5
            ) if hasattr(cv2, 'ximgproc') else disparity
            disparity = disparity_filtered

        # Calculate distance map (in mm)
        distance_map = calculate_distance(disparity)

        # Convert distance to meters for display
        distance_map_m = distance_map / 1000.0

        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

        # Create depth visualization (closer = red, farther = blue)
        depth_vis = np.clip(distance_map_m, 0, 5)  # Clip to 5 meters
        depth_vis = 255 - (depth_vis / 5.0 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # If user clicked, show distance at that point
        if selected_point is not None:
            x, y = selected_point
            if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                dist_mm = distance_map[y, x]
                dist_cm = dist_mm / 10.0
                dist_m = dist_mm / 1000.0

                # Draw crosshair
                cv2.drawMarker(depth_color, (x, y), (0, 255, 0), 
                              cv2.MARKER_CROSS, 20, 2)

                # Display distance with background
                text = f"Distance: {dist_cm:.1f} cm ({dist_m:.2f} m)"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(depth_color, (x + 5, y - text_h - 15), 
                            (x + text_w + 15, y - 5), (0, 0, 0), -1)
                cv2.putText(depth_color, text, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"Point ({x}, {y}): {dist_cm:.1f} cm")

        # Calculate FPS
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = np.mean(fps_list)

        # Add info overlay
        info_text = f"Model: {model_name} | FPS: {avg_fps:.1f} | Frame: {frame_count}"
        cv2.putText(depth_color, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        filter_text = f"Filter: {'ON' if use_filter else 'OFF'}"
        cv2.putText(depth_color, filter_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display windows
        cv2.imshow('Stereo View', frame)
        cv2.imshow('Disparity Map', disparity_color)
        cv2.imshow('Depth Map', depth_color)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            # Save current frame and maps
            cv2.imwrite(f'stereo_frame_{frame_count}.jpg', frame)
            cv2.imwrite(f'disparity_{frame_count}.jpg', disparity_color)
            cv2.imwrite(f'depth_{frame_count}.jpg', depth_color)
            np.save(f'disparity_raw_{frame_count}.npy', disparity)
            print(f"✓ Saved frame {frame_count}")
        elif key == ord('f'):
            use_filter = not use_filter
            print(f"Filtering: {'ON' if use_filter else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            if isinstance(depth_model, SGBMDepth):
                num_disp = depth_model.adjust_num_disp(16)
                print(f"NumDisparities increased to {num_disp}")
        elif key == ord('-') or key == ord('_'):
            if isinstance(depth_model, SGBMDepth):
                num_disp = depth_model.adjust_num_disp(-16)
                print(f"NumDisparities decreased to {num_disp}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {np.mean(fps_list):.1f}")
    print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()