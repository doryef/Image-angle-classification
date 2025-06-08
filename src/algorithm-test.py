import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import sys

class CameraAngleEstimator:
    """
    Camera angle estimation from a single image based on deep learning approaches
    and geometric principles from computer vision research.
    
    This implementation combines several academic approaches:
    1. Vanishing point detection for horizon line estimation
    2. CNN-based direct regression for camera pose
    3. Geometric cues extraction (horizon line, vertical lines)
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CNN model for angle prediction
        self.angle_model = self._create_angle_model()
        
        if model_path:
            self.angle_model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.angle_model.to(self.device)
        self.angle_model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def _create_angle_model(self):
        """
        Create a CNN model for camera angle prediction based on ResNet
        with modified fully connected layers to output camera angles.
        """
        # Use ResNet as backbone with pretrained weights
        model = models.resnet50(pretrained=True)
        
        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Output: [pitch, roll, yaw] in degrees
        )
        
        return model
    
    def detect_vanishing_points(self, image):
        """
        Detect vanishing points in the image using line segment detection
        and RANSAC-based vanishing point estimation.
        
        Based on methods from:
        - Zhai et al. "Detecting Vanishing Points using Global Image Context in a Non-Manhattan World" (CVPR 2016)
        - Barinova et al. "On detection of multiple orthogonal vanishing points in a single image" (CVIU 2012)
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use probabilistic Hough transform to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            # Not enough lines detected for vanishing point analysis
            return None, None, None
        
        # Convert lines to start and end points
        line_points = []
        line_angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Filter out near-horizontal and near-vertical lines
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Group lines into horizontal, vertical, and diagonal clusters
            if 0 <= angle < 30 or 150 < angle <= 180:
                category = "horizontal"
            elif 60 <= angle <= 120:
                category = "vertical" 
            else:
                category = "diagonal"
                
            line_points.append(((x1, y1), (x2, y2)))
            line_angles.append((angle, category))
        
        # Group lines by orientation (vertical, horizontal, diagonal)
        vertical_lines = [line for i, line in enumerate(line_points) 
                         if line_angles[i][1] == "vertical"]
        horizontal_lines = [line for i, line in enumerate(line_points) 
                           if line_angles[i][1] == "horizontal"]
        
        # Find horizontal vanishing point (related to yaw)
        horizontal_vp = self._find_best_vanishing_point(horizontal_lines)
        
        # Find vertical vanishing point (related to pitch)
        vertical_vp = self._find_best_vanishing_point(vertical_lines)
        
        # Estimate horizon line based on vanishing points
        horizon_params = self._estimate_horizon_line(image, horizontal_vp, vertical_vp)
        
        return horizontal_vp, vertical_vp, horizon_params
    
    def _find_best_vanishing_point(self, lines, ransac_iterations=500, inlier_threshold=5.0):
        """
        Find a vanishing point from a set of lines using RANSAC.
        """
        if len(lines) < 2:
            return None
            
        best_vp = None
        max_inliers = 0
        
        for _ in range(ransac_iterations):
            # Randomly select two lines
            i, j = np.random.choice(len(lines), 2, replace=False)
            
            # Get intersection point (potential vanishing point)
            vp = self._get_intersection(lines[i], lines[j])
            
            if vp is None:
                continue
                
            # Count inliers
            inliers = 0
            for line in lines:
                if self._point_line_distance(vp, line) < inlier_threshold:
                    inliers += 1
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_vp = vp
                
        return best_vp
    
    def _get_intersection(self, line1, line2):
        """Calculate intersection of two lines using cross product."""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        # Convert to homogeneous coordinates
        line1_vec = np.cross([x1, y1, 1], [x2, y2, 1])
        line2_vec = np.cross([x3, y3, 1], [x4, y4, 1])
        
        # Get intersection
        intersection = np.cross(line1_vec, line2_vec)
        
        # Check if lines are parallel
        if abs(intersection[2]) < 1e-8:
            return None
            
        # Normalize and return as point
        return (intersection[0] / intersection[2], intersection[1] / intersection[2])
    
    def _point_line_distance(self, point, line):
        """Calculate distance from point to line segment."""
        x, y = point
        (x1, y1), (x2, y2) = line
        
        # Vector from point 1 to point 2
        line_vec = (x2 - x1, y2 - y1)
        # Vector from point 1 to the point
        point_vec = (x - x1, y - y1)
        
        # Line length squared
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        
        # Line is actually a point
        if line_len_sq < 1e-8:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
            
        # Calculate projection of point vector onto line vector (t parameter)
        t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq))
        
        # Calculate closest point on line segment
        closest_x = x1 + t * line_vec[0]
        closest_y = y1 + t * line_vec[1]
        
        # Return distance
        return np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
    
    def _estimate_horizon_line(self, image, horizontal_vp, vertical_vp):
        """
        Estimate the horizon line parameters based on vanishing points.
        Returns (m, b) where y = mx + b is the horizon line.
        """
        if horizontal_vp is None:
            # Fallback: assume horizon is in the middle of the image
            h, w, _ = image.shape
            return (0, h//2)
            
        # If we have a horizontal vanishing point, we can estimate the horizon line
        # as a horizontal line through that point
        return (0, horizontal_vp[1])
    
    def compute_geometric_features(self, image):
        """
        Extract geometric features from the image to use as additional input
        for angle estimation.
        """
        # Detect vanishing points and horizon line
        horizontal_vp, vertical_vp, horizon_params = self.detect_vanishing_points(image)
        
        h, w, _ = image.shape
        features = {}
        
        # Horizon line position relative to image height (pitch indicator)
        if horizon_params:
            m, b = horizon_params
            horizon_pos = b / h  # Normalized horizon position
            features["horizon_position"] = horizon_pos
        else:
            features["horizon_position"] = 0.5  # Default: middle of image
            
        # Vertical vanishing point position (another pitch indicator)
        if vertical_vp:
            x_vp, y_vp = vertical_vp
            if 0 <= y_vp <= h:  # Vanishing point is within the image
                features["vertical_vp_position"] = y_vp / h
            elif y_vp > h:  # Vanishing point is below the image
                features["vertical_vp_position"] = 1.5  # Arbitrary value > 1
            else:  # Vanishing point is above the image
                features["vertical_vp_position"] = -0.5  # Arbitrary value < 0
        else:
            features["vertical_vp_position"] = 0.0  # Default
            
        # Horizontal vanishing point position (yaw indicator)
        if horizontal_vp:
            x_vp, y_vp = horizontal_vp
            if 0 <= x_vp <= w:  # Within image
                features["horizontal_vp_position"] = x_vp / w - 0.5  # Center at 0
            elif x_vp > w:  # Right of image
                features["horizontal_vp_position"] = 0.5  # Right
            else:  # Left of image
                features["horizontal_vp_position"] = -0.5  # Left
        else:
            features["horizontal_vp_position"] = 0.0  # Default: center
            
        return features
    
    def _angles_from_features(self, features):
        """
        Estimate camera angles from geometric features using simple heuristics.
        This serves as a fallback when the CNN model fails or is not available.
        """
        # Estimate pitch from horizon position
        # Above middle: looking down, below middle: looking up
        pitch = (0.5 - features["horizon_position"]) * 90
        
        # Estimate yaw from horizontal vanishing point position
        yaw = features["horizontal_vp_position"] * 90
        
        # Roll is harder to estimate from geometric features alone
        # We would need additional cues like detecting rotated vertical structures
        roll = 0
        
        return pitch, roll, yaw
        
    def predict_from_image_path(self, image_path):
        """Predict camera angles from image path."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        return self.predict_from_image(img)
    
    def predict_from_image(self, image):
        """
        Predict camera angles from an image.
        Returns: pitch, roll, yaw in degrees
        """
        # Get geometric features for traditional approach
        geometric_features = self.compute_geometric_features(image)
        
        # Convert to RGB for model input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        
        # Process image for neural network
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        try:
            # Predict angles with CNN model
            with torch.no_grad():
                predictions = self.angle_model(img_tensor)[0].cpu().numpy()
            
            pitch, roll, yaw = predictions
            
        except Exception as e:
            # Fallback to geometry-based estimation
            print(f"CNN prediction failed: {e}. Using geometric estimation.")
            pitch, roll, yaw = self._angles_from_features(geometric_features)
        
        # Refine angles with geometric features
        pitch_geom, _, yaw_geom = self._angles_from_features(geometric_features)
        
        # When geometric features give strong signals, blend with CNN predictions
        if abs(pitch_geom) > 15:  # Strong geometric pitch signal
            pitch = 0.7 * pitch + 0.3 * pitch_geom
            
        if abs(yaw_geom) > 15:  # Strong geometric yaw signal
            yaw = 0.7 * yaw + 0.3 * yaw_geom
        
        return {
            'pitch': pitch,  # Positive: looking down, Negative: looking up
            'roll': roll,    # Clockwise rotation
            'yaw': yaw       # Positive: right, Negative: left
        }

# Example usage
def visualize_camera_angles(image_path, angles):
    """Visualize the predicted camera angles on the image."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    
    # Draw coordinate system
    length = min(w, h) // 4
    
    # Calculate 3D axes endpoints based on rotation angles
    pitch, roll, yaw = angles['pitch'], angles['roll'], angles['yaw']
    
    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    
    # Create rotation matrices
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Initial axis vectors
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])
    
    # Rotate axes
    x_axis_rotated = R @ x_axis
    y_axis_rotated = R @ y_axis
    z_axis_rotated = R @ z_axis
    
    # Project to 2D image plane (simple orthographic projection)
    x_end = (center_x + int(x_axis_rotated[0]), center_y - int(x_axis_rotated[1]))
    y_end = (center_x + int(y_axis_rotated[0]), center_y - int(y_axis_rotated[1]))
    z_end = (center_x + int(z_axis_rotated[0]), center_y - int(z_axis_rotated[1]))
    
    # Draw axes
    cv2.line(img, (center_x, center_y), x_end, (0, 0, 255), 2)  # X-axis (red)
    cv2.line(img, (center_x, center_y), y_end, (0, 255, 0), 2)  # Y-axis (green)
    cv2.line(img, (center_x, center_y), z_end, (255, 0, 0), 2)  # Z-axis (blue)
    
    # Add text with angle values
    cv2.putText(img, f"Pitch: {pitch:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Roll: {roll:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Yaw: {yaw:.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save and display result
    output_path = image_path.replace('.jpg', '_angles.jpg').replace('.png', '_angles.png')
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to {output_path}")
    
    # Display image (if in interactive environment)
    cv2.imshow("Camera Angle Estimation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Example usage
    image_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"
    
    estimator = CameraAngleEstimator()
    angles = estimator.predict_from_image_path(image_path)
    
    print(f"Estimated camera angles:")
    print(f"Pitch: {angles['pitch']:.2f}° (positive = looking down)")
    print(f"Roll: {angles['roll']:.2f}° (positive = clockwise)")
    print(f"Yaw: {angles['yaw']:.2f}° (positive = right)")
    
    # Visualize results
    visualize_camera_angles(image_path, angles)

if __name__ == "__main__":
    main()