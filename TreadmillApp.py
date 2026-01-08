import cv2
import glob
import os
import numpy as np
import threading
import time
import datetime
from PIL import Image

import sys
import configparser

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from skimage.registration import phase_cross_correlation

config_data = configparser.ConfigParser(allow_no_value=True)
CONFIG_FILE = "config_params.cfg"
assert os.path.exists(CONFIG_FILE)>0,"Config file with name config_params.cfg not found in program path!"
config_data.read(CONFIG_FILE,encoding="UTF-8")

# Ensure [template] section exists for template-based auto-detection
if 'template' not in config_data:
    config_data.add_section('template')
    config_data['template']['active_template'] = 'default'
    config_data['template']['template_directory'] = 'templates/'
    config_data['template']['auto_calibrate_on_startup'] = 'true'
    config_data['template']['matching_threshold'] = '0.50'

    with open(CONFIG_FILE, 'w', encoding="UTF-8") as configfile:
        config_data.write(configfile)
    print("Added [template] section to config")

os.add_dll_directory(config_data.get("paths","vlc_path"))
import vlc

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # use CPU

if config_data.get("main","model_type") == 'tensorflow':
    from tensorflow.keras.models import load_model
elif config_data.get("main","model_type") == 'pytorch':
    import torch
    from pytorch_digits_model import Network
else:
    raise(AssertionError('Bad model type!'))

def set_value(entry,default):
    global config_data
    try:
        val = float(config_data.get("main",entry))
    except:
        val = default
    config_data["main"][entry] = str(val)
    return val

print('Loaded config file parameters are:')
for k in config_data.keys():
    if k!='files':
        print('Section = {0}'.format(k))
        for v in config_data[k].keys():
            print('... {0} = {1}'.format(v,config_data[k][v]))

# required, no defaults
DIGIT_MODEL = config_data.get("paths","model_path") # path to saved Keras model
DEFAULT_PATH = config_data.get("paths","video_path") # path to running videos (MP4)

# params with defaults
float(set_value("zoom_level",1))
PREDICT_INTERVAL = float(set_value("predict_interval",400.0))
SMOOTH_WINDOW = float(set_value("smooth_window",2.0))
DEFAULT_SPEED = float(set_value("default_speed",9.0))
MAX_SPEED = float(set_value("max_speed",20))
EXTRA_PIXEL_RATIO = float(set_value("extra_pixel_ratio",0.08))
SHIFT_CENTER_X = int(set_value("shift_center_x",0))
SHIFT_CENTER_Y = int(set_value("shift_center_y",0))

current_running_speed = -1 # global variable
running_speed_history = [] # keep track of previous speeds

light_green = QColor(144, 238, 144)
light_red = QColor(255, 182, 193)
white = QColor(255, 255, 255)

# FOR DEBUGGING AND DEVELOPMENT
import matplotlib.pyplot as plt
WEBCAM_VIDEO = r"C:\code\TreadmillApp\videos\Video 4.wmv"
DEBUG_SPEED_OVERRIDE = None #13
LOAD_MODEL = True # set false for faster loading in debugging

# correlation image registering
def register_image(image,offset_image):
    MAX_CORRECTION = int(0.10*min(image.shape[0],image.shape[1]))

    # pixel precision first
    shift, error, diffphase = phase_cross_correlation(image, np.fft.fft2(cv2.cvtColor(offset_image,cv2.COLOR_BGR2GRAY)),space='fourier')

    if abs(shift[0])>MAX_CORRECTION:
        shift[0] = 0
    if abs(shift[1])>MAX_CORRECTION:
        shift[1] = 0

    # Combined roll - single allocation instead of two
    shift_y, shift_x = int(shift[0]), int(shift[1])
    if shift_y != 0 or shift_x != 0:
        offset_image = np.roll(offset_image, (shift_y, shift_x), axis=(0, 1))

    return offset_image


# ===== TEMPLATE-BASED AUTO-DETECTION SYSTEM =====

class TemplateManager:
    """
    Manages template files (.npz format) for ROI auto-detection.
    Templates store: image, mask, polygon, ROI coords, digit count, etc.
    """

    def __init__(self, template_dir='templates'):
        """
        Initialize template manager.

        Args:
            template_dir: Directory path for template storage (default: 'templates/')
        """
        self.template_dir = template_dir
        self.current_template = None

        # Create directory if it doesn't exist
        if not os.path.exists(self.template_dir):
            try:
                os.makedirs(self.template_dir)
                print(f"Created template directory: {self.template_dir}")
            except Exception as e:
                print(f"Warning: Could not create template directory: {e}")

    def list_templates(self):
        """
        List all available template names.

        Returns:
            List of template names (without .npz extension)
        """
        try:
            if not os.path.exists(self.template_dir):
                return []

            files = glob.glob(os.path.join(self.template_dir, "*.npz"))
            names = [os.path.splitext(os.path.basename(f))[0] for f in files]
            return sorted(names)
        except Exception as e:
            print(f"Error listing templates: {e}")
            return []

    def template_exists(self, name):
        """
        Check if a template exists.

        Args:
            name: Template name (without .npz extension)

        Returns:
            Boolean indicating if template exists
        """
        path = os.path.join(self.template_dir, f"{name}.npz")
        return os.path.exists(path)

    def load_template(self, name):
        """
        Load a template from disk.

        Args:
            name: Template name (without .npz extension)

        Returns:
            Dictionary with template data, or None if failed
        """
        path = os.path.join(self.template_dir, f"{name}.npz")

        try:
            if not os.path.exists(path):
                print(f"Template not found: {name}")
                return None

            # Load numpy archive
            data = np.load(path, allow_pickle=True)

            # Required keys
            required_keys = ['template_image', 'panel_mask', 'panel_polygon',
                           'roi_box', 'digit_count', 'zoom_level',
                           'timestamp', 'camera_source']

            # Validate all keys present
            for key in required_keys:
                if key not in data:
                    print(f"Template '{name}' missing required key: {key}")
                    return None

            # Convert to regular dict
            template_dict = {key: data[key] for key in required_keys}

            # Add placeholder for cached preprocessed image
            template_dict['template_gray_enhanced'] = None

            # Store as current template
            self.current_template = template_dict

            print(f"Loaded template: {name}")
            return template_dict

        except Exception as e:
            print(f"Error loading template '{name}': {e}")
            return None

    def save_template(self, name, template_data):
        """
        Save a template to disk.

        Args:
            name: Template name (without .npz extension)
            template_data: Dictionary with template data

        Returns:
            Boolean indicating success
        """
        try:
            # Sanitize filename (remove special characters)
            name = ''.join(c for c in name if c.isalnum() or c in ('_', '-'))
            name = name[:30]  # Max 30 characters

            if not name:
                print("Invalid template name")
                return False

            # Required keys
            required_keys = ['template_image', 'panel_mask', 'panel_polygon',
                           'roi_box', 'digit_count', 'zoom_level',
                           'timestamp', 'camera_source']

            # Validate all keys present
            for key in required_keys:
                if key not in template_data:
                    print(f"Template data missing required key: {key}")
                    return False

            # Ensure directory exists
            if not os.path.exists(self.template_dir):
                os.makedirs(self.template_dir)

            # Save as compressed npz file
            path = os.path.join(self.template_dir, f"{name}.npz")
            np.savez_compressed(path, **template_data)

            # Also save template image as PNG for easy viewing
            png_path = os.path.join(self.template_dir, f"{name}.png")
            cv2.imwrite(png_path, template_data['template_image'])

            # Update current template
            self.current_template = template_data

            print(f"Saved template: {name} to {path}")
            print(f"Saved template image: {png_path}")
            return True

        except Exception as e:
            print(f"Error saving template '{name}': {e}")
            return False

    def delete_template(self, name):
        """
        Delete a template from disk.

        Args:
            name: Template name (without .npz extension)

        Returns:
            Boolean indicating success
        """
        path = os.path.join(self.template_dir, f"{name}.npz")

        try:
            if not os.path.exists(path):
                print(f"Template not found: {name}")
                return False

            os.remove(path)
            print(f"Deleted template: {name}")

            # Also delete the PNG file if it exists
            png_path = os.path.join(self.template_dir, f"{name}.png")
            if os.path.exists(png_path):
                os.remove(png_path)
                print(f"Deleted template image: {png_path}")

            # Clear current template if it was deleted
            if self.current_template is not None:
                self.current_template = None

            return True

        except Exception as e:
            print(f"Error deleting template '{name}': {e}")
            return False

    def get_current_template(self):
        """
        Get the currently loaded template.

        Returns:
            Template dictionary or None
        """
        return self.current_template


class ROIMatcher:
    """
    Matches template to live frame using ORB features and homography transformation.
    Returns ROI coordinates and matching ratio.
    """

    def __init__(self):
        """Initialize feature detector, matcher, and preprocessor."""
        # ORB detector with more features for robustness
        self.detector = cv2.ORB_create(
            nfeatures=2000,     # Detect up to 2000 features
            scaleFactor=1.2,    # Pyramid scale factor
            nlevels=8           # Pyramid levels for scale invariance
        )

        # FLANN matcher for fast approximate matching (LSH for binary descriptors)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,     # 12 tables
            key_size=12,        # 20 bits per key
            multi_probe_level=1 # Check 2 neighboring buckets
        )
        search_params = dict(checks=50)  # Number of tree checks
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # CLAHE for contrast enhancement (normalize lighting differences)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_image(self, image):
        """
        Preprocess image for feature detection.
        Converts to grayscale and applies CLAHE contrast enhancement.

        Args:
            image: Color or grayscale image (numpy array)

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply CLAHE to normalize lighting
        enhanced = self.clahe.apply(gray)

        return enhanced

    def preprocess_template(self, template_data):
        """
        Preprocess and cache template grayscale image.

        Args:
            template_data: Template dictionary

        Returns:
            Cached preprocessed grayscale template image
        """
        if template_data.get('template_gray_enhanced') is None:
            template_data['template_gray_enhanced'] = self.preprocess_image(
                template_data['template_image']
            )
        return template_data['template_gray_enhanced']

    def validate_similarity(self, H):
        """
        Validate that similarity transformation is reasonable.
        Rejects excessive rotation or scaling.
        Similarity transform has only 4 DOF: translation (x,y), rotation, uniform scale.

        Args:
            H: 3x3 matrix (similarity in homography format)

        Returns:
            Boolean indicating if transformation is valid
        """
        if H is None:
            return False

        try:
            # Extract 2x2 linear part
            A = H[:2, :2]

            # Extract scale (should be uniform for similarity transform)
            sx = np.sqrt(A[0, 0]**2 + A[1, 0]**2)
            sy = np.sqrt(A[0, 1]**2 + A[1, 1]**2)

            # Average scale (should be nearly equal for true similarity)
            scale = (sx + sy) / 2.0

            # Check uniform scale (75% - 150%)
            if scale < 0.75 or scale > 1.5:
                print(f"Similarity rejected: scale {scale:.2f} outside [0.75, 1.5]")
                return False

            # Verify scales are uniform (within 5% of each other)
            if abs(sx - sy) > 0.05 * scale:
                print(f"Similarity rejected: non-uniform scale sx={sx:.2f}, sy={sy:.2f}")
                return False

            # Extract rotation angle
            angle = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi

            # Check rotation (< 15 degrees)
            if abs(angle) > 15:
                print(f"Similarity rejected: rotation {angle:.1f}° exceeds 15°")
                return False

            # Ensure no perspective (should be [0, 0, 1] in last row)
            if abs(H[2, 0]) > 0.001 or abs(H[2, 1]) > 0.001:
                print(f"Similarity rejected: perspective components present")
                return False

            return True

        except Exception as e:
            print(f"Error validating similarity: {e}")
            return False

    def transform_roi(self, template_roi, H):
        """
        Transform template ROI coordinates to live frame coordinates using homography.

        Args:
            template_roi: [x1, y1, x2, y2] in template coordinates
            H: 3x3 homography matrix

        Returns:
            [x1, y1, x2, y2] in live frame coordinates (bounding box)
        """
        x1, y1, x2, y2 = template_roi

        # Define 4 corners of ROI box
        corners = np.array([
            [x1, y1],  # Top-left
            [x2, y1],  # Top-right
            [x2, y2],  # Bottom-right
            [x1, y2]   # Bottom-left
        ], dtype=np.float32)

        # Reshape for perspectiveTransform (needs shape [N, 1, 2])
        corners = corners.reshape(-1, 1, 2)

        # Apply homography transformation
        transformed = cv2.perspectiveTransform(corners, H)

        # Get bounding box of transformed corners
        x_coords = transformed[:, 0, 0]
        y_coords = transformed[:, 0, 1]

        roi_live = [
            int(np.min(x_coords)),  # x1
            int(np.min(y_coords)),  # y1
            int(np.max(x_coords)),  # x2
            int(np.max(y_coords))   # y2
        ]

        return roi_live

    def validate_roi(self, roi, frame_shape):
        """
        Validate that ROI is within frame bounds and has reasonable size.

        Args:
            roi: [x1, y1, x2, y2]
            frame_shape: (height, width) tuple

        Returns:
            Boolean indicating if ROI is valid
        """
        x1, y1, x2, y2 = roi
        height, width = frame_shape[:2]

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"ROI rejected: out of bounds ({x1},{y1},{x2},{y2}) for frame {width}x{height}")
            return False

        # Check minimum size
        roi_width = x2 - x1
        roi_height = y2 - y1

        if roi_width < 40 or roi_height < 20:
            print(f"ROI rejected: too small {roi_width}x{roi_height} (min 40x20)")
            return False

        # Check maximum size (shouldn't be more than half the frame)
        if roi_width > width * 0.75 or roi_height > height * 0.75:
            print(f"ROI rejected: too large {roi_width}x{roi_height}")
            return False

        # Check aspect ratio (digits are typically wider than tall, 1:1 to 10:1)
        aspect_ratio = roi_width / roi_height
        if aspect_ratio < 1.0 or aspect_ratio > 10.0:
            print(f"ROI rejected: unusual aspect ratio {aspect_ratio:.2f}")
            return False

        return True

    def match_template(self, live_frame, template_data, threshold=0.50,
                       ratio_threshold=0.75, ransac_threshold=5.0, min_matches=15):
        """
        Match template to live frame and compute ROI transformation.
        Main algorithm using ORB features, FLANN matching, and homography.

        Args:
            live_frame: Current webcam frame (H, W, 3) BGR
            template_data: Dictionary with template info
            threshold: Minimum matching ratio threshold (0-1)

        Returns:
            Tuple: ([x1, y1, x2, y2], matching_ratio, H) or (None, matching_ratio, None)
        """
        try:
            # Step 1: Preprocess images (template cached, live preprocessed each time)
            template_gray = self.preprocess_template(template_data)
            live_gray = self.preprocess_image(live_frame)

            # Step 2: Detect features
            # Template: only detect in masked panel region
            kp_template, desc_template = self.detector.detectAndCompute(
                template_gray,
                mask=template_data['panel_mask']  # Only detect on panel
            )

            # Live frame: detect everywhere
            kp_live, desc_live = self.detector.detectAndCompute(live_gray, mask=None)

            # Validate sufficient features found
            if desc_template is None or desc_live is None:
                print("Feature detection failed: no descriptors found")
                return None, 0.0, None

            if len(kp_template) < 10:
                print(f"Too few template features: {len(kp_template)} (need 10+)")
                return None, 0.0, None

            if len(kp_live) < 10:
                print(f"Too few live frame features: {len(kp_live)} (need 10+)")
                return None, 0.0, None

            # Step 3: Match features using FLANN
            matches = self.matcher.knnMatch(desc_template, desc_live, k=2)

            # Apply Lowe's ratio test to filter bad matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:  # Ratio threshold
                        good_matches.append(m)

            if len(good_matches) < min_matches:
                print(f"Too few good matches: {len(good_matches)} (need {min_matches}+)")
                return None, 0.0, None

            # Step 4: Compute homography
            # Extract point coordinates from matches
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches])

            # Reshape for findHomography
            src_pts = src_pts.reshape(-1, 1, 2)
            dst_pts = dst_pts.reshape(-1, 1, 2)

            # Compute similarity transformation (translation + rotation + uniform scale only)
            # This is more robust than full homography or affine
            A, mask_ransac = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,  # Max pixel error for inliers
                maxIters=2000,
                confidence=0.99
            )

            # Convert 2x3 similarity to 3x3 homography format for compatibility
            if A is not None:
                H = np.vstack([A, [0, 0, 1]])
            else:
                H = None

            if H is None:
                print("Similarity transformation computation failed")
                return None, 0.0, None

            # Step 5: Validate similarity transformation
            if not self.validate_similarity(H):
                return None, 0.0, None

            # Calculate confidence based on inliers
            inliers = np.sum(mask_ransac)
            confidence = float(inliers) / len(good_matches)

            if confidence < threshold:
                print(f"Matching ratio too low: {confidence:.1%} < {threshold:.1%}")
                return None, confidence, None

            # Step 6: Transform ROI coordinates
            roi_live = self.transform_roi(template_data['roi_box'], H)

            # Step 7: Validate transformed ROI
            if not self.validate_roi(roi_live, live_frame.shape):
                return None, confidence, None

            # Success!
            print(f"Template matched! Matching ratio: {confidence:.1%}, ROI: {roi_live}")
            return roi_live, confidence, H

        except Exception as e:
            print(f"Error in template matching: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, None

    def match_template_robust(self, live_frame, template_data, threshold=0.50, num_trials=20):
        """
        Robust template matching with multi-start optimization.

        Runs multiple trials with varied parameters and returns best result.
        This dramatically improves reliability by avoiding local minima.

        Args:
            live_frame: Current webcam frame
            template_data: Template dictionary
            threshold: Minimum matching ratio threshold
            num_trials: Number of trials (default 20, takes ~1-2 seconds)

        Returns:
            Tuple: (best_roi, best_matching_ratio, best_H)
        """
        best_roi = None
        best_confidence = 0.0
        best_H = None

        # Parameter variations
        ratio_thresholds = [0.70, 0.75, 0.80]
        ransac_thresholds = [3.0, 5.0, 7.0]
        min_matches_list = [10, 15, 20]

        print(f"Running {num_trials} trials for robust matching...")

        for trial in range(num_trials):
            # Cycle through parameter combinations
            ratio = ratio_thresholds[trial % len(ratio_thresholds)]
            ransac = ransac_thresholds[(trial // len(ratio_thresholds)) % len(ransac_thresholds)]
            min_match = min_matches_list[(trial // (len(ratio_thresholds) * len(ransac_thresholds))) % len(min_matches_list)]

            # Run matching with these parameters
            roi, confidence, H = self.match_template(
                live_frame, template_data, threshold,
                ratio_threshold=ratio,
                ransac_threshold=ransac,
                min_matches=min_match
            )

            # Keep best result
            if confidence > best_confidence:
                best_confidence = confidence
                best_roi = roi
                best_H = H
                print(f"  Trial {trial+1}: matching_ratio={confidence:.1%} (new best!)")

            # Early exit if we get excellent match
            if best_confidence > 0.90:
                print(f"  Early exit at trial {trial+1} with {best_confidence:.1%} matching ratio")
                break

        print(f"Best result: {best_confidence:.1%} matching ratio")
        return best_roi, best_confidence, best_H


class TemplateCaptureDialog(QDialog):
    """
    Interactive dialog for creating ROI templates.
    3-state workflow: DRAW_PANEL → DRAW_ROI → PREVIEW
    User draws panel boundary, then ROI box, then saves template.
    """

    # State constants
    STATE_DRAW_PANEL = 0
    STATE_DRAW_ROI = 1
    STATE_PREVIEW = 2

    def __init__(self, parent, captured_frame, camera_source, zoom_level):
        """
        Initialize template capture dialog.

        Args:
            parent: Parent widget
            captured_frame: Frozen webcam frame (BGR numpy array)
            camera_source: Camera device ID (0, 1, etc.)
            zoom_level: Current zoom level (1-3)
        """
        super().__init__(parent)

        self.captured_frame = captured_frame.copy()  # Store frozen frame
        self.camera_source = camera_source
        self.zoom_level = zoom_level

        # State machine
        self.state = self.STATE_DRAW_PANEL
        self.drawing_mode = 'polygon'  # 'polygon' or 'rectangle'

        # Drawing data
        self.panel_points = []  # List of (x, y) for polygon
        self.roi_corners = []   # List of 2 (x, y) for ROI rectangle
        self.digit_count = 3    # Default digit count
        self.mouse_pos = None   # Current mouse position for preview

        # Template data (filled when saving)
        self.template_data = None
        self.template_name = None

        # Setup UI
        self.setWindowTitle("Create ROI Template")
        self.setModal(True)
        self.resize(900, 750)

        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        """Create and layout all UI widgets."""
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Instructions label
        self.instructions_label = QLabel()
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        self.update_instructions()
        main_layout.addWidget(self.instructions_label)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #ccc;")
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.handle_mouse_press
        self.image_label.mouseMoveEvent = self.handle_mouse_move
        main_layout.addWidget(self.image_label, stretch=1)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Drawing mode combo (only visible in STATE_DRAW_PANEL)
        self.mode_label = QLabel("Mode:")
        controls_layout.addWidget(self.mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Polygon", "Rectangle"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(self.mode_combo)

        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear_clicked)
        controls_layout.addWidget(self.clear_btn)

        controls_layout.addStretch()

        # Digit count spinner (only visible in STATE_DRAW_ROI)
        self.digit_label = QLabel("Digits:")
        self.digit_label.setVisible(False)
        controls_layout.addWidget(self.digit_label)

        self.digit_spin = QSpinBox()
        self.digit_spin.setRange(1, 5)
        self.digit_spin.setValue(3)
        self.digit_spin.valueChanged.connect(self.on_digit_count_changed)
        self.digit_spin.setVisible(False)
        controls_layout.addWidget(self.digit_spin)

        controls_layout.addStretch()

        # Navigation buttons
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.on_back_clicked)
        self.back_btn.setVisible(False)
        controls_layout.addWidget(self.back_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.on_next_clicked)
        self.next_btn.setEnabled(False)
        controls_layout.addWidget(self.next_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        controls_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(controls_layout)

        # Template name input (only visible in STATE_PREVIEW)
        name_layout = QHBoxLayout()
        self.name_label = QLabel("Template name:")
        self.name_label.setVisible(False)
        name_layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.name_input.setText(f"template_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.name_input.setMaxLength(30)
        self.name_input.setVisible(False)
        name_layout.addWidget(self.name_input, stretch=1)

        main_layout.addLayout(name_layout)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666; font-size: 11px; padding: 3px;")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def update_instructions(self):
        """Update instruction text based on current state."""
        if self.state == self.STATE_DRAW_PANEL:
            if self.drawing_mode == 'polygon':
                text = "Step 1 of 3: Draw panel boundary - Click to add points (4+ required), then click Next"
            else:
                text = "Step 1 of 3: Draw panel boundary - Click two corners to define rectangle"
        elif self.state == self.STATE_DRAW_ROI:
            text = "Step 2 of 3: Draw ROI box - Click two corners around the digit display"
        else:  # STATE_PREVIEW
            text = "Step 3 of 3: Review and save - Enter template name and click Save"

        self.instructions_label.setText(text)

    def handle_mouse_press(self, event):
        """Handle mouse click events on image label."""
        if event.button() != Qt.LeftButton:
            return

        # Convert widget coords to image coords
        img_x, img_y = self.widget_to_image_coords(event.pos())

        if img_x is None:
            return

        if self.state == self.STATE_DRAW_PANEL:
            if self.drawing_mode == 'polygon':
                # Add point to polygon
                self.panel_points.append((img_x, img_y))
                self.update_display()
                self.validate_and_update_buttons()

            elif self.drawing_mode == 'rectangle':
                # Add corner (max 2)
                if len(self.panel_points) < 2:
                    self.panel_points.append((img_x, img_y))
                    self.update_display()
                    self.validate_and_update_buttons()

                    if len(self.panel_points) == 2:
                        # Auto-validate rectangle
                        if self.validate_panel_boundary():
                            self.status_label.setText("Rectangle complete - click Next")
                            self.status_label.setStyleSheet("color: green;")

        elif self.state == self.STATE_DRAW_ROI:
            # Add ROI corner (max 2)
            if len(self.roi_corners) < 2:
                self.roi_corners.append((img_x, img_y))
                self.update_display()
                self.validate_and_update_buttons()

                if len(self.roi_corners) == 2:
                    # Auto-validate ROI
                    if self.validate_roi_box():
                        self.status_label.setText("ROI complete - click Next")
                        self.status_label.setStyleSheet("color: green;")

    def handle_mouse_move(self, event):
        """Handle mouse move events for preview drawing."""
        # Convert widget coords to image coords
        img_x, img_y = self.widget_to_image_coords(event.pos())

        if img_x is None:
            return

        self.mouse_pos = (img_x, img_y)
        self.update_display()

    def widget_to_image_coords(self, pos):
        """
        Convert widget click coordinates to image coordinates.
        Handles scaling and centering of displayed image.

        Args:
            pos: QPoint from mouse event

        Returns:
            Tuple (img_x, img_y) or (None, None) if outside image
        """
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None, None

        # Get offsets (image may be centered with padding)
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        x_offset = (label_width - pixmap_width) // 2
        y_offset = (label_height - pixmap_height) // 2

        # Widget coordinates
        widget_x = pos.x()
        widget_y = pos.y()

        # Check if click is within pixmap bounds
        if (widget_x < x_offset or widget_x >= x_offset + pixmap_width or
            widget_y < y_offset or widget_y >= y_offset + pixmap_height):
            return None, None

        # Convert to pixmap coordinates
        pixmap_x = widget_x - x_offset
        pixmap_y = widget_y - y_offset

        # Scale to original image coordinates
        img_height, img_width = self.captured_frame.shape[:2]
        scale_x = img_width / pixmap_width
        scale_y = img_height / pixmap_height

        img_x = int(pixmap_x * scale_x)
        img_y = int(pixmap_y * scale_y)

        # Clamp to image bounds
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))

        return img_x, img_y

    def update_display(self):
        """Redraw frame with current annotations."""
        # Make a copy for drawing
        display_img = self.captured_frame.copy()

        if self.state == self.STATE_DRAW_PANEL:
            self.draw_panel_state(display_img)
        elif self.state == self.STATE_DRAW_ROI:
            self.draw_roi_state(display_img)
        elif self.state == self.STATE_PREVIEW:
            self.draw_preview_state(display_img)

        # Convert to QPixmap (same method as main camera display)
        height, width = display_img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(display_img.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def draw_panel_state(self, img):
        """Draw panel boundary points and lines."""
        if not self.panel_points:
            return

        # Draw points
        for point in self.panel_points:
            cv2.circle(img, point, 5, (0, 255, 0), -1)

        # Draw lines connecting points
        if len(self.panel_points) > 1:
            for i in range(len(self.panel_points) - 1):
                cv2.line(img, self.panel_points[i], self.panel_points[i+1], (0, 255, 0), 2)

            # Close polygon (connect last to first)
            if self.drawing_mode == 'polygon' and len(self.panel_points) > 2:
                cv2.line(img, self.panel_points[-1], self.panel_points[0], (0, 255, 0), 2)
            elif self.drawing_mode == 'rectangle' and len(self.panel_points) == 2:
                # Draw complete rectangle
                x1, y1 = self.panel_points[0]
                x2, y2 = self.panel_points[1]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw preview line from last point to mouse
        if self.mouse_pos and len(self.panel_points) > 0:
            if self.drawing_mode == 'polygon':
                cv2.line(img, self.panel_points[-1], self.mouse_pos, (0, 255, 0), 1)
            elif self.drawing_mode == 'rectangle' and len(self.panel_points) == 1:
                # Preview rectangle
                x1, y1 = self.panel_points[0]
                x2, y2 = self.mouse_pos
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    def draw_roi_state(self, img):
        """Draw panel overlay and ROI box."""
        # Draw panel mask overlay (semi-transparent)
        mask = self.create_panel_mask()
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Draw panel boundary
        if self.drawing_mode == 'polygon':
            pts = np.array(self.panel_points, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        else:  # rectangle
            x1, y1 = self.panel_points[0]
            x2, y2 = self.panel_points[1]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ROI corners and box
        if self.roi_corners:
            for point in self.roi_corners:
                cv2.circle(img, point, 5, (0, 0, 255), -1)

            if len(self.roi_corners) == 2:
                x1, y1 = self.roi_corners[0]
                x2, y2 = self.roi_corners[1]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Draw digit divisions
                digit_count = self.digit_spin.value()
                x1_norm = min(x1, x2)
                x2_norm = max(x1, x2)
                y1_norm = min(y1, y2)
                y2_norm = max(y1, y2)

                roi_width = x2_norm - x1_norm
                digit_width = roi_width / digit_count

                for i in range(1, digit_count):
                    x_div = int(x1_norm + i * digit_width)
                    cv2.line(img, (x_div, y1_norm), (x_div, y2_norm), (255, 255, 0), 1)

            elif self.mouse_pos:
                # Preview ROI box
                x1, y1 = self.roi_corners[0]
                x2, y2 = self.mouse_pos
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def draw_preview_state(self, img):
        """Draw final template preview with all annotations."""
        # Same as ROI state
        self.draw_roi_state(img)

        # Add "PREVIEW" text
        cv2.putText(img, "PREVIEW", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def create_panel_mask(self):
        """Generate binary mask from panel points."""
        height, width = self.captured_frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if self.drawing_mode == 'polygon':
            pts = np.array(self.panel_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif self.drawing_mode == 'rectangle' and len(self.panel_points) == 2:
            x1, y1 = self.panel_points[0]
            x2, y2 = self.panel_points[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            mask[y1:y2, x1:x2] = 255

        return mask

    def validate_panel_boundary(self):
        """Validate panel boundary is acceptable."""
        if self.drawing_mode == 'polygon':
            if len(self.panel_points) < 4:
                self.status_label.setText("Polygon needs at least 4 points")
                self.status_label.setStyleSheet("color: red;")
                return False
        elif self.drawing_mode == 'rectangle':
            if len(self.panel_points) < 2:
                self.status_label.setText("Rectangle needs 2 corners")
                self.status_label.setStyleSheet("color: red;")
                return False

        # Check area
        mask = self.create_panel_mask()
        total_pixels = mask.shape[0] * mask.shape[1]
        panel_pixels = np.sum(mask > 0)
        area_ratio = panel_pixels / total_pixels

        if area_ratio < 0.05:
            self.status_label.setText(f"Panel too small ({area_ratio*100:.1f}% of frame, need >5%)")
            self.status_label.setStyleSheet("color: red;")
            return False

        if area_ratio > 0.80:
            self.status_label.setText(f"Panel too large ({area_ratio*100:.1f}% of frame, need <80%)")
            self.status_label.setStyleSheet("color: red;")
            return False

        self.status_label.setText(f"Panel boundary OK ({area_ratio*100:.1f}% of frame)")
        self.status_label.setStyleSheet("color: green;")
        return True

    def validate_roi_box(self):
        """Validate ROI box is acceptable."""
        if len(self.roi_corners) < 2:
            self.status_label.setText("ROI needs 2 corners")
            self.status_label.setStyleSheet("color: red;")
            return False

        x1, y1 = self.roi_corners[0]
        x2, y2 = self.roi_corners[1]

        # Normalize coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Check dimensions
        roi_width = x2 - x1
        roi_height = y2 - y1

        if roi_width < 40:
            self.status_label.setText(f"ROI too narrow ({roi_width}px, need >40px)")
            self.status_label.setStyleSheet("color: red;")
            return False

        if roi_height < 20:
            self.status_label.setText(f"ROI too short ({roi_height}px, need >20px)")
            self.status_label.setStyleSheet("color: red;")
            return False

        # Check ROI is within panel boundary
        panel_mask = self.create_panel_mask()
        roi_corners_check = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        for corner in roi_corners_check:
            if panel_mask[corner[1], corner[0]] == 0:
                self.status_label.setText("ROI must be entirely within panel boundary")
                self.status_label.setStyleSheet("color: red;")
                return False

        self.status_label.setText(f"ROI OK ({roi_width}x{roi_height}px)")
        self.status_label.setStyleSheet("color: green;")
        return True

    def validate_and_update_buttons(self):
        """Update button states based on current data."""
        if self.state == self.STATE_DRAW_PANEL:
            valid = self.validate_panel_boundary()
            self.next_btn.setEnabled(valid)
        elif self.state == self.STATE_DRAW_ROI:
            valid = self.validate_roi_box()
            self.next_btn.setEnabled(valid)
        elif self.state == self.STATE_PREVIEW:
            self.next_btn.setEnabled(True)
            self.next_btn.setText("Save")

    def on_mode_changed(self, index):
        """Handle drawing mode change."""
        self.drawing_mode = 'polygon' if index == 0 else 'rectangle'
        self.panel_points.clear()
        self.update_instructions()
        self.update_display()
        self.validate_and_update_buttons()

    def on_digit_count_changed(self, value):
        """Handle digit count change."""
        self.digit_count = value
        self.update_display()

    def on_clear_clicked(self):
        """Clear current drawing."""
        if self.state == self.STATE_DRAW_PANEL:
            self.panel_points.clear()
        elif self.state == self.STATE_DRAW_ROI:
            self.roi_corners.clear()

        self.update_display()
        self.validate_and_update_buttons()

    def on_back_clicked(self):
        """Go to previous state."""
        if self.state == self.STATE_DRAW_ROI:
            self.state = self.STATE_DRAW_PANEL
            self.roi_corners.clear()
            self.digit_label.setVisible(False)
            self.digit_spin.setVisible(False)
            self.back_btn.setVisible(False)
            self.mode_label.setVisible(True)
            self.mode_combo.setVisible(True)
            self.next_btn.setText("Next")

        elif self.state == self.STATE_PREVIEW:
            self.state = self.STATE_DRAW_ROI
            self.name_label.setVisible(False)
            self.name_input.setVisible(False)
            self.next_btn.setText("Next")

        self.update_instructions()
        self.update_display()
        self.validate_and_update_buttons()

    def on_next_clicked(self):
        """Advance to next state or save."""
        if self.state == self.STATE_DRAW_PANEL:
            if not self.validate_panel_boundary():
                return

            # Advance to ROI drawing
            self.state = self.STATE_DRAW_ROI
            self.mode_label.setVisible(False)
            self.mode_combo.setVisible(False)
            self.digit_label.setVisible(True)
            self.digit_spin.setVisible(True)
            self.back_btn.setVisible(True)
            self.next_btn.setEnabled(False)

        elif self.state == self.STATE_DRAW_ROI:
            if not self.validate_roi_box():
                return

            # Advance to preview
            self.state = self.STATE_PREVIEW
            self.name_label.setVisible(True)
            self.name_input.setVisible(True)
            self.next_btn.setText("Save")
            self.next_btn.setEnabled(True)

        elif self.state == self.STATE_PREVIEW:
            # Save and close
            self.save_and_close()
            return

        self.update_instructions()
        self.update_display()
        self.validate_and_update_buttons()

    def save_and_close(self):
        """Package template data and close dialog."""
        # Get template name
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a template name.")
            return

        # Sanitize name
        name = ''.join(c for c in name if c.isalnum() or c in ('_', '-'))
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Template name must contain alphanumeric characters.")
            return

        # Normalize ROI coordinates
        x1, y1 = self.roi_corners[0]
        x2, y2 = self.roi_corners[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Create template data dictionary
        self.template_data = {
            'template_image': self.captured_frame.copy(),
            'panel_mask': self.create_panel_mask(),
            'panel_polygon': np.array(self.panel_points, dtype=np.int32),
            'roi_box': np.array([x1, y1, x2, y2], dtype=np.int32),
            'digit_count': self.digit_spin.value(),
            'zoom_level': self.zoom_level,
            'timestamp': datetime.datetime.now().isoformat(),
            'camera_source': self.camera_source
        }

        self.template_name = name

        # Accept dialog (success)
        self.accept()

    def get_template_data(self):
        """
        Get the created template data.
        Called by parent after dialog.exec_() returns Accepted.

        Returns:
            Tuple: (template_data dict, template_name string)
        """
        return self.template_data, self.template_name


# this is the window that playes the selected video with dynamic playrate
class VideoWindow(QMainWindow):

    def __init__(self,initial_video = None,default_speed = 9.0,*args, **kwargs):
        super(VideoWindow, self).__init__(*args, **kwargs)

        self.is_paused = False
        self.default_speed = default_speed  # km/h
        self.sizeHint = lambda: QSize(1280, 900)
        self.setMinimumSize(QSize(300, 200))

        self.move(100, 10)
        self.mainFrame = QFrame()
        self.setCentralWidget(self.mainFrame)
        t_lay_parent = QHBoxLayout()
        t_lay_parent.setContentsMargins(0,0,0,40)

        self.videoFrame = QFrame()
        self.videoFrame.mouseDoubleClickEvent = self.mouseDoubleClickEvent
        t_lay_parent.addWidget(self.videoFrame)

        self.vlcInstance = vlc.Instance('--video-on-top','--disable-screensaver','--high-priority','--loop','--playlist-autostart') # could put in instance parameters here
        self.videoPlayer = self.vlcInstance.media_player_new()
        self.videoPlayer.video_set_mouse_input(False)
        self.videoPlayer.video_set_key_input(False)
        self.videoPlayer.set_mrl("http://xxx.xxx.xxx.xxx", "network-caching=300")
        #self.videoPlayer.audio_set_mute(True)
        self.videoPlayer.audio_set_volume(50)
        if sys.platform.startswith('linux'):  # for Linux using the X Server
            self.videoPlayer.set_xwindow(self.videoFrame.winId())
        elif sys.platform == "win32":  # for Windows
            self.videoPlayer.set_hwnd(self.videoFrame.winId())

        self.hbuttonbox = QHBoxLayout()
        self.hbuttonbox.addStretch(1)
        self.vbuttonbox = QVBoxLayout()
        self.vbuttonbox.addStretch(1)

        self.playbutton = QPushButton("Play",self)
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(self.play_pause)

        #self.stopbutton = QPushButton("Stop",self)
        #self.hbuttonbox.addWidget(self.stopbutton)
        #self.stopbutton.clicked.connect(self.stop)

        self.timetxt = QLabel('0', self)
        self.timetxt.setFont(QFont('Times',11))
        self.timetxt.setAlignment(Qt.AlignCenter)
        self.hbuttonbox.addWidget(self.timetxt)

        self.positionslider = QSlider(Qt.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.setMinimumWidth(500)
        self.positionslider.setTickInterval(1)
        self.positionslider.sliderMoved.connect(self.set_position)
        self.positionslider.sliderPressed.connect(self.set_position)

        self.timer = QTimer(self)
        self.timer.setInterval(500) # polling every 500ms
        self.media_duration=-1
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

        self.timetxt_remain = QLabel('0', self)
        self.timetxt_remain.setFont(QFont('Times',11))
        self.timetxt_remain.setAlignment(Qt.AlignCenter)
        self.hbuttonbox.addWidget(self.timetxt_remain)

        self.speedtxt = QLabel('0', self)
        self.speedtxt.setFont(QFont('Times',12))
        self.speedtxt.setAlignment(Qt.AlignCenter)
        self.hbuttonbox.addWidget(self.speedtxt)

        self.vbuttonbox.addLayout(self.hbuttonbox)
        self.setLayout(self.vbuttonbox)
        '''
        menu_bar = self.menuBar()
        # File menu
        file_menu = menu_bar.addMenu("File")
        # Add actions to file menu
        open_action = QAction("Open video", self)
        #close_action = QAction("Close App", self)
        file_menu.addAction(open_action)
        #file_menu.addAction(close_action)
        open_action.triggered.connect(self.open_file)
        #close_action.triggered.connect(sys.exit)
        '''
        # media object
        self.media = None
        self.media_duration=-1
        #if initial_video is not None and os.path.exists(initial_video):

        print('Loaded default media')
        self.media = vlc.Media(initial_video)
        # setting media to the media player
        self.videoPlayer.set_media(self.media)
        #self.vlcInstance.vlm_set_loop(initial_video, True) # didn't work!
        self.playbutton.setText("Pause")

        #self.videoPlayer.playButton()
        self.mainFrame.setLayout(t_lay_parent)

        self.min_playrate = float(config_data["main"]["min_playrate"])
        self.playrate = 1.0
        self.update_speed()
        self.videoPlayer.play()
        time.sleep(1)
        self.maxtime_ms = self.videoPlayer.get_length()

        self.videoFrame.setMouseTracking(True)
        self.videoFrame.mouseMoveEvent = self.mousemove
        self.last_mouse_move = time.time()

        self.show()

    def mousemove(self,e):
        self.last_mouse_move = time.time()

    def closeEvent(self, event):
        self.videoPlayer.stop()

    def resizeEvent(self, event):
        x = self.rect().getCoords()
        self.playbutton.move(10, x[3] - 35)
        #self.stopbutton.move(120, x[3] - 35)
        self.timetxt.move(105,x[3] - 35)
        self.timetxt.setFixedWidth(100)
        
        self.positionslider.move(200, x[3] - 35)
        self.positionslider.setFixedWidth(x[2]-515)
        
        self.timetxt_remain.move(x[2]-310, x[3] - 35)
        self.timetxt_remain.setFixedWidth(120)
        self.speedtxt.move(x[2] - 190, x[3] - 35)
        self.speedtxt.setFixedWidth(180)
        
        QMainWindow.resizeEvent(self, event)

    def set_position(self):
        """Set the movie position according to the position slider.
        """

        # The vlc MediaPlayer needs a float value between 0 and 1, Qt uses
        # integer variables, so you need a factor; the higher the factor, the
        # more precise are the results (1000 should suffice).

        # Set the media position to where the slider was dragged
        self.timer.stop()
        pos = self.positionslider.value()
        self.videoPlayer.set_position(pos / 1000.0)
        self.timer.start()

    def update_speed(self):
        global current_running_speed

        if DEBUG_SPEED_OVERRIDE is not None:
            self.playrate = max(self.min_playrate,DEBUG_SPEED_OVERRIDE/self.default_speed)
        else:
            if current_running_speed<0:
                self.speedtxt.setStyleSheet("color: red;  background-color: black")
            else:
                self.speedtxt.setStyleSheet("color: black;  background-color: white")
                self.playrate = max(self.min_playrate,current_running_speed/self.default_speed)

        self.speedtxt.setText('%2.1fkmh (%1.2f)' % (current_running_speed,self.playrate))

    def update_ui(self):
        """Updates the user interface"""

        # Set the slider's position to its corresponding media position
        # Note that the setValue function only takes values of type int,
        # so we must first convert the corresponding media position.
        #print('executed update_ui')
        now = time.time()
        if (now - self.last_mouse_move) > 4:
            self.videoFrame.setCursor(QtGui.QCursor(QtCore.Qt.BlankCursor))
        else:
            self.videoFrame.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        pos = self.videoPlayer.get_position()
        media_pos = int(pos * 1000)
        self.positionslider.setValue(media_pos)

        if self.media_duration<=5:
            self.media_duration = self.videoPlayer.get_length()/1000

        remaining = self.media_duration*(1.0-pos)
        if self.media_duration > 10 and remaining<5:
            self.videoPlayer.set_position(0)
            return

        fps = -1
        try:
            fps = self.videoPlayer.get_fps()
        except:
            pass

        fps = fps if fps <150 and fps>5 else -1
        curtime = self.videoPlayer.get_time()
        self.timetxt.setText(self.ms_to_timestr(max(0,curtime))) # + " (%1.0f)" % fps )
        self.timetxt_remain.setText(self.ms_to_timestr(self.maxtime_ms - curtime))  # + " (%1.0f)" % fps )
        self.update_speed()
        self.videoPlayer.set_rate(self.playrate)
        #print("rate is %1.3f" % self.videoPlayer.get_rate())

        # No need to call this function if nothing is played
        if not self.videoPlayer.is_playing():
            self.timer.stop()
            # After the video finished, the play button stills shows "Pause",
            # which is not the desired behavior of a media player.
            # This fixes that "bug".
            if not self.is_paused:
                self.stop()

    def ms_to_timestr(self,ms):
        sec = ms/1000.0
        hours = int(np.floor(sec/60/60))
        sec = sec - hours*60*60
        minutes = int(np.floor(sec/60))
        sec = sec - minutes*60
        return "%02i:%02i:%02.0f" % (hours,minutes,sec)

    def play_pause(self):
        """Toggle play/pause status
        """
        if self.videoPlayer.is_playing():
            self.videoPlayer.pause()
            self.is_paused=True
            self.playbutton.setText("Play")
            self.timer.stop()
        else:
            self.videoPlayer.play()
            self.playbutton.setText("Pause")
            self.is_paused=False
            self.timer.start()

    def stop(self):
        """Stop player
        """
        self.videoPlayer.stop()
        self.playbutton.setText("Play")
    '''
    def open_file(self):
        """Open a media file in a MediaPlayer
        """
        dialog_txt = "Choose Media File"
        filename = QFileDialog.getOpenFileName(self, dialog_txt, os.path.expanduser('~'))
        if not filename:
            print("File was null")
            return
        print("file '%s' loaded" % filename[0],flush=True)
        # getOpenFileName returns a tuple, so use only the actual file name
        self.media = vlc.Media(filename[0])

        # Put the media in the media player
        self.videoPlayer.set_media(self.media)
        #self.vlcInstance.vlm_set_loop(self.media, True)

        # Parse the metadata of the file
        self.media.parse()

        # Set the title of the track as window title
        self.setWindowTitle(self.media.get_meta(0))

        self.videoPlayer.audio_set_volume(0)
        self.playrate = 1.0
        self.media_duration = -1
        self.play_pause()
    '''
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.windowState() == Qt.WindowNoState:
                #self.videoFrame1.hide()
                self.videoFrame.show()
                self.setWindowState(Qt.WindowFullScreen)
            else:
                #self.videoFrame1.show()
                self.setWindowState(Qt.WindowNoState)

# this is the main window that contains list of videos and webcam stream
class MainWindow(QMainWindow):
    # Signal for thread-safe GUI updates from camera thread
    confidence_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("TreadmillApp")
        self.setMinimumSize(1000, 600)

        # State variables - mouse and ROI tracking
        self.mouse_pos = None
        self.selected_ROI = False
        self.coordinate1 = None
        self.coordinate2 = None
        self.coord_capture_context = None  # Store transformation context when coordinates are captured
        self.roi_in_frame_space = False  # True if ROI coordinates are already in frame space (from auto-cal)
        self.roi_lock = threading.Lock()  # Protect ROI coordinate access between threads

        # Frame processing state
        self.frame_shape = None
        self.frame_scaling = None
        self.frame_padding = None
        self.zoom_cut = {}
        self.zoom_multiplier = {}
        self.cached_view_size = (0, 0)  # Cache for QPixmap scaling dimensions
        self.need_transform_recalc = False  # Flag to trigger transformation recalculation after resize

        # File management
        self.filelist = None
        self.chosen_file = None

        # Camera threading
        self.camera_thread = None
        self.latest_full_frame = None  # Store latest unzoomed frame for template capture
        self.show_overlay = False  # Flag to show template overlay instead of live feed
        self.overlay_pixmap = None  # Pixmap to display during overlay mode
        self.overlay_success_message = None  # Message to show after overlay

        # Speed tracking
        self.digitcount = 3
        self.precision = 1
        self.defaultspeed = DEFAULT_SPEED
        self.starttime = time.time()

        # Video window reference
        self.videowindow = None

        # Frame shift state (for manual adjustment)
        self.frame_shift_x = int(float(config_data.get("main", "shift_center_x", fallback="0")))
        self.frame_shift_y = int(float(config_data.get("main", "shift_center_y", fallback="0")))
        self.frame_shift_step = 10  # Pixels per button press

        # Frame flip state (for display orientation)
        self.flip_horizontal = config_data.get("main", "flip_horizontal", fallback="false").lower() == "true"
        self.flip_vertical = config_data.get("main", "flip_vertical", fallback="false").lower() == "true"

        # Debounced config save timer (prevents UI blocking on rapid button presses)
        self.config_save_timer = QTimer()
        self.config_save_timer.setSingleShot(True)
        self.config_save_timer.setInterval(500)  # Save 500ms after last change
        self.config_save_timer.timeout.connect(self.save_configs)

        # Initialize template system for auto-detection
        print("Initializing template system...")
        self.template_manager = TemplateManager()
        self.roi_matcher = ROIMatcher()

        # Cache template config values (avoid repeated string parsing)
        self.auto_calibrate_on_startup_enabled = config_data.get(
            "template", "auto_calibrate_on_startup", fallback="true"
        ).lower() == "true"
        self.matching_threshold = float(config_data.get(
            "template", "matching_threshold", fallback="0.50"
        ))

        # Load predefined ROI from config if available
        try:
            roi_parts = config_data.get("main", "roi").split(",")
            x1, y1, x2, y2, scale = [int(x) for x in roi_parts[:5]]
            # Check for 6th parameter (roi_in_frame_space flag) for backward compatibility
            if len(roi_parts) >= 6:
                try:
                    self.roi_in_frame_space = bool(int(roi_parts[5].strip()))
                except:
                    self.roi_in_frame_space = False  # Default if parsing fails
            else:
                self.roi_in_frame_space = False  # Default to view space for old configs

            assert abs(x1-x2) > 2 and abs(y1-y2) > 2, "Bad ROI in config!"
            self.coordinate1 = (x1, y1, scale)
            self.coordinate2 = (x2, y2, scale)
            self.selected_ROI = True
            print(f"Predefined ROI loaded (frame_space={self.roi_in_frame_space})")
        except Exception as e:
            print(f"No valid ROI in config: {e}")

        # Build UI
        self.setupUi()

        # Load configuration into UI widgets
        self.load_config()

        # Update file list
        self.updatefiles()

        # Start camera
        self.start_camera()

        # Schedule startup auto-calibration (1 second delay to allow camera init)
        if self.auto_calibrate_on_startup_enabled and self.template_combo.currentText() != "(no templates)":
            QTimer.singleShot(1000, self.auto_calibrate_on_startup)

        # Set initial speed
        self.set_running_speed(current_running_speed)

        self.resize(1300,900)

        self.show()

    def setupUi(self):
        """Main UI setup - coordinates sub-methods."""
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Use QSplitter for resizable panels
        from PyQt5.QtWidgets import QSplitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #ddd;
            }
            QSplitter::handle:hover {
                background-color: #bbb;
            }
        """)

        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.main_splitter)

        # Build panels
        self.left_panel = self.build_left_panel()
        self.right_panel = self.build_right_panel()

        # Add to splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)

        # Prevent panels from collapsing completely
        self.main_splitter.setCollapsible(0, False)  # Left panel cannot collapse
        self.main_splitter.setCollapsible(1, False)  # Right panel cannot collapse

        # Set stretch factors (right panel gets more weight)
        self.main_splitter.setStretchFactor(0, 0)  # Left panel: fixed preference
        self.main_splitter.setStretchFactor(1, 1)  # Right panel: expandable

        # Set initial splitter sizes (left panel gets 350px, right panel gets remaining space)
        self.main_splitter.setSizes([350, 650])

        # Enable mouse tracking for ROI selection
        self.setMouseTracking(True)

        # Populate template dropdown
        self.update_template_list()

        # Apply styling
        self.apply_styling()

        # Connect signals
        self.connect_signals()

    def build_left_panel(self):
        """Build left panel with video library and controls."""
        panel = QWidget()
        panel.setMinimumWidth(300)  # Only set minimum, allow resizing

        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video Library Group
        library_group = QGroupBox("Video Library")
        library_layout = QVBoxLayout()

        # Path row
        path_row = QHBoxLayout()
        path_label = QLabel("Path:")
        self.videopathEdit = QLineEdit()
        self.videopathEdit.setText(config_data.get("paths", "video_path", fallback=DEFAULT_PATH))
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self.browse_video_path)

        path_row.addWidget(path_label)
        path_row.addWidget(self.videopathEdit, stretch=1)
        path_row.addWidget(browse_btn)

        # Current video label
        self.currentvideoEdit = QLabel("")
        self.currentvideoEdit.setWordWrap(True)
        self.currentvideoEdit.setStyleSheet("font-size: 11px; color: #666; padding: 5px;")
        self.currentvideoEdit.setFixedHeight(30)

        library_layout.addLayout(path_row)
        library_layout.addWidget(self.currentvideoEdit)
        library_group.setLayout(library_layout)

        # Video list
        self.listWidget = QListWidget()
        self.listWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Video controls group
        controls_group = self.build_video_controls()

        # Add to panel layout
        layout.addWidget(library_group)
        layout.addWidget(self.listWidget, stretch=1)  # Expanding
        layout.addWidget(controls_group)

        return panel

    def build_video_controls(self):
        """Build video controls section."""
        group = QGroupBox("Video Controls")
        layout = QVBoxLayout()

        # Speed row
        speed_row = QHBoxLayout()
        speed_label = QLabel("Speed (km/h):")
        self.speedInput = QDoubleSpinBox()
        self.speedInput.setRange(1.0, 30.0)
        self.speedInput.setSingleStep(0.1)
        self.speedInput.setValue(DEFAULT_SPEED)
        self.speedInput.setSuffix(" km/h")

        speed_row.addWidget(speed_label)
        speed_row.addWidget(self.speedInput, stretch=1)

        # Rating row
        rating_row = QHBoxLayout()
        rating_label = QLabel("Rating:")
        self.scoregoodbutton = QPushButton("✓")
        self.scoregoodbutton.setObjectName("goodButton")
        self.scoregoodbutton.setMaximumWidth(40)
        self.scoregoodbutton.setEnabled(False)

        self.scorebadbutton = QPushButton("✗")
        self.scorebadbutton.setObjectName("badButton")
        self.scorebadbutton.setMaximumWidth(40)
        self.scorebadbutton.setEnabled(False)

        rating_row.addWidget(rating_label)
        rating_row.addWidget(self.scoregoodbutton)
        rating_row.addWidget(self.scorebadbutton)
        rating_row.addStretch()

        # Play button
        self.playButton = QPushButton("Play Video")
        self.playButton.setObjectName("playButton")
        self.playButton.setMinimumHeight(40)
        self.playButton.setEnabled(False)

        layout.addLayout(speed_row)
        layout.addLayout(rating_row)
        layout.addWidget(self.playButton)

        group.setLayout(layout)
        return group

    def browse_video_path(self):
        """Open file dialog to browse for video directory."""
        from PyQt5.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Video Directory", self.videopathEdit.text())
        if directory:
            self.videopathEdit.setText(directory)
            config_data["paths"]["video_path"] = directory
            self.save_configs()

    def build_right_panel(self):
        """Build right panel with camera, speed, and settings."""
        panel = QWidget()
        panel.setMinimumWidth(400)  # Set minimum width to prevent collapse
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # Camera group (gets most space)
        camera_group = self.build_camera_group()

        # Speed group (fixed height)
        speed_group = self.build_speed_group()

        # Settings group (fixed height)
        settings_group = self.build_settings_group()

        # Add to layout with stretch factors
        layout.addWidget(camera_group, stretch=2)  # Gets 2/3 of vertical space
        layout.addWidget(speed_group, stretch=0)   # Fixed ~150px
        layout.addWidget(settings_group, stretch=0) # Fixed ~200px

        return panel

    def build_camera_group(self):
        """Build camera feed section with new frame shift controls."""
        group = QGroupBox("Camera Feed")
        layout = QVBoxLayout()

        # Camera view (expanding)
        self.graphicsView = QLabel()
        self.graphicsView.setObjectName("cameraView")
        self.graphicsView.setMinimumSize(300, 300)  # Reduced for better resizing flexibility
        self.graphicsView.setMaximumHeight(800)  # Prevent vertical expansion over buttons
        self.graphicsView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setStyleSheet("background-color: black; border: 2px solid #ccc;")
        self.graphicsView.setScaledContents(False)

        # Frame shift controls (NEW FEATURE)
        shift_controls = self.build_frame_shift_controls()

        # Camera controls
        camera_controls = self.build_camera_controls()

        layout.addWidget(self.graphicsView, stretch=1)
        layout.addWidget(shift_controls)
        layout.addLayout(camera_controls)

        group.setLayout(layout)
        return group

    def build_frame_shift_controls(self):
        """Build frame shift control buttons (NEW FEATURE)."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)

        # Label
        shift_label = QLabel("Frame Shift:")
        layout.addWidget(shift_label)

        # Arrow buttons
        self.shiftLeftBtn = QPushButton("◄")
        self.shiftLeftBtn.setObjectName("shiftBtn")
        self.shiftLeftBtn.setMaximumWidth(30)
        self.shiftLeftBtn.setMaximumHeight(30)
        self.shiftLeftBtn.setToolTip("Shift frame left by 10 pixels")

        self.shiftRightBtn = QPushButton("►")
        self.shiftRightBtn.setObjectName("shiftBtn")
        self.shiftRightBtn.setMaximumWidth(30)
        self.shiftRightBtn.setMaximumHeight(30)
        self.shiftRightBtn.setToolTip("Shift frame right by 10 pixels")

        self.shiftUpBtn = QPushButton("▲")
        self.shiftUpBtn.setObjectName("shiftBtn")
        self.shiftUpBtn.setMaximumWidth(30)
        self.shiftUpBtn.setMaximumHeight(30)
        self.shiftUpBtn.setToolTip("Shift frame up by 10 pixels")

        self.shiftDownBtn = QPushButton("▼")
        self.shiftDownBtn.setObjectName("shiftBtn")
        self.shiftDownBtn.setMaximumWidth(30)
        self.shiftDownBtn.setMaximumHeight(30)
        self.shiftDownBtn.setToolTip("Shift frame down by 10 pixels")

        # Reset button
        self.shiftResetBtn = QPushButton("Reset")
        self.shiftResetBtn.setObjectName("shiftResetBtn")
        self.shiftResetBtn.setMaximumWidth(60)
        self.shiftResetBtn.setToolTip("Reset frame shift to center (0, 0)")

        # Status label
        self.shiftStatusLabel = QLabel(f"X: {self.frame_shift_x}, Y: {self.frame_shift_y}")
        self.shiftStatusLabel.setStyleSheet("font-size: 12px; color: #000; font-weight: bold; padding: 2px;")

        layout.addWidget(self.shiftLeftBtn)
        layout.addWidget(self.shiftRightBtn)
        layout.addWidget(self.shiftUpBtn)
        layout.addWidget(self.shiftDownBtn)
        layout.addWidget(self.shiftResetBtn)
        layout.addStretch()
        layout.addWidget(self.shiftStatusLabel)

        return widget

    def build_camera_controls(self):
        """Build camera control row with improved flip buttons."""
        layout = QHBoxLayout()

        # Source selection
        source_label = QLabel("Source:")
        self.sourceCombo = QComboBox()
        self.sourceCombo.addItem("Camera 0")
        self.sourceCombo.addItem("Camera 1")
        current_source = int(config_data.get("main", "source", fallback="0"))
        self.sourceCombo.setCurrentIndex(current_source)

        # Tracking checkbox
        self.tracking = QCheckBox("Tracking")
        self.tracking.setChecked(True)

        # Flip buttons - larger with full labels
        self.flipHorizontalBtn = QPushButton("Flip Horizontal")
        self.flipHorizontalBtn.setObjectName("flipBtn")
        self.flipHorizontalBtn.setCheckable(True)
        self.flipHorizontalBtn.setChecked(self.flip_horizontal)
        self.flipHorizontalBtn.setFixedWidth(40)
        self.flipHorizontalBtn.setFixedHeight(25)
        self.flipHorizontalBtn.setStyleSheet(""" QPushButton { padding: 0px; } """)
        self.flipHorizontalBtn.setToolTip("Flip camera view horizontally (mirror left/right)")

        self.flipVerticalBtn = QPushButton("Flip Vertical")
        self.flipVerticalBtn.setObjectName("flipBtn")
        self.flipVerticalBtn.setCheckable(True)
        self.flipVerticalBtn.setChecked(self.flip_vertical)
        self.flipVerticalBtn.setFixedWidth(40)
        self.flipVerticalBtn.setFixedHeight(25)
        self.flipVerticalBtn.setStyleSheet(""" QPushButton { padding: 0px; } """)
        self.flipVerticalBtn.setToolTip("Flip camera view vertically (upside down)")

        layout.addWidget(source_label)
        layout.addWidget(self.sourceCombo)
        layout.addWidget(self.tracking)
        layout.addStretch()
        layout.addWidget(self.flipHorizontalBtn)
        layout.addWidget(self.flipVerticalBtn)

        return layout

    def build_speed_group(self):
        """Build speed display section with template controls."""
        group = QGroupBox("Speed Display & Template")
        main_layout = QHBoxLayout()  # Horizontal split

        # LEFT SECTION: Speed display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # LCD number
        self.lcdNumber = QLCDNumber()
        # Initial digit count - will be updated in load_config() with actual values
        self.lcdNumber.setDigitCount(self.digitcount + (1 if self.precision > 0 else 0))
        self.lcdNumber.setMinimumHeight(50)
        self.lcdNumber.setMaximumHeight(80)
        self.lcdNumber.setSegmentStyle(QLCDNumber.Flat)
        # Set size policy to allow width to shrink
        self.lcdNumber.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.lcdNumber.setStyleSheet("""
            QLCDNumber {
                background-color: #1e1e1e;
                color: #00ff00;
                border: 2px solid #333;
                border-radius: 4px;
            }
        """)

        # Info row
        info_row = QHBoxLayout()
        self.frames_per_sec = QLabel("FPS: --")
        self.frames_per_sec.setStyleSheet("font-size: 11px; color: #666;")

        self.regionselection = QLabel("ROI: Not Set")
        self.regionselection.setStyleSheet("font-size: 11px; color: #666;")

        info_row.addWidget(self.frames_per_sec)
        info_row.addStretch()
        info_row.addWidget(self.regionselection)

        # Confidence progress bar
        self.confidenceBar = QProgressBar()
        self.confidenceBar.setRange(0, 100)
        self.confidenceBar.setValue(0)
        self.confidenceBar.setFormat("Recognition confidence: %p%")
        self.confidenceBar.setTextVisible(True)
        self.confidenceBar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        # Connect signal for thread-safe updates from camera thread
        self.confidence_updated.connect(self.confidenceBar.setValue)

        left_layout.addWidget(self.lcdNumber)
        left_layout.addLayout(info_row)
        left_layout.addWidget(self.confidenceBar)

        # RIGHT SECTION: Template controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # Template dropdown
        template_label = QLabel("Template:")
        template_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.template_combo = QComboBox()
        self.template_combo.setObjectName("template_combo")
        self.template_combo.setMinimumHeight(30)

        # Capture button
        self.capture_template_btn = QPushButton("Capture Template")
        self.capture_template_btn.setObjectName("captureTemplateBtn")
        self.capture_template_btn.setFixedWidth(150)
        self.capture_template_btn.setFixedHeight(30)
        self.capture_template_btn.setStyleSheet(""" QPushButton { padding: 0px; } """)        

        # Auto-calibrate button
        self.auto_calibrate_btn = QPushButton("Auto-Calibrate")
        self.auto_calibrate_btn.setObjectName("autoCalibrateBtn")
        self.auto_calibrate_btn.setFixedWidth(150)
        self.auto_calibrate_btn.setFixedHeight(30)
        self.auto_calibrate_btn.setStyleSheet(""" QPushButton { padding: 0px; } """)        

        right_layout.addWidget(template_label)
        right_layout.addWidget(self.template_combo)
        right_layout.addWidget(self.capture_template_btn)
        right_layout.addWidget(self.auto_calibrate_btn)
        right_layout.addStretch()

        # Add both sections to main horizontal layout
        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(right_widget, stretch=1)

        group.setLayout(main_layout)
        return group

    def build_settings_group(self):
        """Build settings section (all visible, no collapsing)."""
        group = QGroupBox("Settings")
        layout = QFormLayout()
        layout.setSpacing(8)

        # Digit count
        self.digitCountSpin = QSpinBox()
        self.digitCountSpin.setRange(1, 5)
        self.digitCountSpin.setValue(3)
        layout.addRow("Digit Count:", self.digitCountSpin)

        # Precision
        self.precisionSpin = QSpinBox()
        self.precisionSpin.setRange(0, 3)
        self.precisionSpin.setValue(1)
        layout.addRow("Precision:", self.precisionSpin)

        # Zoom level
        self.zoomlevel = QSlider(Qt.Horizontal)
        self.zoomlevel.setRange(1, 3)
        self.zoomlevel.setValue(1)
        self.zoomlevel.setTickPosition(QSlider.TicksBelow)
        self.zoomlevel.setTickInterval(1)
        layout.addRow("Zoom Level:", self.zoomlevel)

        group.setLayout(layout)
        return group

    def shift_frame_left(self):
        """Shift frame left by step pixels."""
        self.frame_shift_x -= self.frame_shift_step
        self.update_frame_shift()

    def shift_frame_right(self):
        """Shift frame right by step pixels."""
        self.frame_shift_x += self.frame_shift_step
        self.update_frame_shift()

    def shift_frame_up(self):
        """Shift frame up by step pixels."""
        self.frame_shift_y -= self.frame_shift_step
        self.update_frame_shift()

    def shift_frame_down(self):
        """Shift frame down by step pixels."""
        self.frame_shift_y += self.frame_shift_step
        self.update_frame_shift()

    def reset_frame_shift(self):
        """Reset frame shift to center (0, 0)."""
        self.frame_shift_x = 0
        self.frame_shift_y = 0
        self.update_frame_shift()

    def update_frame_shift(self):
        """Update status label and apply shift to frame processing."""
        self.shiftStatusLabel.setText(f"X: {self.frame_shift_x}, Y: {self.frame_shift_y}")

        # Update config (in memory)
        config_data["main"]["shift_center_x"] = str(self.frame_shift_x)
        config_data["main"]["shift_center_y"] = str(self.frame_shift_y)

        # Schedule debounced save (prevents UI blocking on rapid button presses)
        self.config_save_timer.stop()  # Reset timer if already running
        self.config_save_timer.start()  # Start/restart 500ms countdown

        # Update globals for camera thread (immediate effect)
        global SHIFT_CENTER_X, SHIFT_CENTER_Y
        SHIFT_CENTER_X = self.frame_shift_x
        SHIFT_CENTER_Y = self.frame_shift_y

    def connect_signals(self):
        """Connect all UI signals to handler methods."""
        # Left panel - video library
        self.videopathEdit.textChanged.connect(self.updatefiles)
        self.listWidget.itemClicked.connect(self.onClickedFile)
        self.listWidget.currentItemChanged.connect(self.onClickedFile)

        # Left panel - video controls
        self.speedInput.valueChanged.connect(self.on_speed_changed)
        self.scoregoodbutton.clicked.connect(self.on_click_button_good)
        self.scorebadbutton.clicked.connect(self.on_click_button_bad)
        self.playButton.clicked.connect(self.play_movie)

        # Camera controls
        self.sourceCombo.currentIndexChanged.connect(self.on_source_changed)
        self.tracking.toggled.connect(self.on_tracking_toggled)
        self.flipHorizontalBtn.clicked.connect(self.on_flip_horizontal)
        self.flipVerticalBtn.clicked.connect(self.on_flip_vertical)
        self.template_combo.currentIndexChanged.connect(self.on_template_changed)
        self.capture_template_btn.clicked.connect(self.capture_template)
        self.auto_calibrate_btn.clicked.connect(self.auto_calibrate_roi)

        # Frame shift controls (NEW)
        self.shiftLeftBtn.clicked.connect(self.shift_frame_left)
        self.shiftRightBtn.clicked.connect(self.shift_frame_right)
        self.shiftUpBtn.clicked.connect(self.shift_frame_up)
        self.shiftDownBtn.clicked.connect(self.shift_frame_down)
        self.shiftResetBtn.clicked.connect(self.reset_frame_shift)

        # Settings
        self.digitCountSpin.valueChanged.connect(self.on_digit_count_changed)
        self.precisionSpin.valueChanged.connect(self.on_precision_changed)
        self.zoomlevel.valueChanged.connect(self.zoom_level_changed)

        # Mouse events on camera view
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.mouseMoveEvent = self.mousemove
        self.graphicsView.mousePressEvent = self.mousedown
        self.graphicsView.mouseReleaseEvent = self.mouseup

    def on_speed_changed(self, value):
        """Handle speed input change."""
        self.defaultspeed = value
        config_data["main"]["default_speed"] = str(value)
        self.save_configs()

    def on_source_changed(self, index):
        """Handle camera source change from combo box."""
        config_data["main"]["source"] = str(index)
        self.save_configs()
        self.start_camera()  # Restart camera with new source

    def on_tracking_toggled(self, checked):
        """Handle tracking checkbox toggle."""
        if checked:
            self.start_camera()

    def on_flip_horizontal(self):
        """Handle horizontal flip button click."""
        self.flip_horizontal = self.flipHorizontalBtn.isChecked()
        config_data["main"]["flip_horizontal"] = str(self.flip_horizontal).lower()
        self.config_save_timer.start()  # Debounced save

    def on_flip_vertical(self):
        """Handle vertical flip button click."""
        self.flip_vertical = self.flipVerticalBtn.isChecked()
        config_data["main"]["flip_vertical"] = str(self.flip_vertical).lower()
        self.config_save_timer.start()  # Debounced save

    def on_digit_count_changed(self, value):
        """Handle digit count change from spin box."""
        self.digitcount = value
        print(f"Digit count changed to: {value}")
        config_data["main"]["digit_count"] = str(value)
        self.save_configs()
        # Update LCD size based on digitcount and precision
        self.update_lcd_size()

    def on_precision_changed(self, value):
        """Handle precision change from spin box."""
        self.precision = value
        print(f"Precision changed to: {value}")
        config_data["main"]["precision"] = str(value)
        self.save_configs()
        # Update LCD size based on digitcount and precision
        self.update_lcd_size()

    def apply_styling(self):
        """Apply modern styling to application."""
        stylesheet = """
        /* Main Window */
        QMainWindow {
            background-color: #f5f5f5;
        }

        /* Group Boxes */
        QGroupBox {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-top: 12px;
            padding: 15px 10px 10px 10px;
            font-weight: bold;
            font-size: 13px;
            color: #333;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
        }

        /* Primary Buttons */
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 13px;
        }

        QPushButton:hover {
            background-color: #1976D2;
        }

        QPushButton:pressed {
            background-color: #0D47A1;
        }

        QPushButton:disabled {
            background-color: #BDBDBD;
            color: #757575;
        }

        /* Play Button */
        QPushButton#playButton {
            background-color: #4CAF50;
            font-size: 14px;
            font-weight: bold;
            min-height: 40px;
        }

        QPushButton#playButton:hover {
            background-color: #45a049;
        }

        QPushButton#playButton:disabled {
            background-color: #A5D6A7;
        }

        /* Rating Buttons */
        QPushButton#goodButton {
            background-color: #4CAF50;
            font-size: 18px;
            max-width: 40px;
        }

        QPushButton#badButton {
            background-color: #f44336;
            font-size: 18px;
            max-width: 40px;
        }

        /* Template Buttons */
        QPushButton#captureTemplateBtn,
        QPushButton#autoCalibrateBtn {
            background-color: #FF9800;
            padding: 6px 12px;
            font-size: 12px;
        }

        QPushButton#captureTemplateBtn:hover,
        QPushButton#autoCalibrateBtn:hover {
            background-color: #F57C00;
        }

        /* Frame Shift Buttons */
        QPushButton#shiftBtn {
            background-color: #607D8B;
            font-size: 16px;
            max-width: 30px;
            max-height: 30px;
            padding: 4px;
        }

        QPushButton#shiftResetBtn {
            background-color: #9E9E9E;
            max-width: 60px;
        }

        /* Flip Buttons */
        QPushButton#flipBtn {
            background-color: #757575;
            min-width: 120px;
            min-height: 28px;
            font-size: 12px;
        }

        QPushButton#flipBtn:hover {
            background-color: #616161;
        }

        QPushButton#flipBtn:checked {
            background-color: #FFA726;
            font-weight: bold;
        }

        QPushButton#flipBtn:checked:hover {
            background-color: #FF9800;
        }

        /* Input Fields */
        QLineEdit,
        QDoubleSpinBox,
        QSpinBox {
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 5px;
            background-color: white;
            selection-background-color: #2196F3;
        }

        QLineEdit:focus,
        QDoubleSpinBox:focus,
        QSpinBox:focus {
            border: 1px solid #2196F3;
            outline: none;
        }

        /* Combo Boxes */
        QComboBox {
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 5px;
            background-color: white;
            min-width: 100px;
        }

        QComboBox:hover {
            border: 1px solid #2196F3;
        }

        /* List Widget */
        QListWidget {
            border: 1px solid #ccc;
            border-radius: 3px;
            background-color: white;
        }

        QListWidget::item {
            padding: 5px;
            border-bottom: 1px solid #eee;
        }

        QListWidget::item:selected {
            background-color: #2196F3;
            color: white;
        }

        QListWidget::item:hover {
            background-color: #e3f2fd;
        }

        /* Checkboxes */
        QCheckBox {
            spacing: 5px;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }

        /* Sliders */
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: white;
            height: 8px;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background: #2196F3;
            border: 1px solid #1976D2;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }

        QSlider::handle:horizontal:hover {
            background: #1976D2;
        }

        /* Camera View */
        QLabel#cameraView {
            background-color: black;
            border: 2px solid #ccc;
            border-radius: 4px;
        }
        """

        self.setStyleSheet(stylesheet)

    def resizeEvent(self, event):
        """Handle window resize - maintain camera view aspect ratio."""
        super().resizeEvent(event)

        # Get camera view size
        if hasattr(self, 'graphicsView'):
            view_size = self.graphicsView.size()
            pixmap = self.graphicsView.pixmap()

            if pixmap and not pixmap.isNull():
                # Scale pixmap to fit view while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    view_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.graphicsView.setPixmap(scaled_pixmap)

                # Trigger recalculation of transformation parameters in camera thread
                self.need_transform_recalc = True

                # Clear any in-progress ROI selection since transforms will change
                if self.coordinate1 is not None and self.coordinate2 is None:
                    self.coordinate1 = None
                    self.coord_capture_context = None

    def calculate_lcd_digit_count(self):
        """Calculate the required LCD digit count based on digitcount and precision.

        Examples:
        - digitcount=3, precision=1 -> "12.3" needs 4 characters (3 digits + decimal point)
        - digitcount=3, precision=0 -> "123" needs 3 characters
        - digitcount=5, precision=2 -> "123.45" needs 6 characters
        """
        return self.digitcount + (1 if self.precision > 0 else 0)

    def update_lcd_size(self):
        """Update LCD digit count and maximum width based on current digitcount and precision."""
        digit_count = self.calculate_lcd_digit_count()
        self.lcdNumber.setDigitCount(digit_count)
        # Each LCD digit is approximately 35-40 pixels wide, plus some padding
        # Use 40 pixels per character + 30 pixels for borders/padding
        max_width = digit_count * 40 + 30
        self.lcdNumber.setMaximumWidth(max_width)
        print(f"LCD updated: {digit_count} digits, max width: {max_width}px")

    def load_config(self):
        """Load configuration into UI widgets."""
        # Video path - already loaded in build_left_panel

        # Speed - already set to DEFAULT_SPEED in build_video_controls

        # Digit count
        digit_count = int(float(config_data.get("main", "digit_count", fallback="3")))
        self.digitCountSpin.setValue(digit_count)
        self.digitcount = digit_count

        # Precision
        precision = int(float(config_data.get("main", "precision", fallback="1")))
        self.precisionSpin.setValue(precision)
        self.precision = precision

        # Update LCD size based on both digitcount and precision
        self.update_lcd_size()

        # Zoom
        zoom = int(float(config_data.get("main", "zoom_level", fallback="1")))
        self.zoomlevel.setValue(zoom)

        # Set active template from config
        active_template = config_data.get("template", "active_template", fallback="default")
        index = self.template_combo.findText(active_template)
        if index >= 0:
            self.template_combo.setCurrentIndex(index)

        # ROI status label
        if self.selected_ROI:
            self.regionselection.setText("ROI: Selected")
        else:
            self.regionselection.setText("ROI: Not Set")

    def on_click_button_good(self):
        if self.chosen_file is not None:
            if self.chosen_file["score"]==1:
                self.chosen_file["score"]=0
                self.scoregoodbutton.setStyleSheet("background-color: white;")
            else:
                self.chosen_file["score"] = 1
                self.scoregoodbutton.setStyleSheet("background-color: green;")
            self.scorebadbutton.setStyleSheet("background-color: white;")
            self.update_filedata()
            self.save_configs()

    def on_click_button_bad(self):
        if self.chosen_file is not None:
            if self.chosen_file["score"] == -1:
                self.chosen_file["score"] = 0
                self.scorebadbutton.setStyleSheet("background-color: white;")
            else:
                self.chosen_file["score"] = -1
                self.scorebadbutton.setStyleSheet("background-color: red;")
            self.scoregoodbutton.setStyleSheet("background-color: white;")
            self.update_filedata()
            self.save_configs()
    def edited_speed(self):
        pass

    def set_running_speed(self,val):
        # Digit count is now managed by update_lcd_size()
        self.lcdNumber.display(val)

    def correct_mouse_coords_for_flip(self, x, y):
        """Correct mouse coordinates to account for view flips.

        When the view is flipped, mouse coordinates need to be inverted to match
        the original frame coordinates for ROI selection.
        """
        view_width = self.graphicsView.width()
        view_height = self.graphicsView.height()

        if self.flip_horizontal:
            x = view_width - x
        if self.flip_vertical:
            y = view_height - y

        return x, y

    def mousemove(self,e):
        if self.frame_shape is not None:
            #zoom_level = self.zoomlevel.value()
            x = e.x()
            y = e.y()
            # Correct coordinates for view flips
            x, y = self.correct_mouse_coords_for_flip(x, y)
            #print("mouse coord current: x=%i y=%i" % (int(x), int(y)))
            self.mouse_pos = (x, y)

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

    # webcam frame click release, gets coordinate 2
    def mouseup(self,e):
        # Record ending (x,y) coordintes on left mouse bottom release
        pass
        '''
        if e.button()==1 and not(self.selected_ROI) and self.frame_shape is not None:
            x = e.x()
            y = e.y()
            self.coordinate2 = (x,y,self.zoomlevel.value())
            print("mouse coord 2 (x, y) = (%i, %i)" % (self.coordinate2[0], self.coordinate2[1]))
            self.selected_ROI = True

            # update config file
            config_data["main"]["roi"] = "%i,%i,%i,%i,%i" % (self.coordinate1[0],self.coordinate1[1],self.coordinate2[0],self.coordinate2[1],self.zoomlevel.value())
            with open(CONFIG_FILE, 'w',encoding="UTF-8") as configfile:
                config_data.write(configfile)
        '''
    def zoom_level_changed(self,value):
        config_data["main"]['zoom_level'] = str(value)
        # Use debounced save to prevent UI blocking
        self.config_save_timer.stop()
        self.config_save_timer.start()

    def save_configs(self):
        with open(CONFIG_FILE, 'w', encoding="UTF-8") as configfile:
            config_data.write(configfile)
    # clicked webcam frame
    def mousedown(self, e):
        # Record starting (x,y) coordinates on left mouse button click
        if e.button()==1 and self.frame_shape is not None:
            x = e.x()
            y = e.y()
            # Correct coordinates for view flips
            x, y = self.correct_mouse_coords_for_flip(x, y)

            # If ROI is complete (both coordinates set), clear it to start fresh
            if self.selected_ROI and self.coordinate1 is not None and self.coordinate2 is not None:
                self.selected_ROI = False
                self.coordinate1 = None
                self.coordinate2 = None
                self.coord_capture_context = None
                self.roi_in_frame_space = False
                print("Cleared existing ROI, click again to start drawing")
                return

            if self.coordinate1 is None:
                self.coordinate1 = (x, y, self.zoomlevel.value())
                self.coordinate2 = None
                self.selected_ROI = False
                print("mouse coord 1 (x, y) = (%i, %i)" % (self.coordinate1[0], self.coordinate1[1]))
            else:
                # Normalize coordinates so ROI can be drawn in any direction
                # This allows drawing from any corner (top-left, bottom-right, etc.)
                x1 = min(self.coordinate1[0], x)
                y1 = min(self.coordinate1[1], y)
                x2 = max(self.coordinate1[0], x)
                y2 = max(self.coordinate1[1], y)

                self.coordinate1 = (x1, y1, self.zoomlevel.value())
                self.coordinate2 = (x2, y2, self.zoomlevel.value())
                print("mouse coord 2 (x, y) = (%i, %i)" % (x, y))
                print("Normalized ROI: (%i, %i) to (%i, %i)" % (x1, y1, x2, y2))
                self.selected_ROI = True

                # Convert to FRAME space immediately and mark as such so it won't be affected by window resize
                if self.frame_padding is not None and self.frame_scaling is not None:
                    # Convert VIEW coordinates to FRAME coordinates
                    zoom_level = self.coordinate1[2]
                    fx1 = (self.coordinate1[0] - self.frame_padding["width"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["width"]
                    fy1 = (self.coordinate1[1] - self.frame_padding["height"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["height"]
                    fx2 = (self.coordinate2[0] - self.frame_padding["width"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["width"]
                    fy2 = (self.coordinate2[1] - self.frame_padding["height"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["height"]

                    # Normalize
                    fx1, fx2 = min(fx1, fx2), max(fx1, fx2)
                    fy1, fy2 = min(fy1, fy2), max(fy1, fy2)

                    # Scale to frame
                    fx1 = int(fx1 * self.frame_scaling["width"])
                    fy1 = int(fy1 * self.frame_scaling["height"])
                    fx2 = int(fx2 * self.frame_scaling["width"])
                    fy2 = int(fy2 * self.frame_scaling["height"])

                    # Store as FRAME coordinates
                    self.coordinate1 = (fx1, fy1, 0)
                    self.coordinate2 = (fx2, fy2, 0)
                    self.roi_in_frame_space = True
                    print("Converted to FRAME space: (%i, %i) to (%i, %i)" % (fx1, fy1, fx2, fy2))
                else:
                    self.roi_in_frame_space = False  # Manual selection: coordinates are in VIEW space

                # update config file
                config_data["main"]["roi"] = "%i,%i,%i,%i,%i,%i" % (
                    self.coordinate1[0], self.coordinate1[1], self.coordinate2[0], self.coordinate2[1],
                    self.zoomlevel.value(), 1 if self.roi_in_frame_space else 0)

                # Stop any pending debounced saves to prevent race conditions
                self.config_save_timer.stop()
                self.save_configs()

        # Clear drawing boxes on right mouse button click
        elif e.button()==2:
            self.selected_ROI = False
            self.coordinate1 = None
            self.coordinate2 = None
            self.coord_capture_context = None
            self.roi_in_frame_space = False
            self.regionselection.setText("No ROI selected")

    # begin recording
    def start_camera(self):
        global current_running_speed
        if self.tracking.isChecked():
            device = self.sourceCombo.currentIndex()
            self.camera_thread = threading.Thread(target=self.camera_update, args=[device])
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            current_running_speed=DEFAULT_SPEED # self.defaultspeed
            self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
            self.frames_per_sec.setText("-")

    # function to capture and analyze webcam stream (running as separate thread)
    def camera_update(self,device):
        global running_speed_history,current_running_speed
        INITIAL_FRAMES = 8 # initial frames to set-up template frame

        DEBUGGING_MODE = False
        if len(WEBCAM_VIDEO)>0:
            DEBUGGING_MODE = True
            assert os.path.exists(WEBCAM_VIDEO),"Webcam video not found!"
            print("!! Loading webcam video instead of actual stream (DEBUGGING)!!")
            self.capture = cv2.VideoCapture(WEBCAM_VIDEO)
            old_images= [None,None,None]
            image_num = 0
            debug_fig,debug_ax = plt.subplots(1, 3, figsize=[3 * 3,3])
            speed_estimation_file = open("speed_predictions_debug.csv","w")
        else:
            self.capture = cv2.VideoCapture(device,cv2.CAP_DSHOW)
        old_device = device
        rectangles = None
        last_prediction_time = time.time() - PREDICT_INTERVAL

        print("Starting video capture")
        init_run=0
        template_frame = 0 # this will be FFT transformed
        # Now using instance variables self.zoom_cut and self.zoom_multiplier
        old_ROI=None

        # Track shift values to detect changes and rebuild template
        last_shift_x = SHIFT_CENTER_X
        last_shift_y = SHIFT_CENTER_Y

        # convert view coordinates to actual coordinates in acquired frame. Need to take into account padding and zooming!
        def view_to_frame(coordinate1, coordinate2):
            """
            Convert VIEW space coordinates to FRAME space coordinates.

            Args:
                coordinate1: (x, y, zoom) in VIEW space
                coordinate2: (x, y, zoom) in VIEW space
            """
            zoom_level = coordinate1[2]

            # remove padded pixels outside frame, convert to zoom=1 scale and add cropped pixels
            x1 = (coordinate1[0] - self.frame_padding["width"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["width"]
            y1 = (coordinate1[1] - self.frame_padding["height"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["height"]

            x2 = (coordinate2[0] - self.frame_padding["width"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["width"]
            y2 = (coordinate2[1] - self.frame_padding["height"]) * self.zoom_multiplier[zoom_level] + self.zoom_cut[zoom_level]["view"]["height"]

            # Normalize coordinates using tuple unpacking (evaluates RHS before assignment)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # scale coordinates to frame
            x1 = int(x1 * self.frame_scaling["width"])
            y1 = int(y1 * self.frame_scaling["height"])
            x2 = int(x2 * self.frame_scaling["width"])
            y2 = int(y2 * self.frame_scaling["height"])

            return x1, y1, x2, y2
        
        last_event = 0
        frame_num = 0
        while True:
            now = time.time()
            if self.capture.isOpened():
                if self.sourceCombo.currentIndex() != old_device or not(self.tracking.isChecked()):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    current_running_speed = self.defaultspeed
                    self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
                    return
                # Read frame
                #if init_run >= 5:
                #    frame = template_frame_real
                #else:
                (status, frame) = self.capture.read()

                if frame is None or not(isinstance(frame,np.ndarray)):
                    if (now - last_event >= 3.0):
                        print('obtained frame is not valid (type %s)!' % str(type(frame)))
                    continue

                # Check if shift values changed - if so, reset template
                if SHIFT_CENTER_X != last_shift_x or SHIFT_CENTER_Y != last_shift_y:
                    print(f"Shift changed: ({last_shift_x},{last_shift_y}) -> ({SHIFT_CENTER_X},{SHIFT_CENTER_Y}), resetting template")
                    init_run = 0
                    template_frame = 0
                    last_shift_x = SHIFT_CENTER_X
                    last_shift_y = SHIFT_CENTER_Y

                #print('frame.shape = %s' % str(frame.shape))
                frame = np.roll(frame,SHIFT_CENTER_X,axis=1)
                frame = np.roll(frame,SHIFT_CENTER_Y,axis=0)

                frame_num+=1

                if init_run<INITIAL_FRAMES:

                    image = QtGui.QImage(frame.data.tobytes(), frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
                    pixmap = QtGui.QPixmap.fromImage(image)

                    # Cache view size for scaling
                    current_view_size = (self.graphicsView.width(), self.graphicsView.height())
                    if current_view_size != self.cached_view_size:
                        self.cached_view_size = current_view_size
                    pixmap = pixmap.scaled(self.cached_view_size[0], self.cached_view_size[1], QtCore.Qt.KeepAspectRatio)

                    # Apply flips if enabled
                    if self.flip_horizontal or self.flip_vertical:
                        transform = QtGui.QTransform()
                        transform.scale(-1 if self.flip_horizontal else 1, -1 if self.flip_vertical else 1)
                        pixmap = pixmap.transformed(transform)

                    init_run+=1
                    template_frame += frame.astype(float)/INITIAL_FRAMES
                    if init_run==INITIAL_FRAMES:
                        #template_frame_real = template_frame.astype(np.uint8)
                        #print('template_frame.shape = %s' % str(template_frame.shape))
                        
                        template_frame = np.fft.fft2(cv2.cvtColor(template_frame.astype(np.uint8),cv2.COLOR_BGR2GRAY))

                        self.frame_shape = {"width":frame.shape[1],"height":frame.shape[0]}  # put this here to make sure we have also padding defined
                        self.frame_scaling = {"width":self.frame_shape["width"]/pixmap.width(),"height":self.frame_shape["height"]/pixmap.height()}
                        self.frame_padding = {"width":int((self.graphicsView.width() - pixmap.width()) / 2),"height":int((self.graphicsView.height() - pixmap.height()) / 2)}

                        for zoom_level in [1,2,3]:
                            self.zoom_cut[zoom_level] = {}
                            self.zoom_cut[zoom_level]["orig"] = {
                                "width":((self.frame_shape["width"] / 2) * (zoom_level-1) / 4),
                                "height":((self.frame_shape["height"] / 2) * (zoom_level-1) / 4),
                            }
                            self.zoom_cut[zoom_level]["view"] = {
                                "width":int(self.zoom_cut[zoom_level]["orig"]["width"]/self.frame_scaling["width"]),
                                "height":int(self.zoom_cut[zoom_level]["orig"]["height"]/self.frame_scaling["height"]),
                            }
                            self.zoom_cut[zoom_level]["orig"]["width"]=int(self.zoom_cut[zoom_level]["orig"]["width"])
                            self.zoom_cut[zoom_level]["orig"]["height"] = int(self.zoom_cut[zoom_level]["orig"]["height"])

                            self.zoom_multiplier[zoom_level] = 1 - (zoom_level-1)/4

                        self.graphicsView.setMouseTracking(True)
                        self.graphicsView.mouseMoveEvent = self.mousemove
                        self.graphicsView.mousePressEvent = self.mousedown
                        self.graphicsView.mouseReleaseEvent = self.mouseup

                else:
                    # Check if we need to recalculate transformation parameters (after window resize)
                    if self.need_transform_recalc:
                        # Get current pixmap to recalculate transformations
                        image = QtGui.QImage(frame.data.tobytes(), frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
                        pixmap = QtGui.QPixmap.fromImage(image)

                        # Update cache
                        current_view_size = (self.graphicsView.width(), self.graphicsView.height())
                        self.cached_view_size = current_view_size
                        pixmap = pixmap.scaled(self.cached_view_size[0], self.cached_view_size[1], QtCore.Qt.KeepAspectRatio)

                        # Apply flips if enabled
                        if self.flip_horizontal or self.flip_vertical:
                            transform = QtGui.QTransform()
                            transform.scale(-1 if self.flip_horizontal else 1, -1 if self.flip_vertical else 1)
                            pixmap = pixmap.transformed(transform)

                        # Recalculate transformation parameters with new pixmap size
                        self.frame_scaling = {"width":self.frame_shape["width"]/pixmap.width(),"height":self.frame_shape["height"]/pixmap.height()}
                        self.frame_padding = {"width":int((self.graphicsView.width() - pixmap.width()) / 2),"height":int((self.graphicsView.height() - pixmap.height()) / 2)}

                        for zoom_level in [1,2,3]:
                            self.zoom_cut[zoom_level] = {}
                            self.zoom_cut[zoom_level]["orig"] = {
                                "width":((self.frame_shape["width"] / 2) * (zoom_level-1) / 4),
                                "height":((self.frame_shape["height"] / 2) * (zoom_level-1) / 4),
                            }
                            self.zoom_cut[zoom_level]["view"] = {
                                "width":int(self.zoom_cut[zoom_level]["orig"]["width"]/self.frame_scaling["width"]),
                                "height":int(self.zoom_cut[zoom_level]["orig"]["height"]/self.frame_scaling["height"]),
                            }
                            self.zoom_cut[zoom_level]["orig"]["width"]=int(self.zoom_cut[zoom_level]["orig"]["width"])
                            self.zoom_cut[zoom_level]["orig"]["height"] = int(self.zoom_cut[zoom_level]["orig"]["height"])

                            self.zoom_multiplier[zoom_level] = 1 - (zoom_level-1)/4

                        self.need_transform_recalc = False
                        print("Recalculated transformation parameters after resize")

                        # If there's a selected ROI in VIEW space, clear it since the transformation changed
                        # (New ROIs are immediately converted to FRAME space, so this only affects legacy ROIs)
                        if self.selected_ROI and not self.roi_in_frame_space:
                            print("Clearing VIEW-space ROI after resize (please redefine)")
                            self.selected_ROI = False
                            self.coordinate1 = None
                            self.coordinate2 = None
                            self.roi_in_frame_space = False

                    frame = register_image(template_frame,frame)

                    # Store clean registered frame (before drawing ROI boxes) for template capture
                    self.latest_full_frame = frame.copy()

                    if self.selected_ROI:
                        # Get ROI coordinates
                        # Check if coordinates are already in frame space (from auto-cal) or view space (from manual click)
                        if self.roi_in_frame_space:
                            # Coordinates are already in frame space (from homography transform)
                            # Use them directly without view_to_frame() transformation
                            x1, y1, x2, y2 = self.coordinate1[0], self.coordinate1[1], self.coordinate2[0], self.coordinate2[1]
                        else:
                            # Coordinates are in view space (from manual click)
                            # Need to convert to frame space
                            x1,y1,x2,y2 = view_to_frame(self.coordinate1, self.coordinate2)

                        ROI = [x1,y1,x2,y2,self.digitcount] # [x1,y1,x2,y2,digit_count]

                        # Validate ROI
                        if (x2-x1)<40 or (y2-y1)<20 or x1<0 or y1<0 or (ROI[2]>frame.shape[1]) or (ROI[3]>frame.shape[0]) or (self.coordinate1[2] != self.coordinate2[2]):
                            print("Bad ROI, resetting (x1=%i, y1=%i, x2=%i, y2=%i, frame.shape=%s)" % (x1,y1,x2,y2,str(frame.shape)))
                            self.selected_ROI = False
                            self.coordinate1 = None
                            self.coordinate2 = None
                            self.coord_capture_context = None
                            self.roi_in_frame_space = False
                            continue

                        # Print ROI only on first use or when it changes significantly
                        if old_ROI is None or abs(old_ROI[0] - ROI[0]) > 10 or abs(old_ROI[1] - ROI[1]) > 10:
                            print("ROI coordinates: x1=%i, y1=%i, x2=%i, y2=%i" % (x1,y1,x2,y2))
                            old_ROI = ROI

                        # if enough time passed, analyze digits
                        if (now - last_prediction_time)*1000 > PREDICT_INTERVAL:
                            predicted_digits,predicted_prob,rectangles,raw_frames = predict_digit(frame, ROI)
                            if predicted_digits is None:
                                # failed, just use last valid prediction
                                predicted_speed = current_running_speed
                                predicted_prob = 0
                            else:
                                # convert individual digits to float using selected precision
                                if 0:#DEBUGGING_MODE:
                                    for k in range(3):
                                        if old_images[k] is not None:
                                            old_images[k].set_data(raw_frames[k])
                                        else:
                                            old_images[k]=debug_ax[k].imshow(raw_frames[k])
                                        if np.random.rand()<1/10 and predicted_digits[k]>-1:
                                            image_num += 1
                                            plt.imsave("%i.png" % (image_num), raw_frames[k])
                                predicted_digits_str = "".join([str(x) if x > -1 else "0" for x in predicted_digits])
                                predicted_digits_str = predicted_digits_str[0:(len(predicted_digits)-self.precision)] + "." + predicted_digits_str[-self.precision:]
                                predicted_speed = float(predicted_digits_str)

                            timestamp = now - self.starttime
                            if DEBUGGING_MODE:
                                speed_estimation_file.writelines("frame=%i, predicted_speed=%f, prob=%f\n" % (frame_num,predicted_speed,predicted_prob))
                                speed_estimation_file.flush()

                            running_speed_history.append((timestamp, predicted_speed,predicted_prob))
                            # a function to obtain speed, could include some fancy smoothing and error-checking
                            current_running_speed = speed_estimator(timestamp,running_speed_history)

                            # only keep last 100 measurements
                            if len(running_speed_history)>50:
                                running_speed_history = running_speed_history[-50:]

                            self.lcdNumber.display("{1:,.{0}f}".format(self.precision,current_running_speed))
                            self.frames_per_sec.setText("%.2f frames/sec" % (1.0 / (max(0.0001,time.time()-now))))
                            # Update confidence bar with digit recognition confidence (thread-safe via signal)
                            self.confidence_updated.emit(int(predicted_prob * 100))
                            last_prediction_time = now

                        frame = cv2.rectangle(frame, (ROI[0],ROI[1]),(ROI[2],ROI[3]), (0, 255, 0),3)
                        if rectangles is not None:
                            for k,rect in enumerate(rectangles):
                                col = (0, 200,200) if (k%2==0) else (200,0,200)
                                frame = cv2.rectangle(frame,
                                                    (rect[0],rect[2]),
                                                    (rect[1],rect[3]),
                                                    col,2)

                        self.regionselection.setText("ROI (%i,%i,%i,%i)" % (x1,y1,x2,y2))

                        #for r in self.digit_ROIs:
                        #    self.frame = cv2.rectangle(self.frame, (r[1][0]+x1,r[1][1]+y1), (r[1][2]+x1,r[1][3]+y1), (0,50,255), 2)
                    elif self.coordinate1 is not None:
                        # Draw preview rectangle while user is selecting second coordinate
                        mouse_pos_with_zoom = self.mouse_pos + (self.coordinate1[2],)  # Add zoom level to mouse_pos
                        x1, y1, x2, y2 = view_to_frame(self.coordinate1, mouse_pos_with_zoom)
                        frame = cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 2)
                        rectangles = None
                        old_ROI = None

                    zoom_level = self.zoomlevel.value()
                    if zoom_level > 1:
                        frame = frame[self.zoom_cut[zoom_level]["orig"]["height"]:-self.zoom_cut[zoom_level]["orig"]["height"], self.zoom_cut[zoom_level]["orig"]["width"]:-self.zoom_cut[zoom_level]["orig"]["width"], :]

                    image = QtGui.QImage(frame.data.tobytes(), frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
                    pixmap = QtGui.QPixmap.fromImage(image)

                    # Cache view size for scaling
                    current_view_size = (self.graphicsView.width(), self.graphicsView.height())
                    if current_view_size != self.cached_view_size:
                        self.cached_view_size = current_view_size
                    pixmap = pixmap.scaled(self.cached_view_size[0], self.cached_view_size[1], QtCore.Qt.KeepAspectRatio)

                    # Apply flips if enabled
                    if self.flip_horizontal or self.flip_vertical:
                        transform = QtGui.QTransform()
                        transform.scale(-1 if self.flip_horizontal else 1, -1 if self.flip_vertical else 1)
                        pixmap = pixmap.transformed(transform)

                # Only update display if not showing overlay
                if not self.show_overlay:
                    self.graphicsView.setPixmap(pixmap)
                else:
                    # Keep showing the overlay
                    if self.overlay_pixmap is not None:
                        self.graphicsView.setPixmap(self.overlay_pixmap)

                key = cv2.waitKey(1)

            else:

                print("Capture terminated!")
                self.capture.release()
                cv2.destroyAllWindows()
                current_running_speed = self.defaultspeed
                self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
                break

    def updatefiles(self):
        # read all files with specific extension
        ROOT_PATH = self.videopathEdit.text() + os.sep
        VALID_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.avi'}

        files = []
        if os.path.isdir(ROOT_PATH):
            for entry in os.scandir(ROOT_PATH):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in VALID_EXTENSIONS:
                        files.append({
                            "full_filename": entry.path,
                            "filename": entry.name,
                            "size": entry.stat().st_size,
                            "rate": DEFAULT_SPEED,
                            "duration": -1,
                            'score': 0
                        })

        # update config file
        count = 0
        old_files = {k:config_data["files"].get(k).split("|") for k in list(config_data["files"])} # separator is pipe |
        for k,old_file in old_files.items():
            for f in files:
                if int(old_file[1])==f["size"]:
                    try:
                        f["rate"] = float(old_file[2])
                        f["duration"] = float(old_file[3])                
                        f['score'] = float(old_file[4])
                        count+=1
                    except:
                        print('failed to read old parameters for video "%s"' % old_file[0])

        print("loaded %i videos with %i new ones" % (len(files),len(files)-count))

        # count the number of frames
        print("computing playtime")
        k = 0
        for f in files:
            if f["duration"]==-1:
                k+=1
                try:
                    video_obj = cv2.VideoCapture(f["full_filename"])
                    frames = video_obj.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps = int(video_obj.get(cv2.CAP_PROP_FPS))
                    # calculate dusration of the video
                    seconds = int(frames / fps)
                    f["duration"] = seconds/60.0
                except:
                    f["duration"] = 35.0
                    print("!! failed to extract video duration for file %s !!" % f["filename"])
            f["playtime"] = f["duration"]*f["rate"]/DEFAULT_SPEED

        files.sort(key= lambda x:x["playtime"], reverse=True) # sort by modified date

        self.filelist = {}
        self.listWidget.clear()
        for f in files:
            label = '%imin - %s' % (round(f["playtime"]),f["filename"])
            self.filelist[label] = f
            item = QListWidgetItem(label)
            if f['score']==1:
                item.setBackground(QBrush(light_green))
            elif f['score']==-1:
                item.setBackground(QBrush(light_red))
            else:
                item.setBackground(QBrush(white))
            self.listWidget.addItem(item)

        #print("saving configs")
        #config_data["files"] = {i:":".join([f["filename"],str(f["size"]),str(f["rate"]),str(round(f["duration"],3))]) for i,f in enumerate(files)}
        #with open(CONFIG_FILE, 'w',encoding="UTF-8") as configfile:
        #    config_data.write(configfile)

    def onClickedFile(self,item=None):
        if item is None:
            item = self.currentItem()
        self.currentvideoEdit.setText(item.text())
        self.playButton.setEnabled(True)
        self.speedInput.setEnabled(True)
        self.scorebadbutton.setEnabled(True)
        self.scoregoodbutton.setEnabled(True)

        #saved_files = {k:config_data["files"].get(k).split(":") for k in list(config_data["files"])}
        selected_file = self.filelist[item.text()]
        self.chosen_file = selected_file
        #for k,file in saved_files.items():
        #    if self.chosen_file[1] in file[0]:
        self.defaultspeed = float(selected_file["rate"])
        self.speedInput.setValue(self.defaultspeed)

        score = float(selected_file["score"])
        self.scoregoodbutton.setStyleSheet("background-color: white;")
        self.scorebadbutton.setStyleSheet("background-color: white;")
        if score==1:
            self.scoregoodbutton.setStyleSheet("background-color: green;")
        if score==-1:
            self.scorebadbutton.setStyleSheet("background-color: red;")
    def update_filedata(self):
        if self.chosen_file is not None:
            old_files = {k:config_data["files"].get(k).split("|") for k in list(config_data["files"])}
            found=False
            new_data_entry = "|".join([
                self.chosen_file["filename"],
                str(self.chosen_file["size"]),
                str(self.chosen_file["rate"]),
                str(round(self.chosen_file["duration"], 3)),
                str(self.chosen_file["score"]),
            ])
            # is the file already in database? If not, add.
            for k,old_file in old_files.items():
                if int(old_file[1])==self.chosen_file['size']:
                    config_data["files"][k] = new_data_entry
                    found=True
                    break
            if found is False:
                config_data["files"][str(len(config_data["files"]))] = new_data_entry

            for index in range(self.listWidget.count()):
                item = self.listWidget.item(index)
                label = item.text()
                itemdata = self.filelist[label]
                if itemdata['full_filename']==self.chosen_file['full_filename']:
                    self.filelist[label]['score']=self.chosen_file['score']
                    if self.chosen_file['score'] == 1:
                        item.setBackground(QBrush(light_green))
                    elif self.chosen_file['score'] == -1:
                        item.setBackground(QBrush(light_red))
                    else:
                        item.setBackground(QBrush(white))

    # ===== TEMPLATE-BASED AUTO-DETECTION INTEGRATION METHODS =====

    def update_template_list(self):
        """Populate template dropdown with available templates."""
        current = self.template_combo.currentText()
        self.template_combo.clear()

        templates = self.template_manager.list_templates()

        if not templates:
            self.template_combo.addItem("(no templates)")
            self.auto_calibrate_btn.setEnabled(False)
        else:
            for name in templates:
                self.template_combo.addItem(name)
            self.auto_calibrate_btn.setEnabled(True)

            # Restore previous selection
            index = self.template_combo.findText(current)
            if index >= 0:
                self.template_combo.setCurrentIndex(index)

    def on_template_changed(self, index):
        """Handle template dropdown selection change."""
        name = self.template_combo.currentText()

        if name and name != "(no templates)":
            config_data["template"]["active_template"] = name
            self.config_save_timer.stop()  # Prevent race conditions
            self.save_configs()
            self.regionselection.setText(f"Template: {name}")
            print(f"Active template: {name}")

    def capture_template(self):
        """Open dialog to capture new template from current frame."""
        # Check camera ready
        if self.frame_shape is None:
            QMessageBox.warning(self, "Camera Not Ready",
                "Wait for camera to initialize.")
            return

        # Get full unzoomed frame (not the zoomed display)
        if self.latest_full_frame is None:
            QMessageBox.warning(self, "No Frame", "No webcam frame available.")
            return

        # Use the full unzoomed frame stored from camera thread
        # Note: OpenCV VideoCapture returns BGR frames
        frame_bgr = self.latest_full_frame.copy()

        # Open template capture dialog
        # Always use zoom_level=1 since we're passing the full frame
        camera_source = self.sourceCombo.currentIndex()

        dialog = TemplateCaptureDialog(self, frame_bgr, camera_source, zoom_level=1)

        if dialog.exec_() == QDialog.Accepted:
            template_data, name = dialog.get_template_data()

            # Save template
            if self.template_manager.save_template(name, template_data):
                config_data["template"]["active_template"] = name
                self.config_save_timer.stop()  # Prevent race conditions
                self.save_configs()

                self.update_template_list()
                index = self.template_combo.findText(name)
                if index >= 0:
                    self.template_combo.setCurrentIndex(index)

                QMessageBox.information(self, "Success",
                    f"Template '{name}' saved.\n\nClick Auto-Cal to detect ROI.")
            else:
                QMessageBox.critical(self, "Failed", f"Failed to save '{name}'.")

    def show_template_overlay(self, live_frame, template_data, roi_coords, confidence, H, success_message=None):
        """
        Display template overlaid on live frame for 2 seconds using falsecolor composite.
        Shows the WARPED template (after similarity transformation) to visualize alignment quality.

        Args:
            live_frame: Current webcam frame (BGR)
            template_data: Template dictionary with 'template_image' key
            roi_coords: Matched ROI coordinates [x1, y1, x2, y2]
            confidence: Matching ratio (0.0 to 1.0)
            H: 3x3 transformation matrix used for warping template to match live frame
            success_message: Optional message to show after overlay (str)
        """
        try:
            print("\n" + "="*60)
            print("SHOWING TEMPLATE OVERLAY IN CAMERA VIEW")
            print("="*60)

            # Get template image
            template_img = template_data['template_image']

            # Warp template to align with live frame using the computed transformation
            template_warped = cv2.warpPerspective(
                template_img,
                H,
                (live_frame.shape[1], live_frame.shape[0])
            )

            # Convert both to grayscale
            live_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
            template_warped_gray = cv2.cvtColor(template_warped, cv2.COLOR_BGR2GRAY)

            # Create falsecolor composite (like MATLAB's imshowpair)
            # Now shows ALIGNED template vs live frame to visualize matching quality
            falsecolor = np.zeros((live_gray.shape[0], live_gray.shape[1], 3), dtype=np.uint8)
            falsecolor[:, :, 0] = template_warped_gray  # Blue = warped template
            falsecolor[:, :, 1] = live_gray             # Green = live frame
            falsecolor[:, :, 2] = template_warped_gray  # Red = warped template

            # Add text overlay
            cv2.rectangle(falsecolor, (5, 5), (550, 85), (0, 0, 0), -1)
            cv2.putText(falsecolor, "TEMPLATE ALIGNMENT CHECK", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(falsecolor, "GREEN=Live | MAGENTA=Template | WHITE=Aligned", (10, 58),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(falsecolor, f"Match Matching ratio: {confidence:.1%}", (10, 78),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            print(f"Falsecolor created: {falsecolor.shape}")

            # Convert to QPixmap
            height, width = falsecolor.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(falsecolor.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Scale to fit camera view
            scaled_pixmap = pixmap.scaled(
                self.graphicsView.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Draw ROI box on TOP of the pixmap (not embedded in the image)
            painter = QPainter(scaled_pixmap)
            painter.setPen(QPen(QColor(0, 255, 255), 4))  # Yellow, 4px thick

            # Calculate scaled ROI coordinates
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height
            x1, y1, x2, y2 = roi_coords
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)

            painter.drawRect(scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)
            painter.end()

            print(f"ROI box drawn on top: ({x1},{y1},{x2},{y2})")

            # Store overlay and set flag
            # The camera thread will pick up the overlay and display it
            self.overlay_pixmap = scaled_pixmap
            self.overlay_success_message = success_message
            self.show_overlay = True

            # Don't call setPixmap here - let camera thread handle display updates
            # This avoids race condition between threads updating the same widget

            print("Overlay displayed in camera view for 2 seconds")
            print("="*60 + "\n")

            # Schedule overlay removal after 2 seconds
            QTimer.singleShot(2000, self.hide_template_overlay)

        except Exception as e:
            print(f"ERROR showing template overlay: {e}")
            import traceback
            traceback.print_exc()

    def hide_template_overlay(self):
        """Hide the template overlay and resume normal camera display."""
        print("Hiding template overlay, resuming normal display")
        self.show_overlay = False
        self.overlay_pixmap = None

        # Success message already shown in overlay - no popup needed
        self.overlay_success_message = None

    def auto_calibrate_roi(self):
        """
        Automatically detect ROI using template matching.
        This is the core method that ties everything together.
        """
        # Get active template
        name = self.template_combo.currentText()

        if not name or name == "(no templates)":
            QMessageBox.warning(self, "No Template",
                "Create a template first (click Capture).")
            return

        # Load template
        template_data = self.template_manager.load_template(name)
        if template_data is None:
            QMessageBox.critical(self, "Load Failed", f"Failed to load '{name}'.")
            return

        # Get full unzoomed frame (not the zoomed display)
        # This ensures coordinates match the template coordinate system
        if self.latest_full_frame is None:
            QMessageBox.warning(self, "No Frame", "No webcam frame available.")
            return

        live_frame = self.latest_full_frame.copy()

        # Match template using robust multi-start optimization
        print(f"\nAuto-calibrating with '{name}'...")
        roi_coords, confidence, H = self.roi_matcher.match_template_robust(
            live_frame, template_data, self.matching_threshold, num_trials=20
        )

        if roi_coords is None:
            # Failed
            msg = (f"Auto-calibration failed (confidence: {confidence:.1%}).\n\n"
                   "Possible reasons:\n"
                   "- Camera/treadmill moved significantly\n"
                   "- Lighting changed dramatically\n"
                   "- Wrong template for current setup\n\n"
                   "Options:\n"
                   "- Retry (adjust position, click Auto-Cal again)\n"
                   "- Create new template (click Capture)\n"
                   "- Manual selection (two clicks on webcam)")

            QMessageBox.warning(self, "Failed", msg)
            self.regionselection.setText(f"Auto-cal failed ({confidence:.0%})")
            self.regionselection.setStyleSheet("color: red;")
            return

        # Success!
        x1, y1, x2, y2 = roi_coords
        digit_count = int(template_data['digit_count'])

        # Coordinates are in full-frame space (zoom_level=1)
        # since template was captured from full frame
        zoom_level = 1

        # Update internal state
        # NOTE: These coordinates are already in FRAME space (transformed by homography)
        # NOT in view space like manual selection!
        self.coordinate1 = (x1, y1, zoom_level)
        self.coordinate2 = (x2, y2, zoom_level)
        self.selected_ROI = True
        self.roi_in_frame_space = True  # Flag: coordinates are already in frame space
        self.digitcount = digit_count

        # Update digit count spin box
        self.digitCountSpin.setValue(digit_count)
        self.lcdNumber.setDigitCount(digit_count + 1)

        # Save to config
        config_data["main"]["roi"] = f"{x1},{y1},{x2},{y2},{zoom_level},1"  # 1 = frame space
        config_data["main"]["digit_count"] = str(digit_count)

        # Stop any pending debounced saves to prevent race conditions
        self.config_save_timer.stop()
        self.save_configs()

        # Update UI
        self.regionselection.setText(f"Auto-cal OK ({confidence:.0%})")
        self.regionselection.setStyleSheet("color: green; font-weight: bold;")

        print(f"Auto-calibration successful: {confidence:.1%}, ROI=({x1},{y1},{x2},{y2})")

        # Show template overlay for debugging (will show for 2 seconds)
        # Success message will appear after overlay
        success_message = (f"ROI detected!\n\n"
                          f"Matching ratio: {confidence:.1%}\n"
                          f"ROI: ({x1}, {y1}, {x2}, {y2})\n"
                          f"Digits: {digit_count}")
        self.show_template_overlay(live_frame, template_data, roi_coords, confidence, H, success_message)

    def auto_calibrate_on_startup(self):
        """
        Perform silent auto-calibration on startup.
        Called 1 second after camera starts.
        """
        print("Attempting startup auto-calibration...")

        # Skip if ROI already selected
        if self.selected_ROI:
            print("ROI already set, skipping")
            return

        # Get template
        name = self.template_combo.currentText()
        if not name or name == "(no templates)":
            return

        template_data = self.template_manager.load_template(name)
        if template_data is None:
            return

        # Get full unzoomed frame
        if self.latest_full_frame is None:
            print("No frame yet, skipping")
            return

        live_frame = self.latest_full_frame.copy()

        # Match using robust multi-start optimization
        roi_coords, confidence, H = self.roi_matcher.match_template_robust(
            live_frame, template_data, self.matching_threshold, num_trials=20
        )

        if roi_coords is not None:
            # Success (silent)
            x1, y1, x2, y2 = roi_coords
            digit_count = int(template_data['digit_count'])

            # Coordinates are in full-frame space (zoom_level=1)
            zoom_level = 1

            self.coordinate1 = (x1, y1, zoom_level)
            self.coordinate2 = (x2, y2, zoom_level)
            self.selected_ROI = True
            self.roi_in_frame_space = True  # Flag: coordinates are already in frame space
            self.digitcount = digit_count

            # Update digit count spin box
            self.digitCountSpin.setValue(digit_count)
            self.lcdNumber.setDigitCount(digit_count + 1)

            config_data["main"]["roi"] = f"{x1},{y1},{x2},{y2},{zoom_level},1"  # 1 = frame space
            config_data["main"]["digit_count"] = str(digit_count)

            # Stop any pending debounced saves to prevent race conditions
            self.config_save_timer.stop()
            self.save_configs()

            self.regionselection.setText(f"Auto-cal OK ({confidence:.0%})")
            self.regionselection.setStyleSheet("color: green; font-weight: bold;")

            print(f"Startup auto-cal successful: {confidence:.1%}")
        else:
            # Failed (silent, just show status)
            self.regionselection.setText(f"Auto-cal failed ({confidence:.0%})")
            self.regionselection.setStyleSheet("color: red;")
            print(f"Startup auto-cal failed: {confidence:.1%}")

    # play selected video
    def play_movie(self):
        # first set and save default speed
        self.defaultspeed = self.speedInput.value()
        for k,f in self.filelist.items():
            if f["size"] == self.chosen_file['size']:
                f["rate"] = self.defaultspeed
                break

        self.update_filedata()

        print("writing updated rate (%.3f)" % self.defaultspeed)
        self.config_save_timer.stop()  # Prevent race conditions
        self.save_configs()

        # finally play video in separate window
        self.videowindow = VideoWindow(self.chosen_file['full_filename'],self.defaultspeed)

    def closeEvent(self, event):
        if self.videowindow is not None:
            self.videowindow.close()

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    sorted_indices = np.argsort(data)
    data, weights = data[sorted_indices], weights[sorted_indices]
    cumulative_weights = np.cumsum(weights)
    midpoint = 0.5 * cumulative_weights[-1]
    if any(weights > midpoint):
        return (data[weights == np.max(weights)])[0]
    median_idx = np.searchsorted(cumulative_weights, midpoint)
    if cumulative_weights[median_idx] == midpoint:
        return np.mean(data[median_idx:median_idx + 2])
    return data[median_idx]

def compute_entropy(softmax_probs):
    """
    Compute normalized entropy from softmax distribution.

    Args:
        softmax_probs: numpy array of softmax probabilities [batch_size, num_classes]

    Returns:
        entropy_normalized: normalized entropy values [0, 1] where 0 is certain, 1 is maximally uncertain
    """
    epsilon = 1e-10
    entropy = -np.sum(softmax_probs * np.log(softmax_probs + epsilon), axis=1)

    # Normalize by max possible entropy (log of num_classes)
    num_classes = softmax_probs.shape[1]
    max_entropy = np.log(num_classes)
    entropy_normalized = entropy / max_entropy

    return entropy_normalized

def compute_calibrated_confidence(softmax_probs, entropy_weight=0.5):
    """
    Compute calibrated confidence score combining max probability and entropy.

    Args:
        softmax_probs: numpy array of softmax probabilities [batch_size, num_classes]
        entropy_weight: weight for entropy penalty (0 to 1), default 0.5

    Returns:
        confidence: calibrated confidence scores [0, 1] for each digit
    """
    # Get max probability for each digit
    max_probs = np.max(softmax_probs, axis=1)

    # Compute normalized entropy (0 = certain, 1 = uncertain)
    entropy_normalized = compute_entropy(softmax_probs)

    # Calibrated confidence: reduce confidence based on entropy
    confidence = max_probs * (1 - entropy_weight * entropy_normalized)

    return confidence

# function to compute speed from current and historical measurements
def speed_estimator(current_time,running_speed_history):
    prev_speeds = []
    prev_probs = []
    for x in reversed(running_speed_history):
        if (current_time - x[0])<SMOOTH_WINDOW:
            prev_speeds.append(x[1])
            prev_probs.append(x[2])
        else:
            break
    if len(prev_speeds)>5:
        display_speed = weighted_median(prev_speeds,prev_probs)
    else:
        display_speed = prev_speeds[0]
    display_speed = max(0.001, min(MAX_SPEED,display_speed))
    return display_speed

# given frame and ROI, return sub-images of digits
def image_preprocessor(img,ROI):
    """Extract digit sub-images from ROI region."""
    ROI_w = ROI[2] - ROI[0]
    ROI_h = ROI[3] - ROI[1]

    # Convert ROI to RGB
    ROI_img = cv2.cvtColor(img[ROI[1]:(ROI[3] + 1), ROI[0]:(ROI[2] + 1), :], cv2.COLOR_BGR2RGB)

    # Use full ROI as bounding box
    box = [0, 0, ROI_w, ROI_h]

    dx = int(np.round(box[2] / ROI[-1]))  # width of one digit box
    extra_pixels_x = int(np.ceil(dx * EXTRA_PIXEL_RATIO))
    extra_pixels_y = 1

    digits = []
    rectangles = []

    # Build lists in correct order directly (no reversal needed)
    for part in range(ROI[-1] - 1, -1, -1):
        rectangles.append([
            ROI[0] + ROI_w - (box[0] + (dx * (part + 1))) - extra_pixels_x,
            ROI[0] + ROI_w - (box[0] + (dx * part)) + extra_pixels_x,
            ROI[1] + ROI_h - box[3] - extra_pixels_y,
            ROI[3] - box[1] + extra_pixels_y
        ])

        # Clamp indices to valid range to avoid negative indices or out-of-bounds
        y_start = max(0, box[1] - extra_pixels_y)
        y_end = min(ROI_img.shape[0], box[1] + box[3] + extra_pixels_y + 1)
        x_start = max(0, box[0] + dx * part - extra_pixels_x)
        x_end = min(ROI_img.shape[1], dx * (part + 1) + extra_pixels_x + 1)

        sub_img = ROI_img[y_start:y_end, x_start:x_end, :]
        sub_img = np.flip(sub_img, axis=0)
        sub_img = np.flip(sub_img, axis=1)
        digits.append(sub_img)

    return digits, rectangles

# resize sub-image with optional padding
def resize_image(img, size=(28,28),PAD = 0):
    # size = h,w
    interpolation = cv2.INTER_LINEAR
    size = size[0]-PAD,size[1]-PAD
    h, w = img.shape[:2]

    # Guard against zero-width or zero-height images
    if w == 0 or h == 0:
        # Return blank image of target size
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)

    aspect_ratio = h/w
    new_aspect_ratio = size[0]/size[1]

    median_color = [int(x) for x in (0.5*np.median(img, axis=(0, 1)))]

    if abs(new_aspect_ratio - aspect_ratio)<1e-6:
        new_im = cv2.resize(img, size, interpolation)
        #new_im = cv2.copyMakeBorder(mask,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        assert new_im.shape == IMG_SIZE
        return new_im
    elif new_aspect_ratio<aspect_ratio: # new image is too narrow, so need padding to width
        tmp_size = [size[0],int(np.round(size[0]/aspect_ratio))]
        new_im = cv2.resize(img,[tmp_size[1],tmp_size[0]],interpolation)
        P = size[1]-new_im.shape[1]
        new_im = cv2.copyMakeBorder(new_im,0,0,int(P/2),0, cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, right
        new_im = cv2.copyMakeBorder(new_im, 0, 0,0,size[1]-new_im.shape[1], cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, right
    else: # new image is too wide, so need padding to height
        tmp_size = [int(np.round(size[1]*aspect_ratio)),size[1]]
        new_im = cv2.resize(img,[tmp_size[1],tmp_size[0]],interpolation)
        P = size[0]-new_im.shape[0]
        new_im = cv2.copyMakeBorder(new_im,int(P/2),0,0,0, cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, right
        new_im = cv2.copyMakeBorder(new_im, 0,size[0]-new_im.shape[0],0,0, cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, righ
    #new_im = cv2.copyMakeBorder(new_im,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right

    assert new_im.shape[:2] == IMG_SIZE,"BUG: Resized image has incorrect shape (%s)!" % str(new_im.shape)

    return new_im

# feed processed images to the predictor model
predictor_model = None

def predict_digit(frame,ROI):
    global predictor_model
    if predictor_model is None:
        return None,None,None,None

    frame_digits, rectangles = image_preprocessor(frame,ROI)

    if len(frame_digits) != ROI[-1]:
        print("Failed to find individual digits (found %i digits out of required %i)!" % (len(frame_digits),ROI[-1]))
        return None,None,None,None

    if USE_PIL_IMAGE:
        frame_digits = [Image.fromarray(x.astype('uint8'), 'RGB') for x in frame_digits]
    else:
        raw_frame_digits = [resize_image(x, IMG_SIZE) for x in frame_digits]
        frame_digits = np.stack(raw_frame_digits, axis=0, dtype=np.float32)
        frame_digits *= (1.0 / 255.0)  # In-place normalization

    # Check if we should use new entropy-based confidence method
    use_calibrated_confidence = config_data.getboolean("main", "use_calibrated_confidence", fallback=False)

    if use_calibrated_confidence and config_data.get("main", "model_type") == 'pytorch':
        # NEW METHOD: Entropy-based calibrated confidence for PyTorch models
        # Get full softmax distribution for entropy computation
        digits, softmax_probs = predictor_model.predict_with_distribution(frame_digits)

        # Compute calibrated per-digit confidence using entropy
        digit_confidences = compute_calibrated_confidence(softmax_probs, entropy_weight=0.5)

        # Aggregate using geometric mean (less sensitive than product)
        prob = np.prod(digit_confidences) ** (1.0 / len(digit_confidences))

        # Apply minimum threshold per digit
        min_digit_confidence = np.min(digit_confidences)
        if min_digit_confidence < 0.30:
            # At least one digit has very low confidence - reject prediction
            return None, None, None, None
    else:
        # OLD METHOD: Original simple approach (default for backward compatibility)
        digits, prob = predictor_model.predict(frame_digits)
        prob = np.prod(prob)

    return digits,prob,rectangles,frame_digits

if __name__ == "__main__":
    if 0:
        app = QtWidgets.QApplication(sys.argv)
        ui = VideoWindow('',4.0)
        sys.exit(app.exec_())
    else:
        USE_PIL_IMAGE = False
        if LOAD_MODEL:
            # load and test predictor model before starting the app
            print("Loading recognition model... ", end="")
            if config_data.get("main", "model_type") == 'tensorflow':
                predictor_model = load_model(DIGIT_MODEL)
                IMG_SIZE = predictor_model.layers[1].input_shape[1:3]
                print("making dummy prediction to initialize predictor... ", end='')
                predictor_model.predict(np.random.rand(10, IMG_SIZE[0], IMG_SIZE[1], 3))
            elif config_data.get("main", "model_type") == 'pytorch':
                USE_PIL_IMAGE = True
                device = torch.device('cpu')
                predictor_model = torch.load(DIGIT_MODEL, map_location=device)
                predictor_model.set_device(device)
                IMG_SIZE = [42,61]
                print("making dummy prediction to initialize predictor... ", end='')
                width, height = 46, 68
                array = np.random.rand(height, width, 3) * 255  # Random values between 0 and 255
                img = Image.fromarray(array.astype('uint8'), 'RGB')
                output,prob = predictor_model.predict([img])
            assert IMG_SIZE[0]>30 and IMG_SIZE[1]>30,"BAD IMAGE SIZE!"
            print("done")

        print("Starting application")
        app = QtWidgets.QApplication(sys.argv)
        ui = MainWindow()
        #QtCore.QTimer.singleShot(0,ui.close)
        sys.exit(app.exec_())


