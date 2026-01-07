# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TreadmillApp is a Python-based video player that dynamically adjusts playback rate based on real-time treadmill speed detection. The application uses computer vision and machine learning to read treadmill speed digits via webcam, then syncs video playback to match the detected speed for an interactive workout experience.

## Core Dependencies

- Python 3
- OpenCV (cv2) - video capture and image processing
- PyQt5 - GUI framework
- VLC with Python bindings - video playback
- TensorFlow/Keras OR PyTorch - digit recognition models
- scikit-image - image registration (phase_cross_correlation)
- NumPy, PIL, matplotlib

## Running the Application

**Main application:**
```bash
python TreadmillApp.py
```

**Windows shortcut (with Anaconda):**
```bash
startup.bat
```
This activates the 'treadmill' conda environment and launches the app.

**Model development/training:**
```bash
python digit_recognition_model_development.py
```
Set the MODE variable inside the file:
- MODE = 0: Prepare training data from frames
- MODE = 1: Train the model
- MODE = 2: Test with video

## Configuration

All application parameters are stored in [config_params.cfg](config_params.cfg):

**[main] section:**
- `model_type`: 'tensorflow' or 'pytorch' - which ML framework to use
- `roi`: Region of interest coordinates for digit detection (x1,y1,x2,y2,zoom_level)
- `predict_interval`: Milliseconds between speed predictions
- `smooth_window`: Time window (seconds) for speed smoothing
- `default_speed`: Default speed when not tracking (km/h)
- `max_speed`: Maximum allowed speed (km/h)
- `min_playrate`: Minimum video playback rate multiplier
- `source`: Camera device ID (0 = default webcam)
- `box_detect`, `median_box`: Digit bounding box detection settings
- `extra_pixel_ratio`: Extra pixels around detected digits
- `shift_center_x`, `shift_center_y`: Frame alignment offsets
- `zoom_level`: Camera zoom level (1-3)

**[paths] section:**
- `video_path`: Directory containing workout videos (MP4/webm)
- `model_path`: Path to trained digit recognition model
- `vlc_path`: VLC installation directory

**[files] section:**
Video library entries with metadata: `filename|size|avg_speed|duration|rating`

## Architecture

### Two-Window Design

1. **MainWindow** ([TreadmillApp.py:384-1117](TreadmillApp.py#L384-L1117))
   - Video library list with file metadata
   - Live webcam preview for speed detection
   - ROI (Region of Interest) selection interface
   - Configuration controls (digit count, precision, zoom)
   - Launches VideoWindow when video is selected

2. **VideoWindow** ([TreadmillApp.py:103-382](TreadmillApp.py#L103-L382))
   - VLC-based video player in separate window
   - Dynamic playback rate adjustment based on global `current_running_speed`
   - Video controls (play/pause, position slider, time display)
   - Updates playrate in real-time via timer (500ms interval)

### Threading Model

The application uses a single background thread for camera processing:

**camera_update thread** ([TreadmillApp.py:737-1100](TreadmillApp.py#L737-L1100)):
- Continuously captures webcam frames
- Performs image registration to compensate for camera shake using phase cross-correlation
- Extracts digit ROI from frames at configurable intervals
- Runs digit recognition inference on extracted regions
- Updates global `current_running_speed` variable
- Maintains `running_speed_history` for temporal smoothing

### Speed Detection Pipeline

1. **Image Registration** ([TreadmillApp.py:83-100](TreadmillApp.py#L83-L100))
   - Uses FFT-based phase cross-correlation to align frames
   - Compensates for camera vibration/movement
   - Template frame created from initial 8 frames

2. **Digit Extraction** ([TreadmillApp.py:1154-1219](TreadmillApp.py#L1154-L1219))
   - Extract ROI from registered frame
   - Optional contour-based bounding box detection
   - Median filtering of historical bounding boxes for stability
   - Split detected region into individual digit sub-images
   - Add padding to maintain aspect ratio

3. **Recognition** (runs every `predict_interval` ms)
   - **TensorFlow/Keras**: Uses loaded Keras model with [TreadmillApp.py:1281-1284](TreadmillApp.py#L1281-L1284)
   - **PyTorch**: Uses Network class from [pytorch_digits_model.py](pytorch_digits_model.py) with custom preprocessing
   - Returns predicted digits and confidence scores

4. **Speed Estimation** ([TreadmillApp.py:1137-1151](TreadmillApp.py#L1137-L1151))
   - Computes weighted median over `smooth_window` time window
   - Weights based on model confidence scores
   - Clamps to [0, max_speed] range

### Model Architecture

**PyTorch Model** ([pytorch_digits_model.py](pytorch_digits_model.py)):
- CNN architecture: 4 conv layers (12→12→24→24 channels) with batch norm and ReLU
- MaxPool after first 2 conv layers
- Dropout (0.35) + Linear classifier
- Custom preprocessing with aspect-ratio-preserving resize and padding
- Input: PIL Images (RGB), Output: digit class + confidence

**TensorFlow/Keras Model**: Legacy support, loaded via `load_model()`

## Development Notes

### Adding New Video Files

Videos are stored in the directory specified by `video_path` in config. The application automatically scans this directory. Metadata is stored in the [files] section of config_params.cfg with format:
```
N = filename|size_bytes|avg_speed_kmh|duration_seconds|rating
```

### Training Custom Digit Recognition Models

Use [digit_recognition_model_development.py](digit_recognition_model_development.py):
1. Set MODE=0 and TRAIN_DATA paths to capture digit frames from your treadmill
2. Manually label captured digits
3. Set MODE=1 to train the model with your labeled data
4. Set MODE=2 to test model performance with video
5. Update `model_path` in config_params.cfg to use new model
6. Set `model_type` to match your framework (tensorflow/pytorch)

### Debugging Flags

Located at [TreadmillApp.py:76-80](TreadmillApp.py#L76-L80):
- `WEBCAM_VIDEO`: Path to pre-recorded video file for testing (instead of live webcam)
- `DEBUG_SPEED_OVERRIDE`: Override detected speed with fixed value
- `LOAD_MODEL`: Set False to skip model loading for faster startup during GUI development

### ROI Selection Workflow

1. Launch app and ensure webcam shows treadmill display
2. Use zoom slider to adjust view
3. Left-click twice on webcam preview to define rectangular ROI around speed digits
4. Right-click to clear selection
5. ROI is automatically saved to config_params.cfg
6. Adjust "digits" radio buttons (1-5) to match number of speed digits displayed

## Critical Global Variables

- `current_running_speed`: Current detected speed (km/h), updated by camera thread, read by video player
- `running_speed_history`: List of (timestamp, speed, confidence) tuples for smoothing
- `config_data`: ConfigParser object holding all settings
- `predictor_model`: Loaded ML model for digit recognition
