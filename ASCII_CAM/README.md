# ASCII Camera - Real-time ASCII Art Webcam with Gesture Controls

An interactive real-time ASCII art webcam application built with OpenCV and MediaPipe. Transform your webcam feed into live ASCII art with hand gesture controls for dynamic filter application and region selection.

## Features

- **Live ASCII Art Conversion**: Real-time webcam feed converted to detailed ASCII art with color preservation
- **Hand Gesture Controls**: 
  - **L-Shape Gesture** (thumb + index extended): Toggle point manipulation mode on/off
  - **Pinch Gesture** (thumb + index touching): Select and drag points to resize/move the ASCII filter region
- **Interactive Point Manipulation**: 4 draggable points create a resizable quadrilateral mask for applying filters to specific regions
- **Multiple Filter Modes**:
  - `0` - None (original feed)
  - `1` - Grayscale
  - `2` - **ASCII Art** (pure white characters on black background inside mask, original feed outside)
  - `3` - Glitch Effect (RGB channel shift + noise)
- **Ultra-Detailed Rendering**: 5×10 pixel cells with 74-character extended set for maximum detail density
- **High-Resolution Display**: Processes at camera resolution, upscales to 1600×900 for crisp viewing
- **Video-Matched Performance**: Requests 960×540 @ 60 FPS to match high-quality video properties
- **Color Preservation**: Maintains original webcam colors with moderate enhancement for visibility
- **Real-time Performance**: 15-30 FPS depending on system (optimized with region-based processing and grayscale reuse)

## Requirements

- Python 3.x
- OpenCV (`opencv-python` package)
- NumPy (usually comes with OpenCV)
- MediaPipe (`mediapipe` package)

## Installation

1. Install the required packages:
   ```bash
   pip install opencv-python numpy mediapipe
   ```

2. The MediaPipe hand landmarker model (`hand_landmarker.task`) will be downloaded automatically on first run.

## Usage

Run the ASCII Camera application:
```bash
python ascii_camera.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `f` | Toggle face detection (ghostly white boxes) |
| `h` | Toggle hand landmark visualization |
| `0` | Filter: None (original feed) |
| `1` | Filter: Grayscale |
| `2` | Filter: **ASCII Art** (default) |
| `3` | Filter: Glitch Effect |

### Hand Gestures

| Gesture | Action |
|---------|--------|
| **L-Shape** (both hands, thumb+index extended) | Toggle point manipulation mode ON/OFF. When enabled, initializes 4 points as a rectangle based on hand positions. |
| **Pinch** (thumb+index touching) | While point manipulation is active: select and drag the nearest point to reshape the filter region. |
| **L-Shape Again** (when points active) | Toggle point manipulation OFF (points persist for next toggle) |

### Point Manipulation Workflow

1. Make an **L-shape with both hands** (thumb and index finger extended on each hand)
2. Four yellow points appear forming a rectangle between your hands
3. **Pinch** (touch thumb to index) on either hand near a point to select it (turns red)
4. Move your pinched hand to **drag the point** and reshape the quadrilateral
5. Release pinch to drop the point
6. Make **L-shape again** to hide points (they persist for next session)
7. Press `2` for ASCII Art filter to see the effect inside your custom region

## How It Works

### ASCII Art Rendering Pipeline

1. **Capture**: Webcam feed at best available resolution (requests 960×540 @ 60 FPS)
2. **Region Processing**: If point manipulation active, crops to quadrilateral mask bounding box
3. **Grayscale Conversion**: Pre-computed once per frame/region for efficiency
4. **Cell Grid**: Divides region into 5×10 pixel cells (~128×48 = 6,144 characters at 640×480)
5. **Brightness Mapping**: Each cell's average brightness maps to a character from 74-symbol set
6. **Color Sampling**: Each cell's average color used for character rendering (or white for pure ASCII mode)
7. **Rendering**: Characters drawn with `cv2.FONT_HERSHEY_SIMPLEX` at scale 0.3, thickness 1
8. **Display Upscale**: Result upscaled to 1600×900 using `INTER_NEAREST` for crisp ASCII preservation

### Hand Gesture Recognition

- **MediaPipe Hand Landmarker** (Tasks API, VIDEO mode) detects 21 landmarks per hand
- **L-Shape Detection**: Measures thumb-to-index distance relative to hand size (wrist to middle MCP)
- **Pinch Detection**: Normalized distance between thumb and index tips < 0.04 threshold
- **Handedness**: Uses MediaPipe's classification or falls back to thumb/index X-position heuristic

### Filter Application

- Quadrilateral mask created from 4 points using `cv2.fillPoly`
- Filter applied only inside mask via `np.where(mask_3ch, filtered_layer, original_frame)`
- Outside mask region shows original webcam feed unchanged

## Technical Specifications

| Aspect | Specification |
|--------|---------------|
| Processing Resolution | Adaptive (requests 960×540, uses webcam capability) |
| ASCII Cell Size | 5×10 pixels |
| Character Grid | ~128 cols × 48 rows = ~6,144 chars/frame (at 640×480) |
| Character Set | 74 symbols (dense to fine: `@%#*+=-:.WM$&8B@QO0ZCLUXJYuunxrqphkbdpwamIzwiI!l;:;,^"'.` + ` ,.-~:;+!*#$%&`) |
| Display Resolution | 1600×900 (upscaled via INTER_NEAREST) |
| Brightness Boost | +15 (moderate) |
| Contrast Enhancement | alpha=1.8 |
| Font | FONT_HERSHEY_SIMPLEX, scale 0.3, thickness 1 |
| Target FPS | 60 (requested), 15-30 actual |
| CPU Optimization | Region-based processing, pre-computed grayscale, INTER_NEAREST upscale |

## Troubleshooting

- **Webcam not opening**: Ensure no other application is using the webcam. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.
- **Model download failed**: Check your internet connection. The script downloads `hand_landmarker.task` from Google's storage APIs.
- **Poor gesture detection**: Ensure good lighting, keep hands visible, and make deliberate gestures.
- **Low FPS**: Close other applications. The app is optimized but real-time ASCII rendering is CPU-intensive.
- **ASCII too small/detail hard to see**: Resize the window larger, sit farther back, or ensure good lighting for contrast.

## Project Structure

```
face_detection_project/
├── ascii_camera.py            # Main ASCII Camera application
├── ascii-camera               # Standalone ASCII camera feed (simpler version)
├── hand_landmarker.task       # MediaPipe model (downloaded automatically)
├── README.md                  # This file
├── PROJECT_EXPLANATION.md     # Technical architecture deep-dive
├── USER_GUIDE.md              # Complete user guide and feature summary
├── LEARNED.md                 # Skills and concepts learned
├── FIXES.md                   # Bug fixes and appearance corrections
├── DETAIL_ENHANCEMENT.md      # Detail improvement history
├── CHANGELOG.md               # Chronological history of all changes
├── ULTRA_DETAIL.md            # Ultra-detail enhancement specifications
├── VIDEO_MATCHING.md          # Video matching analysis and implementation
```

## Acknowledgments

- **OpenCV** team for computer vision library and Haar Cascade face detection
- **MediaPipe** team for Hand Landmarker solution (Tasks API)
- **Google** for hosting MediaPipe model files
- Original ASCII art character set inspiration from various ASCII art communities

## License

This project is for educational and experimental purposes. Feel free to modify and extend for your learning journey.