# ASCII Camera - Real-time Full-Frame ASCII Art Webcam

A simple, lightweight real-time ASCII art webcam application built with OpenCV. Transform your entire webcam feed into live color ASCII art with 16:9 fullscreen support.

## Features

- **Live Full-Frame ASCII Art**: Real-time webcam feed converted to detailed color ASCII art across the entire frame
- **16:9 Full HD Resolution**: Runs at 1920×1080 (Full HD) with automatic fallback to camera's supported resolution
- **Fullscreen Mode**: Launches in fullscreen by default, press `f` to toggle
- **Bright & Detailed Output**: Enhanced contrast (2.2×) and brightness for clear, visible ASCII characters
- **Color Preservation**: Maintains original webcam colors with brightness boost for visibility
- **Optimized Cell Size**: 6×12 pixel cells for balanced detail and readability
- **Real-time Performance**: Smooth 30+ FPS on most systems
- **Zero Dependencies Beyond OpenCV & NumPy**: No MediaPipe or complex ML models required

## Requirements

- Python 3.x
- OpenCV (`opencv-python` package)
- NumPy (usually comes with OpenCV)

## Installation

1. Install the required packages:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

Run the ASCII Camera application:
```bash
python ascii_cam.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `f` | Toggle fullscreen mode |

## How It Works

### ASCII Art Rendering Pipeline

1. **Capture**: Webcam feed at 1920×1080 (16:9) using DirectShow backend on Windows
2. **Grayscale Conversion**: Frame converted to grayscale for brightness calculation
3. **Contrast & Brightness Enhancement**: Applied via `cv2.convertScaleAbs` (alpha=2.2, beta=25)
4. **Cell Grid**: Divides frame into 6×12 pixel cells
5. **Brightness Mapping**: Each cell's average brightness (+45 offset) maps to character from extended 74-symbol set
6. **Color Sampling**: Each cell's average BGR color used for character rendering
7. **Rendering**: Characters drawn with `cv2.FONT_HERSHEY_SIMPLEX` at scale 0.45, thickness 1
8. **Display**: Fullscreen window with ASCII art overlay

### Character Set

74 symbols ordered from darkest to lightest:
```
@%#*+=-:.WM$&8B@QO0ZCLUXJYuunxrqphkbdpwamIzwiI!l;:;,^"'.`
```

## Technical Specifications

| Aspect | Specification |
|--------|---------------|
| Target Resolution | 1920×1080 (16:9 Full HD) |
| ASCII Cell Size | 6×12 pixels |
| Character Grid | ~320 cols × 90 rows = ~28,800 chars/frame (at 1920×1080) |
| Character Set | 74 symbols (dense to fine) |
| Contrast (alpha) | 2.2 |
| Brightness (beta) | 25 |
| Cell Brightness Offset | +45 |
| Font | FONT_HERSHEY_SIMPLEX, scale 0.45, thickness 1 |
| Backend | DirectShow (CAP_DSHOW) on Windows |
| Target FPS | 30+ |

## Troubleshooting

- **Webcam not opening**: Ensure no other application is using the webcam. The script uses `cv2.CAP_DSHOW` backend on Windows for best compatibility.
- **Resolution not 1920×1080**: Some cameras don't support Full HD. The script will use the highest available resolution.
- **ASCII too dim**: The script includes brightness/contrast enhancement. Adjust `alpha`, `beta`, and brightness offset in `apply_ascii_art()` if needed.
- **Low FPS**: Close other applications. The app is optimized for real-time rendering.

## Project Structure

```
ASCII_CAM/
├── ascii_cam.py            # Main entry point - simple full-frame ASCII feed
├── ascii_camera.py         # Advanced version with face/hand detection & gestures
├── face_detection_final.py # Original feature-complete version
├── hand_landmarker.task    # MediaPipe model (for advanced versions)
└── README.md               # This file
```

## Alternative Versions

- **`ascii_camera.py`**: Complex version with face detection, hand landmark tracking, point manipulation, and 4 filter modes (None, Grayscale, ASCII Art, Glitch). Requires MediaPipe.
- **`face_detection_final.py`**: Original feature-complete version with all advanced features.

## Acknowledgments

- **OpenCV** team for computer vision library
- Original ASCII art character set inspiration from various ASCII art communities

## License

This project is for educational and experimental purposes. Feel free to modify and extend for your learning journey.