# Head-Centric Human Detection System

A computer vision system for detecting humans in CCTV/bus imagery using a head-first detection approach with multiple detection methods and cross-validation.

## Features

- **Multi-Method Head Detection**
  - Haar Cascade (frontal and profile faces)
  - Circular shape detection (for people facing away)
  - Skin-tone clustering
  - YOLO fallback detection

- **Advanced Validation**
  - Upper body proportion validation
  - Shoulder detection
  - Cross-validation to filter artifacts (blankets, jackets, bags)
  - Biological plausibility checks

- **Specialized Detection**
  - Blanket-covered human detection
  - Structured grid scanning for bus interiors
  - Seat artifact filtering

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLO

### Install Dependencies

```bash
pip install opencv-python numpy ultralytics
```

### Download YOLO Model

The system uses YOLOv8 for fallback detection. The model will be downloaded automatically on first run, or you can pre-download:

```bash
# Model will be saved as yolov8n.pt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Basic Usage

```python
from main import HeadCentricHumanDetectionSystem, DetectionThresholds

# Initialize with default thresholds
detector = HeadCentricHumanDetectionSystem()

# Or with custom thresholds
thresholds = DetectionThresholds(
    head_min_size=30,
    head_max_size=150,
    yolo_confidence=0.3
)
detector = HeadCentricHumanDetectionSystem(thresholds)

# Detect humans in an image
import cv2
image = cv2.imread("path/to/image.jpg")
detections = detector.detect_humans(image)

# Print results
for det in detections:
    print(f"Head: {det.head_bbox}, Confidence: {det.final_confidence:.2f}, Type: {det.head_type}")
```

### Process Single Image

```python
result = detector.process_image("path/to/image.jpg", save_output=True)
print(f"Detected {result['summary']['total_humans']} humans")
```

### Batch Processing

```python
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = detector.process_batch(image_paths)

# Save results to JSON and CSV
detector.save_results(results, "detection_results.json")
```

### Visualize Detections

```python
annotated = detector.visualize_detections(image, detections)
cv2.imwrite("annotated_output.jpg", annotated)
```

### Command Line

```bash
python main.py
```

This will process images from the `cctv_images/` directory and save results to `head_centric_results/`.

## Detection Types

| Type | Description | Confidence Range |
|------|-------------|------------------|
| `haar_front` | Frontal face detected via Haar Cascade | 0.7 - 0.9 |
| `haar_profile` | Profile face detected via Haar Cascade | 0.7 - 0.85 |
| `circular` | Back of head detected via Hough circles | 0.25 - 0.65 |
| `skin_based` | Head detected via skin-tone clustering | 0.5 - 0.75 |
| `yolo` | Person detected via YOLO model | 0.25 - 0.95 |
| `blanket_covered` | Human detected under blanket/covering | 0.3 - 0.85 |

## Configuration

Key parameters in `DetectionThresholds`:

```python
@dataclass
class DetectionThresholds:
    # Head Detection
    head_min_size: int = 25          # Minimum head size in pixels
    head_max_size: int = 120         # Maximum head size in pixels

    # Circular Detection (Hough)
    hough_param1: int = 120          # Canny edge threshold
    hough_param2: int = 45           # Circle detection strictness
    hough_min_radius: int = 18       # Minimum circle radius
    hough_max_radius: int = 50       # Maximum circle radius

    # Grid Scanning
    num_rows: int = 4                # Number of scan rows
    aisle_width_ratio: float = 0.33  # Aisle width as ratio of image

    # YOLO Fallback
    yolo_confidence: float = 0.25    # Minimum YOLO confidence
```

## Output Format

### JSON Output

```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "total_images": 5,
    "system": "HeadCentricHumanDetectionSystem"
  },
  "results": [
    {
      "image_path": "image1.jpg",
      "detections": [
        {
          "head_bbox": [100, 50, 150, 100],
          "head_confidence": 0.85,
          "head_type": "haar_front",
          "final_confidence": 0.82
        }
      ],
      "summary": {
        "total_humans": 5,
        "detection_types": {"haar_front": 3, "yolo": 2}
      }
    }
  ]
}
```

### CSV Summary

| Image | Total_Humans | Haar_Front | YOLO | Avg_Conf |
|-------|--------------|------------|------|----------|
| img1.jpg | 5 | 3 | 2 | 0.78 |

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_main.py -v

# Run with coverage report
pytest test_main.py --cov=main --cov-report=term-missing

# Current coverage: 85%
```

## Project Structure

```
image_detection/
├── main.py              # Main detection system
├── test_main.py         # Unit tests (102 tests)
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── yolov8n.pt           # YOLO model (auto-downloaded)
├── cctv_images/         # Input images directory
└── head_centric_results/ # Output directory
```

## License

MIT License

## Acknowledgments

- OpenCV for computer vision primitives
- Ultralytics for YOLO implementation
- Haar Cascade classifiers from OpenCV
