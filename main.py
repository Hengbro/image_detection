import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Image Validation
MIN_IMAGE_DIMENSION = 50

# Blanket Detection
BLANKET_MIN_AREA = 3000
BLANKET_MAX_AREA = 50000
BLANKET_ASPECT_RATIO_MIN = 0.3
BLANKET_ASPECT_RATIO_MAX = 2.5
BLANKET_IOU_DUPLICATE_THRESHOLD = 0.5

# Head Detection in Blanket
HEAD_SEARCH_START_RATIO = 0.15
HEAD_SEARCH_END_RATIO = 0.50
MIN_BLANKET_ROI_HEIGHT = 50
MIN_BLANKET_ROI_WIDTH = 30
MIN_HEAD_CONTOUR_AREA = 400
HEAD_CIRCULARITY_MIN = 0.35
HEAD_CIRCULARITY_MAX = 0.9
HEAD_ASPECT_MIN = 0.5
HEAD_ASPECT_MAX = 1.8
BRIGHTNESS_DIFF_THRESHOLD = 15

# Shoulder Detection
SHOULDER_REGION_START_RATIO = 0.35
SHOULDER_REGION_END_RATIO = 0.60
MIN_SHOULDER_ROI_WIDTH = 40
SHOULDER_HOUGH_THRESHOLD = 25
SHOULDER_MIN_LINE_LENGTH = 15
SHOULDER_MAX_LINE_GAP = 20
HORIZONTAL_ANGLE_THRESHOLD = 25  # degrees
SHOULDER_MIN_LENGTH_RATIO = 0.25

# Confidence Scoring
BASE_CONFIDENCE = 0.3
HEAD_CONFIDENCE_BONUS = 0.35
SHOULDER_CONFIDENCE_BONUS = 0.35
COMBINED_CONFIDENCE_BONUS = 0.15

# Circular Head Detection
CEILING_REGION_THRESHOLD = 0.15
UPPER_CORNER_X_THRESHOLD = 0.2
UPPER_CORNER_Y_THRESHOLD = 0.25
CIRCULAR_BRIGHTNESS_MAX = 200
CIRCULAR_BRIGHTNESS_MIN = 20
CIRCULAR_EDGE_SCORE_THRESHOLD = 0.4
CIRCULAR_PERFECT_THRESHOLD = 0.95
CIRCULAR_BASE_CONFIDENCE = 0.25
CIRCULAR_TEXTURE_WEIGHT = 0.12
CIRCULAR_EDGE_WEIGHT = 0.18
CIRCULAR_GRADIENT_WEIGHT = 0.18
TEXTURE_NORMALIZER = 1000.0

# Metallic Surface Detection
METALLIC_SATURATION_THRESHOLD = 30
METALLIC_VALUE_THRESHOLD = 180
METALLIC_LOW_SAT_RATIO = 0.6
METALLIC_HIGH_VAL_RATIO = 0.4

# Pattern Detection
KMEANS_CLUSTERS = 4
KMEANS_ITERATIONS = 10
MIN_PATTERN_LINES = 15
PATTERN_LINE_NORMALIZER = 30
PATTERN_MIN_LINES_THRESHOLD = 20

# Morphological Kernel Sizes
MORPH_KERNEL_LARGE = (11, 11)
MORPH_KERNEL_SMALL = (5, 5)
MORPH_KERNEL_MEDIUM = (7, 7)
LINE_KERNEL_SIZE = (15, 15)

# Canny Edge Detection Thresholds
CANNY_LOW_SOFT = 30
CANNY_HIGH_SOFT = 90
CANNY_LOW_MEDIUM = 40
CANNY_HIGH_MEDIUM = 120
CANNY_LOW_STRONG = 50
CANNY_HIGH_STRONG = 150
CANNY_SHOULDER_LOW = 35
CANNY_SHOULDER_HIGH = 105

# Color Analysis
COLOR_STD_H_MIN = 10
COLOR_STD_H_MAX = 40
COLOR_STD_S_MIN = 15
COLOR_STD_S_MAX = 50
TEXTURE_VAR_MIN = 150
TEXTURE_VAR_MAX = 800

# Skin Detection (YCrCb color space)
SKIN_YCRCB_LOWER = (0, 133, 77)
SKIN_YCRCB_UPPER = (255, 173, 127)

# Hair Detection (HSV color space)
HAIR_HSV_LOWER = (0, 0, 0)
HAIR_HSV_UPPER = (180, 255, 100)

# Radial Gradient Analysis
GRADIENT_SAMPLE_SIZE = 5
GRADIENT_DIFF_MIN = 20
GRADIENT_DIFF_MAX = 80

# Haar Cascade Parameters
HAAR_SCALE_FACTOR_FRONTAL = 1.08
HAAR_MIN_NEIGHBORS_FRONTAL = 4
HAAR_SCALE_FACTOR_PROFILE = 1.1
HAAR_MIN_NEIGHBORS_PROFILE = 3
HAAR_FRONTAL_CONFIDENCE = 0.85
HAAR_PROFILE_CONFIDENCE = 0.75

# Candidate Fusion
FUSION_IOU_THRESHOLD = 0.3
FUSION_VOTE_BONUS = 0.1
FUSION_CONFIDENCE_BONUS = 0.1

# Upper Body Analysis
TORSO_HEIGHT_RATIO = 2.8
TORSO_WIDTH_EXPANSION = 0.4
COLOR_STD_THRESHOLD = 35
TEXTURE_MIN_VARIANCE = 100
HORIZONTAL_ANGLE_LIMIT = 30  # degrees

# Grid Scanner
REGION_OVERLAP_THRESHOLD = 0.3

# Seat Penalty
SEAT_WIDE_ASPECT_THRESHOLD = 1.25
SEAT_WIDE_PENALTY = 0.65
SEAT_TOP_TOUCH_RATIO = 0.15
SEAT_TOP_PENALTY = 0.8
SEAT_CIRCULAR_PENALTY = 0.8
SEAT_FRONT_ROW_PENALTY = 0.8
SEAT_MIN_PENALTY = 0.4
SEAT_FRONT_ROW_THRESHOLD = 2

# Blanket Occupied Seat Inference
BLANKET_EDGE_RATIO_MIN = 0.02
BLANKET_OCCUPIED_ROWS_RATIO = 0.35
BLANKET_SYMMETRY_RATIO_MIN = 0.4
BLANKET_TEXTURE_VAR_MIN = 150
BLANKET_INFERRED_CONFIDENCE = 0.25

# Cross Validation
VALIDATION_HEAD_WEIGHT = 0.3
VALIDATION_BIO_WEIGHT = 0.3
VALIDATION_ARTIFACT_WEIGHT = 0.4
VALIDATION_HEAD_CONF_WEIGHT = 0.4
VALIDATION_SCORE_WEIGHT = 0.6
HEAD_BRIGHTNESS_MIN = 30
HEAD_BRIGHTNESS_MAX = 220
POSITION_UPPER_LIMIT = 0.7
ARTIFACT_CONTEXT_EXPAND_Y_TOP = 30
ARTIFACT_CONTEXT_EXPAND_Y_BOTTOM = 60
ARTIFACT_CONTEXT_EXPAND_X = 10

# YOLO
YOLO_MODEL_PATH = "yolov8m.pt"
YOLO_HEAD_HEIGHT_RATIO = 0.25


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0.0 and 1.0
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


@dataclass
class DetectionThresholds:
    """Configurable thresholds untuk head-centric detection"""
    # Head Detection
    head_min_size: int = 25
    head_max_size: int = 120
    head_aspect_ratio_min: float = 0.5
    head_aspect_ratio_max: float = 1.6
    head_texture_variance_min: float = 150.0
    head_confidence_threshold: float = 0.5

    # Circular Head Detection (Hough) - STRICTER DEFAULTS
    enable_circular_detection: bool = True  # TOGGLE ON/OFF
    hough_param1: int = 120  # Higher = fewer circles detected (was 100)
    hough_param2: int = 45  # Higher = stricter circle detection (was 30, now 45)
    hough_min_radius: int = 18  # Slightly larger minimum (was 15)
    hough_max_radius: int = 50  # Slightly smaller maximum (was 60)
    hough_min_dist: int = 60  # More distance between circles (was 50)

    # Circular Validation Thresholds (NEW)
    circular_texture_variance_min: float = 120.0  # Higher = stricter (was 150, now 250)
    circular_skin_min_ratio: float = 0.08  # Minimum skin presence (was 0.05)
    circular_hair_min_ratio: float = 0.30  # Minimum hair-like colors (was 0.25)
    circular_edge_density_min: float = 0.04  # Stricter minimum (was 0.05)
    circular_edge_density_max: float = 0.22  # Stricter maximum (was 0.25)
    circular_gradient_min: float = 0.25
    circular_position_threshold: float = 0.7  # Max Y position (70% of image)
    circular_confidence_max: float = 0.65  # Max confidence for circular (was 0.7)
    circular_brightness_min: int = 20  # NEW: Reject too dark
    circular_brightness_max: int = 500  # NEW: Reject too bright (lights)
    circular_max_circularity: float = 0.95  # NEW: Reject too perfect circles

    # Skin Detection
    skin_min_area: int = 400
    skin_max_area: int = 6000
    skin_aspect_ratio_min: float = 0.5
    skin_aspect_ratio_max: float = 1.5

    # Upper Body Validation
    torso_head_ratio_min: float = 1.8
    torso_head_ratio_max: float = 3.5
    shoulder_detection_threshold: float = 0.4
    edge_density_min: float = 0.03
    edge_density_max: float = 0.35

    # Artifact Detection
    blanket_blue_threshold: float = 0.65
    uniform_color_threshold: float = 25.0

    # Grid Scanning
    num_rows: int = 4
    aisle_width_ratio: float = 0.33

    # Final Validation
    final_confidence_threshold: float = 0.55
    iou_duplicate_threshold: float = 0.35

    # YOLO Fallback
    yolo_confidence: float = 0.25
    yolo_iou_threshold: float = 0.3

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionThresholds':
        return cls(**data)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Thresholds saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DetectionThresholds':
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Thresholds loaded from {filepath}")
        return cls.from_dict(data)

@dataclass
class HeadCandidate:
    """Kandidat kepala manusia"""
    bbox: List[float]
    confidence: float
    detection_type: str  # 'haar_front', 'haar_profile', 'circular', 'skin_based', 'texture'
    center: Tuple[float, float] = field(default=(0.0, 0.0))

    def __post_init__(self):
        if self.center == (0.0, 0.0):
            self.center = (
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            )

@dataclass
class ScanRegion:
    """Region scanning untuk grid-based approach"""
    bbox: List[float]
    priority: float
    side: str
    row: int

@dataclass
class BlanketRegion:
    """Region yang terdeteksi sebagai selimut"""
    bbox: List[float]
    blanket_confidence: float
    color_info: Dict
    pattern_type: str

@dataclass
class BlanketCoveredHuman:
    """Manusia yang terdeteksi dalam selimut"""
    bbox: List[float]
    confidence: float
    has_head_shape: bool
    has_shoulder_line: bool
    blanket_region: BlanketRegion
    head_bbox: Optional[List[float]] = None
    shoulder_y: Optional[float] = None

@dataclass
class HumanDetection:
    """Final human detection result"""
    head_bbox: List[float]
    head_confidence: float
    head_type: str
    torso_bbox: Optional[List[float]] = None
    body_confidence: float = 0.0
    final_confidence: float = 0.0
    region: Optional[ScanRegion] = None
    is_validated: bool = False
    is_head_based: bool = False
    validation_details: Dict = field(default_factory=dict)


@dataclass
class BlanketCoveredDetection:
    """Detection result untuk manusia tertutup selimut"""
    bbox: List[float]
    confidence: float
    has_head_shape: bool
    has_shoulder_line: bool
    volume_score: float
    texture_info: Dict


class SimplifiedBlanketDetector:
    """
    Strategi Sederhana:
    1. Deteksi area selimut (warna, pattern)
    2. Di dalam selimut, cek bentuk kepala
    3. Cek garis bahu
    4. Jika ada kepala + bahu = manusia
    """

    def __init__(self, thresholds):
        self.thresholds = thresholds

        # Blanket detection thresholds
        self.min_blanket_area = BLANKET_MIN_AREA
        self.max_blanket_area = BLANKET_MAX_AREA

        # Head detection thresholds (relaxed for covered heads)
        self.min_head_ratio = HEAD_SEARCH_START_RATIO
        self.max_head_ratio = HEAD_SEARCH_END_RATIO

        # Shoulder detection
        self.shoulder_region_start = SHOULDER_REGION_START_RATIO
        self.shoulder_region_end = SHOULDER_REGION_END_RATIO

    def detect_humans_in_blankets(self, image: np.ndarray,
                                  region: Optional[Tuple] = None) -> List[BlanketCoveredHuman]:
        """
        Main pipeline: Blanket → Head → Shoulder
        """

        if region:
            x1, y1, x2, y2 = region
            roi = image[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi = image
            offset = (0, 0)

        results = []

        # STEP 1: Deteksi semua area selimut
        blanket_regions = self._detect_blanket_regions(roi)

        logger.info(f"  → Found {len(blanket_regions)} blanket regions")

        # STEP 2: Untuk setiap selimut, cek apakah ada manusia
        for blanket in blanket_regions:
            bx1, by1, bx2, by2 = map(int, blanket.bbox)
            blanket_roi = roi[by1:by2, bx1:bx2]

            if blanket_roi.size == 0:
                continue

            # STEP 3: Cek bentuk kepala di dalam selimut
            has_head, head_bbox = self._detect_head_shape_in_blanket(blanket_roi)

            # STEP 4: Cek garis bahu
            has_shoulder, shoulder_y = self._detect_shoulder_in_blanket(blanket_roi)

            # STEP 5: Validasi - Harus ada kepala ATAU bahu
            if has_head or has_shoulder:
                # Calculate confidence
                confidence = BASE_CONFIDENCE

                if has_head:
                    confidence += HEAD_CONFIDENCE_BONUS

                if has_shoulder:
                    confidence += SHOULDER_CONFIDENCE_BONUS

                # Bonus jika ada keduanya
                if has_head and has_shoulder:
                    confidence = min(confidence + COMBINED_CONFIDENCE_BONUS, 1.0)

                # Global bbox
                global_bbox = [
                    bx1 + offset[0], by1 + offset[1],
                    bx2 + offset[0], by2 + offset[1]
                ]

                # Global head bbox if exists
                global_head_bbox = None
                if head_bbox:
                    global_head_bbox = [
                        head_bbox[0] + bx1 + offset[0],
                        head_bbox[1] + by1 + offset[1],
                        head_bbox[2] + bx1 + offset[0],
                        head_bbox[3] + by1 + offset[1]
                    ]

                result = BlanketCoveredHuman(
                    bbox=global_bbox,
                    confidence=confidence,
                    has_head_shape=has_head,
                    has_shoulder_line=has_shoulder,
                    blanket_region=blanket,
                    head_bbox=global_head_bbox,
                    shoulder_y=shoulder_y + by1 + offset[1] if shoulder_y else None
                )

                results.append(result)
                logger.info(f"  ✓ Human in blanket: head={has_head}, shoulder={has_shoulder}, conf={confidence:.2f}")

        return results

    def _detect_blanket_regions(self, roi: np.ndarray) -> List[BlanketRegion]:
        """
        STEP 1: Deteksi area selimut berdasarkan:
        - Warna uniform/pattern
        - Tekstur khas kain
        - Ukuran yang masuk akal
        """

        blanket_regions = []

        h, w = roi.shape[:2]

        # Convert to different color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Method 1: Deteksi selimut berdasarkan color clustering
        # Blanket biasanya punya warna dominan

        # Quantize colors untuk deteksi warna dominan
        Z = roi.reshape((-1, 3))
        Z = np.float32(Z)

        # K-means clustering untuk cari warna dominan
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_ITERATIONS, 1.0)
        K = KMEANS_CLUSTERS

        ret, label, center = cv2.kmeans(Z, K, None, criteria, KMEANS_ITERATIONS, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)

        # Untuk setiap cluster, cek apakah bisa jadi selimut
        for cluster_idx in range(K):
            # Create mask untuk cluster ini
            mask = (label.flatten() == cluster_idx).astype(np.uint8) * 255
            mask = mask.reshape((h, w))

            # Morphological operations untuk clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_LARGE)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                # Filter by size
                if area < self.min_blanket_area or area > self.max_blanket_area:
                    continue

                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)

                # Aspect ratio check (blanket on person)
                aspect = w_cnt / max(h_cnt, 1)
                if aspect < BLANKET_ASPECT_RATIO_MIN or aspect > BLANKET_ASPECT_RATIO_MAX:
                    continue

                # Extract region
                blanket_roi = roi[y:y + h_cnt, x:x + w_cnt]

                # Analyze if this looks like blanket
                is_blanket, blanket_conf, color_info, pattern = self._analyze_if_blanket(blanket_roi)

                if is_blanket:
                    blanket_regions.append(BlanketRegion(
                        bbox=[x, y, x + w_cnt, y + h_cnt],
                        blanket_confidence=blanket_conf,
                        color_info=color_info,
                        pattern_type=pattern
                    ))

        # Method 2: Deteksi selimut dengan pattern tertentu (kotak-kotak, garis)
        # Ini khusus untuk selimut bermotif seperti di foto Anda

        pattern_blankets = self._detect_patterned_blankets(roi)
        blanket_regions.extend(pattern_blankets)

        # Remove duplicates
        blanket_regions = self._remove_duplicate_blankets(blanket_regions)

        return blanket_regions

    def _analyze_if_blanket(self, roi: np.ndarray) -> Tuple[bool, float, Dict, str]:
        """
        Analisis apakah region ini selimut
        """

        if roi.size == 0:
            return False, 0.0, {}, 'unknown'

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        indicators = []
        color_info = {}

        # 1. Color uniformity (selimut punya warna relatif uniform)
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])

        color_info['h_std'] = float(h_std)
        color_info['s_std'] = float(s_std)
        color_info['v_std'] = float(v_std)

        # Uniform tapi tidak terlalu uniform (bukan dinding/kursi kosong)
        if 10 < h_std < 40 or 15 < s_std < 50:
            indicators.append(0.3)

        # 2. Texture analysis (fabric texture)
        texture_var = np.var(gray)
        color_info['texture_var'] = float(texture_var)

        # Blanket has moderate texture (not too smooth, not too noisy)
        if 150 < texture_var < 800:
            indicators.append(0.3)

        # 3. Edge pattern (fabric folds create edges)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        color_info['edge_density'] = float(edge_density)

        # Moderate edge density
        if 0.03 < edge_density < 0.15:
            indicators.append(0.2)

        # 4. Detect fabric pattern (kotak-kotak, garis-garis)
        pattern_score, pattern_type = self._detect_fabric_pattern(roi)
        color_info['pattern_score'] = pattern_score

        if pattern_score > 0.3:
            indicators.append(0.3)

        confidence = sum(indicators)
        is_blanket = confidence > 0.4

        return is_blanket, confidence, color_info, pattern_type

    def _detect_fabric_pattern(self, roi: np.ndarray) -> Tuple[float, str]:
        """
        Deteksi pattern kain: kotak-kotak (checkered), garis (striped)
        """

        if roi.size == 0:
            return 0.0, 'solid'

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # FFT untuk deteksi pola berulang
        try:
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)

            # Normalize
            magnitude = np.log(magnitude + 1)

            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            # Ambil region di sekitar center (skip DC component)
            mask = np.zeros((h, w))
            cv2.circle(mask, (center_w, center_h), min(h, w) // 4, 1, -1)
            cv2.circle(mask, (center_w, center_h), 5, 0, -1)  # Remove DC

            masked = magnitude * mask

            # Check for peaks (indikasi pattern berulang)
            positive_values = masked[masked > 0]
            if positive_values.size == 0:
                return 0.0, 'solid'
            threshold = np.percentile(positive_values, 95)
            peaks = masked > threshold
            num_peaks = np.sum(peaks)

            # Banyak peak = ada pattern
            if num_peaks > 10:
                # Analyze peak distribution untuk tentukan tipe
                # Horizontal peaks = garis horizontal
                # Vertical peaks = garis vertikal
                # Grid peaks = kotak-kotak

                peak_coords = np.argwhere(peaks)
                if len(peak_coords) > 0:
                    y_variance = np.var(peak_coords[:, 0])
                    x_variance = np.var(peak_coords[:, 1])

                    if abs(y_variance - x_variance) < 100:
                        return min(num_peaks / 50, 1.0), 'checkered'
                    elif y_variance > x_variance:
                        return min(num_peaks / 50, 1.0), 'striped_horizontal'
                    else:
                        return min(num_peaks / 50, 1.0), 'striped_vertical'

                return min(num_peaks / 50, 1.0), 'patterned'

        except ValueError as e:
            logger.debug(f"FFT pattern detection failed due to invalid input: {e}")
        except FloatingPointError as e:
            logger.debug(f"FFT pattern detection failed due to floating point error: {e}")

        return 0.0, 'solid'

    def _detect_patterned_blankets(self, roi: np.ndarray) -> List[BlanketRegion]:
        """
        Deteksi khusus untuk selimut bermotif (seperti selimut kotak-kotak)
        """

        blankets = []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Detect edges dengan multiple thresholds
        edges1 = cv2.Canny(gray, 30, 90)
        edges2 = cv2.Canny(gray, 50, 150)

        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)

        # Detect lines (pola kotak biasanya punya banyak garis)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=30, maxLineGap=15)

        if lines is not None and len(lines) > 20:
            # Banyak garis = kemungkinan pattern
            # Group lines by region

            # Create line density map
            line_map = np.zeros_like(gray)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_map, (x1, y1), (x2, y2), 255, 2)

            # Dilate untuk connect nearby lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            line_map = cv2.dilate(line_map, kernel, iterations=2)

            # Find regions dengan banyak garis
            contours, _ = cv2.findContours(line_map, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < self.min_blanket_area or area > self.max_blanket_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                aspect = w / max(h, 1)
                if aspect < 0.3 or aspect > 2.5:
                    continue

                # Count lines in this region
                region_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if x <= cx <= x + w and y <= cy <= y + h:
                        region_lines.append(line)

                # Jika banyak garis = patterned blanket
                if len(region_lines) > 15:
                    blankets.append(BlanketRegion(
                        bbox=[x, y, x + w, y + h],
                        blanket_confidence=min(len(region_lines) / 30, 1.0),
                        color_info={'num_lines': len(region_lines)},
                        pattern_type='checkered_or_striped'
                    ))

        return blankets

    def _detect_head_shape_in_blanket(self, blanket_roi: np.ndarray) -> Tuple[bool, Optional[List]]:
        """
        STEP 2: Cek bentuk kepala di dalam selimut
        Fokus pada upper portion (20-45% dari atas)
        """

        h, w = blanket_roi.shape[:2]

        if h < MIN_BLANKET_ROI_HEIGHT or w < MIN_BLANKET_ROI_WIDTH:
            return False, None

        gray = cv2.cvtColor(blanket_roi, cv2.COLOR_BGR2GRAY)

        # Define head search region (upper portion)
        head_y_start = int(h * HEAD_SEARCH_START_RATIO)
        head_y_end = int(h * HEAD_SEARCH_END_RATIO)

        head_search_region = gray[head_y_start:head_y_end, :]

        if head_search_region.size == 0:
            return False, None

        # Method 1: Contour-based (cari bentuk bulat/oval)
        edges = cv2.Canny(head_search_region, CANNY_LOW_MEDIUM, CANNY_HIGH_MEDIUM)

        # Morphological close untuk connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SMALL)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < MIN_HEAD_CONTOUR_AREA:
                continue

            # Check circularity/oval
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Head-like shape (moderately round)
            if HEAD_CIRCULARITY_MIN < circularity < HEAD_CIRCULARITY_MAX:
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)

                # Aspect ratio
                aspect = w_cnt / max(h_cnt, 1)
                if HEAD_ASPECT_MIN < aspect < HEAD_ASPECT_MAX:
                    # Found head-like shape!
                    head_bbox = [x, y + head_y_start, x + w_cnt, y + h_cnt + head_y_start]
                    return True, head_bbox

        # Method 2: Brightness-based (kepala sering lebih terang/gelap dari torso)
        upper_half = head_search_region[:head_search_region.shape[0] // 2, :]
        lower_half = head_search_region[head_search_region.shape[0] // 2:, :]

        if upper_half.size > 0 and lower_half.size > 0:
            upper_bright = np.mean(upper_half)
            lower_bright = np.mean(lower_half)

            brightness_diff = abs(upper_bright - lower_bright)

            if brightness_diff > BRIGHTNESS_DIFF_THRESHOLD:
                # Ada perbedaan brightness = kemungkinan kepala
                head_bbox = [0, head_y_start, w, int(h * 0.40)]
                return True, head_bbox

        return False, None

    def _detect_shoulder_in_blanket(self, blanket_roi: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        STEP 3: Deteksi garis bahu horizontal
        Signature paling kuat dari manusia!
        """

        h, w = blanket_roi.shape[:2]

        if h < MIN_BLANKET_ROI_HEIGHT or w < MIN_SHOULDER_ROI_WIDTH:
            return False, None

        gray = cv2.cvtColor(blanket_roi, cv2.COLOR_BGR2GRAY)

        # Define shoulder search region (30-60% dari atas)
        shoulder_y_start = int(h * self.shoulder_region_start)
        shoulder_y_end = int(h * self.shoulder_region_end)

        shoulder_region = gray[shoulder_y_start:shoulder_y_end, :]

        if shoulder_region.size == 0:
            return False, None

        # Detect horizontal edges (shoulder line)
        edges = cv2.Canny(shoulder_region, CANNY_SHOULDER_LOW, CANNY_SHOULDER_HIGH)

        # Hough lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=SHOULDER_HOUGH_THRESHOLD,
            minLineLength=max(SHOULDER_MIN_LINE_LENGTH, w // 5),
            maxLineGap=SHOULDER_MAX_LINE_GAP
        )

        if lines is None:
            return False, None

        # Filter horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                continue

            angle = abs(np.arctan2(y2 - y1, x2 - x1))

            # Horizontal: angle < 25 degrees
            if angle < np.pi / 7.2:  # ~25 degrees
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                horizontal_lines.append({
                    'length': line_length,
                    'y': (y1 + y2) / 2,
                    'angle': angle
                })

        # Jika ada garis horizontal yang cukup panjang = shoulder
        if len(horizontal_lines) > 0:
            # Ambil garis terpanjang
            longest = max(horizontal_lines, key=lambda x: x['length'])

            # Harus cukup panjang (minimal 25% dari lebar)
            if longest['length'] > w * SHOULDER_MIN_LENGTH_RATIO:
                shoulder_y = shoulder_y_start + longest['y']
                return True, shoulder_y

        return False, None

    def _remove_duplicate_blankets(self, blankets: List[BlanketRegion]) -> List[BlanketRegion]:
        """Remove overlapping blanket detections"""

        if len(blankets) <= 1:
            return blankets

        # Sort by confidence
        blankets.sort(key=lambda b: b.blanket_confidence, reverse=True)

        unique = []
        for blanket in blankets:
            is_duplicate = False

            for existing in unique:
                iou = calculate_iou(blanket.bbox, existing.bbox)
                if iou > BLANKET_IOU_DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(blanket)

        return unique


def integrate_with_existing_system(image: np.ndarray,
                                   existing_detections: List,
                                   grid_regions: List,
                                   thresholds: Optional[DetectionThresholds] = None) -> List:

    if thresholds is None:
        thresholds = DetectionThresholds()
    blanket_detector = SimplifiedBlanketDetector(thresholds)

    new_detections = []

    # Scan each grid region yang belum ada deteksi
    for region in grid_regions:
        x1, y1, x2, y2 = map(int, region['bbox'])

        # Check if region already has detection
        has_detection = False
        for det in existing_detections:
            det_x1, det_y1, det_x2, det_y2 = det['bbox']

            # Calculate overlap
            overlap_x = max(0, min(x2, det_x2) - max(x1, det_x1))
            overlap_y = max(0, min(y2, det_y2) - max(y1, det_y1))
            overlap_area = overlap_x * overlap_y

            region_area = (x2 - x1) * (y2 - y1)

            if overlap_area > region_area * 0.3:  # 30% overlap
                has_detection = True
                break

        # Jika belum ada deteksi, coba deteksi blanket-covered
        if not has_detection:
            covered_dets = blanket_detector.detect_humans_in_blankets(
                image,
                region=(x1, y1, x2, y2)
            )

            for det in covered_dets:
                new_detections.append({
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'type': 'blanket_covered',
                    'has_head': det.has_head_shape,
                    'has_shoulder': det.has_shoulder_line,
                    'head_bbox': det.head_bbox
                })

    return existing_detections + new_detections

class HeadDetectionUnit:
    """Unit deteksi kepala dengan multi-method fusion"""

    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

    def detect_all_heads(self, image: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[HeadCandidate]:
        """Deteksi semua kepala dengan multi-method fusion"""
        candidates = []

        # Method 1: Haar Cascade (frontal)
        candidates.extend(self._detect_haar_frontal(image, offset))

        # Method 2: Haar Cascade (profile)
        candidates.extend(self._detect_haar_profile(image, offset))

        # Method 3: Circular shape detection (OPTIONAL - can be disabled)
        if self.thresholds.enable_circular_detection:
            candidates.extend(self._detect_circular_heads(image, offset))

        # Method 4: Skin-tone based detection
        candidates.extend(self._detect_skin_tone_heads(image, offset))

        # Fusion: NMS + voting
        final_heads = self._fuse_candidates(candidates)

        return final_heads

    def _detect_haar_frontal(self, image: np.ndarray, offset: Tuple[int, int]) -> List[HeadCandidate]:
        """Deteksi wajah frontal dengan Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(self.thresholds.head_min_size, self.thresholds.head_min_size),
            maxSize=(self.thresholds.head_max_size, self.thresholds.head_max_size)
        )

        candidates = []
        for (x, y, w, h) in faces:
            candidates.append(HeadCandidate(
                bbox=[float(x + offset[0]), float(y + offset[1]),
                      float(x + w + offset[0]), float(y + h + offset[1])],
                confidence=0.85,
                detection_type='haar_front'
            ))

        return candidates

    def _detect_haar_profile(self, image: np.ndarray, offset: Tuple[int, int]) -> List[HeadCandidate]:
        """Deteksi wajah profile dengan Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(self.thresholds.head_min_size, self.thresholds.head_min_size),
            maxSize=(self.thresholds.head_max_size, self.thresholds.head_max_size)
        )

        candidates = []
        for (x, y, w, h) in profiles:
            candidates.append(HeadCandidate(
                bbox=[float(x + offset[0]), float(y + offset[1]),
                      float(x + w + offset[0]), float(y + h + offset[1])],
                confidence=0.75,
                detection_type='haar_profile'
            ))

        return candidates

    def _detect_circular_heads(self, image: np.ndarray, offset: Tuple[int, int]) -> List[HeadCandidate]:
        """Deteksi kepala berbentuk bulat (untuk yang membelakangi) dengan validasi ketat"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.thresholds.hough_min_dist,
            param1=self.thresholds.hough_param1,
            param2=self.thresholds.hough_param2,
            minRadius=self.thresholds.hough_min_radius,
            maxRadius=self.thresholds.hough_max_radius
        )

        candidates = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            h_img, w_img = gray.shape

            for (x, y, r) in circles[0]:
                # Validate bounds
                if y - r < 0 or y + r >= gray.shape[0] or x - r < 0 or x + r >= gray.shape[1]:
                    continue

                roi = gray[int(y - r):int(y + r), int(x - r):int(x + r)]
                if roi.size == 0:
                    continue

                # CRITICAL: Skip top 15% of image (ceiling fixtures, lights, vents)
                if y < h_img * CEILING_REGION_THRESHOLD:
                    continue

                # CRITICAL: Skip if in upper corners (ceiling mounted objects)
                in_left_corner = (x < w_img * UPPER_CORNER_X_THRESHOLD and y < h_img * UPPER_CORNER_Y_THRESHOLD)
                in_right_corner = (x > w_img * (1 - UPPER_CORNER_X_THRESHOLD) and y < h_img * UPPER_CORNER_Y_THRESHOLD)
                if in_left_corner or in_right_corner:
                    continue

                # VALIDATION 1: Texture variance (not plain surface)
                texture_var = np.var(roi)
                if texture_var < self.thresholds.circular_texture_variance_min:
                    continue

                # VALIDATION 2: Brightness check (reject very bright objects like lights)
                avg_brightness = np.mean(roi)
                if avg_brightness > CIRCULAR_BRIGHTNESS_MAX:
                    continue
                if avg_brightness < CIRCULAR_BRIGHTNESS_MIN:
                    continue

                # VALIDATION 3: Check if it's skin-like or hair-like color
                color_roi = image[int(y - r):int(y + r), int(x - r):int(x + r)]
                if not self._has_human_colors(color_roi):
                    continue

                # VALIDATION 4: Check for metallic/reflective surfaces (AC vents, metal)
                if self._is_metallic_surface(color_roi):
                    continue

                # VALIDATION 5: Position check (heads are typically in middle-upper region)
                # Skip bottom 30% (floor, bags, feet)
                if y > gray.shape[0] * self.thresholds.circular_position_threshold:
                    continue

                # VALIDATION 6: Edge pattern (heads have specific edge patterns)
                edge_score = self._check_head_edge_pattern(roi)
                if edge_score < CIRCULAR_EDGE_SCORE_THRESHOLD:
                    continue

                # VALIDATION 7: Gradient analysis (heads have radial gradients)
                gradient_score = self._check_radial_gradient(roi)
                if gradient_score < self.thresholds.circular_gradient_min:
                    continue

                # VALIDATION 8: Circularity check (too perfect = likely mechanical object)
                circularity = self._check_circularity(roi)
                if circularity > CIRCULAR_PERFECT_THRESHOLD:
                    continue

                # Calculate confidence based on validations
                confidence = (CIRCULAR_BASE_CONFIDENCE +
                              (texture_var / TEXTURE_NORMALIZER) * CIRCULAR_TEXTURE_WEIGHT +
                              edge_score * CIRCULAR_EDGE_WEIGHT +
                              gradient_score * CIRCULAR_GRADIENT_WEIGHT)
                confidence = min(confidence, self.thresholds.circular_confidence_max)

                candidates.append(HeadCandidate(
                    bbox=[float(x - r + offset[0]), float(y - r + offset[1]),
                          float(x + r + offset[0]), float(y + r + offset[1])],
                    confidence=confidence,
                    detection_type='circular'
                ))

        return candidates

    def _is_metallic_surface(self, roi: np.ndarray) -> bool:
        """Detect metallic/reflective surfaces (AC vents, lights)"""
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Metallic surfaces have low saturation and high value
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        low_saturation_ratio = np.sum(s_channel < METALLIC_SATURATION_THRESHOLD) / s_channel.size
        high_value_ratio = np.sum(v_channel > METALLIC_VALUE_THRESHOLD) / v_channel.size

        # If mostly desaturated and bright = metallic
        return low_saturation_ratio > METALLIC_LOW_SAT_RATIO and high_value_ratio > METALLIC_HIGH_VAL_RATIO

    def _check_circularity(self, roi: np.ndarray) -> float:
        """Check how perfectly circular the object is (organic vs mechanical)"""
        if roi.size == 0:
            return 0.0

        # Threshold the image
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        if perimeter == 0:
            return 0.0

        # Calculate circularity: 4π * area / perimeter²
        # Perfect circle = 1.0, irregular shape < 1.0
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        return min(circularity, 1.0)

    def _has_human_colors(self, roi: np.ndarray) -> bool:
        """Check if ROI contains human-like colors (skin, hair, clothing)"""
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        # Check for skin tones
        skin_lower = np.array(SKIN_YCRCB_LOWER, dtype=np.uint8)
        skin_upper = np.array(SKIN_YCRCB_UPPER, dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, skin_lower, skin_upper)
        skin_ratio = np.sum(skin_mask > 0) / max(skin_mask.size, 1)

        # Check for hair colors (black, brown, gray)
        hair_lower = np.array(HAIR_HSV_LOWER, dtype=np.uint8)
        hair_upper = np.array(HAIR_HSV_UPPER, dtype=np.uint8)
        hair_mask = cv2.inRange(hsv, hair_lower, hair_upper)
        hair_ratio = np.sum(hair_mask > 0) / max(hair_mask.size, 1)

        # Must have some skin OR significant hair-like colors
        return skin_ratio > self.thresholds.circular_skin_min_ratio or hair_ratio > self.thresholds.circular_hair_min_ratio

    def _check_head_edge_pattern(self, roi: np.ndarray) -> float:
        """Check if edge pattern matches typical head shape"""
        if roi.size == 0:
            return 0.0

        # Detect edges
        edges = cv2.Canny(roi, 50, 150)

        # Count edge pixels
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels

        # Heads typically have moderate edge density (not too smooth, not too busy)
        min_density = self.thresholds.circular_edge_density_min
        max_density = self.thresholds.circular_edge_density_max

        if min_density < edge_density < max_density:
            return 1.0
        elif edge_density < min_density:
            return 0.0  # Too smooth (might be plain surface)
        else:
            return max(0.0, 1.0 - (edge_density - max_density) / 0.3)

    def _check_radial_gradient(self, roi: np.ndarray) -> float:
        """Check for radial gradient pattern (heads have lighting from center)"""
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return 0.0

        h, w = roi.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Sample brightness at center vs edges
        center_region = roi[
            max(0, center_y - 5):min(h, center_y + 5),
            max(0, center_x - 5):min(w, center_x + 5)
        ]

        if center_region.size == 0:
            return 0.0

        center_brightness = np.mean(center_region)

        # Sample edges (top, bottom, left, right)
        edge_samples = []
        if h > 10:
            edge_samples.append(np.mean(roi[:5, :]))  # Top
            edge_samples.append(np.mean(roi[-5:, :]))  # Bottom
        if w > 10:
            edge_samples.append(np.mean(roi[:, :5]))  # Left
            edge_samples.append(np.mean(roi[:, -5:]))  # Right

        if not edge_samples:
            return 0.0

        edge_brightness = np.mean(edge_samples)

        # Heads often have gradient (center brighter or darker than edges)
        brightness_diff = abs(center_brightness - edge_brightness)

        # Normalize: good gradient difference is 20-80
        if 20 < brightness_diff < 80:
            return min(brightness_diff / 80.0, 1.0)
        else:
            return 0.0

    def _is_head_shape(self, cnt) -> bool:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            return False

        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Convex hull & solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        return (
                self.thresholds.skin_min_area < area < self.thresholds.skin_max_area and
                self.thresholds.skin_aspect_ratio_min < aspect_ratio < self.thresholds.skin_aspect_ratio_max and
                0.55 < circularity < 0.9 and
                solidity > 0.85
        )

    def _detect_skin_tone_heads(self, image: np.ndarray, offset: Tuple[int, int]) -> List[HeadCandidate]:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            if not self._is_head_shape(cnt):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            candidates.append(HeadCandidate(
                bbox=[
                    float(x + offset[0]),
                    float(y + offset[1]),
                    float(x + w + offset[0]),
                    float(y + h + offset[1])
                ],
                confidence=0.80,  # naik karena bentuk divalidasi
                detection_type='skin_shape_head'
            ))

        return candidates


    def _fuse_candidates(self, candidates: List[HeadCandidate]) -> List[HeadCandidate]:
        """Fusi kandidat dengan NMS + voting"""
        if len(candidates) == 0:
            return []

        # Group overlapping candidates
        groups = []
        used = set()

        for i, cand1 in enumerate(candidates):
            if i in used:
                continue

            group = [cand1]
            used.add(i)

            for j, cand2 in enumerate(candidates):
                if j in used or i == j:
                    continue

                iou = calculate_iou(cand1.bbox, cand2.bbox)
                if iou > FUSION_IOU_THRESHOLD:
                    group.append(cand2)
                    used.add(j)

            groups.append(group)

        # For each group, select best candidate (weighted by confidence and votes)
        final_heads = []
        for group in groups:
            # Voting: favor candidates with multiple detection methods
            best = max(group, key=lambda c: c.confidence + len(group) * FUSION_VOTE_BONUS)

            # Average bbox if multiple detections
            if len(group) > 1:
                avg_bbox = [
                    np.mean([c.bbox[0] for c in group]),
                    np.mean([c.bbox[1] for c in group]),
                    np.mean([c.bbox[2] for c in group]),
                    np.mean([c.bbox[3] for c in group])
                ]
                best.bbox = avg_bbox
                best.confidence = min(best.confidence + FUSION_CONFIDENCE_BONUS, 1.0)

            final_heads.append(best)

        return final_heads


class UpperBodyCompositionAnalyzer:
    """Analisis komposisi kepala-bahu-torso"""

    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds

    def validate_human_proportions(self, head_bbox: List[float],
                                   image: np.ndarray) -> Tuple[bool, float, Optional[List[float]], Dict]:
        """
        Validasi proporsi manusia berdasarkan kepala yang terdeteksi
        Returns: (is_valid, confidence, torso_bbox, details)
        """
        hx1, hy1, hx2, hy2 = map(int, head_bbox)
        head_w = hx2 - hx1
        head_h = hy2 - hy1

        # Expected torso region (below head)
        torso_y1 = hy2
        torso_y2 = hy2 + int(head_h * 2.8)
        torso_x1 = hx1 - int(head_w * 0.4)
        torso_x2 = hx2 + int(head_w * 0.4)

        # Clamp to image bounds
        h_img, w_img = image.shape[:2]
        torso_y2 = min(torso_y2, h_img)
        torso_x1 = max(0, torso_x1)
        torso_x2 = min(torso_x2, w_img)

        torso_bbox = [torso_x1, torso_y1, torso_x2, torso_y2]
        torso_roi = image[torso_y1:torso_y2, torso_x1:torso_x2]

        details = {}

        if torso_roi.size == 0:
            return False, 0.0, None, details

        # Analysis 1: Edge density (clothing boundaries)
        gray = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        details['edge_density'] = float(edge_density)

        # Analysis 2: Color consistency (clothing)
        hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:, :, 0])
        details['color_std'] = float(color_std)

        # Analysis 3: Shoulder detection
        has_shoulders, shoulder_conf = self._detect_shoulders(torso_roi, head_w)
        details['has_shoulders'] = has_shoulders
        details['shoulder_confidence'] = shoulder_conf

        # Analysis 4: Texture presence
        texture_var = np.var(gray)
        details['texture_variance'] = float(texture_var)

        # Scoring
        score = 0.0

        if (self.thresholds.edge_density_min < edge_density <
                self.thresholds.edge_density_max):
            score += 0.25

        if color_std < 35:  # Consistent clothing color
            score += 0.2

        if has_shoulders:
            score += 0.3 * shoulder_conf

        if texture_var > 100:  # Has texture (not blank)
            score += 0.25

        is_valid = score > 0.5

        return is_valid, score, torso_bbox if is_valid else None, details

    def _detect_shoulders(self, torso_roi: np.ndarray, head_width: int) -> Tuple[bool, float]:
        """Deteksi bahu (shoulder line)"""
        if torso_roi.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY) if len(torso_roi.shape) == 3 else torso_roi

        # Look for horizontal edge in upper torso (shoulder line)
        upper_third = gray[:max(1, gray.shape[0] // 3), :]

        if upper_third.size == 0:
            return False, 0.0

        # Sobel horizontal edges
        sobelx = cv2.Sobel(upper_third, cv2.CV_64F, 1, 0, ksize=3)
        sobelx_abs = np.uint8(np.abs(sobelx))

        # Find horizontal lines
        lines = cv2.HoughLinesP(
            sobelx_abs,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=max(10, head_width // 3),
            maxLineGap=15
        )

        if lines is not None and len(lines) > 0:
            # Check if lines are roughly horizontal
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1))
                if angle < np.pi / 6:  # Within 30 degrees of horizontal
                    horizontal_lines += 1

            if horizontal_lines > 0:
                confidence = min(horizontal_lines / 3.0, 1.0)
                return True, confidence

        return False, 0.0


class StructuredGridScanner:
    """Pemindaian terstruktur berdasarkan grid layout bus"""

    def __init__(self, image_shape: Tuple[int, int], thresholds: DetectionThresholds):
        self.height, self.width = image_shape
        self.thresholds = thresholds
        self.grid = self._create_bus_layout_grid()

    def _infer_blanket_occupied_seat(
            self,
            roi: np.ndarray,
            region: ScanRegion,
            detected_heads: List[HeadCandidate]
    ) -> bool:

        # Seat only
        if region.side not in ("left", "right"):
            return False

        if detected_heads:
            return False

        h, w = roi.shape[:2]
        if h < 40 or w < 40:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # 1️⃣ Edge density (lower threshold)
        edges = cv2.Canny(blur, 30, 100)
        edge_ratio = np.sum(edges > 0) / edges.size

        if edge_ratio < 0.02:
            return False

        # 2️⃣ Vertical occupancy (more tolerant)
        vertical_projection = np.sum(edges > 0, axis=1)
        occupied_rows = np.count_nonzero(vertical_projection)

        if occupied_rows / h < 0.35:
            return False

        # 3️⃣ Horizontal symmetry (human body cue)
        left_mass = np.sum(edges[:, :w // 2] > 0)
        right_mass = np.sum(edges[:, w // 2:] > 0)

        symmetry_ratio = min(left_mass, right_mass) / max(left_mass, right_mass, 1)

        if symmetry_ratio < 0.4:
            return False

        # 4️⃣ Reject pure seat texture (low variance)
        texture_var = np.var(gray)
        if texture_var < 150:
            return False

        return True

    def _seat_penalty(self, head: HeadCandidate, region: ScanRegion) -> float:
        """
        Soft penalty for seat-like detections.
        Return multiplier in range [0.4, 1.0]
        """

        penalty = 1.0

        x1, y1, x2, y2 = head.bbox
        w = x2 - x1
        h = y2 - y1
        aspect = w / max(h, 1e-6)

        # 1. Wide object (seat back)
        if aspect > 1.25:
            penalty *= 0.65

        # 2. Touching region top (seat headrest)
        region_top = region.bbox[1]
        if abs(y1 - region_top) < h * 0.15:
            penalty *= 0.8

        # 3. Circular-only weak detection
        if head.detection_type == "circular":
            penalty *= 0.8

        # 4. Front rows are stricter (many seat artifacts)
        if region.row < 2:
            penalty *= 0.8

        return max(penalty, 0.4)

    def _create_bus_layout_grid(self) -> List[ScanRegion]:
        """Buat grid berdasarkan layout bus (depan→belakang, kiri→kanan)"""
        regions = []

        num_rows = self.thresholds.num_rows
        row_height = self.height // num_rows
        aisle_width = int(self.width * self.thresholds.aisle_width_ratio)

        for row_idx in range(num_rows):
            y1 = row_idx * row_height
            y2 = (row_idx + 1) * row_height

            # Left seats
            left_width = (self.width - aisle_width) // 2
            regions.append(ScanRegion(
                bbox=[0, y1, left_width, y2],
                priority=float(row_idx),
                side='left',
                row=row_idx
            ))

            # Aisle (for standing passengers)
            regions.append(ScanRegion(
                bbox=[left_width, y1, left_width + aisle_width, y2],
                priority=float(row_idx) + 0.5,
                side='aisle',
                row=row_idx
            ))

            # Right seats
            regions.append(ScanRegion(
                bbox=[left_width + aisle_width, y1, self.width, y2],
                priority=float(row_idx),
                side='right',
                row=row_idx
            ))

        # Sort by priority (front to back)
        regions.sort(key=lambda r: r.priority)

        return regions

    def scan_with_context(
            self,
            image: np.ndarray,
            head_detector: HeadDetectionUnit
    ) -> List[HumanDetection]:

        detections: List[HumanDetection] = []

        for region in self.grid:
            x1, y1, x2, y2 = map(int, region.bbox)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = image[y1:y2, x1:x2]
            if roi is None or roi.size == 0:
                continue

            heads = head_detector.detect_all_heads(
                roi,
                offset=(x1, y1)
            ) or []

            for head in heads:
                penalty = self._seat_penalty(head, region)
                adjusted_conf = head.confidence * penalty

                if adjusted_conf < self.thresholds.final_confidence_threshold:
                    continue

                if self._is_unique_detection(head.bbox, detections):
                    detections.append(
                        HumanDetection(
                            is_head_based=True,
                            head_type=head.detection_type,
                            head_confidence=adjusted_conf,
                            head_bbox=head.bbox,
                            region=region
                        )
                    )

            if not heads:
                if self._infer_blanket_occupied_seat(roi, region, heads):
                    detections.append(
                        HumanDetection(
                            is_head_based=False,
                            head_type="blanket_inferred",
                            head_confidence=0.25,
                            head_bbox=list(region.bbox),
                            region=region
                        )
                    )

        return detections

    def _is_unique_detection(self, new_bbox: List[float],
                             existing: List[HumanDetection]) -> bool:
        """Prevent double counting"""
        for det in existing:
            iou = calculate_iou(new_bbox, det.head_bbox)
            if iou > self.thresholds.iou_duplicate_threshold:
                return False
        return True


class CrossValidationModule:
    """Validasi silang untuk mencegah false positives"""

    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds

    def validate_detections(self, detections: List[HumanDetection],
                            image: np.ndarray) -> List[HumanDetection]:
        """Multi-level validation"""
        validated = []

        for det in detections:
            validation_score = 0.0
            details = {}

            # Check 1: Head quality
            head_quality = self._check_head_quality(det, image)
            validation_score += head_quality * 0.3
            details['head_quality'] = head_quality

            # Check 2: Biological plausibility
            is_plausible, plausibility_score = self._check_biological_plausibility(det, image)
            if is_plausible:
                validation_score += plausibility_score * 0.3
            details['biological_plausibility'] = plausibility_score

            # Check 3: Not an artifact
            is_artifact, artifact_score = self._is_likely_artifact(det, image)
            if not is_artifact:
                validation_score += (1.0 - artifact_score) * 0.4
            details['artifact_score'] = artifact_score

            # Combine with head confidence
            final_score = (validation_score * 0.6 + det.head_confidence * 0.4)

            det.validation_details = details
            det.final_confidence = final_score

            # Only keep if passes threshold
            if final_score > self.thresholds.final_confidence_threshold:
                det.is_validated = True
                validated.append(det)

        return validated

    def _check_head_quality(self, det: HumanDetection, image: np.ndarray) -> float:
        """Check kualitas deteksi kepala"""
        hx1, hy1, hx2, hy2 = map(int, det.head_bbox)

        # Clamp to image bounds
        h_img, w_img = image.shape[:2]
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(w_img, hx2), min(h_img, hy2)

        head_roi = image[hy1:hy2, hx1:hx2]

        if head_roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

        # Check texture variance
        texture_var = np.var(gray)
        texture_score = min(texture_var / 500.0, 1.0)

        # Check if not too dark or too bright
        brightness = np.mean(gray)
        brightness_score = 1.0 if 30 < brightness < 220 else 0.5

        return (texture_score * 0.6 + brightness_score * 0.4)

    def _check_biological_plausibility(self, det: HumanDetection,
                                       image: np.ndarray) -> Tuple[bool, float]:
        """Check if detection looks like a real human"""
        hx1, hy1, hx2, hy2 = map(int, det.head_bbox)

        # Size check
        w = hx2 - hx1
        h = hy2 - hy1

        size_ok = (self.thresholds.head_min_size < w < self.thresholds.head_max_size and
                   self.thresholds.head_min_size < h < self.thresholds.head_max_size)

        if not size_ok:
            return False, 0.0

        # Aspect ratio check
        aspect = w / max(h, 1)
        aspect_ok = (self.thresholds.head_aspect_ratio_min < aspect <
                     self.thresholds.head_aspect_ratio_max)

        if not aspect_ok:
            return False, 0.3

        # Position check (heads typically in upper 60% of image)
        h_img = image.shape[0]
        head_center_y = (hy1 + hy2) / 2
        position_score = 1.0 if head_center_y < h_img * 0.7 else 0.7

        score = 0.5 + position_score * 0.5

        return True, score

    def _is_likely_artifact(self, det: HumanDetection, image: np.ndarray) -> Tuple[bool, float]:
        """Detect if this is a false positive (jacket, blanket, bag)"""
        hx1, hy1, hx2, hy2 = map(int, det.head_bbox)

        # Expand region to check surroundings
        h_img, w_img = image.shape[:2]
        context_y1 = max(0, hy1 - 30)
        context_y2 = min(h_img, hy2 + 60)
        context_x1 = max(0, hx1 - 10)
        context_x2 = min(w_img, hx2 + 10)

        context_roi = image[context_y1:context_y2, context_x1:context_x2]

        if context_roi.size == 0:
            return True, 1.0

        artifact_score = 0.0

        # Check for blanket (large uniform blue area)
        hsv = cv2.cvtColor(context_roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([130, 255, 255]))
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

        if blue_ratio > self.thresholds.blanket_blue_threshold:
            artifact_score += 0.5

        # Check for uniform color (not varied like human face/body)
        gray = cv2.cvtColor(context_roi, cv2.COLOR_BGR2GRAY)
        color_std = np.std(gray)

        if color_std < self.thresholds.uniform_color_threshold:
            artifact_score += 0.3

        is_artifact = artifact_score > 0.5

        return is_artifact, artifact_score


class HeadCentricHumanDetectionSystem:
    """Sistem deteksi manusia dengan head-centric approach"""

    def __init__(self, thresholds: Optional[DetectionThresholds] = None):
        # Load or create thresholds
        if thresholds is None:
            threshold_file = "head_centric_thresholds.json"
            if os.path.exists(threshold_file):
                self.thresholds = DetectionThresholds.load(threshold_file)
                logger.info("Loaded thresholds from file")
            else:
                self.thresholds = DetectionThresholds()
                self.thresholds.save(threshold_file)
                logger.info("Created default thresholds")
        else:
            self.thresholds = thresholds

        # Initialize core components
        self.head_detector = HeadDetectionUnit(self.thresholds)
        self.body_analyzer = UpperBodyCompositionAnalyzer(self.thresholds)
        self.grid_scanner = None  # Initialized per image
        self.cross_validator = CrossValidationModule(self.thresholds)

        # YOLO fallback (lazy loaded)
        self._yolo_model = None

        # Statistics
        self.stats = defaultdict(int)

    @property
    def yolo_model(self):
        """Lazy load YOLO model on first access."""
        if self._yolo_model is None:
            logger.info("Loading YOLO model...")
            self._yolo_model = YOLO(YOLO_MODEL_PATH)
        return self._yolo_model

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate input image before processing.

        Args:
            image: Input image array

        Raises:
            ValueError: If image is invalid
        """
        if image is None:
            raise ValueError("Invalid image: image is None")

        if not isinstance(image, np.ndarray):
            raise ValueError(f"Invalid image: expected numpy array, got {type(image).__name__}")

        if image.size == 0:
            raise ValueError("Invalid image: image is empty")

        if len(image.shape) != 3:
            raise ValueError(f"Invalid image: expected 3 dimensions (H, W, C), got {len(image.shape)}")

        if image.shape[2] != 3:
            raise ValueError(f"Invalid image: expected 3 color channels (BGR), got {image.shape[2]}")

        height, width = image.shape[:2]
        if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
            raise ValueError(f"Invalid image: dimensions {width}x{height} too small (minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})")

    def detect_humans(self, image: np.ndarray) -> List[HumanDetection]:
        """Main detection pipeline with head-centric approach"""

        self._validate_image(image)

        logger.info("=" * 60)
        logger.info("HEAD-CENTRIC DETECTION PIPELINE")
        logger.info("=" * 60)

        # Step 1: Initialize grid scanner
        self.grid_scanner = StructuredGridScanner(image.shape[:2], self.thresholds)
        logger.info(f"Step 1: Grid initialized ({len(self.grid_scanner.grid)} regions)")

        # Step 2: Structured scanning for heads
        logger.info("Step 2: Scanning for heads (multi-method)...")
        head_detections = self.grid_scanner.scan_with_context(image, self.head_detector)
        logger.info(f"  → Found {len(head_detections)} head candidates")

        # Step 3: Validate each head with torso analysis
        logger.info("Step 3: Validating with upper body composition...")
        validated_detections = []
        for idx, head_det in enumerate(head_detections):
            has_body, body_conf, torso_bbox, details = self.body_analyzer.validate_human_proportions(
                head_det.head_bbox, image
            )

            head_det.torso_bbox = torso_bbox
            head_det.body_confidence = body_conf

            # Add to validated list regardless (will be filtered later)
            validated_detections.append(head_det)

            if has_body:
                logger.info(f"  ✓ Head #{idx + 1}: body_conf={body_conf:.2f}")
            else:
                logger.info(f"  ✗ Head #{idx + 1}: no clear torso")

        # Step 4: Cross-validation
        logger.info("Step 4: Cross-validation (artifact filtering)...")
        final_detections = self.cross_validator.validate_detections(
            validated_detections, image
        )
        logger.info(f"  → {len(final_detections)} humans validated")

        # Step 5: YOLO fallback
        logger.info("Step 5: YOLO fallback check...")
        yolo_additions = self._yolo_fallback(image, final_detections)
        if yolo_additions:
            logger.info(f"  → Added {len(yolo_additions)} from YOLO")
            final_detections.extend(yolo_additions)

        # Step 6: Final verification
        self._verify_count_logic(final_detections, image)

        # Update statistics
        self.stats['total_processed'] += 1
        self.stats['humans_detected'] += len(final_detections)

        return final_detections

    def _yolo_fallback(self, image: np.ndarray,
                       existing_detections: List[HumanDetection]) -> List[HumanDetection]:
        """YOLO fallback untuk menangkap missed detections"""
        fallback_detections = []

        results = self.yolo_model(image, conf=self.thresholds.yolo_confidence, verbose=False)

        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.cpu().numpy()
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # person class
                        bbox = box.xyxy[0].tolist()
                        conf = float(box.conf[0])

                        # Check if already detected
                        already_detected = any(
                            calculate_iou(bbox, ex.head_bbox) > self.thresholds.yolo_iou_threshold
                            for ex in existing_detections
                        )

                        if not already_detected:
                            # YOLO gives full body bbox, estimate head
                            x1, y1, x2, y2 = bbox
                            body_h = y2 - y1
                            head_h = body_h * 0.25  # Head is ~25% of body
                            head_bbox = [x1, y1, x2, y1 + head_h]

                            detection = HumanDetection(
                                head_bbox=head_bbox,
                                head_confidence=conf,
                                head_type='yolo',
                                torso_bbox=bbox,
                                body_confidence=conf,
                                final_confidence=conf,
                                is_validated=True
                            )
                            fallback_detections.append(detection)
                            self.stats['yolo_fallback'] += 1

        return fallback_detections

    def _verify_count_logic(self, detections: List[HumanDetection], image: np.ndarray):
        """Final sanity check"""
        num_heads = len(detections)

        logger.info("=" * 60)
        logger.info(f"FINAL COUNT: {num_heads} humans detected")
        logger.info("=" * 60)

        # Breakdown by detection type
        type_counts = defaultdict(int)
        for det in detections:
            type_counts[det.head_type] += 1

        logger.info("Detection breakdown:")
        for dtype, count in sorted(type_counts.items()):
            logger.info(f"  - {dtype}: {count}")

        # Confidence distribution
        high_conf = sum(1 for d in detections if d.final_confidence > 0.8)
        med_conf = sum(1 for d in detections if 0.6 <= d.final_confidence <= 0.8)
        low_conf = sum(1 for d in detections if d.final_confidence < 0.6)

        logger.info("Confidence distribution:")
        logger.info(f"  - High (>0.8): {high_conf}")
        logger.info(f"  - Medium (0.6-0.8): {med_conf}")
        logger.info(f"  - Low (<0.6): {low_conf}")

    def visualize_detections(self, image: np.ndarray,
                             detections: List[HumanDetection]) -> np.ndarray:
        """Visualisasi hasil deteksi"""
        annotated = image.copy()

        for idx, det in enumerate(detections):
            # Draw head bbox
            hx1, hy1, hx2, hy2 = map(int, det.head_bbox)

            # Color by detection type
            color_map = {
                'haar_front': (0, 255, 0),  # Green
                'haar_profile': (0, 255, 255),  # Yellow
                'circular': (255, 0, 255),  # Magenta
                'skin_based': (255, 165, 0),  # Orange
                'yolo': (0, 128, 255)  # Blue
            }
            color = color_map.get(det.head_type, (128, 128, 128))

            # Draw head
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 2)

            # Draw torso if available
            if det.torso_bbox:
                tx1, ty1, tx2, ty2 = map(int, det.torso_bbox)
                cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), color, 1)

            # Label
            label = f"#{idx + 1} {det.head_type} {det.final_confidence:.2f}"
            cv2.putText(annotated, label, (hx1, hy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add summary
        summary = f"Total Humans: {len(detections)}"
        cv2.putText(annotated, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated

    def process_image(self, image_path: str, save_output: bool = True) -> Dict:
        """Process satu gambar"""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing: {image_path}")
        logger.info(f"{'=' * 70}")

        result = {
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "detections": [],
            "summary": {},
            "error": None
        }

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            # Run detection
            detections = self.detect_humans(image)

            # Convert to dict
            for det in detections:
                det_dict = {
                    'head_bbox': det.head_bbox,
                    'head_confidence': det.head_confidence,
                    'head_type': det.head_type,
                    'torso_bbox': det.torso_bbox,
                    'body_confidence': det.body_confidence,
                    'final_confidence': det.final_confidence,
                    'is_validated': det.is_validated,
                    'validation_details': det.validation_details
                }
                result["detections"].append(det_dict)

            # Create summary
            result["summary"] = {
                "total_humans": len(detections),
                "detection_types": {
                    "haar_front": sum(1 for d in detections if d.head_type == 'haar_front'),
                    "haar_profile": sum(1 for d in detections if d.head_type == 'haar_profile'),
                    "circular": sum(1 for d in detections if d.head_type == 'circular'),
                    "skin_based": sum(1 for d in detections if d.head_type == 'skin_based'),
                    "yolo": sum(1 for d in detections if d.head_type == 'yolo')
                },
                "confidence_stats": {
                    "high": sum(1 for d in detections if d.final_confidence > 0.8),
                    "medium": sum(1 for d in detections if 0.6 <= d.final_confidence <= 0.8),
                    "low": sum(1 for d in detections if d.final_confidence < 0.6),
                    "average": float(np.mean([d.final_confidence for d in detections])) if detections else 0.0
                }
            }

            # Save annotated image
            if save_output:
                annotated = self.visualize_detections(image, detections)

                output_dir = Path("head_centric_results")
                output_dir.mkdir(exist_ok=True)

                output_path = output_dir / f"{Path(image_path).stem}_headcentric.jpg"
                cv2.imwrite(str(output_path), annotated)
                logger.info(f"✓ Saved: {output_path}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Error: {e}")

        return result

    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process batch gambar"""
        results = []

        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH PROCESSING: {len(image_paths)} images")
        logger.info(f"{'=' * 70}\n")

        for idx, img_path in enumerate(image_paths):
            logger.info(f"[{idx + 1}/{len(image_paths)}]")
            result = self.process_image(img_path)
            results.append(result)

            # Print progress
            if result["summary"]:
                summary = result["summary"]
                print(f"  → Humans: {summary['total_humans']} | "
                      f"Avg Conf: {summary['confidence_stats']['average']:.2f}")

        return results

    def save_results(self, results: List[Dict], output_file: str = "head_centric_results.json"):
        """Save hasil ke JSON"""
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_images": len(results),
                "system": "HeadCentricHumanDetectionSystem",
                "approach": "head_first_torso_validation"
            },
            "statistics": dict(self.stats),
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"✓ Results saved: {output_file}")

        # Save summary CSV
        self._save_summary_csv(results, output_file.replace('.json', '_summary.csv'))

    def _save_summary_csv(self, results: List[Dict], csv_file: str):
        """Save summary to CSV"""
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Image", "Total_Humans", "Haar_Front", "Haar_Profile",
                "Circular", "Skin_Based", "YOLO",
                "High_Conf", "Med_Conf", "Low_Conf", "Avg_Conf", "Error"
            ])

            # Data
            for result in results:
                summary = result.get("summary", {})
                types = summary.get("detection_types", {})
                conf_stats = summary.get("confidence_stats", {})

                writer.writerow([
                    Path(result["image_path"]).name,
                    summary.get("total_humans", 0),
                    types.get("haar_front", 0),
                    types.get("haar_profile", 0),
                    types.get("circular", 0),
                    types.get("skin_based", 0),
                    types.get("yolo", 0),
                    conf_stats.get("high", 0),
                    conf_stats.get("medium", 0),
                    conf_stats.get("low", 0),
                    f"{conf_stats.get('average', 0):.2f}",
                    result.get("error", "")
                ])

        logger.info(f"✓ Summary CSV saved: {csv_file}")


def main():
    """Main function"""

    print("\n" + "=" * 70)
    print("HEAD-CENTRIC HUMAN DETECTION SYSTEM v3.0")
    print("=" * 70)
    print("\n🎯 APPROACH:")
    print("  1. HEAD-FIRST: Detect all heads using multi-method fusion")
    print("     • Haar Cascade (frontal + profile)")
    print("     • Circular shape detection (for back heads)")
    print("     • Skin-tone clustering")
    print("  2. TORSO VALIDATION: Verify head-torso proportions")
    print("  3. GRID SCANNING: Structured front→back, left→right")
    print("  4. CROSS-VALIDATION: Filter artifacts (blankets, jackets)")
    print("  5. YOLO FALLBACK: Catch missed detections")
    print("=" * 70)

    # Find test images
    test_images = []
    for test_dir in ["cctv_images", "bus_images", "test_images", "images"]:
        if os.path.exists(test_dir):
            for ext in ['.jpg', '.jpeg', '.png']:
                test_images.extend(Path(test_dir).glob(f"*{ext}"))
            if test_images:
                break

    if not test_images:
        print("\n⚠️  No test images found!")
        print("Please create one of these folders and add images:")
        print("  - cctv_images")
        print("  - bus_images")
        print("  - test_images")
        print("  - images")

        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        print(f"\n✓ Created '{test_dir}' folder. Add images and run again.")
        return

    print(f"\n✓ Found {len(test_images)} test images")

    # Show images
    print("\nImages to process:")
    for idx, img in enumerate(test_images[:10], 1):
        print(f"  {idx}. {img.name}")
    if len(test_images) > 10:
        print(f"  ... and {len(test_images) - 10} more")

    # Initialize system
    print("\n" + "=" * 70)
    print("Initializing Detection System...")
    print("=" * 70)

    try:
        detector = HeadCentricHumanDetectionSystem()

        print("\n⚙️  Configuration:")
        print(f"  - Head size range: {detector.thresholds.head_min_size}-{detector.thresholds.head_max_size} px")
        print(f"  - Grid layout: {detector.thresholds.num_rows} rows")
        print(f"  - Final confidence threshold: {detector.thresholds.final_confidence_threshold}")
        print(f"  - YOLO confidence: {detector.thresholds.yolo_confidence}")
        print(f"\n  🔵 Circular Detection: {'ENABLED' if detector.thresholds.enable_circular_detection else 'DISABLED'}")
        if detector.thresholds.enable_circular_detection:
            print(f"     - Hough param2: {detector.thresholds.hough_param2} (higher = stricter)")
            print(f"     - Texture min: {detector.thresholds.circular_texture_variance_min}")
            print(f"     - Max confidence: {detector.thresholds.circular_confidence_max}")

        print(f"\n💡 To disable circular detection:")
        print(f"   Edit 'head_centric_thresholds.json' → set 'enable_circular_detection': false")
        print(f"\n💡 To make circular detection stricter:")
        print(f"   Increase: hough_param2 (40→50), circular_texture_variance_min (200→300)")
        print(f"   Decrease: circular_confidence_max (0.7→0.6)")

        # Process images
        num_to_process = min(5, len(test_images))
        print(f"\nProcessing {num_to_process} images...\n")

        results = detector.process_batch([str(p) for p in test_images[:num_to_process]])

        # Save results
        detector.save_results(results)

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        total_humans = sum(r["summary"].get("total_humans", 0) for r in results if r.get("summary"))

        print(f"Images Processed: {len(results)}")
        print(f"Total Humans Detected: {total_humans}")

        print("\nDetection Methods:")
        for method in ["haar_front", "haar_profile", "circular", "skin_based", "yolo"]:
            count = sum(
                r["summary"].get("detection_types", {}).get(method, 0)
                for r in results if r.get("summary")
            )
            if count > 0:
                print(f"  - {method.replace('_', ' ').title()}: {count}")

        print("\nOutput Files:")
        print("  - head_centric_results.json (detailed results)")
        print("  - head_centric_results_summary.csv (summary)")
        print("  - head_centric_results/ (annotated images)")

        print("\n" + "=" * 70)
        print("✓ Processing complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()