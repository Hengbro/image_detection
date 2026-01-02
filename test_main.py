"""Unit tests for the head-centric human detection system."""

import pytest
import numpy as np
import cv2
import json
import tempfile
import os

from main import (
    # Utility function
    calculate_iou,
    # Constants
    MIN_IMAGE_DIMENSION,
    BLANKET_MIN_AREA,
    BLANKET_MAX_AREA,
    HEAD_CIRCULARITY_MIN,
    HEAD_CIRCULARITY_MAX,
    # Dataclasses
    DetectionThresholds,
    HeadCandidate,
    ScanRegion,
    BlanketRegion,
    HumanDetection,
    BlanketCoveredHuman,
    # Classes
    SimplifiedBlanketDetector,
    HeadDetectionUnit,
    UpperBodyCompositionAnalyzer,
    StructuredGridScanner,
    CrossValidationModule,
    HeadCentricHumanDetectionSystem,
)


class TestCalculateIoU:
    """Tests for the calculate_iou utility function."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU of 1.0."""
        bbox = [0, 0, 100, 100]
        assert calculate_iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        bbox1 = [0, 0, 50, 50]
        bbox2 = [100, 100, 150, 150]
        assert calculate_iou(bbox1, bbox2) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IoU between 0 and 1."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]
        iou = calculate_iou(bbox1, bbox2)
        assert 0.0 < iou < 1.0
        # Expected: intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        assert iou == pytest.approx(2500 / 17500)

    def test_one_box_inside_other(self):
        """When one box is inside another, IoU should be ratio of areas."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [25, 25, 75, 75]
        iou = calculate_iou(bbox1, bbox2)
        # intersection = 50*50 = 2500, union = 10000 + 2500 - 2500 = 10000
        assert iou == pytest.approx(2500 / 10000)

    def test_touching_boxes(self):
        """Boxes that only touch (no area overlap) should have IoU of 0.0."""
        bbox1 = [0, 0, 50, 50]
        bbox2 = [50, 0, 100, 50]
        assert calculate_iou(bbox1, bbox2) == 0.0

    def test_float_coordinates(self):
        """Should work with float coordinates."""
        bbox1 = [0.0, 0.0, 100.5, 100.5]
        bbox2 = [0.0, 0.0, 100.5, 100.5]
        assert calculate_iou(bbox1, bbox2) == pytest.approx(1.0)


class TestDetectionThresholds:
    """Tests for the DetectionThresholds dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        thresholds = DetectionThresholds()
        assert thresholds.head_min_size == 25
        assert thresholds.head_max_size == 120
        assert thresholds.yolo_confidence == 0.25

    def test_custom_values(self):
        """Test creating thresholds with custom values."""
        thresholds = DetectionThresholds(head_min_size=30, head_max_size=150)
        assert thresholds.head_min_size == 30
        assert thresholds.head_max_size == 150

    def test_to_dict(self):
        """Test conversion to dictionary."""
        thresholds = DetectionThresholds()
        data = thresholds.to_dict()
        assert isinstance(data, dict)
        assert 'head_min_size' in data
        assert data['head_min_size'] == 25

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {'head_min_size': 40, 'head_max_size': 100}
        thresholds = DetectionThresholds.from_dict(data)
        assert thresholds.head_min_size == 40
        assert thresholds.head_max_size == 100

    def test_save_and_load(self):
        """Test saving and loading from file."""
        thresholds = DetectionThresholds(head_min_size=35)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            thresholds.save(filepath)
            loaded = DetectionThresholds.load(filepath)
            assert loaded.head_min_size == 35
        finally:
            os.unlink(filepath)


class TestHeadCandidate:
    """Tests for the HeadCandidate dataclass."""

    def test_center_calculation(self):
        """Test that center is calculated correctly from bbox."""
        candidate = HeadCandidate(
            bbox=[0, 0, 100, 100],
            confidence=0.9,
            detection_type='haar_front'
        )
        assert candidate.center == (50.0, 50.0)

    def test_custom_center(self):
        """Test that custom center overrides calculation."""
        candidate = HeadCandidate(
            bbox=[0, 0, 100, 100],
            confidence=0.9,
            detection_type='haar_front',
            center=(25.0, 25.0)
        )
        assert candidate.center == (25.0, 25.0)

    def test_confidence_range(self):
        """Test confidence value storage."""
        candidate = HeadCandidate(
            bbox=[0, 0, 50, 50],
            confidence=0.75,
            detection_type='circular'
        )
        assert candidate.confidence == 0.75


class TestScanRegion:
    """Tests for the ScanRegion dataclass."""

    def test_creation(self):
        """Test ScanRegion creation."""
        region = ScanRegion(
            bbox=[0, 0, 100, 200],
            priority=1.0,
            side='left',
            row=0
        )
        assert region.bbox == [0, 0, 100, 200]
        assert region.side == 'left'
        assert region.row == 0


class TestHumanDetection:
    """Tests for the HumanDetection dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        detection = HumanDetection(
            head_bbox=[10, 10, 50, 50],
            head_confidence=0.8,
            head_type='haar_front'
        )
        assert detection.torso_bbox is None
        assert detection.body_confidence == 0.0
        assert detection.final_confidence == 0.0
        assert detection.is_validated is False


class TestImageValidation:
    """Tests for image validation in HeadCentricHumanDetectionSystem."""

    def test_valid_image(self):
        """Valid image should not raise exception."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should not raise
        HeadCentricHumanDetectionSystem._validate_image(image)

    def test_none_image(self):
        """None image should raise ValueError."""
        with pytest.raises(ValueError, match="image is None"):
            HeadCentricHumanDetectionSystem._validate_image(None)

    def test_wrong_type(self):
        """Non-numpy array should raise ValueError."""
        with pytest.raises(ValueError, match="expected numpy array"):
            HeadCentricHumanDetectionSystem._validate_image([[1, 2], [3, 4]])

    def test_empty_image(self):
        """Empty image should raise ValueError."""
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="image is empty"):
            HeadCentricHumanDetectionSystem._validate_image(image)

    def test_wrong_dimensions(self):
        """Image with wrong dimensions should raise ValueError."""
        image = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(ValueError, match="expected 3 dimensions"):
            HeadCentricHumanDetectionSystem._validate_image(image)

    def test_wrong_channels(self):
        """Image with wrong channel count should raise ValueError."""
        image = np.zeros((100, 100, 4), dtype=np.uint8)  # RGBA instead of BGR
        with pytest.raises(ValueError, match="expected 3 color channels"):
            HeadCentricHumanDetectionSystem._validate_image(image)

    def test_too_small_image(self):
        """Image smaller than minimum should raise ValueError."""
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            HeadCentricHumanDetectionSystem._validate_image(image)

    def test_minimum_size_image(self):
        """Image at minimum size should be valid."""
        image = np.zeros((MIN_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION, 3), dtype=np.uint8)
        # Should not raise
        HeadCentricHumanDetectionSystem._validate_image(image)


class TestHeadDetectionUnit:
    """Tests for the HeadDetectionUnit class."""

    @pytest.fixture
    def detector(self):
        """Create a HeadDetectionUnit instance."""
        thresholds = DetectionThresholds()
        return HeadDetectionUnit(thresholds)

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.thresholds is not None
        assert detector.face_cascade is not None
        assert detector.profile_cascade is not None

    def test_has_human_colors_valid_roi(self, detector):
        """Test _has_human_colors with valid ROI."""
        # Create a skin-tone colored image (BGR format)
        # Skin tone in YCrCb: Y=any, Cr=133-173, Cb=77-127
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        roi[:, :] = [120, 150, 200]  # BGR that converts to skin-like YCrCb
        result = detector._has_human_colors(roi)
        assert result in (True, False)  # Can be numpy.bool_ or Python bool

    def test_has_human_colors_empty_roi(self, detector):
        """Test _has_human_colors with empty ROI returns False."""
        roi = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector._has_human_colors(roi) is False

    def test_has_human_colors_tiny_roi(self, detector):
        """Test _has_human_colors with 1x1 ROI (edge case)."""
        roi = np.zeros((1, 1, 3), dtype=np.uint8)
        roi[0, 0] = [120, 150, 200]  # BGR color
        # Should not crash due to division by zero
        result = detector._has_human_colors(roi)
        assert result in (True, False)  # Can be numpy.bool_ or Python bool

    def test_check_head_edge_pattern_empty(self, detector):
        """Test _check_head_edge_pattern with empty ROI."""
        roi = np.zeros((0, 0), dtype=np.uint8)
        assert detector._check_head_edge_pattern(roi) == 0.0

    def test_check_head_edge_pattern_valid(self, detector):
        """Test _check_head_edge_pattern with valid ROI."""
        roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = detector._check_head_edge_pattern(roi)
        assert 0.0 <= result <= 1.0

    def test_check_radial_gradient_empty(self, detector):
        """Test _check_radial_gradient with empty ROI."""
        roi = np.zeros((0, 0), dtype=np.uint8)
        assert detector._check_radial_gradient(roi) == 0.0

    def test_check_radial_gradient_small(self, detector):
        """Test _check_radial_gradient with too small ROI."""
        roi = np.zeros((5, 5), dtype=np.uint8)
        assert detector._check_radial_gradient(roi) == 0.0

    def test_is_metallic_surface_empty(self, detector):
        """Test _is_metallic_surface with empty ROI."""
        roi = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector._is_metallic_surface(roi) is False

    def test_check_circularity_empty(self, detector):
        """Test _check_circularity with empty ROI."""
        roi = np.zeros((0, 0), dtype=np.uint8)
        assert detector._check_circularity(roi) == 0.0

    def test_fuse_candidates_empty(self, detector):
        """Test _fuse_candidates with empty list."""
        result = detector._fuse_candidates([])
        assert result == []

    def test_fuse_candidates_single(self, detector):
        """Test _fuse_candidates with single candidate."""
        candidate = HeadCandidate(
            bbox=[0, 0, 50, 50],
            confidence=0.8,
            detection_type='haar_front'
        )
        result = detector._fuse_candidates([candidate])
        assert len(result) == 1

    def test_fuse_candidates_overlapping(self, detector):
        """Test _fuse_candidates merges overlapping candidates."""
        candidates = [
            HeadCandidate(bbox=[0, 0, 50, 50], confidence=0.8, detection_type='haar_front'),
            HeadCandidate(bbox=[5, 5, 55, 55], confidence=0.7, detection_type='circular'),
        ]
        result = detector._fuse_candidates(candidates)
        # Should merge into one
        assert len(result) == 1
        # Confidence should be boosted
        assert result[0].confidence >= 0.8


class TestSimplifiedBlanketDetector:
    """Tests for the SimplifiedBlanketDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a SimplifiedBlanketDetector instance."""
        thresholds = DetectionThresholds()
        return SimplifiedBlanketDetector(thresholds)

    def test_initialization(self, detector):
        """Test detector initialization with constants."""
        assert detector.min_blanket_area == BLANKET_MIN_AREA
        assert detector.max_blanket_area == BLANKET_MAX_AREA

    def test_detect_humans_in_blankets_empty_image(self, detector):
        """Test with minimal valid image."""
        # Use a non-zero image to avoid empty array issues in FFT
        image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        result = detector.detect_humans_in_blankets(image)
        assert isinstance(result, list)

    def test_detect_head_shape_small_roi(self, detector):
        """Test _detect_head_shape_in_blanket with too small ROI."""
        roi = np.zeros((30, 20, 3), dtype=np.uint8)
        has_head, bbox = detector._detect_head_shape_in_blanket(roi)
        assert has_head is False
        assert bbox is None

    def test_detect_shoulder_small_roi(self, detector):
        """Test _detect_shoulder_in_blanket with too small ROI."""
        roi = np.zeros((30, 30, 3), dtype=np.uint8)
        has_shoulder, shoulder_y = detector._detect_shoulder_in_blanket(roi)
        assert has_shoulder is False
        assert shoulder_y is None


class TestUpperBodyCompositionAnalyzer:
    """Tests for the UpperBodyCompositionAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an UpperBodyCompositionAnalyzer instance."""
        thresholds = DetectionThresholds()
        return UpperBodyCompositionAnalyzer(thresholds)

    def test_validate_human_proportions_empty_torso(self, analyzer):
        """Test with head bbox that results in empty torso region."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Head at bottom of image, no room for torso
        head_bbox = [25, 90, 75, 100]
        is_valid, conf, torso_bbox, details = analyzer.validate_human_proportions(head_bbox, image)
        assert isinstance(is_valid, bool)
        assert isinstance(details, dict)

    def test_detect_shoulders_empty(self, analyzer):
        """Test _detect_shoulders with empty ROI."""
        roi = np.zeros((0, 0, 3), dtype=np.uint8)
        has_shoulders, conf = analyzer._detect_shoulders(roi, 50)
        assert has_shoulders is False
        assert conf == 0.0


class TestStructuredGridScanner:
    """Tests for the StructuredGridScanner class."""

    @pytest.fixture
    def scanner(self):
        """Create a StructuredGridScanner instance."""
        thresholds = DetectionThresholds()
        return StructuredGridScanner((480, 640), thresholds)

    def test_grid_creation(self, scanner):
        """Test that grid is created with expected regions."""
        assert len(scanner.grid) > 0
        # Should have regions for each row (left, aisle, right)
        assert len(scanner.grid) == scanner.thresholds.num_rows * 3

    def test_grid_regions_have_correct_attributes(self, scanner):
        """Test that grid regions have required attributes."""
        for region in scanner.grid:
            assert hasattr(region, 'bbox')
            assert hasattr(region, 'side')
            assert hasattr(region, 'row')
            assert len(region.bbox) == 4

    def test_is_unique_detection_empty_list(self, scanner):
        """Test _is_unique_detection with empty existing list."""
        bbox = [0, 0, 50, 50]
        assert scanner._is_unique_detection(bbox, []) is True

    def test_is_unique_detection_duplicate(self, scanner):
        """Test _is_unique_detection identifies duplicates."""
        existing = [
            HumanDetection(
                head_bbox=[0, 0, 50, 50],
                head_confidence=0.8,
                head_type='haar_front'
            )
        ]
        # Same bbox should be identified as duplicate
        assert scanner._is_unique_detection([0, 0, 50, 50], existing) is False
        # Different bbox should be unique
        assert scanner._is_unique_detection([200, 200, 250, 250], existing) is True


class TestCrossValidationModule:
    """Tests for the CrossValidationModule class."""

    @pytest.fixture
    def validator(self):
        """Create a CrossValidationModule instance."""
        thresholds = DetectionThresholds()
        return CrossValidationModule(thresholds)

    def test_validate_detections_empty(self, validator):
        """Test with empty detection list."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = validator.validate_detections([], image)
        assert result == []

    def test_check_head_quality_valid(self, validator):
        """Test _check_head_quality with valid detection."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detection = HumanDetection(
            head_bbox=[50, 50, 100, 100],
            head_confidence=0.8,
            head_type='haar_front'
        )
        quality = validator._check_head_quality(detection, image)
        assert 0.0 <= quality <= 1.0

    def test_check_biological_plausibility(self, validator):
        """Test _check_biological_plausibility."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detection = HumanDetection(
            head_bbox=[50, 50, 100, 100],
            head_confidence=0.8,
            head_type='haar_front'
        )
        is_plausible, score = validator._check_biological_plausibility(detection, image)
        assert isinstance(is_plausible, bool)
        assert 0.0 <= score <= 1.0


class TestConstants:
    """Tests to verify constants are properly defined."""

    def test_min_image_dimension(self):
        """Test MIN_IMAGE_DIMENSION is reasonable."""
        assert MIN_IMAGE_DIMENSION > 0
        assert MIN_IMAGE_DIMENSION <= 100

    def test_blanket_area_range(self):
        """Test blanket area constants are valid."""
        assert BLANKET_MIN_AREA > 0
        assert BLANKET_MAX_AREA > BLANKET_MIN_AREA

    def test_circularity_range(self):
        """Test circularity constants are in valid range."""
        assert 0.0 <= HEAD_CIRCULARITY_MIN < HEAD_CIRCULARITY_MAX <= 1.0


class TestIntegration:
    """Integration tests for the full detection pipeline."""

    @pytest.fixture
    def detector(self):
        """Create a HeadCentricHumanDetectionSystem instance."""
        thresholds = DetectionThresholds()
        return HeadCentricHumanDetectionSystem(thresholds)

    def test_detect_humans_blank_image(self, detector):
        """Test detection on blank image."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.detect_humans(image)
        assert isinstance(result, list)

    def test_detect_humans_random_image(self, detector):
        """Test detection on random noise image."""
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = detector.detect_humans(image)
        assert isinstance(result, list)

    def test_yolo_lazy_loading(self, detector):
        """Test that YOLO model is lazy loaded."""
        # Before accessing, _yolo_model should be None
        assert detector._yolo_model is None
        # After accessing property, model should be loaded
        _ = detector.yolo_model
        assert detector._yolo_model is not None

    def test_visualize_detections(self, detector):
        """Test visualization doesn't crash."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [
            HumanDetection(
                head_bbox=[50, 50, 100, 100],
                head_confidence=0.8,
                head_type='haar_front',
                final_confidence=0.75
            )
        ]
        result = detector.visualize_detections(image, detections)
        assert result.shape == image.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
