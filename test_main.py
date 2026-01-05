"""Unit tests for the head-centric human detection system."""

import os
import tempfile

import numpy as np
import pytest

from main import (
    # Utility function
    calculate_iou,
    integrate_with_existing_system,
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
    HumanDetection,
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


class TestHeadDetectionUnitAdvanced:
    """Advanced tests for HeadDetectionUnit methods with low coverage."""

    @pytest.fixture
    def detector(self):
        """Create a HeadDetectionUnit instance."""
        thresholds = DetectionThresholds()
        return HeadDetectionUnit(thresholds)

    # Tests for _check_radial_gradient
    def test_check_radial_gradient_valid_roi(self, detector):
        """Test _check_radial_gradient with valid ROI that has gradient."""
        # Create a gradient image (bright center, dark edges)
        roi = np.zeros((50, 50), dtype=np.uint8)
        center_y, center_x = 25, 25
        for y in range(50):
            for x in range(50):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # Bright center, darker edges
                roi[y, x] = max(0, 200 - int(dist * 4))
        score = detector._check_radial_gradient(roi)
        assert 0.0 <= score <= 1.0

    def test_check_radial_gradient_small_roi(self, detector):
        """Test _check_radial_gradient with too small ROI."""
        roi = np.zeros((5, 5), dtype=np.uint8)
        score = detector._check_radial_gradient(roi)
        assert score == 0.0

    def test_check_radial_gradient_flat_image(self, detector):
        """Test _check_radial_gradient with flat (no gradient) image."""
        roi = np.full((30, 30), 128, dtype=np.uint8)
        score = detector._check_radial_gradient(roi)
        # Flat image has no gradient, should return 0.0
        assert score == 0.0

    def test_check_radial_gradient_strong_gradient(self, detector):
        """Test _check_radial_gradient with strong gradient."""
        # Create strong gradient (center much brighter than edges)
        roi = np.zeros((40, 40), dtype=np.uint8)
        roi[15:25, 15:25] = 200  # Bright center
        roi[:5, :] = 50  # Dark top
        roi[-5:, :] = 50  # Dark bottom
        roi[:, :5] = 50  # Dark left
        roi[:, -5:] = 50  # Dark right
        score = detector._check_radial_gradient(roi)
        assert 0.0 <= score <= 1.0

    # Tests for _check_circularity
    def test_check_circularity_valid_circle(self, detector):
        """Test _check_circularity with a circular shape."""
        roi = np.zeros((100, 100), dtype=np.uint8)
        # Draw a filled circle
        import cv2
        cv2.circle(roi, (50, 50), 30, 255, -1)
        circularity = detector._check_circularity(roi)
        # Circle should have high circularity (close to 1.0)
        assert 0.7 <= circularity <= 1.0

    def test_check_circularity_square(self, detector):
        """Test _check_circularity with a square shape."""
        roi = np.zeros((100, 100), dtype=np.uint8)
        roi[25:75, 25:75] = 255  # Fill a square
        circularity = detector._check_circularity(roi)
        # Square has lower circularity than circle
        assert 0.0 <= circularity <= 1.0

    def test_check_circularity_no_contours(self, detector):
        """Test _check_circularity with empty image."""
        roi = np.zeros((50, 50), dtype=np.uint8)
        circularity = detector._check_circularity(roi)
        assert circularity == 0.0

    # Tests for _is_metallic_surface
    def test_is_metallic_surface_metallic(self, detector):
        """Test _is_metallic_surface with metallic-like colors."""
        # Create image with low saturation, high value (metallic appearance)
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        # BGR: bright gray/silver
        roi[:, :] = [200, 200, 200]
        result = detector._is_metallic_surface(roi)
        assert isinstance(result, bool)

    def test_is_metallic_surface_colorful(self, detector):
        """Test _is_metallic_surface with colorful image."""
        # Create colorful image (should not be metallic)
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        roi[:, :] = [0, 0, 255]  # Red in BGR
        result = detector._is_metallic_surface(roi)
        assert isinstance(result, bool)

    def test_is_metallic_surface_empty(self, detector):
        """Test _is_metallic_surface with empty ROI."""
        roi = np.zeros((0, 0, 3), dtype=np.uint8)
        result = detector._is_metallic_surface(roi)
        assert result is False

    # Tests for _detect_circular_heads
    def test_detect_circular_heads_blank_image(self, detector):
        """Test _detect_circular_heads with blank image."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        candidates = detector._detect_circular_heads(image, (0, 0))
        assert isinstance(candidates, list)

    def test_detect_circular_heads_with_circle(self, detector):
        """Test _detect_circular_heads with image containing circles."""
        import cv2
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add some texture and a circle-like shape
        image[:, :] = [100, 100, 100]
        # Add skin-tone colored circle
        cv2.circle(image, (150, 200), 30, [140, 140, 200], -1)
        candidates = detector._detect_circular_heads(image, (0, 0))
        assert isinstance(candidates, list)

    def test_detect_circular_heads_with_offset(self, detector):
        """Test _detect_circular_heads with non-zero offset."""
        image = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        candidates = detector._detect_circular_heads(image, (50, 100))
        assert isinstance(candidates, list)
        # Any candidates should have offset applied
        for candidate in candidates:
            assert candidate.bbox[0] >= 50
            assert candidate.bbox[1] >= 100

    # Tests for _check_head_edge_pattern
    def test_check_head_edge_pattern_moderate_edges(self, detector):
        """Test _check_head_edge_pattern with moderate edge density."""
        # Create image with moderate edges
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[::3, :, :] = 200  # Horizontal lines every 3 rows
        score = detector._check_head_edge_pattern(roi)
        assert 0.0 <= score <= 1.0

    def test_check_head_edge_pattern_smooth(self, detector):
        """Test _check_head_edge_pattern with smooth (no edges) image."""
        roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        score = detector._check_head_edge_pattern(roi)
        # Too smooth should return low score
        assert score == 0.0

    def test_check_head_edge_pattern_too_many_edges(self, detector):
        """Test _check_head_edge_pattern with too many edges."""
        # Create very busy image
        roi = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        score = detector._check_head_edge_pattern(roi)
        assert 0.0 <= score <= 1.0


class TestStructuredGridScannerAdvanced:
    """Advanced tests for StructuredGridScanner with low coverage."""

    @pytest.fixture
    def scanner(self):
        """Create a StructuredGridScanner instance."""
        thresholds = DetectionThresholds()
        return StructuredGridScanner((480, 640), thresholds)

    @pytest.fixture
    def head_detector(self):
        """Create a HeadDetectionUnit instance."""
        thresholds = DetectionThresholds()
        return HeadDetectionUnit(thresholds)

    # Tests for _seat_penalty
    def test_seat_penalty_normal_head(self, scanner):
        """Test _seat_penalty with normal head-like detection."""
        head = HeadCandidate(
            bbox=[100, 200, 150, 260],  # Normal aspect ratio
            confidence=0.8,
            detection_type='haar_front'
        )
        region = ScanRegion(
            bbox=[50.0, 100.0, 300.0, 400.0],
            priority=1.0,
            side='left',
            row=2
        )
        penalty = scanner._seat_penalty(head, region)
        assert 0.4 <= penalty <= 1.0

    def test_seat_penalty_wide_object(self, scanner):
        """Test _seat_penalty with wide object (seat-like)."""
        head = HeadCandidate(
            bbox=[100, 200, 200, 240],  # Wide aspect ratio
            confidence=0.6,
            detection_type='circular'
        )
        region = ScanRegion(
            bbox=[50.0, 100.0, 300.0, 400.0],
            priority=1.0,
            side='left',
            row=1
        )
        penalty = scanner._seat_penalty(head, region)
        # Wide object should get penalty
        assert penalty < 1.0

    def test_seat_penalty_touching_region_top(self, scanner):
        """Test _seat_penalty with object touching region top."""
        region = ScanRegion(
            bbox=[50.0, 100.0, 300.0, 400.0],
            priority=1.0,
            side='left',
            row=0
        )
        head = HeadCandidate(
            bbox=[100, 102, 150, 152],  # Close to region top (100)
            confidence=0.7,
            detection_type='circular'
        )
        penalty = scanner._seat_penalty(head, region)
        # Should have penalty for touching top
        assert penalty < 1.0

    def test_seat_penalty_front_row(self, scanner):
        """Test _seat_penalty for front row detection."""
        head = HeadCandidate(
            bbox=[100, 200, 150, 250],
            confidence=0.7,
            detection_type='circular'
        )
        region = ScanRegion(
            bbox=[50.0, 100.0, 300.0, 400.0],
            priority=1.0,
            side='left',
            row=0  # Front row
        )
        penalty = scanner._seat_penalty(head, region)
        # Front row should have stricter penalty
        assert penalty < 1.0

    def test_seat_penalty_minimum(self, scanner):
        """Test _seat_penalty returns at least 0.4."""
        head = HeadCandidate(
            bbox=[100, 100, 200, 120],  # Very wide, touching top
            confidence=0.5,
            detection_type='circular'
        )
        region = ScanRegion(
            bbox=[50.0, 100.0, 300.0, 400.0],
            priority=1.0,
            side='left',
            row=0
        )
        penalty = scanner._seat_penalty(head, region)
        assert penalty >= 0.4

    # Tests for scan_with_context
    def test_scan_with_context_blank_image(self, scanner, head_detector):
        """Test scan_with_context with blank image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = scanner.scan_with_context(image, head_detector)
        assert isinstance(detections, list)

    def test_scan_with_context_textured_image(self, scanner, head_detector):
        """Test scan_with_context with textured image."""
        image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        detections = scanner.scan_with_context(image, head_detector)
        assert isinstance(detections, list)


class TestIntegrateWithExistingSystem:
    """Tests for the integrate_with_existing_system function."""

    def test_integrate_empty_inputs(self):
        """Test with empty inputs."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = integrate_with_existing_system(
            image,
            existing_detections=[],
            grid_regions=[]
        )
        assert isinstance(result, list)
        assert len(result) == 0

    def test_integrate_with_existing_detections(self):
        """Test with existing detections."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        existing = [{'bbox': [50, 50, 100, 100]}]
        grid_regions = [{'bbox': [50, 50, 100, 100]}]
        result = integrate_with_existing_system(
            image,
            existing_detections=existing,
            grid_regions=grid_regions
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_integrate_with_uncovered_regions(self):
        """Test with grid regions that don't have detections."""
        image = np.random.randint(50, 150, (200, 200, 3), dtype=np.uint8)
        existing = [{'bbox': [0, 0, 50, 50]}]
        grid_regions = [
            {'bbox': [0, 0, 50, 50]},  # Has detection
            {'bbox': [100, 100, 150, 150]}  # No detection
        ]
        result = integrate_with_existing_system(
            image,
            existing_detections=existing,
            grid_regions=grid_regions
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_integrate_with_custom_thresholds(self):
        """Test with custom thresholds."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        thresholds = DetectionThresholds(head_min_size=30)
        result = integrate_with_existing_system(
            image,
            existing_detections=[],
            grid_regions=[{'bbox': [0, 0, 100, 100]}],
            thresholds=thresholds
        )
        assert isinstance(result, list)

    def test_integrate_partial_overlap(self):
        """Test with partial overlap between detection and region."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        existing = [{'bbox': [40, 40, 80, 80]}]
        grid_regions = [{'bbox': [50, 50, 100, 100]}]  # 30% overlap
        result = integrate_with_existing_system(
            image,
            existing_detections=existing,
            grid_regions=grid_regions
        )
        assert isinstance(result, list)


class TestProcessingFunctions:
    """Tests for process_image and process_batch functions."""

    @pytest.fixture
    def detector(self):
        """Create a HeadCentricHumanDetectionSystem instance."""
        thresholds = DetectionThresholds()
        return HeadCentricHumanDetectionSystem(thresholds)

    def test_process_image_nonexistent_file(self, detector):
        """Test process_image with nonexistent file."""
        result = detector.process_image("/nonexistent/image.jpg", save_output=False)
        assert isinstance(result, dict)
        assert result["error"] is not None
        assert "Cannot load image" in result["error"]

    def test_process_image_valid_image(self, detector):
        """Test process_image with a valid image file."""
        import cv2
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        try:
            # Create and save a test image
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_image)

            result = detector.process_image(temp_path, save_output=False)
            assert isinstance(result, dict)
            assert result["error"] is None
            assert "detections" in result
            assert "summary" in result
            assert result["image_path"] == temp_path
        finally:
            os.unlink(temp_path)

    def test_process_image_with_save_output(self, detector):
        """Test process_image with save_output=True."""
        import cv2
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        try:
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_image)

            result = detector.process_image(temp_path, save_output=True)
            assert isinstance(result, dict)
            assert result["error"] is None
        finally:
            os.unlink(temp_path)
            # Clean up output directory if created
            import shutil
            if os.path.exists("head_centric_results"):
                shutil.rmtree("head_centric_results")

    def test_process_batch_empty_list(self, detector):
        """Test process_batch with empty list."""
        results = detector.process_batch([])
        assert isinstance(results, list)
        assert len(results) == 0

    def test_process_batch_with_images(self, detector):
        """Test process_batch with multiple images."""
        import cv2
        temp_paths = []
        try:
            # Create temporary image files
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    temp_path = f.name
                    temp_paths.append(temp_path)
                test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                cv2.imwrite(temp_path, test_image)

            results = detector.process_batch(temp_paths)
            assert isinstance(results, list)
            assert len(results) == 2
            for result in results:
                assert isinstance(result, dict)
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)
            import shutil
            if os.path.exists("head_centric_results"):
                shutil.rmtree("head_centric_results")

    def test_process_batch_mixed_valid_invalid(self, detector):
        """Test process_batch with mix of valid and invalid paths."""
        import cv2
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        try:
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(temp_path, test_image)

            results = detector.process_batch([temp_path, "/nonexistent/image.jpg"])
            assert len(results) == 2
            assert results[0]["error"] is None
            assert results[1]["error"] is not None
        finally:
            os.unlink(temp_path)
            import shutil
            if os.path.exists("head_centric_results"):
                shutil.rmtree("head_centric_results")


class TestSaveResults:
    """Tests for save_results and _save_summary_csv methods."""

    @pytest.fixture
    def detector(self):
        """Create a HeadCentricHumanDetectionSystem instance."""
        thresholds = DetectionThresholds()
        return HeadCentricHumanDetectionSystem(thresholds)

    def test_save_results_empty(self, detector):
        """Test save_results with empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.json")
            detector.save_results([], output_file)
            assert os.path.exists(output_file)
            assert os.path.exists(output_file.replace('.json', '_summary.csv'))

    def test_save_results_with_data(self, detector):
        """Test save_results with actual results."""
        results = [
            {
                "image_path": "/test/image1.jpg",
                "timestamp": "2024-01-01T00:00:00",
                "detections": [
                    {
                        "head_bbox": [50, 50, 100, 100],
                        "head_confidence": 0.8,
                        "head_type": "haar_front",
                        "final_confidence": 0.75
                    }
                ],
                "summary": {
                    "total_humans": 1,
                    "detection_types": {
                        "haar_front": 1,
                        "haar_profile": 0,
                        "circular": 0,
                        "skin_based": 0,
                        "yolo": 0
                    },
                    "confidence_stats": {
                        "high": 0,
                        "medium": 1,
                        "low": 0,
                        "average": 0.75
                    }
                },
                "error": None
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.json")
            detector.save_results(results, output_file)

            # Verify JSON was created
            assert os.path.exists(output_file)
            import json
            with open(output_file, 'r') as f:
                data = json.load(f)
            assert "metadata" in data
            assert "results" in data
            assert len(data["results"]) == 1

            # Verify CSV was created
            csv_file = output_file.replace('.json', '_summary.csv')
            assert os.path.exists(csv_file)

    def test_save_results_with_error(self, detector):
        """Test save_results with error in results."""
        results = [
            {
                "image_path": "/test/image1.jpg",
                "timestamp": "2024-01-01T00:00:00",
                "detections": [],
                "summary": {},
                "error": "Cannot load image"
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.json")
            detector.save_results(results, output_file)
            assert os.path.exists(output_file)


class TestCrossValidationAdvanced:
    """Advanced tests for CrossValidationModule with low coverage."""

    @pytest.fixture
    def validator(self):
        """Create a CrossValidationModule instance."""
        thresholds = DetectionThresholds()
        return CrossValidationModule(thresholds)

    def test_validate_detections_with_multiple(self, validator):
        """Test validate_detections with multiple detections."""
        image = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        detections = [
            HumanDetection(
                head_bbox=[50, 50, 100, 100],
                head_confidence=0.9,
                head_type='haar_front'
            ),
            HumanDetection(
                head_bbox=[150, 150, 200, 200],
                head_confidence=0.7,
                head_type='circular'
            )
        ]
        result = validator.validate_detections(detections, image)
        assert isinstance(result, list)

    def test_is_likely_artifact_valid(self, validator):
        """Test _is_likely_artifact with valid detection."""
        image = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        detection = HumanDetection(
            head_bbox=[50, 100, 100, 150],  # Not at top
            head_confidence=0.8,
            head_type='haar_front'
        )
        is_artifact, artifact_score = validator._is_likely_artifact(detection, image)
        assert isinstance(is_artifact, bool)
        assert isinstance(artifact_score, float)

    def test_is_likely_artifact_top_of_image(self, validator):
        """Test _is_likely_artifact with detection at top of image."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detection = HumanDetection(
            head_bbox=[50, 5, 100, 30],  # At top of image
            head_confidence=0.6,
            head_type='circular'
        )
        is_artifact, reason = validator._is_likely_artifact(detection, image)
        assert isinstance(is_artifact, bool)

    def test_check_head_quality_small_head(self, validator):
        """Test _check_head_quality with small head detection."""
        image = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        detection = HumanDetection(
            head_bbox=[50, 50, 60, 60],  # Very small
            head_confidence=0.5,
            head_type='circular'
        )
        quality = validator._check_head_quality(detection, image)
        assert 0.0 <= quality <= 1.0


class TestYOLOFallback:
    """Tests for YOLO fallback detection."""

    @pytest.fixture
    def detector(self):
        """Create a HeadCentricHumanDetectionSystem instance."""
        thresholds = DetectionThresholds()
        return HeadCentricHumanDetectionSystem(thresholds)

    def test_yolo_fallback_blank_image(self, detector):
        """Test _yolo_fallback with blank image."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        existing = []
        result = detector._yolo_fallback(image, existing)
        assert isinstance(result, list)

    def test_yolo_fallback_with_existing_detections(self, detector):
        """Test _yolo_fallback with existing detections."""
        image = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        existing = [
            HumanDetection(
                head_bbox=[50, 50, 100, 100],
                head_confidence=0.8,
                head_type='haar_front'
            )
        ]
        result = detector._yolo_fallback(image, existing)
        assert isinstance(result, list)


class TestVerifyCountLogic:
    """Tests for _verify_count_logic method."""

    @pytest.fixture
    def detector(self):
        """Create a HeadCentricHumanDetectionSystem instance."""
        thresholds = DetectionThresholds()
        return HeadCentricHumanDetectionSystem(thresholds)

    def test_verify_count_logic_empty(self, detector):
        """Test _verify_count_logic with empty detections."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Should not raise any exceptions
        detector._verify_count_logic([], image)

    def test_verify_count_logic_multiple_types(self, detector):
        """Test _verify_count_logic with multiple detection types."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [
            HumanDetection(
                head_bbox=[50, 50, 100, 100],
                head_confidence=0.9,
                head_type='haar_front',
                final_confidence=0.85
            ),
            HumanDetection(
                head_bbox=[150, 50, 200, 100],
                head_confidence=0.7,
                head_type='circular',
                final_confidence=0.65
            ),
            HumanDetection(
                head_bbox=[50, 150, 100, 200],
                head_confidence=0.8,
                head_type='yolo',
                final_confidence=0.75
            )
        ]
        # Should not raise any exceptions
        detector._verify_count_logic(detections, image)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
