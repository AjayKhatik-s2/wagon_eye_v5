"""
Spatio-Temporal Door State Verification Module (Enhanced)

Lightweight post-decision filter that validates YOLO "OPEN" door predictions
using temporal analysis of consecutive door crops. Detects reflection artifacts,
tracks gap evolution, verifies motion consistency, and performs feature-vector
temporal aggregation across a rolling window of 20 frames.

Enhancements over v1:
- Multi-scale gap analysis (top/middle/bottom slices)
- Panel separation validation (dark gap vs bright reflection)
- Feature-vector temporal aggregation with exponential weighting
- Directional motion analysis via optical flow
- Texture uniformity check for reflection detection
- Weighted temporal confidence smoothing

This module does NOT modify YOLO inference, weights, or confidence thresholds.
It operates purely as a verification layer after the existing FSM decision.

Integration point: DoorTrack.get_decision() in door_tracker.py
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TemporalReasoningConfig:
    """Configuration for temporal verification."""
    # Buffer settings
    buffer_size: int = 20              # Rolling window of door crops (increased from 15)
    min_frames_for_analysis: int = 8   # Minimum frames before analysis activates

    # Reflection detection
    reflection_brightness_threshold: int = 220   # Pixel intensity to flag bright lines
    reflection_spatial_tolerance: float = 3.0    # Max px displacement for "static" line
    reflection_persistence_ratio: float = 0.75   # Fraction of frames line must persist
    texture_uniformity_threshold: float = 0.25   # CV below this = uniform texture (reflection)

    # Gap evolution
    gap_min_edge_strength: float = 30.0   # Sobel gradient threshold for edge detection
    gap_widening_slope: float = 0.3       # Min slope (px/frame) to consider widening
    gap_narrowing_slope: float = -0.3     # Max slope for narrowing
    gap_stable_tolerance: float = 0.2     # Slope magnitude below this = stable
    multi_scale_slices: int = 3           # Number of vertical slices for multi-scale analysis

    # Panel separation validation
    panel_separation_darkness_threshold: float = 0.7  # Gap region must be this much darker
    # than surrounding panels (ratio: gap_brightness / panel_brightness)
    # Below this ratio = true dark gap; above = bright reflection

    # Motion consistency
    motion_min_diff_threshold: float = 8.0    # Min mean pixel diff to count as motion
    motion_structural_cv_threshold: float = 0.4  # CV above this = structured (real motion)
    motion_min_moving_frames: float = 0.3     # Fraction of frames with structural motion

    # Optical flow motion direction analysis
    flow_min_magnitude: float = 1.0       # Min optical flow magnitude to count as motion
    flow_horizontal_ratio: float = 0.6    # Fraction of flow that must be horizontal (panel sliding)

    # Feature aggregation
    feature_vector_dim: int = 32           # Dimension of per-crop feature vector
    feature_decay_rate: float = 0.85       # Exponential decay for temporal weighting
    feature_consistency_threshold: float = 0.15  # Below this = temporally stable features

    # Verification decision
    override_confidence_reduction: float = 0.35   # Max confidence reduction on override
    confirmation_confidence_boost: float = 0.05   # Confidence boost on confirmation

    # Crop normalization
    crop_resize_width: int = 64    # Normalized crop width for analysis
    crop_resize_height: int = 128  # Normalized crop height for analysis


# =============================================================================
# TEMPORAL CROP BUFFER
# =============================================================================

@dataclass
class CropEntry:
    """Single entry in the temporal crop buffer."""
    frame_idx: int
    crop_gray: np.ndarray         # Grayscale normalized crop
    bbox: np.ndarray              # Original bounding box [x1, y1, x2, y2]
    edge_map: np.ndarray          # Canny edge map of the crop
    brightness_profile: np.ndarray   # Column-averaged brightness profile
    texture_cv: float             # Coefficient of variation of texture (new)
    feature_vector: np.ndarray    # Compact feature vector for temporal aggregation (new)


class TemporalCropBuffer:
    """
    Per-track rolling buffer of door crops for temporal analysis.

    Stores grayscale door ROI crops with precomputed features
    (edge maps, brightness profiles, feature vectors) to avoid
    redundant computation.
    """

    def __init__(self, config: TemporalReasoningConfig = None):
        self.config = config or TemporalReasoningConfig()
        self.entries: deque = deque(maxlen=self.config.buffer_size)

    def add_crop(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_idx: int
    ):
        """
        Extract, normalize, and store a door crop with precomputed features.

        Args:
            frame: Full BGR frame
            bbox: Door bounding box [x1, y1, x2, y2]
            frame_idx: Current frame index
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]

        # Convert to grayscale
        if len(crop.shape) == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop.copy()

        # Resize to normalized dimensions for consistent analysis
        crop_resized = cv2.resize(
            crop_gray,
            (self.config.crop_resize_width, self.config.crop_resize_height),
            interpolation=cv2.INTER_AREA
        )

        # Precompute edge map
        edge_map = cv2.Canny(crop_resized, 50, 150)

        # Precompute column-averaged brightness profile (vertical bright lines)
        brightness_profile = np.mean(crop_resized, axis=0)

        # Compute texture coefficient of variation
        texture_cv = self._compute_texture_cv(crop_resized)

        # Compute compact feature vector
        feature_vector = self._compute_feature_vector(crop_resized, edge_map, brightness_profile)

        entry = CropEntry(
            frame_idx=frame_idx,
            crop_gray=crop_resized,
            bbox=np.array(bbox),
            edge_map=edge_map,
            brightness_profile=brightness_profile,
            texture_cv=texture_cv,
            feature_vector=feature_vector
        )
        self.entries.append(entry)

    def _compute_texture_cv(self, crop_gray: np.ndarray) -> float:
        """
        Compute coefficient of variation of texture (Laplacian variance).

        Low CV → uniform texture (likely reflection/shine)
        High CV → varied texture (likely real depth/opening)
        """
        laplacian = cv2.Laplacian(crop_gray, cv2.CV_64F)
        lap_abs = np.abs(laplacian)
        mean_val = np.mean(lap_abs)
        if mean_val < 1e-6:
            return 0.0
        return float(np.std(lap_abs) / mean_val)

    def _compute_feature_vector(
        self,
        crop_gray: np.ndarray,
        edge_map: np.ndarray,
        brightness_profile: np.ndarray
    ) -> np.ndarray:
        """
        Compute a compact 32-dimensional feature vector for temporal aggregation.

        Components (8 each):
        - Edge orientation histogram (8 bins)
        - Brightness statistics (8 dims)
        - Gradient statistics (8 dims)
        - Spatial statistics (8 dims)
        """
        features = []

        # 1. Edge orientation histogram (8 bins)
        sobel_x = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        angle = np.arctan2(sobel_y, sobel_x + 1e-8)  # [-pi, pi]

        # Only use edges with significant magnitude
        mask = magnitude > 20.0
        if np.any(mask):
            hist, _ = np.histogram(angle[mask], bins=8, range=(-np.pi, np.pi),
                                   weights=magnitude[mask])
            total = np.sum(hist)
            if total > 0:
                hist = hist / total
        else:
            hist = np.zeros(8)
        features.extend(hist.tolist())

        # 2. Brightness statistics (8 dims)
        bp = brightness_profile
        if len(bp) > 0:
            features.extend([
                np.mean(bp) / 255.0,
                np.std(bp) / 128.0,
                np.max(bp) / 255.0,
                np.min(bp) / 255.0,
                float(np.median(bp)) / 255.0,
                float(np.percentile(bp, 25)) / 255.0,
                float(np.percentile(bp, 75)) / 255.0,
                float(np.sum(bp > 200)) / max(len(bp), 1)  # Fraction of bright columns
            ])
        else:
            features.extend([0.0] * 8)

        # 3. Gradient statistics (8 dims)
        grad_mag = magnitude
        features.extend([
            np.mean(grad_mag) / 128.0,
            np.std(grad_mag) / 128.0,
            np.max(grad_mag) / 255.0 if grad_mag.size > 0 else 0.0,
            float(np.sum(edge_map > 0)) / max(edge_map.size, 1),  # Edge density
            np.mean(np.abs(sobel_x)) / 128.0,  # Vertical edge strength
            np.mean(np.abs(sobel_y)) / 128.0,  # Horizontal edge strength
            float(np.mean(np.abs(sobel_x)) / (np.mean(np.abs(sobel_y)) + 1e-6)),  # V/H ratio
            float(np.std(np.mean(np.abs(sobel_x), axis=0))) / 64.0  # Vertical edge spread
        ])

        # 4. Spatial statistics (8 dims) - top/middle/bottom analysis
        h = crop_gray.shape[0]
        third = max(h // 3, 1)
        top = crop_gray[:third, :]
        mid = crop_gray[third:2*third, :]
        bot = crop_gray[2*third:, :]

        features.extend([
            np.mean(top) / 255.0,
            np.mean(mid) / 255.0,
            np.mean(bot) / 255.0,
            np.std(top) / 128.0,
            np.std(mid) / 128.0,
            np.std(bot) / 128.0,
            abs(np.mean(top) - np.mean(bot)) / 255.0,  # Top-bottom gradient
            abs(np.mean(top) - np.mean(mid)) / 255.0   # Top-middle gradient
        ])

        return np.array(features[:32], dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.entries)

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for analysis."""
        return self.size >= (self.config.min_frames_for_analysis if self.config else 8)

    def get_entries(self) -> List[CropEntry]:
        """Get all entries as a list."""
        return list(self.entries)

    def clear(self):
        """Clear the buffer."""
        self.entries.clear()


# =============================================================================
# GAP TREND CLASSIFICATION
# =============================================================================

class GapTrend(Enum):
    """Classification of gap evolution over time."""
    WIDENING = "WIDENING"
    NARROWING = "NARROWING"
    STABLE = "STABLE"
    NO_GAP = "NO_GAP"


# =============================================================================
# REFLECTION DETECTOR (Enhanced)
# =============================================================================

class ReflectionDetector:
    """
    Detects static vertical bright lines that indicate reflections/shine
    rather than true door openings.

    Reflections produce bright vertical bands that remain spatially fixed
    across frames without progressive edge displacement. True door openings
    show progressive edge movement as the gap widens.

    Enhanced with:
    - Texture uniformity check (reflections have uniform low-texture bright regions)
    - Edge displacement analysis
    """

    def __init__(self, config: TemporalReasoningConfig):
        self.config = config

    def detect(self, entries: List[CropEntry]) -> Tuple[bool, float, str]:
        """
        Analyze temporal window for static reflection artifacts.

        Args:
            entries: List of CropEntry from temporal buffer

        Returns:
            (is_reflection, confidence, reason)
        """
        if len(entries) < 3:
            return False, 0.0, "insufficient_frames"

        threshold = self.config.reflection_brightness_threshold
        spatial_tol = self.config.reflection_spatial_tolerance
        persistence_ratio = self.config.reflection_persistence_ratio

        # Find bright columns in each frame's brightness profile
        bright_columns_per_frame = []
        for entry in entries:
            profile = entry.brightness_profile
            bright_cols = np.where(profile > threshold)[0]
            bright_columns_per_frame.append(set(bright_cols.tolist()))

        if not bright_columns_per_frame:
            return False, 0.0, "no_bright_lines"

        # Find columns that are bright across most frames (static bright lines)
        all_bright_cols = set()
        for bc in bright_columns_per_frame:
            all_bright_cols.update(bc)

        if not all_bright_cols:
            return False, 0.0, "no_bright_lines"

        static_line_count = 0
        total_candidate_cols = 0

        for col in all_bright_cols:
            total_candidate_cols += 1
            frames_with_col = 0
            for bc_set in bright_columns_per_frame:
                for c in bc_set:
                    if abs(c - col) <= spatial_tol:
                        frames_with_col += 1
                        break

            if frames_with_col / len(entries) >= persistence_ratio:
                static_line_count += 1

        if total_candidate_cols == 0:
            return False, 0.0, "no_candidates"

        reflection_ratio = static_line_count / total_candidate_cols

        # Primary reflection criterion
        is_reflection = (
            static_line_count >= 2 and reflection_ratio > 0.5
        ) or (
            static_line_count >= 1 and reflection_ratio > 0.7
        )

        # Enhanced check 1: Texture uniformity
        # Reflections have low texture CV; real openings have high texture CV
        texture_is_uniform = self._check_texture_uniformity(entries)
        if texture_is_uniform:
            # Strengthen reflection signal
            if static_line_count >= 1:
                is_reflection = True
        else:
            # High texture variation — likely real structure, weaken reflection
            if reflection_ratio < 0.8:
                is_reflection = False

        # Enhanced check 2: Edge displacement analysis
        edge_displacement = self._compute_edge_displacement(entries)
        if edge_displacement > 2.0:
            # Significant edge movement → probably real, not a reflection
            is_reflection = False

        confidence = min(1.0, reflection_ratio * (static_line_count / max(1, len(all_bright_cols))))

        reason = (
            f"static_lines={static_line_count}, "
            f"ratio={reflection_ratio:.2f}, "
            f"edge_disp={edge_displacement:.1f}, "
            f"texture_uniform={texture_is_uniform}"
        )

        return is_reflection, confidence, reason

    def _check_texture_uniformity(self, entries: List[CropEntry]) -> bool:
        """
        Check if bright regions have uniform texture (reflection) vs
        varied texture (real opening with depth).

        Reflections produce spatially uniform brightness with low texture.
        Real openings show depth-varying texture even in bright regions.
        """
        texture_cvs = [entry.texture_cv for entry in entries]
        avg_cv = np.mean(texture_cvs)
        return avg_cv < self.config.texture_uniformity_threshold

    def _compute_edge_displacement(self, entries: List[CropEntry]) -> float:
        """
        Compute average edge displacement across frames.

        Measures how much vertical edge positions shift between consecutive
        frames. Static reflections have near-zero displacement.
        """
        if len(entries) < 2:
            return 0.0

        displacements = []
        for i in range(1, len(entries)):
            prev_edges = entries[i - 1].edge_map
            curr_edges = entries[i].edge_map

            # Find vertical edge column positions (sum along rows)
            prev_profile = np.sum(prev_edges, axis=0).astype(float)
            curr_profile = np.sum(curr_edges, axis=0).astype(float)

            # Compute centroid of edge mass
            if np.sum(prev_profile) > 0 and np.sum(curr_profile) > 0:
                cols = np.arange(len(prev_profile))
                prev_centroid = np.average(cols, weights=prev_profile)
                curr_centroid = np.average(cols, weights=curr_profile)
                displacements.append(abs(curr_centroid - prev_centroid))

        return np.mean(displacements) if displacements else 0.0


# =============================================================================
# GAP EVOLUTION ANALYZER (Enhanced with Multi-Scale + Panel Separation)
# =============================================================================

class GapEvolutionAnalyzer:
    """
    Tracks the evolution of the visible door gap width across frames.

    Uses vertical edge profiles (Sobel x-gradient) to measure gap width
    at each frame, then fits a linear trend to classify the gap behavior
    as WIDENING, NARROWING, STABLE, or NO_GAP.

    Enhanced with:
    - Multi-scale analysis (top/middle/bottom slices)
    - Panel separation validation (dark gap vs bright reflection)
    """

    def __init__(self, config: TemporalReasoningConfig):
        self.config = config

    def analyze(self, entries: List[CropEntry]) -> Tuple[GapTrend, float, str]:
        """
        Analyze gap evolution across temporal window with multi-scale analysis.

        Args:
            entries: List of CropEntry from temporal buffer

        Returns:
            (trend, slope, reason)
        """
        if len(entries) < 3:
            return GapTrend.STABLE, 0.0, "insufficient_frames"

        # Multi-scale gap analysis: measure at top, middle, bottom
        n_slices = self.config.multi_scale_slices
        slice_gap_widths = [[] for _ in range(n_slices)]
        overall_gap_widths = []
        panel_separation_scores = []

        for entry in entries:
            h = entry.crop_gray.shape[0]
            slice_height = max(h // n_slices, 1)

            slice_gaps = []
            for s in range(n_slices):
                y_start = s * slice_height
                y_end = min((s + 1) * slice_height, h)
                slice_crop = entry.crop_gray[y_start:y_end, :]
                gap_w = self._measure_gap_width(slice_crop)
                slice_gap_widths[s].append(gap_w)
                slice_gaps.append(gap_w)

            # Overall gap width as average across slices
            overall_gap = np.mean(slice_gaps)
            overall_gap_widths.append(overall_gap)

            # Panel separation validation
            sep_score = self._validate_panel_separation(entry.crop_gray)
            panel_separation_scores.append(sep_score)

        # Check if there's any detectable gap at all
        max_gap = max(overall_gap_widths) if overall_gap_widths else 0
        if max_gap < 2.0:
            return GapTrend.NO_GAP, 0.0, "no_measurable_gap"

        # Fit linear regression to gap widths over time
        x = np.arange(len(overall_gap_widths), dtype=float)
        gap_arr = np.array(overall_gap_widths, dtype=float)

        if np.var(x) < 1e-8:
            slope = 0.0
        else:
            slope = np.cov(x, gap_arr)[0, 1] / np.var(x)

        # Multi-scale consistency: check if gap trend is consistent across slices
        slice_slopes = []
        for s in range(n_slices):
            s_arr = np.array(slice_gap_widths[s], dtype=float)
            if np.var(x) > 1e-8 and len(s_arr) == len(x):
                s_slope = np.cov(x, s_arr)[0, 1] / np.var(x)
            else:
                s_slope = 0.0
            slice_slopes.append(s_slope)

        # Cross-slice consistency: real openings have consistent gap across slices
        slice_slope_std = np.std(slice_slopes)
        is_consistent_across_slices = slice_slope_std < abs(slope) + 0.5

        # Panel separation: if gap is brighter than panels → reflection, not real opening
        avg_panel_sep = np.mean(panel_separation_scores) if panel_separation_scores else 1.0
        has_true_dark_gap = avg_panel_sep < self.config.panel_separation_darkness_threshold

        # Classify trend
        if slope > self.config.gap_widening_slope and is_consistent_across_slices:
            trend = GapTrend.WIDENING
        elif slope < self.config.gap_narrowing_slope:
            trend = GapTrend.NARROWING
        elif abs(slope) < self.config.gap_stable_tolerance:
            trend = GapTrend.STABLE
        else:
            trend = GapTrend.STABLE  # Borderline → stable

        # If gap is bright (not dark) → probably reflection, override to NO_GAP
        if not has_true_dark_gap and trend != GapTrend.WIDENING:
            trend = GapTrend.NO_GAP

        reason = (
            f"slope={slope:.3f}, max_gap={max_gap:.1f}, "
            f"gap_range=[{min(overall_gap_widths):.1f}, {max(overall_gap_widths):.1f}], "
            f"slice_consistency={slice_slope_std:.2f}, "
            f"panel_sep={avg_panel_sep:.2f}, dark_gap={has_true_dark_gap}"
        )

        return trend, slope, reason

    def _measure_gap_width(self, crop_gray: np.ndarray) -> float:
        """
        Measure door gap width using vertical edge detection.

        Uses Sobel x-gradient to find strong vertical edges (door panel
        boundaries), then measures the distance between the two strongest
        edge peaks as the gap width.
        """
        if crop_gray.size == 0:
            return 0.0

        # Compute Sobel x-gradient (vertical edges)
        sobel_x = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs = np.abs(sobel_x)

        # Average across rows to get column-wise edge profile
        edge_profile = np.mean(sobel_abs, axis=0)

        # Find peaks above threshold
        threshold = self.config.gap_min_edge_strength
        peaks = np.where(edge_profile > threshold)[0]

        if len(peaks) < 2:
            return 0.0

        # Gap width = distance between the two strongest peaks
        peak_strengths = edge_profile[peaks]
        sorted_indices = np.argsort(peak_strengths)[::-1]

        # Take top 2 strongest peaks
        top_peaks = sorted(peaks[sorted_indices[:2]])

        return float(top_peaks[1] - top_peaks[0])

    def _validate_panel_separation(self, crop_gray: np.ndarray) -> float:
        """
        Validate whether the gap between door panels is truly dark (real opening)
        or bright (reflection/shine artifact).

        Returns:
            Ratio of gap brightness to panel brightness.
            < 0.7 → true dark gap (real opening)
            > 0.7 → bright gap (likely reflection)
        """
        if crop_gray.size == 0:
            return 1.0

        w = crop_gray.shape[1]
        if w < 6:
            return 1.0

        # Divide into three vertical zones: left panel, center gap, right panel
        third = max(w // 3, 1)
        left_panel = crop_gray[:, :third]
        center_gap = crop_gray[:, third:2*third]
        right_panel = crop_gray[:, 2*third:]

        panel_brightness = (np.mean(left_panel) + np.mean(right_panel)) / 2.0
        gap_brightness = np.mean(center_gap)

        if panel_brightness < 1e-6:
            return 1.0

        return float(gap_brightness / panel_brightness)


# =============================================================================
# MOTION CONSISTENCY CHECKER (Enhanced with Optical Flow)
# =============================================================================

class MotionConsistencyChecker:
    """
    Validates pixel-level structural motion in door crops.

    Distinguishes physical panel motion (structured, directional pixel
    changes) from lighting fluctuation (uniform, diffuse changes) by
    analyzing the spatial distribution of inter-frame differences.

    Enhanced with:
    - Directional motion analysis via optical flow
    - Horizontal vs. diffuse motion classification
    """

    def __init__(self, config: TemporalReasoningConfig):
        self.config = config

    def check(self, entries: List[CropEntry]) -> Tuple[bool, float, str]:
        """
        Check for structural motion consistency.

        Args:
            entries: List of CropEntry from temporal buffer

        Returns:
            (has_structural_motion, motion_score, reason)
        """
        if len(entries) < 2:
            return False, 0.0, "insufficient_frames"

        structural_motion_frames = 0
        total_diff_frames = 0
        motion_magnitudes = []
        directional_frames = 0

        for i in range(1, len(entries)):
            prev_crop = entries[i - 1].crop_gray.astype(float)
            curr_crop = entries[i].crop_gray.astype(float)

            # Compute absolute difference
            diff = np.abs(curr_crop - prev_crop)
            mean_diff = np.mean(diff)
            motion_magnitudes.append(mean_diff)

            # Only analyze frames with sufficient change
            if mean_diff > self.config.motion_min_diff_threshold:
                total_diff_frames += 1

                # Compute coefficient of variation of the diff image
                std_diff = np.std(diff)
                cv = std_diff / (mean_diff + 1e-6)

                if cv > self.config.motion_structural_cv_threshold:
                    structural_motion_frames += 1

                # Directional motion analysis via optical flow
                is_directional = self._check_directional_motion(
                    entries[i - 1].crop_gray,
                    entries[i].crop_gray
                )
                if is_directional:
                    directional_frames += 1

        n_pairs = len(entries) - 1
        if n_pairs == 0:
            return False, 0.0, "no_pairs"

        motion_fraction = total_diff_frames / n_pairs

        if total_diff_frames > 0:
            structural_ratio = structural_motion_frames / total_diff_frames
            directional_ratio = directional_frames / total_diff_frames
        else:
            structural_ratio = 0.0
            directional_ratio = 0.0

        # Has structural motion if enough frames show directional changes
        has_motion = (
            structural_ratio > self.config.motion_min_moving_frames
            and total_diff_frames >= 2
        )

        # Boost signal if motion is directional (horizontal panel sliding)
        if directional_ratio > 0.5 and total_diff_frames >= 2:
            has_motion = True

        avg_magnitude = np.mean(motion_magnitudes) if motion_magnitudes else 0.0

        reason = (
            f"motion_frac={motion_fraction:.2f}, "
            f"structural_ratio={structural_ratio:.2f}, "
            f"directional_ratio={directional_ratio:.2f}, "
            f"avg_mag={avg_magnitude:.1f}"
        )

        return has_motion, structural_ratio, reason

    def _check_directional_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> bool:
        """
        Check if motion between frames is directionally coherent (horizontal
        panel sliding) rather than diffuse (lighting fluctuation).

        Uses Farneback optical flow on the small normalized crops.
        """
        try:
            prev_u8 = prev_gray.astype(np.uint8)
            curr_u8 = curr_gray.astype(np.uint8)

            flow = cv2.calcOpticalFlowFarneback(
                prev_u8, curr_u8,
                None,
                pyr_scale=0.5, levels=2, winsize=9,
                iterations=2, poly_n=5, poly_sigma=1.1,
                flags=0
            )

            # Compute flow magnitude and angle
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            significant = mag > self.config.flow_min_magnitude

            if np.sum(significant) < 10:
                return False

            # Check if flow is predominantly horizontal
            horizontal_flow = np.abs(flow[..., 0])
            vertical_flow = np.abs(flow[..., 1])

            total_horiz = np.sum(horizontal_flow[significant])
            total_vert = np.sum(vertical_flow[significant])
            total_flow = total_horiz + total_vert

            if total_flow < 1e-6:
                return False

            horiz_ratio = total_horiz / total_flow
            return horiz_ratio > self.config.flow_horizontal_ratio

        except Exception:
            return False


# =============================================================================
# TEMPORAL FEATURE AGGREGATOR (New)
# =============================================================================

class TemporalFeatureAggregator:
    """
    Aggregates per-crop feature vectors across the temporal window using
    exponentially-weighted averaging.

    Stable features over time → static scene (reflection, no real motion)
    Changing features over time → dynamic scene (real door opening/closing)

    This provides a holistic "temporal consistency score" that complements
    the individual analyzers (reflection, gap, motion).
    """

    def __init__(self, config: TemporalReasoningConfig):
        self.config = config

    def aggregate(self, entries: List[CropEntry]) -> Tuple[float, str]:
        """
        Compute temporal feature consistency score.

        Args:
            entries: List of CropEntry from temporal buffer

        Returns:
            (consistency_score, reason)
            - Low score → features are changing (real motion)
            - High score → features are stable (static/reflection)
        """
        if len(entries) < 3:
            return 0.5, "insufficient_frames"

        feature_vectors = [entry.feature_vector for entry in entries]
        n = len(feature_vectors)

        # Compute exponentially-weighted average feature vector
        # Recent frames get higher weight
        decay = self.config.feature_decay_rate
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= np.sum(weights)

        # Weighted mean feature vector
        stacked = np.stack(feature_vectors, axis=0)  # (N, 32)
        weighted_mean = np.average(stacked, axis=0, weights=weights)

        # Compute weighted variance (temporal feature stability)
        diffs = stacked - weighted_mean[np.newaxis, :]
        weighted_var = np.average(diffs**2, axis=0, weights=weights)
        feature_std = np.sqrt(np.mean(weighted_var))

        # Compute pairwise feature drift (how much features change frame-to-frame)
        drifts = []
        for i in range(1, n):
            drift = np.linalg.norm(feature_vectors[i] - feature_vectors[i-1])
            drifts.append(drift)

        avg_drift = np.mean(drifts) if drifts else 0.0

        # Consistency score: higher = more stable/static
        # Normalize by feature dimension for interpretability
        dim = self.config.feature_vector_dim
        consistency = 1.0 - min(1.0, feature_std / (0.3 * np.sqrt(dim)))

        reason = (
            f"feature_std={feature_std:.4f}, "
            f"avg_drift={avg_drift:.4f}, "
            f"consistency={consistency:.3f}"
        )

        return float(consistency), reason


# =============================================================================
# TEMPORAL DOOR VERIFIER (Top-Level, Enhanced)
# =============================================================================

class TemporalDoorVerifier:
    """
    Top-level temporal verification for door OPEN predictions.

    Aggregates ReflectionDetector, GapEvolutionAnalyzer,
    MotionConsistencyChecker, and TemporalFeatureAggregator to decide
    whether an OPEN prediction should be confirmed or overridden.

    Decision logic:
    - OVERRIDE to CLOSED if: reflection detected AND no gap evolution
      AND no structural motion AND features are temporally stable
    - CONFIRM OPEN if: consistent gap widening OR sustained structural
      motion detected OR features are temporally changing
    - PASS THROUGH if: insufficient evidence either way (keep original)

    Confidence adjustment uses weighted temporal smoothing rather than
    a fixed reduction.

    Usage:
        verifier = TemporalDoorVerifier()

        # In DoorTrack.get_decision():
        if decision == OPEN and crop_buffer.is_ready():
            verified, reason, adj_conf = verifier.verify_open_prediction(
                track_id, crop_buffer, original_confidence
            )
    """

    def __init__(self, config: TemporalReasoningConfig = None):
        self.config = config or TemporalReasoningConfig()
        self.reflection_detector = ReflectionDetector(self.config)
        self.gap_analyzer = GapEvolutionAnalyzer(self.config)
        self.motion_checker = MotionConsistencyChecker(self.config)
        self.feature_aggregator = TemporalFeatureAggregator(self.config)

        # Track override history for logging
        self.override_log: Dict[int, List[Dict]] = {}

    def verify_open_prediction(
        self,
        track_id: int,
        crop_buffer: TemporalCropBuffer,
        original_confidence: float
    ) -> Tuple[bool, str, float]:
        """
        Verify whether an OPEN prediction should be confirmed or overridden.

        Args:
            track_id: Door track ID
            crop_buffer: Rolling buffer of door crops
            original_confidence: Original confidence from FSM/majority voting

        Returns:
            (verified: bool, reason: str, adjusted_confidence: float)
            - verified=True: OPEN confirmed, keep the state
            - verified=False: OPEN overridden → should become CLOSED
        """
        if not crop_buffer.is_ready():
            return True, "buffer_not_ready", original_confidence

        entries = crop_buffer.get_entries()

        # Run all four analyzers
        is_reflection, refl_conf, refl_reason = self.reflection_detector.detect(entries)
        gap_trend, gap_slope, gap_reason = self.gap_analyzer.analyze(entries)
        has_motion, motion_score, motion_reason = self.motion_checker.check(entries)
        feature_consistency, feat_reason = self.feature_aggregator.aggregate(entries)

        # Decision logic with enhanced feature aggregation
        should_override = False
        should_confirm = False
        reasons = []
        override_strength = 0.0  # 0-1 scale for weighted confidence reduction

        # ------ Override indicators ------

        if is_reflection:
            reasons.append(f"reflection_detected({refl_reason})")
            override_strength += 0.3

        if gap_trend in (GapTrend.NO_GAP, GapTrend.STABLE):
            reasons.append(f"gap_{gap_trend.value}({gap_reason})")
            override_strength += 0.25

        if not has_motion:
            reasons.append(f"no_structural_motion({motion_reason})")
            override_strength += 0.2

        if feature_consistency > (1.0 - self.config.feature_consistency_threshold):
            reasons.append(f"features_stable({feat_reason})")
            override_strength += 0.25

        # Strong override: multiple indicators
        if override_strength >= 0.7:
            should_override = True

        # Moderate override: reflection + no gap
        elif is_reflection and gap_trend in (GapTrend.NO_GAP,):
            should_override = True
            override_strength = max(override_strength, 0.6)

        # Moderate override: no gap AND no motion (even without reflection)
        elif gap_trend == GapTrend.NO_GAP and not has_motion:
            should_override = True
            override_strength = max(override_strength, 0.5)

        # ------ Confirmation indicators ------

        if gap_trend == GapTrend.WIDENING:
            should_confirm = True
            reasons.append(f"gap_widening({gap_reason})")

        if has_motion and gap_trend != GapTrend.NO_GAP:
            should_confirm = True
            reasons.append(f"structural_motion({motion_reason})")

        if feature_consistency < self.config.feature_consistency_threshold:
            should_confirm = True
            reasons.append(f"features_changing({feat_reason})")

        # Confirmation overrides override
        if should_confirm and should_override:
            should_override = False

        # Compute adjusted confidence with weighted temporal smoothing
        if should_override:
            # Scale confidence reduction by override strength
            max_reduction = self.config.override_confidence_reduction
            reduction = max_reduction * min(1.0, override_strength)
            adjusted_conf = max(0.1, original_confidence - reduction)
            reason_str = f"OVERRIDDEN: {'; '.join(reasons)}"

            # Log override
            if track_id not in self.override_log:
                self.override_log[track_id] = []
            self.override_log[track_id].append({
                'frame_range': (entries[0].frame_idx, entries[-1].frame_idx),
                'reason': reason_str,
                'original_confidence': original_confidence,
                'adjusted_confidence': adjusted_conf,
                'override_strength': override_strength
            })

            return False, reason_str, adjusted_conf

        elif should_confirm:
            adjusted_conf = min(1.0, original_confidence + self.config.confirmation_confidence_boost)
            reason_str = f"CONFIRMED: {'; '.join(reasons)}"
            return True, reason_str, adjusted_conf

        else:
            # Ambiguous → pass through with original decision
            reason_str = f"PASS_THROUGH: {'; '.join(reasons) if reasons else 'no_clear_signal'}"
            return True, reason_str, original_confidence

    def get_override_summary(self) -> Dict[int, int]:
        """Get summary of overrides per track for reporting."""
        return {
            track_id: len(overrides)
            for track_id, overrides in self.override_log.items()
        }

    def reset(self):
        """Reset state for new video."""
        self.override_log.clear()
