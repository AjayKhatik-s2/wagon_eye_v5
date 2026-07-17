"""
Snapshot Selector Module - Select best representative frames for each door.

Scores frames based on completeness, visibility, sharpness, 
center alignment, and detection confidence.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SnapshotCandidate:
    """Single frame candidate for snapshot selection."""
    frame_idx: int
    frame: np.ndarray
    bbox: np.ndarray
    class_name: str
    confidence: float
    
    # Computed scores
    completeness_score: float = 0.0
    size_score: float = 0.0
    sharpness_score: float = 0.0
    center_score: float = 0.0
    confidence_score: float = 0.0
    total_score: float = 0.0


class SnapshotSelector:
    """
    Selects best representative snapshot for each tracked door.
    
    Scoring weights:
    - Completeness: 2.0 (heavily penalize clipped boxes)
    - Size: 1.5 (prefer larger, more visible doors)
    - Sharpness: 1.0 (prefer sharp, focused images)
    - Center: 0.5 (slight preference for center-frame)
    - Confidence: 1.5 (trust high-confidence detections)
    """
    
    def __init__(
        self,
        completeness_weight: float = 2.0,
        size_weight: float = 1.5,
        sharpness_weight: float = 1.0,
        center_weight: float = 1.5,
        confidence_weight: float = 1.5,
        border_tolerance: int = 10,
        balance_weight: float = 1.0
    ):
        self.completeness_weight = completeness_weight
        self.size_weight = size_weight
        self.sharpness_weight = sharpness_weight
        self.center_weight = center_weight
        self.confidence_weight = confidence_weight
        self.border_tolerance = border_tolerance
        self.balance_weight = balance_weight
    
    def compute_completeness(
        self, 
        bbox: np.ndarray, 
        frame_width: int, 
        frame_height: int
    ) -> float:
        """
        Compute how complete the bounding box is (not clipped).
        
        Returns 1.0 if fully inside, 0.0 if touching/crossing edges.
        """
        x1, y1, x2, y2 = bbox
        tol = self.border_tolerance
        
        # Check each edge
        edge_penalties = 0
        if x1 <= tol:
            edge_penalties += 1
        if y1 <= tol:
            edge_penalties += 1
        if x2 >= frame_width - tol:
            edge_penalties += 1
        if y2 >= frame_height - tol:
            edge_penalties += 1
        
        # 4 edges total
        return 1.0 - (edge_penalties / 4.0)
    
    def compute_size_score(
        self, 
        bbox: np.ndarray, 
        frame_width: int, 
        frame_height: int
    ) -> float:
        """
        Compute size/visibility score.
        
        Normalized to ~30% of frame being optimal.
        """
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        
        # Optimal area is ~10% of frame
        optimal_ratio = 0.10
        actual_ratio = area / frame_area
        
        # Score decreases if too small or too large
        if actual_ratio <= optimal_ratio:
            return actual_ratio / optimal_ratio
        else:
            # Slight penalty for taking up too much frame
            return max(0, 1.0 - (actual_ratio - optimal_ratio))
    
    def compute_sharpness(self, frame: np.ndarray, bbox: np.ndarray) -> float:
        """
        Compute sharpness score using Laplacian variance.
        
        Returns normalized score in [0, 1].
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize (typical range 0-500+)
        return min(variance / 200.0, 1.0)
    
    def compute_center_score(
        self, 
        bbox: np.ndarray, 
        frame_width: int, 
        frame_height: int
    ) -> float:
        """
        Compute center alignment score.
        
        Higher score for boxes closer to frame center.
        """
        x1, y1, x2, y2 = bbox
        
        # Box center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Frame center
        fcx = frame_width / 2
        fcy = frame_height / 2
        
        # Distance from center
        dist = np.sqrt((cx - fcx)**2 + (cy - fcy)**2)
        max_dist = np.sqrt(fcx**2 + fcy**2)
        
        return 1.0 - (dist / max_dist)
    
    def compute_balance_score(
        self,
        bbox: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Penalize boxes where the door is partially cut off at edges.
        
        Checks margin on each side relative to box size.
        Returns 1.0 for well-balanced, 0.0 for edge-hugging boxes.
        """
        x1, y1, x2, y2 = bbox
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        
        # How much margin exists on each side (relative to box size)
        left_margin = x1 / box_w
        right_margin = (frame_width - x2) / box_w
        top_margin = y1 / box_h
        bottom_margin = (frame_height - y2) / box_h
        
        # Minimum margin ratio — need at least 0.3x box size as margin
        min_h_margin = min(left_margin, right_margin)
        min_v_margin = min(top_margin, bottom_margin)
        
        h_score = min(min_h_margin / 0.3, 1.0)
        v_score = min(min_v_margin / 0.3, 1.0)
        
        return (h_score + v_score) / 2.0
    
    def score_candidate(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        confidence: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total score for a snapshot candidate.
        
        Returns (total_score, score_breakdown).
        """
        h, w = frame.shape[:2]
        
        # Compute individual scores
        completeness = self.compute_completeness(bbox, w, h)
        size = self.compute_size_score(bbox, w, h)
        sharpness = self.compute_sharpness(frame, bbox)
        center = self.compute_center_score(bbox, w, h)
        balance = self.compute_balance_score(bbox, w, h)
        
        # Weighted sum
        total = (
            completeness * self.completeness_weight +
            size * self.size_weight +
            sharpness * self.sharpness_weight +
            center * self.center_weight +
            confidence * self.confidence_weight +
            balance * self.balance_weight
        )
        
        breakdown = {
            'completeness': completeness,
            'size': size,
            'sharpness': sharpness,
            'center': center,
            'confidence': confidence,
            'balance': balance,
            'total': total
        }
        
        return total, breakdown
    
    def select_best(
        self,
        candidates: List[SnapshotCandidate]
    ) -> Optional[SnapshotCandidate]:
        """
        Select best snapshot from list of candidates.
        
        Returns the candidate with highest total score.
        """
        if not candidates:
            return None
        
        best_candidate = None
        best_score = -float('inf')
        
        for candidate in candidates:
            score, breakdown = self.score_candidate(
                candidate.frame,
                candidate.bbox,
                candidate.confidence
            )
            
            candidate.total_score = score
            candidate.completeness_score = breakdown['completeness']
            candidate.size_score = breakdown['size']
            candidate.sharpness_score = breakdown['sharpness']
            candidate.center_score = breakdown['center']
            candidate.confidence_score = breakdown['confidence']
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate


def annotate_snapshot(
    frame: np.ndarray,
    bbox: np.ndarray,
    door_id: int,
    door_state: str,
    state_colors: Dict[str, Tuple[int, int, int]],
    confidence: float = 0.0
) -> np.ndarray:
    """
    Annotate snapshot with door ID, state, and confidence.
    
    Args:
        frame: Original frame
        bbox: [x1, y1, x2, y2] bounding box
        door_id: Door number (sequential)
        door_state: Final determined state
        state_colors: Dict mapping state to BGR color
        confidence: Detection confidence score
        
    Returns:
        Annotated frame copy
    """
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Get color for state
    color = state_colors.get(door_state.lower(), (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    
    # Prepare label with confidence
    state_display = door_state.upper().replace('_', ' ')
    label = f"Door #{door_id}: {state_display} ({confidence:.0%})"
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw label background above bounding box
    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        annotated,
        (x1, label_y - text_h - 10),
        (x1 + text_w + 10, label_y + 5),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        annotated,
        label,
        (x1 + 5, label_y - 5),
        font,
        font_scale,
        (0, 0, 0),  # Black text
        thickness
    )
    
    return annotated


def refined_crop_around_detection(
    frame: np.ndarray,
    bbox: np.ndarray,
    width_expand: float = 0.40,
    height_expand: float = 0.40,
    min_frame_ratio_w: float = 0.60,
    min_frame_ratio_h: float = 0.60,
    edge_threshold: float = 0.05,
    small_area_threshold: float = 0.03,
    fallback_roi_ratio: float = 0.70
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refined bounding box crop that ALWAYS centers the door with full context.
    
    Strategy:
    1. Compute bbox center, expand proportionally (40% each side)
    2. Enforce minimum crop = 60% of frame dimensions
    3. Shift-to-fit boundary clipping (no black borders)
    4. Fallback ROI template for edge/small detections
    
    Args:
        frame: Raw frame (NOT annotated — annotation happens after crop)
        bbox: [x1, y1, x2, y2] bounding box of the detection
        width_expand: Fractional expansion on each side horizontally (0.40 = 40%)
        height_expand: Fractional expansion on each side vertically (0.40 = 40%)
        min_frame_ratio_w: Minimum crop width as fraction of frame width
        min_frame_ratio_h: Minimum crop height as fraction of frame height
        edge_threshold: Fraction of frame dimension — bbox within this margin
                        of any edge is considered "at edge"
        small_area_threshold: Bbox area / frame area below this triggers fallback
        fallback_roi_ratio: Fraction of frame to use for fallback center ROI
        
    Returns:
        (cropped_frame, adjusted_bbox) — the cropped region and the bbox
        coordinates remapped to the cropped frame coordinate system
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # --- Bbox metrics ---
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)
    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0
    bbox_area = bbox_w * bbox_h
    frame_area = w * h
    
    # --- Check if fallback ROI is needed ---
    edge_margin_x = w * edge_threshold
    edge_margin_y = h * edge_threshold
    is_at_edge = (
        x1 < edge_margin_x or
        y1 < edge_margin_y or
        x2 > w - edge_margin_x or
        y2 > h - edge_margin_y
    )
    is_small = (bbox_area / frame_area) < small_area_threshold
    
    use_fallback = is_at_edge and is_small
    
    if use_fallback:
        # --- Fallback: use center ROI template ---
        roi_w = int(w * fallback_roi_ratio)
        roi_h = int(h * fallback_roi_ratio)
        crop_x1 = (w - roi_w) // 2
        crop_y1 = (h - roi_h) // 2
        crop_x2 = crop_x1 + roi_w
        crop_y2 = crop_y1 + roi_h
    else:
        # --- Step 1: Proportional expansion around bbox center ---
        expand_w = int(bbox_w * width_expand)
        expand_h = int(bbox_h * height_expand)
        
        crop_x1 = int(bbox_cx - bbox_w / 2 - expand_w)
        crop_y1 = int(bbox_cy - bbox_h / 2 - expand_h)
        crop_x2 = int(bbox_cx + bbox_w / 2 + expand_w)
        crop_y2 = int(bbox_cy + bbox_h / 2 + expand_h)
        
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        # --- Step 2: Enforce minimum crop dimensions ---
        min_w = int(w * min_frame_ratio_w)
        min_h = int(h * min_frame_ratio_h)
        
        if crop_w < min_w:
            deficit = min_w - crop_w
            crop_x1 -= deficit // 2
            crop_x2 += deficit - deficit // 2
        
        if crop_h < min_h:
            deficit = min_h - crop_h
            crop_y1 -= deficit // 2
            crop_y2 += deficit - deficit // 2
    
    # --- Step 3: Shift-to-fit boundary clipping ---
    # Instead of black borders, shift the crop window inward
    crop_w_final = crop_x2 - crop_x1
    crop_h_final = crop_y2 - crop_y1
    
    # Horizontal shift
    if crop_x1 < 0:
        crop_x2 -= crop_x1  # shift right
        crop_x1 = 0
    if crop_x2 > w:
        crop_x1 -= (crop_x2 - w)  # shift left
        crop_x2 = w
    # Final clamp (if crop is larger than frame)
    crop_x1 = max(0, crop_x1)
    crop_x2 = min(w, crop_x2)
    
    # Vertical shift
    if crop_y1 < 0:
        crop_y2 -= crop_y1  # shift down
        crop_y1 = 0
    if crop_y2 > h:
        crop_y1 -= (crop_y2 - h)  # shift up
        crop_y2 = h
    # Final clamp
    crop_y1 = max(0, crop_y1)
    crop_y2 = min(h, crop_y2)
    
    # --- Safety: if crop region is too small, return full frame ---
    if crop_x2 - crop_x1 < 50 or crop_y2 - crop_y1 < 50:
        adjusted_bbox = np.array([x1, y1, x2, y2])
        return frame.copy(), adjusted_bbox
    
    # --- Step 4: Remap bbox to cropped coordinate system ---
    adj_x1 = x1 - crop_x1
    adj_y1 = y1 - crop_y1
    adj_x2 = x2 - crop_x1
    adj_y2 = y2 - crop_y1
    adjusted_bbox = np.array([adj_x1, adj_y1, adj_x2, adj_y2])
    
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return cropped, adjusted_bbox


# Backward-compatible wrapper (old interface)
def crop_around_detection(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding_factor: float = 1.5,
    min_width: int = 640,
    min_height: int = 480,
    label_extra_top: int = 60
) -> np.ndarray:
    """
    Legacy wrapper — calls refined_crop_around_detection internally.
    
    Kept for backward compatibility. New code should use
    refined_crop_around_detection() directly.
    """
    cropped, _ = refined_crop_around_detection(
        frame, bbox,
        width_expand=0.40,
        height_expand=0.40,
        min_frame_ratio_w=0.60,
        min_frame_ratio_h=0.60
    )
    return cropped

