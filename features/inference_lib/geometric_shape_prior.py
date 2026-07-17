"""
Geometric Shape Prior Module - Validates door detections using geometric constraints.

Implements:
- Aspect ratio validation (doors have typical height/width ratios)
- Vertical edge dominance (doors have strong vertical edges)
- Border completeness (doors have visible edges at boundaries)

These checks reject false detections from reflections, glare patches,
and other non-door objects before they enter the tracking pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GeometricPriorConfig:
    """Configuration for geometric shape prior validation."""
    
    # Aspect ratio constraints
    # Train doors are typically taller than wide (aspect ratio = width/height)
    min_aspect_ratio: float = 0.3    # Narrowest acceptable door
    max_aspect_ratio: float = 1.2    # Widest acceptable door (some doors are wide)
    preferred_aspect_ratio: float = 0.5  # Typical train door aspect ratio
    aspect_tolerance: float = 0.3    # Tolerance around preferred
    
    # Vertical edge dominance
    # Doors should have strong vertical edges (frame, panels)
    min_vertical_edge_ratio: float = 1.2   # Vertical edges should exceed horizontal
    edge_canny_low: int = 50
    edge_canny_high: int = 150
    min_edge_density: float = 0.02   # Minimum edge pixel ratio
    
    # Border completeness
    # Doors should have visible edges at their boundaries
    border_sample_width: int = 5     # Pixels from edge to sample
    min_border_edge_ratio: float = 0.08  # Minimum ratio of border with edges
    min_sides_with_edges: int = 2    # Minimum sides with visible edges
    
    # Overall thresholds
    min_structure_score: float = 0.4     # Minimum combined score to accept
    require_aspect_ratio: bool = True
    require_vertical_edges: bool = True
    require_border_completeness: bool = True


class GeometricShapePrior:
    """
    Validates door detections using geometric shape priors.
    
    A valid train door detection should exhibit:
    1. Appropriate aspect ratio (taller than wide, within bounds)
    2. Dominant vertical edges (door frame, panels)
    3. Complete borders (visible edges on at least 2 sides)
    
    False positives from reflections, glare, and flat surfaces
    typically fail these geometric constraints.
    """
    
    def __init__(self, config: Optional[GeometricPriorConfig] = None):
        self.config = config or GeometricPriorConfig()
    
    def check_aspect_ratio(self, bbox: np.ndarray) -> Tuple[bool, float]:
        """
        Validate aspect ratio of bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (is_valid, aspect_score)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height <= 0 or width <= 0:
            return False, 0.0
        
        aspect_ratio = width / height
        
        # Check if within acceptable range
        if aspect_ratio < self.config.min_aspect_ratio:
            return False, 0.0
        if aspect_ratio > self.config.max_aspect_ratio:
            return False, 0.0
        
        # Score based on distance from preferred ratio
        distance_from_preferred = abs(aspect_ratio - self.config.preferred_aspect_ratio)
        aspect_score = max(0, 1.0 - distance_from_preferred / self.config.aspect_tolerance)
        
        # Clamp score to [0, 1]
        aspect_score = min(1.0, aspect_score)
        
        return True, aspect_score
    
    def check_vertical_edge_dominance(
        self, 
        frame: np.ndarray, 
        bbox: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if vertical edges dominate over horizontal edges.
        
        Train doors have strong vertical edges from the door frame,
        panels, and window frames. Reflections and glare typically
        don't exhibit this vertical structure.
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (is_valid, edge_dominance_score)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return False, 0.0
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Compute Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Vertical edges
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Horizontal edges
        
        # Sum of absolute gradients
        vertical_energy = np.sum(np.abs(sobel_x))
        horizontal_energy = np.sum(np.abs(sobel_y))
        
        if horizontal_energy < 1e-6:
            # Avoid division by zero
            edge_ratio = 10.0 if vertical_energy > 0 else 0.0
        else:
            edge_ratio = vertical_energy / horizontal_energy
        
        # Check if vertical edges dominate
        is_valid = edge_ratio >= self.config.min_vertical_edge_ratio
        
        # Score: how much vertical edges exceed the minimum ratio
        if edge_ratio < self.config.min_vertical_edge_ratio:
            score = edge_ratio / self.config.min_vertical_edge_ratio
        else:
            # Score increases with edge dominance, capped at 1.0
            score = min(1.0, edge_ratio / 2.0)  # Score 1.0 at ratio=2.0+
        
        # Also check minimum edge density (reject flat/uniform regions)
        edges = cv2.Canny(
            gray, 
            self.config.edge_canny_low, 
            self.config.edge_canny_high
        )
        edge_density = np.count_nonzero(edges) / edges.size
        
        if edge_density < self.config.min_edge_density:
            return False, score * 0.5  # Penalize low-texture regions
        
        return is_valid, score
    
    def check_border_completeness(
        self, 
        frame: np.ndarray, 
        bbox: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if the detection has visible edges at its borders.
        
        Real doors have visible frame edges at their boundaries.
        Reflections and glare patches often lack definite borders.
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (is_valid, border_score)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        
        # Clamp to frame bounds
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w, x2), min(h, y2)
        
        if x2_c <= x1_c or y2_c <= y1_c:
            return False, 0.0
        
        crop = frame[y1_c:y2_c, x1_c:x2_c]
        
        if crop.size == 0:
            return False, 0.0
        
        # Convert to grayscale and detect edges
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        edges = cv2.Canny(
            gray, 
            self.config.edge_canny_low, 
            self.config.edge_canny_high
        )
        
        crop_h, crop_w = edges.shape
        border_w = min(self.config.border_sample_width, crop_w // 4, crop_h // 4)
        
        if border_w < 1:
            return False, 0.0
        
        # Sample borders
        border_scores = []
        
        # Top border
        if y1 > 0:  # Not clipped at top
            top_border = edges[:border_w, :]
            top_ratio = np.count_nonzero(top_border) / top_border.size if top_border.size > 0 else 0
            border_scores.append(('top', top_ratio))
        
        # Bottom border
        if y2 < h:  # Not clipped at bottom
            bottom_border = edges[-border_w:, :]
            bottom_ratio = np.count_nonzero(bottom_border) / bottom_border.size if bottom_border.size > 0 else 0
            border_scores.append(('bottom', bottom_ratio))
        
        # Left border
        if x1 > 0:  # Not clipped at left
            left_border = edges[:, :border_w]
            left_ratio = np.count_nonzero(left_border) / left_border.size if left_border.size > 0 else 0
            border_scores.append(('left', left_ratio))
        
        # Right border
        if x2 < w:  # Not clipped at right
            right_border = edges[:, -border_w:]
            right_ratio = np.count_nonzero(right_border) / right_border.size if right_border.size > 0 else 0
            border_scores.append(('right', right_ratio))
        
        if not border_scores:
            return False, 0.0
        
        # Count sides with sufficient edges
        sides_with_edges = sum(
            1 for _, ratio in border_scores 
            if ratio >= self.config.min_border_edge_ratio
        )
        
        is_valid = sides_with_edges >= self.config.min_sides_with_edges
        
        # Average border score
        avg_score = np.mean([ratio for _, ratio in border_scores])
        # Normalize: min_border_edge_ratio should give ~0.5 score
        normalized_score = min(1.0, avg_score / (self.config.min_border_edge_ratio * 2))
        
        # Bonus for having multiple sides with edges
        if sides_with_edges >= 3:
            normalized_score = min(1.0, normalized_score * 1.2)
        
        return is_valid, normalized_score
    
    def validate_detection(
        self, 
        frame: np.ndarray, 
        bbox: np.ndarray
    ) -> Tuple[bool, float, dict]:
        """
        Validate a door detection using all geometric priors.
        
        Args:
            frame: Full BGR frame (raw, unprocessed)
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (is_valid, overall_score, detail_scores)
        """
        scores = {}
        passed_checks = 0
        total_checks = 0
        
        # Check 1: Aspect ratio
        if self.config.require_aspect_ratio:
            total_checks += 1
            valid_ar, ar_score = self.check_aspect_ratio(bbox)
            scores['aspect_ratio'] = {
                'valid': valid_ar,
                'score': ar_score
            }
            if valid_ar:
                passed_checks += 1
        
        # Check 2: Vertical edge dominance
        if self.config.require_vertical_edges:
            total_checks += 1
            valid_ve, ve_score = self.check_vertical_edge_dominance(frame, bbox)
            scores['vertical_edges'] = {
                'valid': valid_ve,
                'score': ve_score
            }
            if valid_ve:
                passed_checks += 1
        
        # Check 3: Border completeness
        if self.config.require_border_completeness:
            total_checks += 1
            valid_bc, bc_score = self.check_border_completeness(frame, bbox)
            scores['border_completeness'] = {
                'valid': valid_bc,
                'score': bc_score
            }
            if valid_bc:
                passed_checks += 1
        
        # Compute overall score (weighted average)
        individual_scores = [s['score'] for s in scores.values()]
        overall_score = np.mean(individual_scores) if individual_scores else 0.0
        
        # Detection is valid if:
        # 1. Passes all required checks, OR
        # 2. Overall score exceeds threshold
        is_valid = (
            passed_checks == total_checks or 
            overall_score >= self.config.min_structure_score
        )
        
        return is_valid, overall_score, scores
    
    def filter_detections(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Filter detections using geometric shape priors.
        
        Args:
            frame: Full BGR frame (raw, unprocessed)
            boxes: Array of [x1, y1, x2, y2] boxes
            confidences: Array of confidence scores
            class_ids: Array of class IDs
            
        Returns:
            (filtered_boxes, filtered_confidences, filtered_class_ids, valid_indices)
        """
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([]), []
        
        valid_indices = []
        
        for i, bbox in enumerate(boxes):
            is_valid, score, _ = self.validate_detection(frame, bbox)
            if is_valid:
                valid_indices.append(i)
        
        if not valid_indices:
            return np.array([]), np.array([]), np.array([]), []
        
        return (
            boxes[valid_indices],
            confidences[valid_indices],
            class_ids[valid_indices],
            valid_indices
        )


def get_default_prior() -> GeometricShapePrior:
    """Get geometric shape prior with default configuration."""
    return GeometricShapePrior()


def get_strict_prior() -> GeometricShapePrior:
    """Get geometric shape prior with stricter thresholds."""
    config = GeometricPriorConfig(
        min_vertical_edge_ratio=1.5,
        min_border_edge_ratio=0.12,
        min_sides_with_edges=3,
        min_structure_score=0.5
    )
    return GeometricShapePrior(config)
