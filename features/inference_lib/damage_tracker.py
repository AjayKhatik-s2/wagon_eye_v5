"""
Damage Tracker Module - Kalman Filter + Hungarian Algorithm for Top-View Damage Tracking

Adapted from the right/left camera door_tracker.py to track floor damage detections
across frames. Uses the same proven approach:
- Kalman Filter with constant-velocity model
- Hungarian algorithm (scipy.optimize.linear_sum_assignment) for optimal matching
- Track lifecycle: TENTATIVE → CONFIRMED → LOST → DELETED
- Per-track memory for class voting and confidence aggregation
- Best snapshot selection per track

Author: Damage Inspection System
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DamageTrackerConfig:
    """Configuration for the damage tracking system."""
    # Confidence threshold
    confidence_threshold: float = 0.50
    
    # Kalman Filter settings
    process_noise_moving: float = 0.5
    process_noise_stopped: float = 0.01
    measurement_noise: float = 1.0
    velocity_threshold: float = 2.0
    
    # Association settings
    max_center_distance: float = 200.0   # Max center distance for matching (generous for moving train)
    iou_weight: float = 0.0             # Disable IoU cost (top-view: damage moves too much between frames)
    distance_weight: float = 1.0        # Use pure distance matching
    min_iou_threshold: float = 0.0      # No IoU gate (moving train = near-zero IoU between frames)
    
    # Track management
    max_age: int = 30                   # Max frames to keep lost track
    n_init: int = 2                     # Hits before confirmed (lower for fast-moving damages)
    min_hits_for_decision: int = 2      # Min hits for reliable output
    
    # Prediction buffer
    prediction_buffer_size: int = 30
    
    # Exit detection
    exit_margin: int = 20


# =============================================================================
# KALMAN FILTER
# =============================================================================

class KalmanFilter:
    """
    Kalman Filter with constant-velocity state model.
    
    State: [cx, cy, w, h, vx, vy]
    Measurement: [cx, cy, w, h]
    """
    
    def __init__(self, config: DamageTrackerConfig):
        self.config = config
        self.dt = 1.0
        self.dim_x = 6
        self.dim_z = 4
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0],
            [0, 1, 0, 0, 0, self.dt],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=np.float64)
        
        # Measurement noise
        self.R = np.eye(self.dim_z) * config.measurement_noise
    
    def get_process_noise(self, velocity: float) -> np.ndarray:
        """Get adaptive process noise based on velocity."""
        if abs(velocity) < self.config.velocity_threshold:
            q = self.config.process_noise_stopped
        else:
            q = self.config.process_noise_moving
        
        Q = np.diag([q, q, q * 0.1, q * 0.1, q * 2, q * 2])
        return Q
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize track state from first measurement."""
        mean = np.zeros(self.dim_x)
        mean[:4] = measurement
        mean[4:] = 0
        
        covariance = np.diag([
            10, 10, 10, 10,    # Position/size uncertainty
            100, 100,           # Velocity uncertainty (unknown)
        ]).astype(np.float64)
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        velocity = np.sqrt(mean[4]**2 + mean[5]**2)
        Q = self.get_process_noise(velocity)
        
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        
        return mean, covariance
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with new measurement."""
        y = measurement - self.H @ mean
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        mean = mean + K @ y
        covariance = (np.eye(self.dim_x) - K @ self.H) @ covariance
        
        return mean, covariance


# =============================================================================
# TRACK MEMORY
# =============================================================================

@dataclass
class PredictionEntry:
    """Single prediction entry in track buffer."""
    frame_idx: int
    class_name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


class TrackMemory:
    """
    Per-track memory buffer for temporal aggregation.
    Stores recent class predictions and computes majority voting.
    """
    
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.predictions: deque = deque(maxlen=buffer_size)
        self.class_counts: Dict[str, int] = {}
        self.class_confidences: Dict[str, List[float]] = {}
        self.total_frames = 0
    
    def add_prediction(self, frame_idx: int, class_name: str, confidence: float):
        """Add a new prediction to the buffer."""
        entry = PredictionEntry(
            frame_idx=frame_idx,
            class_name=class_name.lower(),
            confidence=confidence
        )
        self.predictions.append(entry)
        
        cls = class_name.lower()
        self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
        if cls not in self.class_confidences:
            self.class_confidences[cls] = []
        self.class_confidences[cls].append(confidence)
        self.total_frames += 1
    
    def get_majority_class(self) -> Tuple[str, float]:
        """Get majority class with weighted voting."""
        if not self.predictions:
            return "unknown", 0.0
        
        weighted_counts: Dict[str, float] = {}
        weighted_confs: Dict[str, float] = {}
        total_weights: Dict[str, float] = {}
        
        for entry in self.predictions:
            cls = entry.class_name
            weight = entry.confidence
            
            weighted_counts[cls] = weighted_counts.get(cls, 0.0) + weight
            weighted_confs[cls] = weighted_confs.get(cls, 0.0) + entry.confidence * weight
            total_weights[cls] = total_weights.get(cls, 0.0) + weight
        
        if not weighted_counts:
            return "unknown", 0.0
        
        majority_class = max(weighted_counts, key=weighted_counts.get)
        total_w = total_weights.get(majority_class, 1.0)
        avg_conf = weighted_confs.get(majority_class, 0.0) / total_w if total_w > 0 else 0.0
        
        return majority_class, avg_conf


# =============================================================================
# DETECTION & TRACK
# =============================================================================

class TrackState(Enum):
    """Track lifecycle state."""
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4


@dataclass
class DamageDetection:
    """Single damage detection from YOLO."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    class_name: str           # e.g., "floor_damage", "inner_wall_damage", "outer_wall_damage"
    confidence: float
    frame_idx: int = 0
    
    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    def to_measurement(self) -> np.ndarray:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return np.array([cx, cy, w, h])


class DamageTrack:
    """
    Single damage track with Kalman filter and memory.
    Represents one physical damage region throughout its visibility.
    """
    
    _next_id = 1
    
    def __init__(self, detection: DamageDetection, kf: KalmanFilter, config: DamageTrackerConfig):
        self.track_id = DamageTrack._next_id
        DamageTrack._next_id += 1
        
        self.config = config
        self.kf = kf
        
        # Initialize Kalman filter state
        measurement = detection.to_measurement()
        self.mean, self.covariance = kf.initiate(measurement)
        
        # Track state
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        
        # Last detection info
        self.last_detection = detection
        self.last_class = detection.class_name
        self.last_confidence = detection.confidence
        
        # First/last seen
        self.first_frame = detection.frame_idx
        self.last_frame = detection.frame_idx
        
        # Memory for class voting
        self.memory = TrackMemory(config.prediction_buffer_size)
        self.memory.add_prediction(detection.frame_idx, detection.class_name, detection.confidence)
        
        # Best snapshot tracking
        self.best_snapshot = None
        self.best_snapshot_bbox = None
        self.best_snapshot_conf = 0.0
        self.best_frame_idx = detection.frame_idx
        self.best_snapshot_is_gap = False  # Whether best snapshot is from a gap frame
        
        # Backup snapshot (best non-gap frame, used as fallback)
        self.backup_snapshot = None
        self.backup_snapshot_bbox = None
        self.backup_snapshot_conf = 0.0
        self.backup_frame_idx = -1
        
        # Wagon number (assigned later)
        self.wagon_number: Optional[int] = None
    
    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1
    
    def predict(self):
        """Predict next state using Kalman filter."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: DamageDetection, frame: Optional[np.ndarray] = None, is_gap_frame: bool = False):
        """Update track with matched detection."""
        measurement = detection.to_measurement()
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        
        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection
        self.last_class = detection.class_name
        self.last_confidence = detection.confidence
        self.last_frame = detection.frame_idx
        
        self.memory.add_prediction(detection.frame_idx, detection.class_name, detection.confidence)
        
        # Update best snapshot — prefer non-gap frames
        if frame is not None and detection.confidence > self.best_snapshot_conf:
            if not is_gap_frame:
                # Non-gap frame with higher conf → becomes best snapshot
                self.best_snapshot = frame.copy()
                self.best_snapshot_bbox = detection.bbox.copy()
                self.best_snapshot_conf = detection.confidence
                self.best_frame_idx = detection.frame_idx
                self.best_snapshot_is_gap = False
            elif self.best_snapshot is None:
                # Gap frame but no snapshot yet → store as fallback
                self.best_snapshot = frame.copy()
                self.best_snapshot_bbox = detection.bbox.copy()
                self.best_snapshot_conf = detection.confidence
                self.best_frame_idx = detection.frame_idx
                self.best_snapshot_is_gap = True
        
        # Always maintain backup snapshot from non-gap frames
        if frame is not None and not is_gap_frame:
            if detection.confidence > self.backup_snapshot_conf:
                self.backup_snapshot = frame.copy()
                self.backup_snapshot_bbox = detection.bbox.copy()
                self.backup_snapshot_conf = detection.confidence
                self.backup_frame_idx = detection.frame_idx
        
        # Update track state
        if self.state == TrackState.TENTATIVE:
            if self.hits >= self.config.n_init:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
    
    def mark_lost(self):
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
    
    def mark_deleted(self):
        self.state = TrackState.DELETED
    
    def should_delete(self) -> bool:
        if self.time_since_update > self.config.max_age:
            return True
        if self.state == TrackState.TENTATIVE and self.time_since_update > 3:
            return True
        return False
    
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box as [x1, y1, x2, y2] from Kalman state."""
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    @property
    def center(self) -> np.ndarray:
        return self.mean[:2]
    
    @property
    def velocity(self) -> np.ndarray:
        return self.mean[4:6]
    
    @property
    def velocity_magnitude(self) -> float:
        return np.sqrt(self.mean[4]**2 + self.mean[5]**2)


# =============================================================================
# DAMAGE TRACKER (Main Class)
# =============================================================================

def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


class DamageTracker:
    """
    Multi-object tracker for damage detections using Kalman filter + Hungarian algorithm.
    
    Adapted from the right/left camera DoorTracker for top-view damage tracking.
    Uses the same proven approach:
    1. Predict all track positions using Kalman filter
    2. Compute cost matrix (distance + IoU) between predictions and detections
    3. Solve assignment using Hungarian algorithm
    4. Update matched tracks, create new tracks, handle lost tracks
    """
    
    def __init__(self, config: DamageTrackerConfig = None):
        self.config = config or DamageTrackerConfig()
        self.kf = KalmanFilter(self.config)
        
        self.tracks: List[DamageTrack] = []
        self.completed_tracks: List[DamageTrack] = []  # Tracks saved before deletion
        self.frame_idx = 0
        self._gap_frames = set()  # Gap frame indices for snapshot filtering
        
        # Reset track ID counter
        DamageTrack.reset_id_counter()
    
    def set_gap_frames(self, gap_frames: set):
        """Set gap frame indices so tracker can avoid gap frames for snapshots."""
        self._gap_frames = gap_frames
    
    def reset(self):
        """Reset tracker for new video."""
        self.tracks.clear()
        self.completed_tracks.clear()
        self.frame_idx = 0
        self._gap_frames = set()
        DamageTrack.reset_id_counter()
    
    def update(
        self,
        detections: List[DamageDetection],
        frame: Optional[np.ndarray] = None,
        frame_width: int = 0,
        frame_height: int = 0
    ) -> List[DamageTrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of DamageDetection objects for current frame
            frame: Current frame (for snapshot capture)
            frame_width: Frame width (for exit detection)
            frame_height: Frame height
            
        Returns:
            List of confirmed tracks
        """
        self.frame_idx += 1
        
        # Set frame_idx on detections
        for det in detections:
            det.frame_idx = self.frame_idx
        
        # Step 1: Predict all existing tracks
        for track in self.tracks:
            track.predict()
        
        # Step 2: Associate detections with tracks using Hungarian algorithm
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate(
            self.tracks, detections
        )
        
        # Step 3: Update matched tracks
        for track_idx, det_idx in matched_pairs:
            is_gap = self.frame_idx in self._gap_frames if self._gap_frames else False
            self.tracks[track_idx].update(detections[det_idx], frame, is_gap_frame=is_gap)
        
        # Step 4: Handle unmatched tracks (mark as lost)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_lost()
        
        # Step 5: Create new tracks from unmatched detections
        for det_idx in unmatched_detections:
            new_track = DamageTrack(
                detections[det_idx],
                self.kf,
                self.config
            )
            # Store initial snapshot (mark if gap frame)
            if frame is not None:
                is_gap = self.frame_idx in self._gap_frames if self._gap_frames else False
                new_track.best_snapshot = frame.copy()
                new_track.best_snapshot_bbox = detections[det_idx].bbox.copy()
                new_track.best_snapshot_conf = detections[det_idx].confidence
                new_track.best_snapshot_is_gap = is_gap
                if not is_gap:
                    new_track.backup_snapshot = frame.copy()
                    new_track.backup_snapshot_bbox = detections[det_idx].bbox.copy()
                    new_track.backup_snapshot_conf = detections[det_idx].confidence
                    new_track.backup_frame_idx = detections[det_idx].frame_idx
            self.tracks.append(new_track)
        
        # Step 6: Save confirmed tracks that are about to be deleted
        for t in self.tracks:
            if t.should_delete() and t.hits >= self.config.min_hits_for_decision:
                self.completed_tracks.append(t)
        
        # Step 7: Delete stale tracks
        self.tracks = [t for t in self.tracks if not t.should_delete()]
        
        # Return confirmed tracks only
        return [t for t in self.tracks if t.is_confirmed()]
    
    def _associate(
        self,
        tracks: List[DamageTrack],
        detections: List[DamageDetection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with tracks using Hungarian algorithm.
        
        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Build cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for t, track in enumerate(tracks):
            track_bbox = track.tlbr
            
            for d, det in enumerate(detections):
                det_bbox = det.bbox
                
                # Distance cost
                track_center = track.center
                det_center = det.center
                distance = np.linalg.norm(track_center - det_center)
                
                # Normalize distance
                dist_cost = min(distance / self.config.max_center_distance, 1.0)
                
                # IoU cost
                iou = compute_iou(track_bbox, det_bbox)
                iou_cost = 1.0 - iou
                
                # Combined cost (pure distance for top-view moving train)
                cost = (self.config.distance_weight * dist_cost + 
                       self.config.iou_weight * iou_cost)
                
                # Gate: if distance is too large, set cost to very high
                if distance > self.config.max_center_distance:
                    cost = 1e5
                
                cost_matrix[t, d] = cost
        
        # Solve assignment using Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out matches with high cost
        matched_pairs = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(tracks)))
        
        for t, d in zip(track_indices, det_indices):
            if cost_matrix[t, d] > 1.0:
                # Cost too high — not a valid match
                continue
            
            matched_pairs.append((t, d))
            unmatched_detections.discard(d)
            unmatched_tracks.discard(t)
        
        return matched_pairs, list(unmatched_detections), list(unmatched_tracks)
    
    def get_all_tracks(self) -> List[DamageTrack]:
        """Get all tracks (including tentative and lost)."""
        return self.tracks
    
    def get_confirmed_tracks(self) -> List[DamageTrack]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed()]
    
    def get_final_damage_states(self) -> Dict[int, dict]:
        """
        Get final damage state for all tracks that were ever confirmed.
        Includes both active tracks AND completed tracks (saved before deletion).
        
        Returns:
            Dict mapping track_id to damage info
        """
        final_states = {}
        
        # Combine active + completed tracks
        all_tracks = list(self.tracks) + list(self.completed_tracks)
        
        for track in all_tracks:
            if track.hits >= self.config.min_hits_for_decision:
                majority_class, avg_confidence = track.memory.get_majority_class()
                
                final_states[track.track_id] = {
                    'class_name': majority_class,
                    'confidence': avg_confidence,
                    'best_confidence': track.best_snapshot_conf,
                    'total_hits': track.hits,
                    'first_frame': track.first_frame,
                    'last_frame': track.last_frame,
                    'best_snapshot': track.best_snapshot,
                    'best_snapshot_bbox': track.best_snapshot_bbox,
                    'best_frame_idx': track.best_frame_idx,
                    'wagon_number': track.wagon_number,
                }
                
                # If best snapshot is from gap frame, swap with backup
                if track.best_snapshot_is_gap and track.backup_snapshot is not None:
                    print(f"  [Snapshot] Track {track.track_id}: swapping gap snapshot "
                          f"(f{track.best_frame_idx}) with backup (f{track.backup_frame_idx})")
                    final_states[track.track_id]['best_snapshot'] = track.backup_snapshot
                    final_states[track.track_id]['best_snapshot_bbox'] = track.backup_snapshot_bbox
                    final_states[track.track_id]['best_confidence'] = track.backup_snapshot_conf
                    final_states[track.track_id]['best_frame_idx'] = track.backup_frame_idx
        
        return final_states


def yolo_to_damage_detections(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: Dict[int, str],
    frame_idx: int = 0
) -> List[DamageDetection]:
    """
    Convert YOLO detection outputs to DamageDetection objects.
    
    Args:
        boxes: Array of [x1, y1, x2, y2] bounding boxes
        confidences: Array of confidence scores
        class_ids: Array of class IDs
        class_names: Dict mapping class ID to class name
        frame_idx: Current frame index
        
    Returns:
        List of DamageDetection objects
    """
    detections = []
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        cls_name = class_names.get(int(cls_id), "unknown")
        det = DamageDetection(
            bbox=np.array(box),
            class_name=cls_name,
            confidence=float(conf),
            frame_idx=frame_idx
        )
        detections.append(det)
    return detections
