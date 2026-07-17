"""
Production-Ready Door Tracking and State Decision System

Designed for video-based door detection on moving trains with fixed cameras.
Features:
- Kalman Filter with constant-velocity model and adaptive process noise
- Hungarian algorithm for data association with direction gating
- Temporal confirmation system with class-dependent thresholds
- Finite State Machine per door with hysteresis
- Stable door-level events (not frame-level predictions)

Author: Door Inspection System
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from collections import deque
import time

# Spatio-temporal verification (post-decision filter for OPEN predictions).
# Lives as a sibling module inside this package (ported from legacy
# RIGHT_UP/temporal_reasoning.py).  Package-relative import so it actually
# resolves when door_tracker is loaded as features.inference_lib.door_tracker;
# the absolute fallback keeps standalone use from hard-failing.
try:
    from .temporal_reasoning import (
        TemporalCropBuffer, TemporalDoorVerifier,
        TemporalReasoningConfig
    )
    TEMPORAL_REASONING_AVAILABLE = True
except ImportError:
    try:
        from temporal_reasoning import (
            TemporalCropBuffer, TemporalDoorVerifier,
            TemporalReasoningConfig
        )
        TEMPORAL_REASONING_AVAILABLE = True
    except ImportError:
        TEMPORAL_REASONING_AVAILABLE = False


# =============================================================================
# CONTEXT SIMILARITY CHECKER (inlined from detection_stabilizer)
# =============================================================================

class ContextSimilarityChecker:
    """
    Compares surrounding context for identity reinforcement.
    
    Stable surroundings reinforce track identity, while unstable
    surroundings reduce confidence in matches.
    """
    
    def __init__(self, context_expand_scale: float = 1.4, histogram_bins: int = 8):
        self.context_expand_scale = context_expand_scale
        self.histogram_bins = histogram_bins
    
    def get_expanded_crop(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        scale: float = None
    ) -> np.ndarray:
        """Get an expanded region around the bounding box."""
        if scale is None:
            scale = self.context_expand_scale
        
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        # Expand
        new_w, new_h = w * scale, h * scale
        
        # New bounds
        nx1 = int(cx - new_w / 2)
        ny1 = int(cy - new_h / 2)
        nx2 = int(cx + new_w / 2)
        ny2 = int(cy + new_h / 2)
        
        # Clamp to frame
        fh, fw = frame.shape[:2]
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(fw, nx2), min(fh, ny2)
        
        if nx2 <= nx1 or ny2 <= ny1:
            return np.array([])
        
        return frame[ny1:ny2, nx1:nx2].copy()
    
    def compute_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram for a crop."""
        if crop is None or crop.size == 0:
            return np.array([])
        
        bins = self.histogram_bins
        
        if len(crop.shape) == 3:
            hist = cv2.calcHist(
                [crop], [0, 1, 2], None,
                [bins, bins, bins],
                [0, 256, 0, 256, 0, 256]
            )
        else:
            hist = cv2.calcHist([crop], [0], None, [bins * 3], [0, 256])
        
        cv2.normalize(hist, hist)
        return hist.flatten()
    
    def compute_context_similarity(
        self,
        frame: np.ndarray,
        bbox1: np.ndarray,
        bbox2: np.ndarray
    ) -> float:
        """Compute similarity between surrounding contexts of two boxes."""
        context1 = self.get_expanded_crop(frame, bbox1)
        context2 = self.get_expanded_crop(frame, bbox2)
        
        if context1.size == 0 or context2.size == 0:
            return 0.5
        
        hist1 = self.compute_histogram(context1)
        hist2 = self.compute_histogram(context2)
        
        if hist1.size == 0 or hist2.size == 0:
            return 0.5
        
        similarity = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        return (similarity + 1) / 2


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrackerConfig:
    """Configuration for the door tracking system."""
    # Confidence thresholds (class-dependent)
    open_confidence_threshold: float = 0.80    # OPEN is safety-critical
    closed_confidence_threshold: float = 0.68  # CLOSED/OTHER threshold
    
    # Kalman Filter settings
    process_noise_moving: float = 0.5    # Process noise when moving
    process_noise_stopped: float = 0.01  # Process noise when stopped
    measurement_noise: float = 1.0       # Measurement noise
    velocity_threshold: float = 2.0      # Velocity below this = stopped
    
    # Association settings
    max_center_distance: float = 150.0   # Max center distance for matching
    iou_weight: float = 0.5              # Weight for IOU cost
    distance_weight: float = 0.5         # Weight for distance cost
    min_iou_threshold: float = 0.1       # Minimum IoU for valid match
    direction_gate_enabled: bool = True  # Reject backward matches
    
    # Spatial hysteresis (Layer 3)
    iou_confirm_threshold: float = 0.2   # Tight threshold for confirming match
    iou_new_id_threshold: float = 0.05   # Loose threshold for permitting new ID
    
    # Temporal ID inertia (Layer 2) - balanced setting
    new_id_delay_frames: int = 8         # Frames before unmatched detection becomes new ID (balanced)
    
    # Context similarity (Layer 5)
    use_context_similarity: bool = True  # Enable context-based matching
    context_weight: float = 0.2          # Weight for context similarity in cost
    
    # Track management
    max_age: int = 30                    # Max frames to keep lost track
    n_init: int = 3                      # Hits before confirmed
    min_hits_for_decision: int = 3       # Min hits for state decision
    
    # State machine hysteresis
    open_confirmation_frames: int = 5    # Consecutive frames to confirm OPEN
    closed_confirmation_frames: int = 5  # Consecutive frames to confirm CLOSED
    
    # Temporal buffer
    prediction_buffer_size: int = 30     # Frames to keep in buffer
    
    # Exit detection
    exit_margin: int = 20                # Pixels from edge to consider exit
    
    # Track revival restrictions (NEW)
    enable_track_revival: bool = True    # Allow reviving retired tracks
    revival_min_iou: float = 0.4         # Minimum IoU with retired track to revive
    revival_min_context_sim: float = 0.6 # Minimum context similarity to revive
    revival_max_frames: int = 60         # Maximum frames since deletion to allow revival


# =============================================================================
# KALMAN FILTER
# =============================================================================

class KalmanFilter:
    """
    Kalman Filter with constant-velocity state model.
    
    State: [cx, cy, w, h, vx, vy] - center x/y, width, height, velocity x/y
    Measurement: [cx, cy, w, h] - center x/y, width, height
    
    Supports adaptive process noise for stop-and-go motion.
    """
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.dt = 1.0  # Time step (1 frame)
        
        # State dimension = 6, Measurement dimension = 4
        self.dim_x = 6
        self.dim_z = 4
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0],      # cx = cx + vx*dt
            [0, 1, 0, 0, 0, self.dt],      # cy = cy + vy*dt
            [0, 0, 1, 0, 0, 0],            # w = w
            [0, 0, 0, 1, 0, 0],            # h = h
            [0, 0, 0, 0, 1, 0],            # vx = vx
            [0, 0, 0, 0, 0, 1],            # vy = vy
        ], dtype=np.float64)
        
        # Measurement matrix (observe position and size, not velocity)
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
            # Stopped: very low process noise
            q = self.config.process_noise_stopped
        else:
            # Moving: normal process noise
            q = self.config.process_noise_moving
        
        # Process noise covariance (higher for velocity components)
        Q = np.diag([q, q, q*0.1, q*0.1, q*2, q*2])
        return Q
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track state from first measurement.
        
        Args:
            measurement: [cx, cy, w, h]
            
        Returns:
            (mean, covariance)
        """
        mean = np.zeros(self.dim_x)
        mean[:4] = measurement  # Position and size
        mean[4:] = 0  # Initial velocity is zero
        
        # Initial covariance
        covariance = np.diag([
            10,   # cx uncertainty
            10,   # cy uncertainty
            10,   # w uncertainty
            10,   # h uncertainty
            100,  # vx uncertainty (unknown)
            100,  # vy uncertainty (unknown)
        ]).astype(np.float64)
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state.
        
        Args:
            mean: Current state mean
            covariance: Current state covariance
            
        Returns:
            (predicted_mean, predicted_covariance)
        """
        # Get velocity for adaptive noise
        velocity = np.sqrt(mean[4]**2 + mean[5]**2)
        Q = self.get_process_noise(velocity)
        
        # Predict
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        
        return mean, covariance
    
    def update(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray, 
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state with new measurement.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: [cx, cy, w, h]
            
        Returns:
            (updated_mean, updated_covariance)
        """
        # Innovation
        y = measurement - self.H @ mean
        
        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R
        
        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update
        mean = mean + K @ y
        covariance = (np.eye(self.dim_x) - K @ self.H) @ covariance
        
        return mean, covariance


# =============================================================================
# DOOR STATE MACHINE
# =============================================================================

class DoorState(Enum):
    """Door state enumeration."""
    UNKNOWN = "UNKNOWN"
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    PARTIAL_CLOSED = "PARTIAL_CLOSED"  # Added: for partial_closed detections
    DAMAGE = "DAMAGE"  # Added: for damage detections from updated model
    OTHER = "OTHER"


class StateMachine:
    """
    Finite State Machine for door state transitions.
    
    Implements hysteresis to prevent state flicker:
    - OPEN requires multiple consecutive confirmations
    - CLOSED only after sustained evidence
    - OPEN events raised only once per door
    
    State diagram:
    UNKNOWN → CLOSED (default if confident)
    UNKNOWN → OPEN (if confirmed)
    CLOSED → OPEN (if confirmed for N frames)
    OPEN → CLOSED (only with strong evidence, rare reversal)
    """
    
    def __init__(self, track_id: int, config: TrackerConfig):
        self.track_id = track_id
        self.config = config
        
        # Current state
        self.state = DoorState.UNKNOWN
        self.previous_state = DoorState.UNKNOWN
        
        # Confirmation counters
        self.open_streak = 0
        self.closed_streak = 0
        self.partial_closed_streak = 0  # Added for partial_closed tracking
        
        # Event tracking
        self.open_event_raised = False
        self.state_history: List[Tuple[int, DoorState]] = []
        
        # Timestamps
        self.last_state_change_frame = 0
    
    def update(
        self, 
        predicted_class: str, 
        confidence: float, 
        frame_idx: int
    ) -> Optional[str]:
        """
        Update state machine with new prediction.
        
        Args:
            predicted_class: Class prediction ("open", "closed", "other")
            confidence: Confidence score
            frame_idx: Current frame index
            
        Returns:
            Event string if state event should be raised, None otherwise
        """
        event = None
        predicted_class = predicted_class.lower()
        
        # Check confidence thresholds
        # IMPORTANT: Check partial_closed BEFORE closed (since partial_closed contains "closed")
        is_partial_closed_confident = (
            "partial" in predicted_class and 
            confidence >= self.config.closed_confidence_threshold
        )
        is_open_confident = (
            "open" in predicted_class and 
            "partial" not in predicted_class and  # Exclude partial_closed
            confidence >= self.config.open_confidence_threshold
        )
        is_closed_confident = (
            "closed" in predicted_class and 
            "partial" not in predicted_class and  # Exclude partial_closed
            confidence >= self.config.closed_confidence_threshold
        )
        is_other_confident = (
            predicted_class == "other" and 
            confidence >= self.config.closed_confidence_threshold
        )
        is_damage_confident = (
            "damage" in predicted_class and
            confidence >= self.config.closed_confidence_threshold
        )
        
        # Update streaks
        if is_damage_confident:
            # Damage is treated as a special state — breaks all other streaks
            self.open_streak = 0
            self.closed_streak = 0
            self.partial_closed_streak = getattr(self, 'partial_closed_streak', 0)
            self.partial_closed_streak = 0
            self.damage_streak = getattr(self, 'damage_streak', 0) + 1
        elif is_open_confident:
            self.open_streak += 1
            self.closed_streak = 0
            self.partial_closed_streak = getattr(self, 'partial_closed_streak', 0)
            self.partial_closed_streak = 0
            self.damage_streak = 0
        elif is_partial_closed_confident:
            self.partial_closed_streak = getattr(self, 'partial_closed_streak', 0) + 1
            self.open_streak = 0
            self.closed_streak = 0
            self.damage_streak = 0
        elif is_closed_confident or is_other_confident:
            self.closed_streak += 1
            self.open_streak = 0
            self.partial_closed_streak = getattr(self, 'partial_closed_streak', 0)
            self.partial_closed_streak = 0
            self.damage_streak = 0
        else:
            # Low confidence - don't break streaks immediately
            self.open_streak = max(0, self.open_streak - 1)
            self.closed_streak = max(0, self.closed_streak - 1)
            if hasattr(self, 'partial_closed_streak'):
                self.partial_closed_streak = max(0, self.partial_closed_streak - 1)
        
        # State transitions with hysteresis
        if self.state == DoorState.UNKNOWN:
            # From UNKNOWN
            if self.open_streak >= self.config.open_confirmation_frames:
                self._transition_to(DoorState.OPEN, frame_idx)
                if not self.open_event_raised:
                    event = "DOOR_OPEN"
                    self.open_event_raised = True
            elif getattr(self, 'partial_closed_streak', 0) >= self.config.closed_confirmation_frames:
                self._transition_to(DoorState.PARTIAL_CLOSED, frame_idx)
            elif self.closed_streak >= self.config.closed_confirmation_frames:
                self._transition_to(DoorState.CLOSED, frame_idx)
        
        elif self.state == DoorState.CLOSED:
            # From CLOSED → can go to OPEN or PARTIAL_CLOSED with confirmation
            if self.open_streak >= self.config.open_confirmation_frames:
                self._transition_to(DoorState.OPEN, frame_idx)
                if not self.open_event_raised:
                    event = "DOOR_OPEN"
                    self.open_event_raised = True
            elif getattr(self, 'partial_closed_streak', 0) >= self.config.closed_confirmation_frames:
                self._transition_to(DoorState.PARTIAL_CLOSED, frame_idx)
        
        elif self.state == DoorState.PARTIAL_CLOSED:
            # From PARTIAL_CLOSED → can go to OPEN or CLOSED
            if self.open_streak >= self.config.open_confirmation_frames:
                self._transition_to(DoorState.OPEN, frame_idx)
                if not self.open_event_raised:
                    event = "DOOR_OPEN"
                    self.open_event_raised = True
            elif self.closed_streak >= self.config.closed_confirmation_frames:
                self._transition_to(DoorState.CLOSED, frame_idx)
        
        elif self.state == DoorState.OPEN:
            # From OPEN → rarely go back to CLOSED (strong evidence needed)
            if self.closed_streak >= self.config.closed_confirmation_frames * 2:
                self._transition_to(DoorState.CLOSED, frame_idx)
            elif getattr(self, 'partial_closed_streak', 0) >= self.config.closed_confirmation_frames * 2:
                self._transition_to(DoorState.PARTIAL_CLOSED, frame_idx)
        
        return event
    
    def _transition_to(self, new_state: DoorState, frame_idx: int):
        """Internal state transition."""
        self.previous_state = self.state
        self.state = new_state
        self.last_state_change_frame = frame_idx
        self.state_history.append((frame_idx, new_state))
    
    def get_state(self) -> DoorState:
        """Get current door state."""
        return self.state
    
    def has_raised_open_event(self) -> bool:
        """Check if OPEN event was raised for this door."""
        return self.open_event_raised


# =============================================================================
# TRACK MEMORY (Per-Track Buffer)
# =============================================================================

@dataclass
class PredictionEntry:
    """Single prediction entry in track buffer."""
    frame_idx: int
    class_name: str
    confidence: float
    illumination_quality: float = 1.0  # Quality score from illumination processor
    timestamp: float = field(default_factory=time.time)


class TrackMemory:
    """
    Per-track memory buffer for temporal aggregation.
    
    Stores recent class predictions, confidences, and computes
    majority voting with weighted confidence averaging.
    """
    
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.predictions: deque = deque(maxlen=buffer_size)
        
        # Aggregated statistics
        self.class_counts: Dict[str, int] = {}
        self.class_confidences: Dict[str, List[float]] = {}
        self.total_frames = 0
    
    def add_prediction(
        self, 
        frame_idx: int, 
        class_name: str, 
        confidence: float,
        illumination_quality: float = 1.0
    ):
        """Add a new prediction to the buffer."""
        entry = PredictionEntry(
            frame_idx=frame_idx,
            class_name=class_name.lower(),
            confidence=confidence,
            illumination_quality=illumination_quality
        )
        self.predictions.append(entry)
        
        # Update aggregates
        cls = class_name.lower()
        self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
        if cls not in self.class_confidences:
            self.class_confidences[cls] = []
        self.class_confidences[cls].append(confidence)
        self.total_frames += 1
    
    def get_majority_class(self) -> Tuple[str, float]:
        """
        Get majority class with quality-weighted voting.
        
        Weights each vote by confidence * illumination_quality so that
        frames with poor lighting (glare, shadows) have reduced influence.
        
        Returns:
            (class_name, weighted_confidence)
        """
        if not self.predictions:
            return "unknown", 0.0
        
        # Quality-weighted voting: each prediction contributes
        # weight = confidence * illumination_quality
        weighted_counts: Dict[str, float] = {}
        weighted_confs: Dict[str, float] = {}
        total_weights: Dict[str, float] = {}
        
        for entry in self.predictions:
            cls = entry.class_name
            # Weight combines detection confidence and illumination quality
            weight = entry.confidence * entry.illumination_quality
            
            weighted_counts[cls] = weighted_counts.get(cls, 0.0) + weight
            weighted_confs[cls] = weighted_confs.get(cls, 0.0) + entry.confidence * weight
            total_weights[cls] = total_weights.get(cls, 0.0) + weight
        
        if not weighted_counts:
            return "unknown", 0.0
        
        # Find majority class based on weighted counts
        majority_class = max(weighted_counts, key=weighted_counts.get)
        
        # Compute weighted average confidence for majority class
        total_w = total_weights.get(majority_class, 1.0)
        if total_w > 0:
            avg_conf = weighted_confs.get(majority_class, 0.0) / total_w
        else:
            avg_conf = 0.0
        
        return majority_class, avg_conf
    
    def get_open_ratio(self) -> float:
        """Get ratio of OPEN predictions in buffer."""
        if not self.predictions:
            return 0.0
        
        open_count = sum(
            1 for p in self.predictions 
            if "open" in p.class_name
        )
        return open_count / len(self.predictions)
    
    def get_recent_confidence(self, n: int = 5) -> float:
        """Get average confidence of last N predictions."""
        if not self.predictions:
            return 0.0
        
        recent = list(self.predictions)[-n:]
        return np.mean([p.confidence for p in recent])


# =============================================================================
# DOOR TRACK
# =============================================================================

class TrackState(Enum):
    """Track lifecycle state."""
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4


@dataclass
class Detection:
    """Single detection from YOLO."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    class_name: str           # "open", "closed", "other"
    confidence: float         # Detection confidence
    frame_idx: int = 0
    illumination_quality: float = 1.0  # Frame illumination quality (0-1)
    
    @property
    def center(self) -> np.ndarray:
        """Get bounding box center."""
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
        """Convert to Kalman filter measurement [cx, cy, w, h]."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return np.array([cx, cy, w, h])


class DoorTrack:
    """
    Single door track with Kalman filter, memory, and state machine.
    
    Represents one physical door throughout its visibility.
    """
    
    _next_id = 1
    
    def __init__(
        self, 
        detection: Detection,
        kf: KalmanFilter,
        config: TrackerConfig
    ):
        # Assign unique ID
        self.track_id = DoorTrack._next_id
        DoorTrack._next_id += 1
        
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
        
        # Memory and state machine
        self.memory = TrackMemory(config.prediction_buffer_size)
        self.state_machine = StateMachine(self.track_id, config)
        
        # Spatio-temporal crop buffer for OPEN verification
        if TEMPORAL_REASONING_AVAILABLE:
            self.crop_buffer = TemporalCropBuffer()
        else:
            self.crop_buffer = None
        
        # Reference to shared temporal verifier (set by DoorTracker)
        self.temporal_verifier: Optional['TemporalDoorVerifier'] = None
        
        # Add first prediction with illumination quality
        self.memory.add_prediction(
            detection.frame_idx,
            detection.class_name,
            detection.confidence,
            detection.illumination_quality
        )
        
        # Exit tracking
        self.exited = False
        self.exit_frame = None
        
        # Wagon number (assigned by wagon timeline integration)
        self.wagon_number: Optional[int] = None
    
    @classmethod
    def reset_id_counter(cls):
        """Reset track ID counter (for new video)."""
        cls._next_id = 1
    
    def predict(self):
        """Predict next state using Kalman filter."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: Detection, frame: Optional[np.ndarray] = None):
        """Update track with matched detection."""
        measurement = detection.to_measurement()
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measurement
        )
        
        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection
        self.last_class = detection.class_name
        self.last_confidence = detection.confidence
        self.last_frame = detection.frame_idx
        
        # Add to memory with illumination quality
        self.memory.add_prediction(
            detection.frame_idx,
            detection.class_name,
            detection.confidence,
            detection.illumination_quality
        )
        
        # Add crop to temporal buffer for spatio-temporal verification
        if self.crop_buffer is not None and frame is not None:
            self.crop_buffer.add_crop(frame, detection.bbox, detection.frame_idx)
        
        # Update track state
        if self.state == TrackState.TENTATIVE:
            if self.hits >= self.config.n_init:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
    
    def mark_lost(self):
        """Mark track as lost (no match this frame)."""
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
    
    def mark_deleted(self):
        """Mark track for deletion."""
        self.state = TrackState.DELETED
    
    def should_delete(self) -> bool:
        """Check if track should be deleted."""
        # Delete if lost too long
        if self.time_since_update > self.config.max_age:
            return True
        # Delete if tentative and lost
        if self.state == TrackState.TENTATIVE and self.time_since_update > 3:
            return True
        return False
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED
    
    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        return self.mean[4:6]
    
    @property
    def velocity_magnitude(self) -> float:
        """Get velocity magnitude."""
        return np.sqrt(self.mean[4]**2 + self.mean[5]**2)
    
    @property
    def center(self) -> np.ndarray:
        """Get current center position."""
        return self.mean[:2]
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box as [x1, y1, x2, y2]."""
        cx, cy, w, h = self.mean[:4]
        return np.array([
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ])
    
    def get_decision(self, frame_idx: int) -> Tuple[DoorState, float, Optional[str]]:
        """
        Get final door state decision with temporal aggregation.
        
        Returns:
            (final_state, confidence, event_if_any)
        """
        # Get majority class from memory
        majority_class, weighted_conf = self.memory.get_majority_class()
        
        # If not enough hits, use majority class directly without FSM
        if self.hits < self.config.min_hits_for_decision:
            # Map class name to DoorState
            # IMPORTANT: Only mark as OPEN if confidence meets threshold
            if "damage" in majority_class:
                return DoorState.DAMAGE, weighted_conf, None
            elif "open" in majority_class:
                if weighted_conf >= self.config.open_confidence_threshold:
                    # Ensure open_event_raised is set even without FSM transition
                    if not self.state_machine.open_event_raised:
                        self.state_machine.open_event_raised = True
                    return DoorState.OPEN, weighted_conf, None
                else:
                    return DoorState.PARTIAL_CLOSED, weighted_conf, None
            elif "closed" in majority_class:
                return DoorState.CLOSED, weighted_conf, None
            elif majority_class != "unknown":
                return DoorState.CLOSED, weighted_conf, None
            return DoorState.CLOSED, weighted_conf, None
        
        # Update state machine
        event = self.state_machine.update(majority_class, weighted_conf, frame_idx)
        
        # If FSM still at UNKNOWN, fallback to majority class
        fsm_state = self.state_machine.get_state()
        if fsm_state == DoorState.UNKNOWN:
            if "damage" in majority_class:
                return DoorState.DAMAGE, weighted_conf, event
            elif "open" in majority_class:
                if weighted_conf >= self.config.open_confidence_threshold:
                    # Ensure open_event_raised is set even without FSM transition
                    if not self.state_machine.open_event_raised:
                        self.state_machine.open_event_raised = True
                    return DoorState.OPEN, weighted_conf, event
                else:
                    return DoorState.PARTIAL_CLOSED, weighted_conf, event
            else:
                return DoorState.CLOSED, weighted_conf, event
        
        # IMPORTANT: Final enforcement - if FSM says OPEN but weighted confidence
        # is below threshold, downgrade to PARTIAL_CLOSED for reporting consistency
        if fsm_state == DoorState.OPEN and weighted_conf < self.config.open_confidence_threshold:
            return DoorState.PARTIAL_CLOSED, weighted_conf, event
        
        # =================================================================
        # Spatio-Temporal Verification Layer (post-decision filter)
        # Only activates for OPEN predictions with sufficient crop history
        # =================================================================
        if (
            fsm_state == DoorState.OPEN
            and self.temporal_verifier is not None
            and self.crop_buffer is not None
            and self.crop_buffer.is_ready()
        ):
            verified, reason, adj_conf = self.temporal_verifier.verify_open_prediction(
                self.track_id, self.crop_buffer, weighted_conf
            )
            if not verified:
                # Override: OPEN → CLOSED (shine/reflection false positive)
                return DoorState.CLOSED, adj_conf, event
            else:
                # Confirmed or pass-through: keep OPEN with adjusted confidence
                return DoorState.OPEN, adj_conf, event
        
        return fsm_state, weighted_conf, event


# =============================================================================
# DECISION LOGIC (Aggregation & Events)
# =============================================================================

class DecisionLogic:
    """
    Temporal aggregation and decision logic for door states.
    
    Handles:
    - Aggregating predictions across frames
    - Determining final door state per track
    - Raising door-level events (not frame-level)
    """
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.events: List[Dict] = []
        self.door_decisions: Dict[int, DoorState] = {}
    
    def process_track(
        self, 
        track: DoorTrack, 
        frame_idx: int
    ) -> Tuple[DoorState, float, Optional[str]]:
        """
        Process a track and get current decision.
        
        Returns:
            (state, confidence, event)
        """
        state, conf, event = track.get_decision(frame_idx)
        
        # Record event if raised
        if event:
            self.events.append({
                'event': event,
                'track_id': track.track_id,
                'frame_idx': frame_idx,
                'confidence': conf,
                'center': track.center.tolist(),
                'bbox': track.tlbr.tolist()
            })
        
        # Update decision record
        self.door_decisions[track.track_id] = state
        
        return state, conf, event
    
    def get_all_open_doors(self) -> List[int]:
        """Get list of track IDs for doors classified as OPEN."""
        return [
            tid for tid, state in self.door_decisions.items()
            if state == DoorState.OPEN
        ]
    
    def get_event_count(self, event_type: str = "DOOR_OPEN") -> int:
        """Get count of specific event type."""
        return sum(1 for e in self.events if e['event'] == event_type)
    
    def get_all_events(self) -> List[Dict]:
        """Get all recorded events."""
        return self.events.copy()


# =============================================================================
# MAIN TRACKER
# =============================================================================

class DoorTracker:
    """
    Main door tracking system.
    
    Combines:
    - Kalman Filter for motion prediction
    - Hungarian algorithm for data association
    - Direction gating to reject backward matches
    - Track lifecycle management
    - Decision logic for door state events
    """
    
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.kf = KalmanFilter(self.config)
        self.decision_logic = DecisionLogic(self.config)
        
        # Context similarity checker (Layer 5)
        self.context_checker = ContextSimilarityChecker()
        
        # Spatio-temporal verifier (shared across all tracks)
        if TEMPORAL_REASONING_AVAILABLE:
            self.temporal_verifier = TemporalDoorVerifier()
        else:
            self.temporal_verifier = None
        
        # Active tracks
        self.tracks: List[DoorTrack] = []
        self.deleted_tracks: List[DoorTrack] = []
        
        # Frame tracking
        self.frame_idx = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Pending detections for temporal ID inertia (Layer 2)
        # Key: spatial hash, Value: (detection, consecutive_frame_count, last_frame_idx)
        self.pending_detections: Dict[str, Tuple[Detection, int, int]] = {}
        
        # Reset track IDs
        DoorTrack.reset_id_counter()
    
    def reset(self):
        """Reset tracker for new video."""
        self.tracks.clear()
        self.deleted_tracks.clear()
        self.pending_detections.clear()
        self.frame_idx = 0
        self.decision_logic = DecisionLogic(self.config)
        if self.temporal_verifier is not None:
            self.temporal_verifier.reset()
        DoorTrack.reset_id_counter()
    
    def update(
        self, 
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
        frame_width: int = 0,
        frame_height: int = 0
    ) -> List[DoorTrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from YOLO
            frame: Current video frame (required for context similarity)
            frame_width: Frame width for exit detection
            frame_height: Frame height for exit detection
            
        Returns:
            List of confirmed tracks with decisions
        """
        self.frame_idx += 1
        if frame_width > 0:
            self.frame_width = frame_width
        if frame_height > 0:
            self.frame_height = frame_height
        
        # Set frame index on detections
        for det in detections:
            det.frame_idx = self.frame_idx
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = \
            self._associate(detections, frame)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            self.tracks[track_idx].update(detections[det_idx], frame=frame)
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_lost()
            
            # Check for exit
            if self._check_exit(track):
                track.exited = True
                track.exit_frame = self.frame_idx
        
        # Create new tracks for unmatched detections with temporal ID inertia
        # Layer 2: Require multiple consecutive frames before creating new ID
        # NEW: First try to revive deleted tracks if spatial overlap + context similarity are high
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            
            # Try to revive a deleted track first (restricted revival)
            revived_track = self._try_revive_track(det, frame)
            if revived_track is not None:
                # Revival successful - don't create new track
                revived_track.temporal_verifier = self.temporal_verifier
                self.tracks.append(revived_track)
                continue
            
            # Check if this detection should create a new track
            if self._should_create_new_track(det):
                new_track = DoorTrack(
                    det,
                    self.kf,
                    self.config
                )
                # Attach shared temporal verifier
                new_track.temporal_verifier = self.temporal_verifier
                self.tracks.append(new_track)
        
        # Clean up old pending detections
        self._cleanup_pending_detections()
        
        # Delete old tracks
        self._cleanup_tracks()
        
        # Process decisions for confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.is_confirmed():
                # Get decision
                state, conf, event = self.decision_logic.process_track(
                    track, self.frame_idx
                )
                confirmed_tracks.append(track)
        
        return confirmed_tracks
    
    def _associate(
        self, 
        detections: List[Detection],
        frame: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm.
        
        Cost = distance_weight * norm_dist + iou_weight * (1-IoU) + context_weight * (1-Similarity)
        
        Direction gating rejects backward matches.
        
        Returns:
            (matched_track_indices, matched_det_indices, 
             unmatched_track_indices, unmatched_det_indices)
        """
        if len(self.tracks) == 0:
            return [], [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks))), []
        
        # Build cost matrix
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets))
        
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                # Center distance
                track_center = track.center
                det_center = det.center
                dist = np.linalg.norm(track_center - det_center)
                
                # Normalize distance
                norm_dist = min(dist / self.config.max_center_distance, 1.0)
                
                # IoU
                iou = self._compute_iou(track.tlbr, det.bbox)
                
                # Combined cost (Layer 5: incorporate context similarity)
                base_cost = (
                    self.config.distance_weight * norm_dist +
                    self.config.iou_weight * (1 - iou)
                )
                
                if self.config.use_context_similarity and frame is not None:
                    # Compute context similarity between track (last position) and detection
                    similarity = self.context_checker.compute_context_similarity(
                        frame, track.tlbr, det.bbox
                    )
                    # Add context-based cost component
                    cost = (
                        (1.0 - self.config.context_weight) * base_cost +
                        self.config.context_weight * (1.0 - similarity)
                    )
                else:
                    cost = base_cost
                
                # Direction gating: reject backward matches
                if self.config.direction_gate_enabled:
                    # Track velocity indicates expected movement direction
                    vx = track.velocity[0]
                    
                    # Detection should be in same direction as velocity
                    dx = det_center[0] - track_center[0]
                    
                    # If track is moving right (vx > 0), detection should be right
                    # If moving left (vx < 0), detection should be left
                    # If stopped, no gating
                    if abs(vx) > self.config.velocity_threshold:
                        if (vx > 0 and dx < -10) or (vx < 0 and dx > 10):
                            cost = 1e6  # Effectively block this match
                
                # Gate by max distance
                if dist > self.config.max_center_distance:
                    cost = 1e6
                
                # Gate by minimum IoU threshold
                if iou < self.config.min_iou_threshold:
                    cost = 1e6
                
                cost_matrix[t, d] = cost
        
        # Sanitize cost matrix — replace NaN/Inf with high cost
        cost_matrix = np.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)
        
        # Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by cost threshold with spatial hysteresis (Layer 3)
        matched_tracks = []
        matched_dets = []
        
        for t, d in zip(track_indices, det_indices):
            if cost_matrix[t, d] < 1.0:  # Valid match
                # Additional IoU check for confirmation (tight threshold)
                track = self.tracks[t]
                det = detections[d]
                iou = self._compute_iou(track.tlbr, det.bbox)
                
                # Spatial hysteresis: use tight threshold for confirming match
                if iou >= self.config.iou_confirm_threshold:
                    matched_tracks.append(t)
                    matched_dets.append(d)
                elif iou >= self.config.iou_new_id_threshold:
                    # In buffer zone - still match but with lower confidence
                    # This prevents new ID creation but allows match
                    matched_tracks.append(t)
                    matched_dets.append(d)
                # Below iou_new_id_threshold: don't match, will create new ID
        
        # Find unmatched
        unmatched_tracks = [
            t for t in range(n_tracks) if t not in matched_tracks
        ]
        
        # Layer 3: Spatial hysteresis for unmatched detections
        # Only allow new ID if IoU with ALL tracks is below loose threshold
        truly_unmatched_dets = []
        for d in range(n_dets):
            if d not in matched_dets:
                # Check if this detection overlaps with any track
                det = detections[d]
                max_iou_with_any_track = 0.0
                for track in self.tracks:
                    iou = self._compute_iou(track.tlbr, det.bbox)
                    max_iou_with_any_track = max(max_iou_with_any_track, iou)
                
                # Only create new ID if clearly separate from all tracks
                if max_iou_with_any_track < self.config.iou_new_id_threshold:
                    truly_unmatched_dets.append(d)
                # Else: in buffer zone, don't create new ID (will try to match later)
        
        return matched_tracks, matched_dets, unmatched_tracks, truly_unmatched_dets
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        # Cast to float64 to prevent integer overflow
        b1 = np.asarray(box1, dtype=np.float64)
        b2 = np.asarray(box2, dtype=np.float64)
        
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return float(inter_area / union_area)
    
    def _check_exit(self, track: DoorTrack) -> bool:
        """Check if track has exited the frame."""
        if self.frame_width == 0:
            return False
        
        cx = track.center[0]
        margin = self.config.exit_margin
        
        # Check if center is near edge and velocity points outward
        vx = track.velocity[0]
        
        # Exiting right
        if cx > self.frame_width - margin and vx > 0:
            return True
        
        # Exiting left
        if cx < margin and vx < 0:
            return True
        
        return False
    
    def _cleanup_tracks(self):
        """Remove deleted tracks."""
        tracks_to_keep = []
        
        for track in self.tracks:
            if track.should_delete():
                track.mark_deleted()
                # Store deletion frame for revival restriction
                track.deletion_frame = self.frame_idx
                self.deleted_tracks.append(track)
            else:
                tracks_to_keep.append(track)
        
        self.tracks = tracks_to_keep
    
    def _try_revive_track(
        self, 
        detection: Detection, 
        frame: Optional[np.ndarray] = None
    ) -> Optional[DoorTrack]:
        """
        Try to revive a recently deleted track if the detection matches well.
        
        Restricted revival requires BOTH:
        1. High spatial overlap (IoU >= revival_min_iou)
        2. High context similarity (>= revival_min_context_sim)
        
        This prevents false positives from similar-looking but different doors
        and ensures only truly matching tracks are revived.
        
        Args:
            detection: Unmatched detection to check for revival
            frame: Current frame for context similarity check
            
        Returns:
            Revived DoorTrack if successful, None otherwise
        """
        if not self.config.enable_track_revival:
            return None
        
        if not self.deleted_tracks:
            return None
        
        best_match = None
        best_score = 0.0
        
        for track in self.deleted_tracks:
            # Check deletion age
            if not hasattr(track, 'deletion_frame'):
                continue
            
            frames_since_deletion = self.frame_idx - track.deletion_frame
            if frames_since_deletion > self.config.revival_max_frames:
                continue
            
            # Check spatial overlap (IoU)
            iou = self._compute_iou(track.tlbr, detection.bbox)
            if iou < self.config.revival_min_iou:
                continue
            
            # Check context similarity (if frame available)
            context_sim = 1.0  # Default high if no frame
            if frame is not None and self.config.use_context_similarity:
                context_sim = self.context_checker.compute_context_similarity(
                    frame, track.tlbr, detection.bbox
                )
                if context_sim < self.config.revival_min_context_sim:
                    continue
            
            # Score based on IoU and context similarity
            combined_score = 0.5 * iou + 0.5 * context_sim
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = track
        
        if best_match is not None:
            # Revive the track
            # Remove from deleted_tracks
            self.deleted_tracks.remove(best_match)
            
            # Update track with new detection
            best_match.update(detection)
            best_match.state = TrackState.CONFIRMED  # Reset to confirmed
            best_match.time_since_update = 0
            
            return best_match
        
        return None
    
    def _get_detection_hash(self, detection: Detection) -> str:
        """
        Create a spatial hash for a detection based on location.
        
        Used to track pending detections across frames.
        """
        cx, cy = detection.center
        # Round to grid of ~50 pixels for spatial binning
        grid_x = int(cx / 50)
        grid_y = int(cy / 50)
        return f"{grid_x}_{grid_y}"
    
    def _should_create_new_track(self, detection: Detection) -> bool:
        """
        Check if unmatched detection should create a new track.
        
        Layer 2: Temporal ID Inertia - Require multiple consecutive
        frames of unmatched detection before creating new ID.
        
        Uses proximity search to associate detections across frames 
        even if they are moving.
        """
        best_match_key = None
        min_dist = 200.0  # Search radius (pixels) - enough for moving train
        
        for key, (stored_det, count, last_frame) in self.pending_detections.items():
            dist = np.linalg.norm(stored_det.center - detection.center)
            if dist < min_dist:
                min_dist = dist
                best_match_key = key
        
        if best_match_key is not None:
            _, count, last_frame = self.pending_detections[best_match_key]
            
            # Check if this is from consecutive frame (or small gap)
            if self.frame_idx - last_frame <= 2:  # Allow 1 frame gap
                new_count = count + 1
                
                # Update entry with current detection and frame
                del self.pending_detections[best_match_key]
                new_key = f"pending_{self.frame_idx}_{int(detection.center[0])}"
                self.pending_detections[new_key] = (detection, new_count, self.frame_idx)
                
                # Check if we've waited long enough
                if new_count >= self.config.new_id_delay_frames:
                    # Create track and remove from pending
                    del self.pending_detections[new_key]
                    return True
                else:
                    # Still waiting
                    return False
            else:
                # Gap too large - remove old entry
                del self.pending_detections[best_match_key]
        
        # No match found - start new pending tracking
        new_key = f"pending_{self.frame_idx}_{int(detection.center[0])}"
        self.pending_detections[new_key] = (detection, 1, self.frame_idx)
        return False
    
    def _cleanup_pending_detections(self):
        """Remove stale pending detections."""
        stale_keys = []
        for det_hash, (_, _, last_frame) in self.pending_detections.items():
            # Remove if not seen in last 3 frames
            if self.frame_idx - last_frame > 3:
                stale_keys.append(det_hash)
        
        for key in stale_keys:
            del self.pending_detections[key]
    
    def get_confirmed_tracks(self) -> List[DoorTrack]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed()]
    
    def get_all_tracks(self) -> List[DoorTrack]:
        """Get all active tracks."""
        return self.tracks.copy()
    
    def get_open_door_count(self) -> int:
        """Get number of doors classified as OPEN."""
        return len(self.decision_logic.get_all_open_doors())
    
    def get_events(self) -> List[Dict]:
        """Get all door-level events."""
        return self.decision_logic.get_all_events()
    
    def get_final_door_states(self) -> Dict[int, Dict]:
        """
        Get final state for all tracked doors.
        
        Returns:
            Dict mapping track_id to {state, confidence, first_frame, last_frame}
        """
        results = {}
        
        # Include both active and deleted tracks
        all_tracks = self.tracks + self.deleted_tracks
        
        for track in all_tracks:
            # Use get_decision which has fallback logic to avoid UNKNOWN
            state, conf, _ = track.get_decision(track.last_frame)
            
            results[track.track_id] = {
                'state': state.value,
                'confidence': conf,
                'first_frame': track.first_frame,
                'last_frame': track.last_frame,
                'total_hits': track.hits,
                'open_event_raised': track.state_machine.has_raised_open_event(),
                'wagon_number': track.wagon_number
            }
        
        return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def yolo_to_detections(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: Dict[int, str],
    illumination_quality: float = 1.0
) -> List[Detection]:
    """
    Convert YOLO output to Detection objects.
    
    Args:
        boxes: Array of [x1, y1, x2, y2] boxes
        confidences: Array of confidence scores
        class_ids: Array of class IDs
        class_names: Dict mapping class ID to name
        illumination_quality: Frame illumination quality score (0-1)
        
    Returns:
        List of Detection objects
    """
    detections = []
    
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        cls_name = class_names.get(int(cls_id), "unknown")
        
        det = Detection(
            bbox=np.array(box),
            class_name=cls_name,
            confidence=float(conf),
            illumination_quality=illumination_quality
        )
        detections.append(det)
    
    return detections


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    config = TrackerConfig(
        open_confidence_threshold=0.80,
        closed_confidence_threshold=0.68,
        max_age=30,
        n_init=3
    )
    
    tracker = DoorTracker(config)
    
    # Simulate detections
    print("Door Tracker initialized successfully!")
    print(f"Config: {config}")
