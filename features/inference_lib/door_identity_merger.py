"""
Door Identity Merger Module - Post-tracking identity consolidation.

Prevents the same physical door from appearing multiple times in the final PDF
by merging fragmented tracks that represent the same door.

This module runs once per video AFTER tracking completes but BEFORE snapshot
selection and PDF generation. It does NOT affect live tracking behavior.

Similarity Functions:
1. Spatial proximity of mean/median bounding box centers
2. Temporal adjacency/overlap of visibility intervals
3. Context similarity using HSV histogram (from detection_stabilizer)
4. Structural similarity of representative snapshots (SSIM)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Ported from RIGHT_UP/door_identity_merger.py.  Originally imported
# ContextSimilarityChecker as a top-level sibling; we now sit inside
# `features.inference_lib`, so use a relative import.
from .door_tracker import ContextSimilarityChecker


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MergeConfig:
    """Configuration for door identity merging."""
    
    # Spatial similarity thresholds
    # Max normalized center distance for spatial match (0-1 scale)
    # CONSERVATIVE: Only merge if very close (8% of diagonal)
    max_spatial_distance: float = 0.08
    
    # Minimum spatial similarity required (hard gate)
    # Tracks must be spatially close to even consider merging
    min_spatial_similarity: float = 0.5
    
    # Temporal adjacency thresholds
    # Max frame gap to consider tracks as temporally adjacent
    max_temporal_gap: int = 30
    # Min overlap frames to boost temporal similarity
    min_temporal_overlap: int = 5
    
    # Context similarity threshold (HSV histogram correlation)
    min_context_similarity: float = 0.6
    
    # Structural similarity threshold (SSIM)
    min_structural_similarity: float = 0.5
    
    # Combined merge threshold (must exceed this to merge)
    # VERY CONSERVATIVE: set high to avoid merging different doors
    merge_threshold: float = 0.85
    
    # Weights for combining similarity metrics
    # Spatial is most important to avoid merging different doors
    spatial_weight: float = 0.40
    temporal_weight: float = 0.15
    context_weight: float = 0.25
    structural_weight: float = 0.20
    
    # Snapshot comparison size
    snapshot_resize: Tuple[int, int] = (128, 128)


# =============================================================================
# TRACK SUMMARY
# =============================================================================

@dataclass
class TrackSummary:
    """Summary of a door track for merging."""
    track_id: int
    first_frame: int
    last_frame: int
    total_hits: int
    
    # Spatial info (computed from track history)
    mean_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    median_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    mean_bbox: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    
    # State info
    final_state: str = "UNKNOWN"
    confidence: float = 0.0
    open_event_raised: bool = False
    
    # Context histogram (precomputed)
    context_histogram: Optional[np.ndarray] = None
    
    # Representative snapshot
    snapshot: Optional[np.ndarray] = None
    snapshot_bbox: Optional[np.ndarray] = None


# =============================================================================
# DOOR IDENTITY MERGER
# =============================================================================

class DoorIdentityMerger:
    """
    Merges fragmented door tracks that represent the same physical door.
    
    This is a post-processing step that runs once after tracking completes.
    It compares all finalized tracks pairwise and merges those with high
    combined similarity scores.
    
    Key properties:
    - Conservative: uses strict threshold to avoid merging different doors
    - Deterministic: same input always produces same output
    - Non-breaking: does not affect live tracking behavior
    """
    
    def __init__(self, config: Optional[MergeConfig] = None):
        self.config = config or MergeConfig()
        self.context_checker = ContextSimilarityChecker()
    
    # -------------------------------------------------------------------------
    # SPATIAL SIMILARITY
    # -------------------------------------------------------------------------
    
    def compute_spatial_similarity(
        self,
        track1: TrackSummary,
        track2: TrackSummary,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Compute spatial proximity similarity between two tracks.
        
        Uses normalized distance between mean/median centers.
        Returns 1.0 for identical positions, 0.0 for distant positions.
        """
        # Normalize centers by frame dimensions
        diag = np.sqrt(frame_width**2 + frame_height**2)
        if diag == 0:
            return 0.0
        
        # Distance between mean centers
        mean_dist = np.linalg.norm(track1.mean_center - track2.mean_center)
        norm_mean_dist = mean_dist / diag
        
        # Distance between median centers
        median_dist = np.linalg.norm(track1.median_center - track2.median_center)
        norm_median_dist = median_dist / diag
        
        # Average normalized distance
        avg_dist = (norm_mean_dist + norm_median_dist) / 2
        
        # Convert to similarity (0-1, closer = higher)
        if avg_dist >= self.config.max_spatial_distance:
            return 0.0
        
        similarity = 1.0 - (avg_dist / self.config.max_spatial_distance)
        return float(np.clip(similarity, 0.0, 1.0))
    
    # -------------------------------------------------------------------------
    # TEMPORAL SIMILARITY
    # -------------------------------------------------------------------------
    
    def compute_temporal_similarity(
        self,
        track1: TrackSummary,
        track2: TrackSummary
    ) -> float:
        """
        Compute temporal adjacency/overlap similarity.
        
        High similarity for overlapping or adjacent visibility intervals.
        Low similarity for tracks separated by many frames.
        """
        # Compute overlap
        overlap_start = max(track1.first_frame, track2.first_frame)
        overlap_end = min(track1.last_frame, track2.last_frame)
        overlap = max(0, overlap_end - overlap_start + 1)
        
        if overlap >= self.config.min_temporal_overlap:
            # Overlapping tracks - high similarity
            # More overlap = higher similarity
            max_duration = max(
                track1.last_frame - track1.first_frame + 1,
                track2.last_frame - track2.first_frame + 1
            )
            if max_duration > 0:
                overlap_ratio = overlap / max_duration
                return float(np.clip(0.7 + 0.3 * overlap_ratio, 0.0, 1.0))
            return 0.7
        
        # No overlap - compute gap
        if track1.last_frame < track2.first_frame:
            gap = track2.first_frame - track1.last_frame
        elif track2.last_frame < track1.first_frame:
            gap = track1.first_frame - track2.last_frame
        else:
            gap = 0
        
        if gap > self.config.max_temporal_gap:
            return 0.0
        
        # Adjacent tracks - similarity decreases with gap
        similarity = 1.0 - (gap / self.config.max_temporal_gap)
        return float(np.clip(similarity * 0.7, 0.0, 1.0))  # Cap at 0.7 for non-overlapping
    
    # -------------------------------------------------------------------------
    # CONTEXT SIMILARITY
    # -------------------------------------------------------------------------
    
    def compute_context_similarity(
        self,
        track1: TrackSummary,
        track2: TrackSummary
    ) -> float:
        """
        Compute context similarity using precomputed HSV histograms.
        
        Uses histogram correlation for robust matching.
        """
        if track1.context_histogram is None or track2.context_histogram is None:
            return 0.5  # Neutral if no histogram available
        
        if track1.context_histogram.size == 0 or track2.context_histogram.size == 0:
            return 0.5
        
        # Histogram correlation
        similarity = cv2.compareHist(
            track1.context_histogram.astype(np.float32),
            track2.context_histogram.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Normalize to 0-1 (correlation is -1 to 1)
        return float(np.clip((similarity + 1) / 2, 0.0, 1.0))
    
    # -------------------------------------------------------------------------
    # STRUCTURAL SIMILARITY
    # -------------------------------------------------------------------------
    
    def compute_structural_similarity(
        self,
        track1: TrackSummary,
        track2: TrackSummary
    ) -> float:
        """
        Compute structural similarity between representative snapshots.
        
        Uses normalized cross-correlation or histogram comparison on
        resized door region crops.
        """
        if track1.snapshot is None or track2.snapshot is None:
            return 0.5  # Neutral if no snapshot
        
        try:
            # Resize to common size
            size = self.config.snapshot_resize
            snap1 = cv2.resize(track1.snapshot, size)
            snap2 = cv2.resize(track2.snapshot, size)
            
            # Convert to grayscale
            if len(snap1.shape) == 3:
                gray1 = cv2.cvtColor(snap1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = snap1
            
            if len(snap2.shape) == 3:
                gray2 = cv2.cvtColor(snap2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = snap2
            
            # Normalized cross-correlation
            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)
            similarity = float(result[0, 0])
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    # -------------------------------------------------------------------------
    # COMBINED SIMILARITY
    # -------------------------------------------------------------------------
    
    def compute_combined_similarity(
        self,
        track1: TrackSummary,
        track2: TrackSummary,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted combination of all similarity metrics.
        
        Returns:
            (combined_score, breakdown_dict)
        """
        spatial = self.compute_spatial_similarity(
            track1, track2, frame_width, frame_height
        )
        temporal = self.compute_temporal_similarity(track1, track2)
        context = self.compute_context_similarity(track1, track2)
        structural = self.compute_structural_similarity(track1, track2)
        
        # Weighted average
        combined = (
            self.config.spatial_weight * spatial +
            self.config.temporal_weight * temporal +
            self.config.context_weight * context +
            self.config.structural_weight * structural
        )
        
        breakdown = {
            'spatial': spatial,
            'temporal': temporal,
            'context': context,
            'structural': structural,
            'combined': combined
        }
        
        return combined, breakdown
    
    # -------------------------------------------------------------------------
    # MERGE DECISION
    # -------------------------------------------------------------------------
    
    def should_merge(
        self,
        track1: TrackSummary,
        track2: TrackSummary,
        frame_width: int,
        frame_height: int
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determine if two tracks should be merged.
        
        Uses conservative threshold to avoid merging different doors.
        Applies spatial hard gate FIRST - tracks must be spatially close.
        
        Returns:
            (should_merge, combined_score, breakdown)
        """
        # HARD GATE: Check spatial similarity first
        # Tracks must be spatially close to even consider merging
        spatial = self.compute_spatial_similarity(
            track1, track2, frame_width, frame_height
        )
        
        if spatial < self.config.min_spatial_similarity:
            # Not close enough - don't merge regardless of other metrics
            breakdown = {
                'spatial': spatial,
                'temporal': 0.0,
                'context': 0.0,
                'structural': 0.0,
                'combined': 0.0,
                'rejected_reason': 'spatial_gate'
            }
            return False, 0.0, breakdown
        
        # Passed spatial gate - compute full similarity
        combined, breakdown = self.compute_combined_similarity(
            track1, track2, frame_width, frame_height
        )
        
        should = combined >= self.config.merge_threshold
        
        return should, combined, breakdown
    
    # -------------------------------------------------------------------------
    # TRACK SUMMARY EXTRACTION
    # -------------------------------------------------------------------------
    
    def extract_track_summary(
        self,
        track,  # DoorTrack object
        final_state_info: Dict,
        best_snapshot_data: Optional[Dict],
        frame: Optional[np.ndarray] = None
    ) -> TrackSummary:
        """
        Extract summary information from a DoorTrack for merging.
        
        Args:
            track: DoorTrack object
            final_state_info: Dict with 'state', 'confidence', etc.
            best_snapshot_data: Dict with 'frame', 'bbox', etc.
            frame: Optional frame for context histogram computation
        """
        summary = TrackSummary(
            track_id=track.track_id,
            first_frame=track.first_frame,
            last_frame=track.last_frame,
            total_hits=track.hits,
            final_state=final_state_info.get('state', 'UNKNOWN'),
            confidence=final_state_info.get('confidence', 0.0),
            open_event_raised=final_state_info.get('open_event_raised', False)
        )
        
        # Extract spatial info from Kalman filter state
        # mean = [cx, cy, w, h, vx, vy]
        if hasattr(track, 'mean') and track.mean is not None:
            cx, cy = track.mean[0], track.mean[1]
            w, h = track.mean[2], track.mean[3]
            summary.mean_center = np.array([cx, cy])
            summary.median_center = np.array([cx, cy])  # Use current as median approximation
            summary.mean_bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        # Extract snapshot
        if best_snapshot_data is not None:
            snap_frame = best_snapshot_data.get('frame')
            snap_bbox = best_snapshot_data.get('bbox')
            
            if snap_frame is not None and snap_bbox is not None:
                # Crop door region
                x1, y1, x2, y2 = [int(v) for v in snap_bbox]
                h_frame, w_frame = snap_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                
                if x2 > x1 and y2 > y1:
                    summary.snapshot = snap_frame[y1:y2, x1:x2].copy()
                    summary.snapshot_bbox = np.array([x1, y1, x2, y2])
                    
                    # Compute context histogram
                    context_crop = self.context_checker.get_expanded_crop(
                        snap_frame, snap_bbox
                    )
                    if context_crop is not None and context_crop.size > 0:
                        summary.context_histogram = self.context_checker.compute_histogram(
                            context_crop
                        )
        
        return summary
    
    # -------------------------------------------------------------------------
    # MERGE EXECUTION
    # -------------------------------------------------------------------------
    
    def merge_track_pair(
        self,
        summary1: TrackSummary,
        summary2: TrackSummary,
        snapshot_data1: Optional[Dict],
        snapshot_data2: Optional[Dict]
    ) -> Tuple[int, int, Dict]:
        """
        Merge two tracks, selecting canonical ID and best snapshot.
        
        Returns:
            (canonical_id, removed_id, merged_snapshot_data)
        """
        # Canonical = older track (lower ID)
        if summary1.track_id < summary2.track_id:
            canonical, removed = summary1, summary2
            canonical_snap, removed_snap = snapshot_data1, snapshot_data2
        else:
            canonical, removed = summary2, summary1
            canonical_snap, removed_snap = snapshot_data2, snapshot_data1
        
        # Select best snapshot based on score
        merged_snap = canonical_snap
        if canonical_snap is not None and removed_snap is not None:
            can_score = canonical_snap.get('score', 0.0)
            rem_score = removed_snap.get('score', 0.0)
            if rem_score > can_score:
                merged_snap = removed_snap
        elif removed_snap is not None:
            merged_snap = removed_snap
        
        # Update merged snapshot with canonical's state info
        if merged_snap is not None:
            # Use the more confident state, preferring OPEN if either has it
            if canonical.open_event_raised or removed.open_event_raised:
                merged_snap['state'] = 'OPEN'
            elif 'open' in canonical.final_state.lower():
                merged_snap['state'] = canonical.final_state
            elif 'open' in removed.final_state.lower():
                merged_snap['state'] = removed.final_state
            else:
                # Use higher confidence state
                if canonical.confidence >= removed.confidence:
                    merged_snap['state'] = canonical.final_state
                else:
                    merged_snap['state'] = removed.final_state
            
            # Use max confidence
            merged_snap['confidence'] = max(canonical.confidence, removed.confidence)
        
        return canonical.track_id, removed.track_id, merged_snap
    
    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------
    
    def merge_all_tracks(
        self,
        tracks: List,  # List[DoorTrack]
        final_states: Dict[int, Dict],
        best_snapshots: Dict[int, Dict],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Dict, Dict[int, Dict]]:
        """
        Merge all fragmented tracks that represent the same physical door.
        
        This is the main entry point called from door_processor.py.
        
        Args:
            tracks: All tracks (active + deleted)
            final_states: Dict from get_final_door_states()
            best_snapshots: Dict mapping track_id to snapshot data
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            (merge_result, updated_snapshots)
            
            merge_result contains:
            - 'removed_ids': List of track IDs that were merged into others
            - 'merge_map': Dict mapping removed_id -> canonical_id
            - 'merge_count': Number of merges performed
            
            updated_snapshots: Updated best_snapshots dict
        """
        if len(tracks) < 2:
            return {'removed_ids': [], 'merge_map': {}, 'merge_count': 0}, best_snapshots
        
        # Extract summaries for all tracks
        summaries: Dict[int, TrackSummary] = {}
        for track in tracks:
            if track.track_id in final_states:
                summary = self.extract_track_summary(
                    track,
                    final_states[track.track_id],
                    best_snapshots.get(track.track_id)
                )
                summaries[track.track_id] = summary
        
        if len(summaries) < 2:
            return {'removed_ids': [], 'merge_map': {}, 'merge_count': 0}, best_snapshots
        
        # Find all merge pairs
        merge_pairs: List[Tuple[int, int, float]] = []
        track_ids = sorted(summaries.keys())
        
        for i, id1 in enumerate(track_ids):
            for id2 in track_ids[i+1:]:
                should, score, breakdown = self.should_merge(
                    summaries[id1],
                    summaries[id2],
                    frame_width,
                    frame_height
                )
                
                if should:
                    merge_pairs.append((id1, id2, score))
        
        if not merge_pairs:
            return {'removed_ids': [], 'merge_map': {}, 'merge_count': 0}, best_snapshots
        
        # Sort by score (highest first) for greedy merging
        merge_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy merge with union-find for transitive closure
        parent: Dict[int, int] = {tid: tid for tid in track_ids}
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                # Canonical = lower ID
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py
        
        # Apply merges
        for id1, id2, score in merge_pairs:
            union(id1, id2)
        
        # Build merge result
        merge_map: Dict[int, int] = {}
        removed_ids: List[int] = []
        
        for tid in track_ids:
            canonical = find(tid)
            if canonical != tid:
                merge_map[tid] = canonical
                removed_ids.append(tid)
        
        # Update snapshots
        updated_snapshots = best_snapshots.copy()
        
        for removed_id, canonical_id in merge_map.items():
            if removed_id in summaries and canonical_id in summaries:
                _, _, merged_snap = self.merge_track_pair(
                    summaries[canonical_id],
                    summaries[removed_id],
                    updated_snapshots.get(canonical_id),
                    updated_snapshots.get(removed_id)
                )
                
                # Update canonical's snapshot
                if merged_snap is not None:
                    updated_snapshots[canonical_id] = merged_snap
                
                # Remove merged track's snapshot
                if removed_id in updated_snapshots:
                    del updated_snapshots[removed_id]
        
        merge_result = {
            'removed_ids': removed_ids,
            'merge_map': merge_map,
            'merge_count': len(removed_ids)
        }
        
        if removed_ids:
            print(f"Door Identity Merger: Merged {len(removed_ids)} duplicate track(s)")
            for removed_id, canonical_id in merge_map.items():
                print(f"  Track {removed_id} -> Track {canonical_id}")
        
        return merge_result, updated_snapshots


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_merge_config() -> MergeConfig:
    """Get default merge configuration (already conservative)."""
    return MergeConfig()


def get_conservative_merge_config() -> MergeConfig:
    """Get ultra-conservative merge configuration - almost never merges."""
    return MergeConfig(
        max_spatial_distance=0.05,      # Very close only
        min_spatial_similarity=0.7,     # High spatial requirement
        max_temporal_gap=15,            # Small gap only
        merge_threshold=0.90,           # Very high threshold
    )


def get_moderate_merge_config() -> MergeConfig:
    """Get moderate merge configuration - balanced approach."""
    return MergeConfig(
        max_spatial_distance=0.10,
        min_spatial_similarity=0.4,
        max_temporal_gap=45,
        merge_threshold=0.75,
    )

