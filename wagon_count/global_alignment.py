"""
global_alignment.py  --  Phase-1 cross-camera gap fusion (standalone)
=====================================================================

This is the standalone Phase-1 version of the alignment module: it carries
ONLY the gap-level fusion logic.  The legacy v3 functions that depend on
RIGHT_UP/train_session.CameraEvidence have been removed -- they belong to
the production reporting pipeline, which is out of scope for this package.

Pipeline:
    1) match_support_to_master       support gap -> closest master gap
    2) cluster_unmatched_supports    group leftover support gaps in time
    3) decide_inserted_gaps          accept clusters with quorum + confidence
    4) fuse_master_timeline          combine real master gaps + accepted inserts
    5) build_global_wagons           emit GW_1..GW_N with inherited classification
    6) assemble_global_train_state   end-to-end with deterministic fallback

Determinism: deterministic sort keys throughout; no randomness; same input
yields the same output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from global_train_state import (
    GapEvent,
    LocalCameraTracks,
    SegmentClass,
    GlobalWagon,
    GlobalTrainState,
    GapCorrection,
    _MasterClassification,
    MASTER_CAMERA,
    ALL_CAMERAS,
)


# -----------------------------------------------------------------------------
# Core temporal-IoU math (kept here so this module has no external deps
# beyond global_train_state).
# -----------------------------------------------------------------------------

def compute_temporal_iou(
    a_start: float, a_end: float,
    b_start: float, b_end: float,
) -> Tuple[float, float]:
    """Return (IoU, overlap_seconds) of two closed intervals."""
    if a_end <= a_start or b_end <= b_start:
        return 0.0, 0.0
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    overlap = max(0.0, inter_end - inter_start)
    if overlap <= 0.0:
        return 0.0, 0.0
    union = (a_end - a_start) + (b_end - b_start) - overlap
    if union <= 0.0:
        return 0.0, 0.0
    return overlap / union, overlap


# -----------------------------------------------------------------------------
# Tuning knobs
# -----------------------------------------------------------------------------

PHASE1_DEFAULTS = {
    # Matching a support gap to a master gap
    "match_time_window_sec": 1.0,
    "match_min_iou": 0.2,

    # Inserting a missed master gap (gap-recovery quorum)
    "insert_min_support": 2,
    "insert_max_spread_sec": 1.5,
    "insert_min_confidence": 0.4,
    "insert_min_distance_to_master_sec": 1.0,
}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _gap_to_interval(g: GapEvent) -> Tuple[float, float, float]:
    return g.start_time, g.end_time, g.center_time


def _interval_iou(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    iou, _ = compute_temporal_iou(a_s, a_e, b_s, b_e)
    return iou


# -----------------------------------------------------------------------------
# Step A -- match each support gap to a master gap
# -----------------------------------------------------------------------------

def match_support_to_master(
    master_gaps: List[GapEvent],
    support_gaps: List[GapEvent],
    match_time_window_sec: float = 1.0,
    match_min_iou: float = 0.2,
) -> Tuple[Dict[int, int], List[GapEvent]]:
    """Match each support gap to its best master gap.

    Returns
    -------
    matched : dict   support.track_id -> master.track_id
    leftover : list  support gaps that could not be matched
    """
    matched: Dict[int, int] = {}
    leftover: List[GapEvent] = []
    if not master_gaps:
        return matched, list(support_gaps)

    m_intervals = [(g.track_id, g.start_time, g.end_time, g.center_time) for g in master_gaps]

    for sg in support_gaps:
        best_score = -1.0
        best_master_id = -1
        s_s, s_e, s_c = _gap_to_interval(sg)
        for (m_id, m_s, m_e, m_c) in m_intervals:
            iou = _interval_iou(s_s, s_e, m_s, m_e)
            dt = abs(s_c - m_c)
            time_score = max(0.0, 1.0 - dt / max(match_time_window_sec, 1e-3))
            score = max(iou, time_score) if (iou >= match_min_iou or dt <= match_time_window_sec) else -1.0
            if score > best_score:
                best_score = score
                best_master_id = m_id

        if best_master_id >= 0 and best_score >= 0.0:
            matched[sg.track_id] = best_master_id
        else:
            leftover.append(sg)

    return matched, leftover


# -----------------------------------------------------------------------------
# Step B -- cluster unmatched supports across cameras
# -----------------------------------------------------------------------------

def cluster_unmatched_supports(
    leftovers_per_camera: Dict[str, List[GapEvent]],
    spread_sec: float,
) -> List[List[GapEvent]]:
    """Sweep over the union of leftover gaps sorted by center_time."""
    all_gaps: List[GapEvent] = []
    for cam, gs in leftovers_per_camera.items():
        all_gaps.extend(gs)
    if not all_gaps:
        return []
    all_gaps.sort(key=lambda g: (g.center_time, g.camera_id, g.track_id))

    clusters: List[List[GapEvent]] = []
    current: List[GapEvent] = [all_gaps[0]]
    cluster_center = all_gaps[0].center_time

    for g in all_gaps[1:]:
        if abs(g.center_time - cluster_center) <= spread_sec:
            current.append(g)
            cluster_center = sum(x.center_time for x in current) / len(current)
        else:
            clusters.append(current)
            current = [g]
            cluster_center = g.center_time
    clusters.append(current)
    return clusters


# -----------------------------------------------------------------------------
# Step C -- decide which clusters become inserted master gaps
# -----------------------------------------------------------------------------

def decide_inserted_gaps(
    clusters: List[List[GapEvent]],
    master_gaps: List[GapEvent],
    *,
    min_support: int,
    max_spread_sec: float,
    min_confidence: float,
    min_distance_to_master_sec: float,
    master_fps: float,
) -> List[GapCorrection]:
    inserted: List[GapCorrection] = []
    if not clusters:
        return inserted

    master_centers = [g.center_time for g in master_gaps]

    for cluster in clusters:
        cams = {g.camera_id for g in cluster}
        if len(cams) < min_support:
            continue
        centers = sorted(g.center_time for g in cluster)
        spread = centers[-1] - centers[0]
        if spread > max_spread_sec:
            continue
        mean_conf = float(sum(g.confidence for g in cluster) / len(cluster))
        if mean_conf < min_confidence:
            continue
        center = sum(centers) / len(centers)
        if master_centers and min(abs(center - mc) for mc in master_centers) < min_distance_to_master_sec:
            continue

        inserted.append(GapCorrection(
            inserted_at_master_time=center,
            inserted_at_master_frame=int(round(center * master_fps)),
            supporting_cameras=sorted(cams),
            mean_confidence=mean_conf,
            time_spread_sec=spread,
            contributing_track_ids={g.camera_id: g.track_id for g in cluster},
        ))

    inserted.sort(key=lambda c: c.inserted_at_master_time)
    return inserted


# -----------------------------------------------------------------------------
# Step D -- fuse the corrected master gap list
# -----------------------------------------------------------------------------

def fuse_master_timeline(
    master_tracks: LocalCameraTracks,
    support_tracks: List[LocalCameraTracks],
    *,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[List[GapEvent], List[GapCorrection], Dict[str, List[GapEvent]]]:
    cfg = dict(PHASE1_DEFAULTS)
    if config:
        cfg.update(config)

    master_gaps = list(master_tracks.gaps)
    master_fps = master_tracks.fps

    leftovers_per_cam: Dict[str, List[GapEvent]] = {}
    for st in support_tracks:
        matched, leftover = match_support_to_master(
            master_gaps, st.gaps,
            match_time_window_sec=cfg["match_time_window_sec"],
            match_min_iou=cfg["match_min_iou"],
        )
        leftovers_per_cam[st.camera_id] = leftover
        if verbose:
            print(f"[FUSE/{st.camera_id}] matched={len(matched)}  leftover={len(leftover)}  "
                  f"(of {len(st.gaps)} support gaps)")

    clusters = cluster_unmatched_supports(leftovers_per_cam, spread_sec=cfg["insert_max_spread_sec"])
    if verbose:
        print(f"[FUSE] {len(clusters)} cross-camera cluster(s) of unmatched support gaps")

    inserts = decide_inserted_gaps(
        clusters,
        master_gaps,
        min_support=cfg["insert_min_support"],
        max_spread_sec=cfg["insert_max_spread_sec"],
        min_confidence=cfg["insert_min_confidence"],
        min_distance_to_master_sec=cfg["insert_min_distance_to_master_sec"],
        master_fps=master_fps,
    )
    if verbose:
        print(f"[FUSE] {len(inserts)} gap(s) will be inserted into master timeline")
        for c in inserts:
            print(f"   + insert @ t={c.inserted_at_master_time:.2f}s  "
                  f"f={c.inserted_at_master_frame}  "
                  f"supports={'/'.join(c.supporting_cameras)}  "
                  f"conf={c.mean_confidence:.2f}  spread={c.time_spread_sec:.2f}s")

    next_synth_id = -1
    synth_gaps: List[GapEvent] = []
    for c in inserts:
        f = c.inserted_at_master_frame
        synth_gaps.append(GapEvent(
            track_id=next_synth_id,
            camera_id=f"FUSED({'+'.join(c.supporting_cameras)})",
            start_frame=max(0, f - 1),
            end_frame=f + 1,
            confidence=c.mean_confidence,
            hit_count=len(c.contributing_track_ids),
            center_x_trajectory=[],
            fps=master_fps,
            temporal_consistency_score=1.0,
            class_label="gap_inserted",
        ))
        next_synth_id -= 1

    fused = sorted(master_gaps + synth_gaps, key=lambda g: g.center_time)
    return fused, inserts, leftovers_per_cam


# -----------------------------------------------------------------------------
# Step E -- rebuild GlobalWagons, inheriting RIGHT_UP classification
# -----------------------------------------------------------------------------

def build_global_wagons(
    fused_gaps: List[GapEvent],
    *,
    master_total_frames: int,
    master_fps: float,
    initial_classifications: List[_MasterClassification],
    support_camera_ids: List[str],
    master_camera_id: str = MASTER_CAMERA,
) -> List[GlobalWagon]:
    if master_total_frames <= 0:
        return []

    boundaries: List[int] = []
    for g in fused_gaps:
        f = int(round(g.center_frame))
        f = max(0, min(master_total_frames - 1, f))
        boundaries.append(f)
    boundaries.sort()

    def label_for_frame(frame_idx: int) -> Tuple[str, float]:
        for c in initial_classifications:
            if c.start_frame <= frame_idx <= c.end_frame:
                return c.label, c.confidence
        if not initial_classifications:
            return SegmentClass.UNKNOWN, 0.0
        nearest = min(initial_classifications,
                      key=lambda c: min(abs(c.start_frame - frame_idx),
                                        abs(c.end_frame - frame_idx)))
        return nearest.label, nearest.confidence

    segs: List[Tuple[int, int]] = []
    prev = 0
    for b in boundaries:
        if b <= prev:
            continue
        segs.append((prev, b - 1))
        prev = b
    if prev <= master_total_frames - 1:
        segs.append((prev, master_total_frames - 1))

    # --- Startup false-engine guard: delay initialization -------------------
    # NEVER assume the first detected segment is the ENGINE.  If the LEADING
    # segment classifies as UNKNOWN -- i.e. the model gave no stable, confident
    # evidence (low-confidence loco-front, bare track, or background as the
    # train enters frame; an uncertain 'engine' read is demoted to UNKNOWN
    # upstream in tracker_engine._label_to_class) -- it is a phantom leading
    # wagon.  Drop ONLY that single leading segment so GW_1 re-bases onto the
    # first stably-classified real wagon.  We never drop a real
    # WAGON/ENGINE/BRAKE_VAN segment, so real wagons are never removed and
    # never renumbered relative to each other -- only the phantom disappears.
    # Skipped entirely when classification was unavailable (would label
    # everything UNKNOWN) or when only one segment exists (never empty a train).
    if len(segs) > 1 and initial_classifications:
        lead_sf, lead_ef = segs[0]
        lead_label, _lead_conf = label_for_frame((lead_sf + lead_ef) // 2)
        if lead_label == SegmentClass.UNKNOWN:
            segs = segs[1:]

    wagons: List[GlobalWagon] = []
    fused_sorted = sorted(fused_gaps, key=lambda g: g.center_frame)
    for i, (sf, ef) in enumerate(segs, start=1):
        center_frame = (sf + ef) // 2
        label, conf = label_for_frame(center_frame)
        gw = GlobalWagon(
            global_id=f"GW_{i}",
            wagon_index=i,
            start_frame_master=sf,
            end_frame_master=ef,
            start_time=sf / master_fps if master_fps > 0 else 0.0,
            end_time=(ef + 1) / master_fps if master_fps > 0 else 0.0,
            classification=label,
            classification_confidence=conf,
            supporting_cameras=[master_camera_id]
            + [c for c in support_camera_ids if c != master_camera_id],
        )

        leading = None
        trailing = None
        for g in fused_sorted:
            cf = int(round(g.center_frame))
            # Gap whose center is at or before sf IS the leading boundary
            # of this segment.  Strict `<` would miss the boundary-frame case.
            if cf <= sf:
                leading = g
            elif cf > ef and trailing is None:
                trailing = g
                break
        if leading is not None:
            gw.leading_gap = {
                "source": "master" if leading.track_id > 0 else "fused",
                "camera_id": leading.camera_id,
                "track_id": leading.track_id,
                "center_time": round(leading.center_time, 4),
            }
        else:
            gw.leading_gap = {"source": "video_start"}
        if trailing is not None:
            gw.trailing_gap = {
                "source": "master" if trailing.track_id > 0 else "fused",
                "camera_id": trailing.camera_id,
                "track_id": trailing.track_id,
                "center_time": round(trailing.center_time, 4),
            }
        else:
            gw.trailing_gap = {"source": "video_end"}

        if (leading is not None and leading.track_id < 0) or \
           (trailing is not None and trailing.track_id < 0):
            parent_idx = next(
                (c.segment_index for c in initial_classifications
                 if c.start_frame <= sf <= c.end_frame),
                None,
            )
            if parent_idx is not None:
                gw.split_from_global_id = f"PRE_SEG_{parent_idx}"

        wagons.append(gw)

    return wagons


# -----------------------------------------------------------------------------
# Step F -- pure-master fallback
# -----------------------------------------------------------------------------

def build_wagons_pure_master(
    master_tracks: LocalCameraTracks,
    initial_classifications: List[_MasterClassification],
) -> List[GlobalWagon]:
    fused = sorted(master_tracks.gaps, key=lambda g: g.center_time)
    return build_global_wagons(
        fused,
        master_total_frames=master_tracks.total_frames,
        master_fps=master_tracks.fps,
        initial_classifications=initial_classifications,
        support_camera_ids=[master_tracks.camera_id],
        master_camera_id=master_tracks.camera_id,
    )


# -----------------------------------------------------------------------------
# Step G -- end-to-end
# -----------------------------------------------------------------------------

def assemble_global_train_state(
    *,
    master_tracks: LocalCameraTracks,
    support_tracks: List[LocalCameraTracks],
    initial_classifications: List[_MasterClassification],
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> GlobalTrainState:
    cfg = dict(PHASE1_DEFAULTS)
    if config:
        cfg.update(config)

    per_local_counts: Dict[str, int] = {master_tracks.camera_id: master_tracks.local_wagon_count}
    per_gap_counts: Dict[str, int] = {master_tracks.camera_id: len(master_tracks.gaps)}
    per_status: Dict[str, str] = {master_tracks.camera_id: "ok"}
    for st in support_tracks:
        per_local_counts[st.camera_id] = st.local_wagon_count
        per_gap_counts[st.camera_id] = len(st.gaps)
        per_status[st.camera_id] = "ok"

    if verbose:
        print(f"[FUSE] master({master_tracks.camera_id}) wagons={master_tracks.local_wagon_count} "
              f"gaps={len(master_tracks.gaps)}")
        for st in support_tracks:
            print(f"[FUSE] support({st.camera_id}) wagons={st.local_wagon_count} "
                  f"gaps={len(st.gaps)}")

    fallback_used = False
    fallback_reason = ""

    try:
        fused_gaps, corrections, _leftovers = fuse_master_timeline(
            master_tracks, support_tracks, config=cfg, verbose=verbose,
        )
        wagons = build_global_wagons(
            fused_gaps,
            master_total_frames=master_tracks.total_frames,
            master_fps=master_tracks.fps,
            initial_classifications=initial_classifications,
            support_camera_ids=[st.camera_id for st in support_tracks],
            master_camera_id=master_tracks.camera_id,
        )
    except Exception as e:
        fallback_used = True
        fallback_reason = f"fusion error: {type(e).__name__}: {e}"
        if verbose:
            print(f"[FUSE] {fallback_reason} -- falling back to pure RIGHT_UP")
        corrections = []
        wagons = build_wagons_pure_master(master_tracks, initial_classifications)

    if not wagons:
        fallback_used = True
        if not fallback_reason:
            fallback_reason = "no wagons produced; using pure RIGHT_UP build"
        wagons = build_wagons_pure_master(master_tracks, initial_classifications)
        corrections = []

    state = GlobalTrainState(
        total_wagons=len(wagons),
        wagons=wagons,
        master_camera=master_tracks.camera_id,
        master_fps=master_tracks.fps,
        master_total_frames=master_tracks.total_frames,
        per_camera_local_counts=per_local_counts,
        per_camera_gap_counts=per_gap_counts,
        per_camera_status=per_status,
        corrections_applied=corrections,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    return state
