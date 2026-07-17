"""Door feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Per wagon, for each side camera (RIGHT_UP / LEFT_UP):

    1. Iterate cached JPEGs in wagon_cache/<GW_n>/<camera>/.
    2. Score illumination quality per frame (legacy IlluminationProcessor).
    3. Run YOLO door_state.pt on the raw frame.
    4. Filter detections through the geometric shape prior (aspect ratio,
       vertical-edge dominance, border completeness).
    5. Feed surviving detections + quality score into the legacy
       DoorTracker (Kalman + Hungarian + per-track 30-frame quality-
       weighted majority vote + state machine with 2x hysteresis on
       OPEN -> CLOSED transitions + sticky DAMAGE state).
    6. After all frames, finalize the tracker -> per-track {state,
       confidence, snapshot}.
    7. Run DoorIdentityMerger to collapse fragmented tracks of the same
       physical door (spatial + temporal + context + structural).
    8. Pick the dominant door state per CAMERA SIDE.

The per-CAMERA dominant state IS the per-side door state (RIGHT_UP -> right
door, LEFT_UP -> left door).  Same convention the legacy combined report
used.

Output JSON shape (per wagon):
    {
        "global_id":   "GW_7",
        "feature":     "door",
        "status":      "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "left_door":   "CLOSED" | "OPEN" | "PARTIAL" | "DAMAGED" | "NO_DATA",
        "left_door_confidence":  0.91,
        "right_door":  "...",
        "right_door_confidence": 0.83,
        "tracks": [
            {camera_id, track_id, state, confidence, first_frame,
             last_frame, total_hits},
            ...
        ],
        "supporting_cameras": ["LEFT_UP", "RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.frame_quality import (
    detection_quality, snapshot_score, expand_bbox, _DOOR_BBOX_EXPAND_FRAC,
)

from features._common import (
    load_yolo, iter_wagon_frames, list_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer, feature_camera_dir,
    DEVICE, HALF,
)

# Mature intelligence ported from legacy
from features.inference_lib.door_tracker import (
    DoorTracker, TrackerConfig, DoorState, yolo_to_detections,
)
from features.inference_lib.door_identity_merger import (
    DoorIdentityMerger, MergeConfig,
)
from features.inference_lib.illumination_processor import (
    IlluminationProcessor, IlluminationConfig,
)
from features.inference_lib.geometric_shape_prior import (
    GeometricShapePrior, GeometricPriorConfig,
)
from features._evidence import (
    BestFrameTracker, atomic_camera_evidence,
    save_jpeg, safe_crop, write_metadata, draw_annotated_bbox,
)


FEATURE_NAME = "door"


# -----------------------------------------------------------------------------
# Canonicalization of legacy state strings
# -----------------------------------------------------------------------------

# DoorTracker.DoorState values are: UNKNOWN / CLOSED / OPEN / PARTIAL_CLOSED /
# DAMAGE / OTHER.  Map to the v4 canonical vocabulary.
_STATE_TO_CANONICAL = {
    "OPEN":            C.DOOR_OPEN,
    "CLOSED":          C.DOOR_CLOSED,
    "PARTIAL_CLOSED":  C.DOOR_PARTIAL,
    "PARTIAL":         C.DOOR_PARTIAL,
    "DAMAGE":          C.DOOR_DAMAGED,
    "DAMAGED":         C.DOOR_DAMAGED,
    "OTHER":           C.NO_DATA,
    "UNKNOWN":         C.NO_DATA,
}


def _canonical(state_value: str) -> str:
    s = str(state_value or "").strip().upper()
    return _STATE_TO_CANONICAL.get(s, s if s else C.NO_DATA)


# -----------------------------------------------------------------------------
# Per-camera tracker run
# -----------------------------------------------------------------------------

def _run_tracker_one_camera(
    yolo_model,
    illumination: IlluminationProcessor,
    geo_prior: GeometricShapePrior,
    tracker_config: TrackerConfig,
    merger_config: MergeConfig,
    cache_root: str,
    gw_id: str,
    camera_id: str,
) -> Tuple[List[Dict[str, Any]], int, int, int, Dict[str, "BestFrameTracker"]]:
    """Run the full per-camera door pipeline on one wagon.

    Returns:
        (track_decisions, n_frames, width, height, evidence_candidates)

    ``evidence_candidates`` is a ``{canonical_state -> BestFrameTracker}`` map.
    For each door state observed on this camera we keep the single highest
    snapshot-quality frame (legacy ``_score_detection``: area + horizontal
    centre + confidence + crop quality, with an edge-hugging penalty).  The
    caller picks the bucket matching the wagon's reported side-state so the
    persisted snapshot actually shows that (often anomalous) state, falling
    back to the globally best-scored frame when no such frame exists.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True)
    if not paths:
        return [], 0, 0, 0, {}, {"tracks": [], "events": []}

    # Fresh tracker per (gw, camera).  Wagons are independent in the new
    # train-state-native world, so each one resets the tracker.
    tracker = DoorTracker(config=tracker_config)
    tracker.reset()

    frame_w, frame_h = 0, 0
    used = 0
    cands: Dict[str, BestFrameTracker] = {}

    # Per-frame confirmed-track positions for the Stage-4b overlay.  We record
    # the Kalman-smoothed tlbr + FSM state of every CONFIRMED track after each
    # tracker step -- including frames where the door was only predicted (no
    # detection) -- exactly mirroring how the legacy door_processor wrote its
    # `_tracked.mp4` (draws `track.tlbr` for confirmed tracks every frame, so
    # boxes glide smoothly through detection-less frames).  Persisting it lets
    # the visualization-only renderer replay the motion WITHOUT re-running any
    # detector.  Keyed by track_id.
    trajectory: Dict[int, Dict[str, Any]] = {}
    # Ordered absolute cache frame indices, one entry per tracker.update() step.
    # DoorTracker numbers its events with an INTERNAL 1-based step counter
    # (self.frame_idx, ++ per update), but the renderer keys the event banner by
    # ABSOLUTE cache frame index.  _snapshot_confirmed runs exactly once per
    # tracker.update(), so recording `fi` here builds the step->absolute map used
    # to translate event frames at finalize -- otherwise the banner fires on the
    # wrong frame / a neighbouring wagon's span.
    step_to_abs: List[int] = []

    def _snapshot_confirmed(frame_index: int) -> None:
        step_to_abs.append(int(frame_index))
        for t in tracker.tracks:
            if not t.is_confirmed():
                continue
            try:
                bb = [float(v) for v in t.tlbr]
            except Exception:
                continue
            # ITEM 4: expand the persisted overlay box so the processed-video
            # rectangle visually contains the WHOLE door (matches the expanded
            # evidence crop below).  Clipped to the frame; still a clean rect.
            bb = expand_bbox(bb, _DOOR_BBOX_EXPAND_FRAC, frame_w, frame_h)
            try:
                vel = [float(t.velocity[0]), float(t.velocity[1])]
            except Exception:
                vel = [0.0, 0.0]
            # Persist the RAW legacy fields the overlay needs to reproduce the
            # exact legacy door annotation: the raw DoorState value (for colour
            # + label), last_class (UNKNOWN colour/label fallback), the raw
            # last-frame confidence, and the velocity vector (arrow).
            entry = trajectory.setdefault(int(t.track_id), {
                "camera_id": camera_id,
                "track_id":  int(t.track_id),
                "frames":    [],
            })
            entry["frames"].append({
                "frame_idx":  int(frame_index),
                "bbox":       bb,
                "state_raw":  str(t.state_machine.get_state().value),
                "last_class": str(getattr(t, "last_class", "") or ""),
                "confidence": float(getattr(t, "last_confidence", 0.0) or 0.0),
                "velocity":   vel,
            })

    # ------- frame loop (stable interior only) -------
    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True):
        if frame_w == 0:
            frame_h, frame_w = frame.shape[:2]
        used += 1

        # 1) quality score (does NOT alter the frame; YOLO sees raw bytes,
        #    matching the legacy pipeline).
        try:
            ill_res = illumination.process_frame(frame, frame_idx=fi)
            quality = float(getattr(ill_res, "quality_score", 1.0))
        except Exception:
            quality = 1.0

        # 2) YOLO detection on raw frame
        # half/device from the process-resolved DEVICE: on GPU this stays
        # half=True (identical to pre-migration); on CPU it drops to FP32.
        try:
            results = yolo_model(frame, verbose=False, half=HALF, device=DEVICE)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            _snapshot_confirmed(fi)
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy().astype(int)

        # 3) confidence floor (the tracker has its own per-class thresholds
        #    too; this gate just discards obviously-noisy detections early).
        min_conf = float(tracker_config.closed_confidence_threshold)
        keep = confs >= min_conf
        boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
        if len(boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            _snapshot_confirmed(fi)
            continue

        # 4) geometric prior filter (aspect ratio / vertical-edge / border)
        try:
            boxes, confs, clss, _idx = geo_prior.filter_detections(
                frame, boxes, confs, clss,
            )
        except Exception:
            pass

        # 5) convert to Detection objects + feed tracker
        names = getattr(yolo_model, "names", {}) or {}
        detections = yolo_to_detections(
            boxes=boxes, confidences=confs, class_ids=clss,
            class_names=names, illumination_quality=quality,
        )
        tracker.update(
            detections=detections,
            frame=frame,
            frame_width=frame_w,
            frame_height=frame_h,
        )
        _snapshot_confirmed(fi)

        # 6) evidence:  bucket each surviving detection by its canonical state
        # and keep the single highest snapshot-QUALITY frame per state (legacy
        # _score_detection -- area + horizontal centre + confidence + crop
        # quality, edge-hugging penalty).  This is what makes the persisted
        # snapshot sharp / centred / non-edge instead of merely high-confidence.
        for bbox, conf, cls_id in zip(boxes, confs, clss):
            cls_name = str(names.get(int(cls_id), "")).lower()
            canon = _canonical(cls_name)
            bbox_list = [float(bbox[0]), float(bbox[1]),
                         float(bbox[2]), float(bbox[3])]
            # Score on the RAW detection box (true area / centre / quality).
            crop_q = detection_quality(frame, bbox_list)
            sc = snapshot_score(bbox_list, float(conf), crop_q,
                                frame_w, frame_h)
            # ITEM 4: persist an EXPANDED box so the evidence crop + annotated
            # frame + metadata bbox visually contain the WHOLE door, consistent
            # with the expanded overlay box.
            bbox_store = expand_bbox(bbox_list, _DOOR_BBOX_EXPAND_FRAC,
                                     frame_w, frame_h)
            bucket = cands.setdefault(canon, BestFrameTracker())
            if sc > bucket.score:
                bucket.update(
                    score=sc, frame=frame, bbox=bbox_store, frame_idx=fi,
                    state=canon, confidence=float(conf),
                    raw_class=cls_name, quality=float(crop_q),
                )

    # ------- finalize -------
    # Bundle the per-frame trajectory + door-level events for the overlay.
    def _abs_event_frame(rel: Any) -> int:
        # DoorTracker numbers events with a 1-based internal step counter; step k
        # corresponds to step_to_abs[k-1] (the absolute cache frame index).
        try:
            k = int(rel)
        except (TypeError, ValueError):
            return -1
        if 1 <= k <= len(step_to_abs):
            return step_to_abs[k - 1]
        return -1

    overlay = {
        "tracks": list(trajectory.values()),
        "events": [
            {"frame_idx": _abs_event_frame(e.get("frame_idx", -1)),
             "event":     str(e.get("event", "")),
             "track_id":  int(e.get("track_id", -1)),
             "camera_id": camera_id}
            for e in (tracker.get_events() or [])
        ],
    }
    final_states = tracker.get_final_door_states()
    if not final_states:
        return [], used, frame_w, frame_h, cands, overlay

    # Run identity merger on the final track set (collapses fragmented IDs
    # of the same physical door).  Operates on the live + deleted track
    # objects exposed by the tracker.
    try:
        merger = DoorIdentityMerger(config=merger_config)
        all_tracks_objs = list(tracker.tracks) + list(tracker.deleted_tracks)
        merged_groups = merger.merge_all_tracks(all_tracks_objs)
        # merge_all_tracks returns mapping {canonical_id: [member_ids]};
        # we keep the canonical id for each group as the surviving track.
        if isinstance(merged_groups, dict) and merged_groups:
            merged_ids = set(merged_groups.keys())
        elif isinstance(merged_groups, list) and merged_groups:
            merged_ids = set(merged_groups)
        else:
            merged_ids = set(final_states.keys())
    except Exception:
        merged_ids = set(final_states.keys())   # fallback: keep everything

    decisions: List[Dict[str, Any]] = []
    all_tracks = list(tracker.tracks) + list(tracker.deleted_tracks)
    by_id = {t.track_id: t for t in all_tracks}

    for tid, state_dict in final_states.items():
        if merged_ids and tid not in merged_ids:
            continue
        tr = by_id.get(tid)
        mean_cx = float(np.mean([d['bbox'][[0,2]].mean()
                                 for d in (tr.detections if tr else [])])) \
            if (tr and getattr(tr, "detections", None)) else 0.0
        decisions.append({
            "camera_id":   camera_id,
            "track_id":    tid,
            "state":       _canonical(state_dict.get("state")),
            "confidence":  float(state_dict.get("confidence", 0.0) or 0.0),
            "first_frame": int(state_dict.get("first_frame", 0)),
            "last_frame":  int(state_dict.get("last_frame", 0)),
            "total_hits":  int(state_dict.get("total_hits", 0)),
            "mean_center_x": mean_cx,
        })
    return decisions, used, frame_w, frame_h, cands, overlay


# -----------------------------------------------------------------------------
# Per-side decision picker
# -----------------------------------------------------------------------------

def _pick_side_state(track_decisions: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Pick the dominant door state for one camera/side.

    Priority order:
        1. Any DAMAGED track  -> DAMAGED (terminal in the FSM)
        2. Any OPEN track     -> OPEN  (safety-critical; legacy code biases here)
        3. Any PARTIAL track  -> PARTIAL
        4. Most-frequent CLOSED-class result by total_hits, confidence weighted
        5. NO_DATA
    """
    if not track_decisions:
        return C.NO_DATA, 0.0

    def _max_conf(items):
        return max(items, key=lambda d: (d["total_hits"], d["confidence"]))

    damaged = [d for d in track_decisions if d["state"] == C.DOOR_DAMAGED]
    if damaged:
        best = _max_conf(damaged)
        return C.DOOR_DAMAGED, best["confidence"]

    opens = [d for d in track_decisions if d["state"] == C.DOOR_OPEN]
    if opens:
        best = _max_conf(opens)
        return C.DOOR_OPEN, best["confidence"]

    partials = [d for d in track_decisions if d["state"] == C.DOOR_PARTIAL]
    if partials:
        best = _max_conf(partials)
        return C.DOOR_PARTIAL, best["confidence"]

    closeds = [d for d in track_decisions if d["state"] == C.DOOR_CLOSED]
    if closeds:
        best = _max_conf(closeds)
        return C.DOOR_CLOSED, best["confidence"]

    return C.NO_DATA, 0.0


def _resolve_evidence(
    cands: Dict[str, "BestFrameTracker"], reported_state: str,
) -> "BestFrameTracker":
    """Pick the evidence frame for one side.

    Prefer the highest snapshot-quality frame that actually shows the wagon's
    reported side-state (anomaly-central: an OPEN/DAMAGED snapshot for an
    OPEN/DAMAGED door).  If no frame of that state was captured, fall back to
    the globally best-scored frame on the camera.
    """
    bucket = cands.get(reported_state)
    if bucket is not None and bucket.has_data():
        return bucket
    best = BestFrameTracker()
    for b in cands.values():
        if b.has_data() and b.score > best.score:
            best = b
    return best


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

_SIDE_FOR_CAMERA = {C.CAMERA_RIGHT_UP: "right", C.CAMERA_LEFT_UP: "left"}


def _process_wagon_camera_door(
    yolo_model, illumination, geo_prior, tracker_cfg, merger_cfg,
    cache_root: str, gw, camera_id: str, feature_out: str,
    evidence_root: Optional[str], verbose: bool,
) -> str:
    """Run the door pipeline for ONE side camera on ONE wagon; write
    door/<CAMERA>/<gw>.json (RIGHT_UP -> right door, LEFT_UP -> left door)."""
    gw_id = gw.global_id
    side = _SIDE_FOR_CAMERA[camera_id]

    decisions, used, _, _, cands, overlay = _run_tracker_one_camera(
        yolo_model, illumination, geo_prior, tracker_cfg, merger_cfg,
        cache_root, gw_id, camera_id,
    )
    if used == 0:
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            camera_id=camera_id, side=side,
            door_state=C.NO_DATA, door_confidence=0.0,
            tracks=[], supporting_cameras=[],
        ))
        return C.STATUS_NO_FRAMES

    st, cf = _pick_side_state(decisions)
    # frames existed but no confirmed track -> conservative legacy CLOSED default
    if st == C.NO_DATA:
        st, cf = C.DOOR_CLOSED, 0.0
    best = _resolve_evidence(cands, st)

    evidence_paths: Dict[str, str] = {}
    if evidence_root and (best.has_data() or overlay.get("tracks") or overlay.get("events")):
        final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
        crop_img = safe_crop(best.frame, best.bbox, pad=12) if best.has_data() else None
        with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME,
                                    camera_id) as ev_tmp:
            if best.has_data():
                annotated = draw_annotated_bbox(
                    best.frame, best.bbox,
                    label=f"{best.meta.get('state','?')} "
                          f"{best.meta.get('confidence',0.0):.2f}",
                    color=(0, 255, 255),
                )
                save_jpeg(os.path.join(ev_tmp, f"{side}_best.jpg"), annotated)
                if crop_img is not None:
                    save_jpeg(os.path.join(ev_tmp, f"{side}_crop.jpg"), crop_img)
                write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                    "global_id": gw_id, "feature": FEATURE_NAME,
                    "camera_id": camera_id, "side": side,
                    "sides": {side: {
                        "camera_id": camera_id, "frame_idx": best.frame_idx,
                        "bbox": best.bbox, "state": best.meta.get("state"),
                        "confidence": best.meta.get("confidence"),
                        "raw_class": best.meta.get("raw_class"),
                        "quality": best.meta.get("quality"),
                    }},
                })
            if overlay.get("tracks") or overlay.get("events"):
                write_metadata(os.path.join(ev_tmp, "overlay.json"), {
                    "global_id": gw_id, "feature": FEATURE_NAME,
                    "camera_id": camera_id,
                    "tracks": overlay.get("tracks", []),
                    "events": overlay.get("events", []),
                })
        if best.has_data():
            evidence_paths[f"{side}_best"] = os.path.join(final_dir, f"{side}_best.jpg")
            if crop_img is not None:
                evidence_paths[f"{side}_crop"] = os.path.join(final_dir, f"{side}_crop.jpg")

    payload: Dict[str, Any] = {
        "global_id": gw_id,
        "feature":   FEATURE_NAME,
        "camera_id": camera_id,
        "side":      side,
        "status":    C.STATUS_OK,
        "door_state":      st,
        "door_confidence": round(float(cf), 4),
        # convenience side-keyed fields so the Stage-4 adapter is trivial
        f"{side}_door":            st,
        f"{side}_door_confidence": round(float(cf), 4),
        "tracks":      decisions,
        "supporting_cameras": [camera_id],
        "frame_count": used,
    }
    payload["evidence"] = evidence_paths
    write_per_wagon_json(feature_out, gw_id, payload)
    if verbose:
        print(f"  [door/{camera_id}/{gw_id}] {side}={st} ({cf:.2f})  "
              f"tracks={len(decisions)}  frames={used}")
    return C.STATUS_OK


def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,   # enables evidence persistence
    cameras: Optional[List[str]] = None,
    confidence: float = C.CONF_DOOR,
    every_nth: int = 1,
    max_frames: int = 0,           # 0 = unbounded (legacy used the whole wagon)
    verbose: bool = True,
) -> Dict[str, str]:
    """Run the door feature per side camera, writing the per-camera layout
    wagon_states/door/<CAMERA>/<gw>.json.  RIGHT_UP produces ONLY the right
    door; LEFT_UP produces ONLY the left door -- so a late camera can never
    overwrite the other side's result."""
    del every_nth, max_frames  # kept for API symmetry; we iterate every frame

    model_path = os.path.join(feature_models_dir, C.MODEL_DOOR_STATE)
    yolo_model = load_yolo(model_path)

    target_cams = [c for c in C.SIDE_CAMERAS if (cameras is None or c in cameras)]
    if not target_cams:
        return {}
    timer = FeatureTimer("door")
    summary: Dict[str, str] = {}

    # Shared per-process helpers (loaded once across wagons + cameras)
    illumination = IlluminationProcessor(IlluminationConfig())
    geo_prior    = GeometricShapePrior(GeometricPriorConfig())
    tracker_cfg  = TrackerConfig()
    merger_cfg   = MergeConfig()

    if yolo_model is None and verbose:
        print(f"[FEAT/door] WARNING: {model_path} missing; emitting NO_DATA.")
    if verbose:
        print(f"[FEAT/door] running on {len(state.wagons)} wagons, cameras={target_cams} "
              f"(conf>={confidence}, legacy DoorTracker + IdentityMerger + "
              f"GeometricPrior + IlluminationQuality)")

    for cam in target_cams:
        side = _SIDE_FOR_CAMERA[cam]
        feature_out = feature_camera_dir(output_dir, FEATURE_NAME, cam)
        for gw in state.wagons:
            gw_id = gw.global_id
            t0 = time.time()
            try:
                if yolo_model is None:
                    write_per_wagon_json(feature_out, gw_id, empty_payload(
                        gw_id, FEATURE_NAME, C.NO_DATA,
                        camera_id=cam, side=side,
                        door_state=C.NO_DATA, door_confidence=0.0,
                        tracks=[], supporting_cameras=[],
                        error="door_state.pt not present",
                    ))
                    summary[gw_id] = C.NO_DATA
                    continue
                summary[gw_id] = _process_wagon_camera_door(
                    yolo_model, illumination, geo_prior, tracker_cfg, merger_cfg,
                    cache_root, gw, cam, feature_out, evidence_root, verbose,
                )
            except Exception as e:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_FAILED,
                    camera_id=cam, side=side,
                    door_state=C.NO_DATA,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=2),
                ))
                summary[gw_id] = C.STATUS_FAILED
                if verbose:
                    print(f"  [door/{cam}/{gw_id}] FAILED: {e}")
            finally:
                timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/door] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
