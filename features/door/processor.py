"""Door feature processor -- PRODUCTION behaviour on the GlobalTrain wagon cache.

Milestone 1 reproduces the deployed production side-camera door logic EXACTLY,
expressed on the wagon_cache instead of production's own per-camera segment
extraction. The EXTERNAL CONTRACT is unchanged -- ``run()`` signature, the
per-camera output ``wagon_states/door/<CAMERA>/GW_n.json`` payload keys
(``right_door``/``left_door`` + ``*_door_confidence``, ``door_state``,
``tracks``, ``supporting_cameras``, ``frame_count``, ``evidence``), and the
evidence layout -- so orchestrator / lifecycle_runner / fusion / reporting /
rendering consume it without modification. Only the internals differ.

Production door logic (the side-camera ``side_damage.pt`` model emits both door
and side-damage classes; this processor consumes ONLY the door classes --
``damage`` is owned by the damage processor):

  1. For each side camera (RIGHT_UP -> right door, LEFT_UP -> left door), iterate
     the wagon's interior frames (production skips a fixed 10-frame margin at
     each end -- ``_iter_segment_frames`` edge_skip).
  2. Run ``side_damage.pt`` at the PRODUCTION per-camera confidence
     (RIGHT_UP 0.85 / LEFT_UP 0.88 -- notebook-authoritative; see
     WAGONEYE_V5_STAGE3_FEATURE_AUDIT.md V-1). Collect ``door_open`` /
     ``door_close`` detections (frame, conf, bbox).
  3. Band detections by frame proximity (``gap_tolerance = 5``), keeping each
     band's highest-confidence frame (production ``_analyze_detection_bands``).
  4. ``door_state = OPEN if any door_open band else CLOSED`` -- the exact
     production two-state rule (``'open' if wagon_door_open else 'closed'``).
     Production doors are never PARTIAL/DAMAGED, so those states never occur.
  5. ``door_close_detected = any door_close band`` (additive audit field,
     matching production's side JSON).
  6. Evidence = the winning band's best frame, annotated (door_open red,
     door_close green -- production colours).

Deliberately NO Kalman/Hungarian tracking, FSM hysteresis, identity merging,
illumination scoring, or geometric-shape prior: production does not use them for
doors, and milestone 1 must not add behaviour the production system lacks. Those
belong to the shelved v4-native door path (recoverable from git history).

Graceful degradation: when the production model is absent the loader raises
``MissingProductionModel`` and every wagon is written ``NO_DATA`` (with the
reason) -- no dummy inference, no fabricated door state.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2

from core import constants as C
from core import config as CFG
from core import production_models as PM
from core.global_state_loader import GlobalTrainState

from features._common import (
    list_wagon_frames, write_per_wagon_json, feature_camera_dir,
    FeatureTimer, DEVICE, HALF, batched_detect,
)
from features._evidence import (
    atomic_camera_evidence, read_cached_frame,
    save_jpeg, safe_crop, write_metadata, draw_annotated_bbox,
)


FEATURE_NAME = "door"

# RIGHT_UP -> right door, LEFT_UP -> left door (production side->door convention).
_SIDE_FOR_CAMERA = {C.CAMERA_RIGHT_UP: "right", C.CAMERA_LEFT_UP: "left"}

# PRODUCTION per-camera detection confidence for the side model (notebook-
# authoritative: DAMAGE_CONFIDENCE = 0.85 right_up / 0.88 left_up). One model
# gates both doors and side damage; this is the door pass.
_SIDE_DOOR_CONF = {C.CAMERA_RIGHT_UP: 0.85, C.CAMERA_LEFT_UP: 0.88}

# Fixed edge margin skipped at each end of a wagon's frame span (production
# _iter_segment_frames edge_skip_frames = 10).
_EDGE_SKIP_FRAMES = 10

# Band gap tolerance for grouping door detections (production
# _analyze_detection_bands gap_tolerance = 5).
_BAND_GAP_TOLERANCE = 5

# Production side-damage door class names + annotation colours (BGR).
_CLASS_DOOR_OPEN = "door_open"
_CLASS_DOOR_CLOSE = "door_close"
_COLOR_DOOR_OPEN = (0, 0, 255)     # red
_COLOR_DOOR_CLOSE = (0, 255, 0)    # green


# -----------------------------------------------------------------------------
# Band grouping (port of production _analyze_detection_bands)
# -----------------------------------------------------------------------------

def _analyze_bands(
    detections: List[Tuple[int, float, float, float, float, float]],
    gap_tolerance: int,
) -> List[Dict[str, Any]]:
    """Group ``(frame, conf, x1, y1, x2, y2)`` detections into bands.

    A new band starts when the frame gap exceeds ``gap_tolerance + 1``. Each
    band records its highest-confidence frame as the representative
    (``best_frame`` / ``best_conf`` / ``best_bbox``), matching production.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[0])
    bands: List[Dict[str, Any]] = []
    cur = {
        "band_id": 1, "start": dets[0][0], "end": dets[0][0],
        "frames": [dets[0][0]], "confs": [dets[0][1]], "dets": [dets[0]],
    }
    for d in dets[1:]:
        if d[0] - cur["end"] <= gap_tolerance + 1:
            cur["end"] = d[0]
            if d[0] not in cur["frames"]:
                cur["frames"].append(d[0])
            cur["confs"].append(d[1])
            cur["dets"].append(d)
        else:
            bands.append(cur)
            cur = {
                "band_id": len(bands) + 1, "start": d[0], "end": d[0],
                "frames": [d[0]], "confs": [d[1]], "dets": [d],
            }
    bands.append(cur)
    for b in bands:
        best = max(b["dets"], key=lambda d: d[1])
        b["best_frame"] = int(best[0])
        b["best_conf"] = float(best[1])
        b["best_bbox"] = [float(best[2]), float(best[3]), float(best[4]), float(best[5])]
        b["avg_conf"] = sum(b["confs"]) / len(b["confs"])
        b["frame_count"] = len(set(b["frames"]))
    return bands


def _parse_frame_index(path: str) -> int:
    """Absolute cache frame index from ``frame_NNNNNN.jpg`` (or -1)."""
    try:
        return int(os.path.basename(path).split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def _interior_frames(paths: List[str]) -> List[str]:
    """Frames with the fixed production edge margin removed from each end.

    Mirrors ``_iter_segment_frames``: when the span is too short to leave an
    interior (``2 * edge_skip``), no interior frames exist -> no detections ->
    the wagon reports the production CLOSED default.
    """
    n = len(paths)
    if n <= 2 * _EDGE_SKIP_FRAMES:
        return []
    return paths[_EDGE_SKIP_FRAMES:n - _EDGE_SKIP_FRAMES]


# -----------------------------------------------------------------------------
# Payload builders (keep the exact door JSON contract fusion/reporting read)
# -----------------------------------------------------------------------------

def _base_payload(gw_id: str, camera_id: str, side: str, status: str) -> Dict[str, Any]:
    return {
        "global_id": gw_id,
        "feature": FEATURE_NAME,
        "camera_id": camera_id,
        "side": side,
        "status": status,
    }


def _empty_door_payload(
    gw_id: str, camera_id: str, side: str, status: str, **extra: Any,
) -> Dict[str, Any]:
    """NO_FRAMES / NO_DATA / FAILED payload with the full door key surface so
    the fusion adapter reads a clean NO_DATA rather than a missing key."""
    p = _base_payload(gw_id, camera_id, side, status)
    p.update({
        "door_state": C.NO_DATA,
        "door_confidence": 0.0,
        f"{side}_door": C.NO_DATA,
        f"{side}_door_confidence": 0.0,
        "door_close_detected": False,
        "tracks": [],
        "supporting_cameras": [],
        "frame_count": 0,
        "evidence": {},
    })
    p.update(extra)
    return p


def _bands_to_tracks(
    bands: List[Dict[str, Any]], camera_id: str, door_state: str,
) -> List[Dict[str, Any]]:
    """Represent each detection band as a track record (preserves the
    ``tracks[]`` key shape reporting/rendering may read)."""
    tracks: List[Dict[str, Any]] = []
    for b in bands:
        cx = (b["best_bbox"][0] + b["best_bbox"][2]) / 2.0
        tracks.append({
            "camera_id": camera_id,
            "track_id": int(b["band_id"]),
            "state": door_state,
            "confidence": round(float(b["best_conf"]), 4),
            "first_frame": int(b["start"]),
            "last_frame": int(b["end"]),
            "total_hits": int(b["frame_count"]),
            "mean_center_x": float(cx),
        })
    return tracks


# -----------------------------------------------------------------------------
# Detection collection (default per-frame == byte-identical; batched == opt-in)
# -----------------------------------------------------------------------------

def _collect_door_batched(model, cam_conf, interior):
    """OPT-IN batched collection (FEATURE_BATCH_SIZE>1): decode in chunks, run
    batched inference, and cache decoded frames that carried a door detection so
    evidence reuses them (no re-read).  Produces the SAME (frame,conf,bbox)
    records as the per-frame path; enable only after on-host parity validation."""
    names = getattr(model, "names", {}) or {}
    open_dets: List[Tuple[int, float, float, float, float, float]] = []
    close_dets: List[Tuple[int, float, float, float, float, float]] = []
    used = 0
    frame_cache: Dict[int, Any] = {}
    bs = CFG.FEATURE_BATCH_SIZE
    for j in range(0, len(interior), bs):
        fis: List[int] = []
        frames: List[Any] = []
        for p in interior[j:j + bs]:
            fr = cv2.imread(p)
            if fr is None:
                continue
            fis.append(_parse_frame_index(p))
            frames.append(fr)
        if not frames:
            continue
        used += len(frames)
        per = batched_detect(model, frames, confidence=cam_conf)
        for fi, fr, dets in zip(fis, frames, per):
            hit = False
            for d in dets:
                cname = d["class_name"]
                rec = (fi, d["confidence"], d["bbox"][0], d["bbox"][1],
                       d["bbox"][2], d["bbox"][3])
                if cname == _CLASS_DOOR_OPEN:
                    open_dets.append(rec); hit = True
                elif cname == _CLASS_DOOR_CLOSE:
                    close_dets.append(rec); hit = True
            if hit:
                frame_cache[fi] = fr
    return open_dets, close_dets, used, frame_cache


# -----------------------------------------------------------------------------
# Per-wagon, per-camera door pass
# -----------------------------------------------------------------------------

def _process_wagon_camera_door(
    model, cam_conf: float, cache_root: str, gw, camera_id: str, side: str,
    feature_out: str, evidence_root: Optional[str], verbose: bool,
) -> str:
    gw_id = gw.global_id
    paths = list_wagon_frames(cache_root, gw_id, camera_id)  # sorted; no adaptive trim
    if not paths:
        write_per_wagon_json(feature_out, gw_id, _empty_door_payload(
            gw_id, camera_id, side, C.STATUS_NO_FRAMES))
        return C.STATUS_NO_FRAMES

    interior = _interior_frames(paths)
    open_dets: List[Tuple[int, float, float, float, float, float]] = []
    close_dets: List[Tuple[int, float, float, float, float, float]] = []
    used = 0
    frame_cache: Dict[int, Any] = {}
    names = getattr(model, "names", {}) or {}

    if CFG.FEATURE_BATCH_SIZE > 1:
        open_dets, close_dets, used, frame_cache = _collect_door_batched(
            model, cam_conf, interior)
    else:
        # === DEFAULT per-frame path -- VERBATIM pre-optimization behaviour ===
        for p in interior:
            frame = cv2.imread(p)
            if frame is None:
                continue
            used += 1
            fi = _parse_frame_index(p)
            # Production ran the side model with conf=DAMAGE_CONFIDENCE, so the
            # confidence floor is applied at inference time (identical outcome).
            try:
                res = model(frame, verbose=False, half=HALF, device=DEVICE, conf=cam_conf)[0]
            except Exception:
                continue
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for bbox, conf, cid in zip(boxes, confs, clss):
                cname = str(names.get(int(cid), "")).lower()
                rec = (fi, float(conf), float(bbox[0]), float(bbox[1]),
                       float(bbox[2]), float(bbox[3]))
                if cname == _CLASS_DOOR_OPEN:
                    open_dets.append(rec)
                elif cname == _CLASS_DOOR_CLOSE:
                    close_dets.append(rec)
                # 'damage' class is owned by the damage processor -- ignored here.

    open_bands = _analyze_bands(open_dets, _BAND_GAP_TOLERANCE)
    close_bands = _analyze_bands(close_dets, _BAND_GAP_TOLERANCE)

    # Production rule: door_status = 'open' if any door_open band else 'closed'.
    if open_bands:
        door_state = C.DOOR_OPEN
        winning = max(open_bands, key=lambda b: b["best_conf"])
        conf = winning["best_conf"]
        reported_class = _CLASS_DOOR_OPEN
        evidence_color = _COLOR_DOOR_OPEN
        src_bands = open_bands
    else:
        door_state = C.DOOR_CLOSED
        if close_bands:
            winning = max(close_bands, key=lambda b: b["best_conf"])
            conf = winning["best_conf"]
        else:
            winning = None
            conf = 0.0
        reported_class = _CLASS_DOOR_CLOSE
        evidence_color = _COLOR_DOOR_CLOSE
        src_bands = close_bands

    door_close_detected = bool(close_bands)
    tracks = _bands_to_tracks(src_bands, camera_id, door_state)

    # ---- evidence: the winning band's best frame, annotated ----
    evidence_paths: Dict[str, str] = {}
    if evidence_root and winning is not None:
        best_fi = winning["best_frame"]
        best_bbox = winning["best_bbox"]
        # Reuse the already-decoded frame from the batched path (identical bytes);
        # default path has an empty cache -> falls back to the original re-read.
        best_frame = frame_cache.get(best_fi)
        if best_frame is None:
            best_frame = read_cached_frame(
                cache_root, gw_id, C.CAMERA_FOLDER[camera_id], best_fi)
        if best_frame is not None:
            crop_img = safe_crop(best_frame, best_bbox, pad=12)
            with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME,
                                        camera_id) as ev_tmp:
                annotated = draw_annotated_bbox(
                    best_frame, best_bbox,
                    label=f"{reported_class.upper()} {conf:.2f}",
                    color=evidence_color,
                )
                save_jpeg(os.path.join(ev_tmp, f"{side}_best.jpg"), annotated)
                if crop_img is not None:
                    save_jpeg(os.path.join(ev_tmp, f"{side}_crop.jpg"), crop_img)
                write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                    "global_id": gw_id, "feature": FEATURE_NAME,
                    "camera_id": camera_id, "side": side,
                    "sides": {side: {
                        "camera_id": camera_id, "frame_idx": int(best_fi),
                        "bbox": best_bbox, "state": door_state,
                        "confidence": round(float(conf), 4),
                        "raw_class": reported_class,
                    }},
                })
            final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
            evidence_paths[f"{side}_best"] = os.path.join(final_dir, f"{side}_best.jpg")
            if crop_img is not None:
                evidence_paths[f"{side}_crop"] = os.path.join(final_dir, f"{side}_crop.jpg")

    payload = _base_payload(gw_id, camera_id, side, C.STATUS_OK)
    payload.update({
        "door_state": door_state,
        "door_confidence": round(float(conf), 4),
        f"{side}_door": door_state,
        f"{side}_door_confidence": round(float(conf), 4),
        "door_close_detected": door_close_detected,
        "tracks": tracks,
        "supporting_cameras": [camera_id],
        "frame_count": used,
        "evidence": evidence_paths,
    })
    write_per_wagon_json(feature_out, gw_id, payload)
    if verbose:
        print(f"  [door/{camera_id}/{gw_id}] {side}={door_state} ({conf:.2f})  "
              f"open_bands={len(open_bands)} close_bands={len(close_bands)} "
              f"frames={used}")
    return C.STATUS_OK


# -----------------------------------------------------------------------------
# Public entry (signature preserved -- called per camera by lifecycle_runner)
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    confidence: float = C.CONF_DOOR,
    every_nth: int = 1,
    max_frames: int = 0,
    verbose: bool = True,
) -> Dict[str, str]:
    """Run the production door feature per side camera, writing the per-camera
    layout ``wagon_states/door/<CAMERA>/<gw>.json``.

    Signature is unchanged for the orchestrator. ``feature_models_dir``,
    ``confidence``, ``every_nth`` and ``max_frames`` are accepted for API
    stability but not used: the model is the PRODUCTION ``side_damage.pt``
    (resolved via ``core.production_models``), the confidence is the production
    per-camera value, and every interior frame is processed (as production did).
    """
    del feature_models_dir, confidence, every_nth, max_frames  # see docstring

    target_cams = [c for c in C.SIDE_CAMERAS if (cameras is None or c in cameras)]
    if not target_cams:
        return {}

    timer = FeatureTimer("door")
    summary: Dict[str, str] = {}

    if verbose:
        print(f"[FEAT/door] running on {len(state.wagons)} wagons, "
              f"cameras={target_cams} (PRODUCTION side_damage.pt; "
              f"conf right_up=0.85 left_up=0.88; band gap_tol=5)")

    for cam in target_cams:
        side = _SIDE_FOR_CAMERA[cam]
        feature_out = feature_camera_dir(output_dir, FEATURE_NAME, cam)
        cam_conf = _SIDE_DOOR_CONF[cam]

        # Load the PRODUCTION model (cached). Absent -> clear error -> NO_DATA.
        model = None
        model_err: Optional[str] = None
        try:
            model = PM.load_for(FEATURE_NAME, cam)
        except PM.MissingProductionModel as e:
            model_err = str(e)
            if verbose:
                print(f"[FEAT/door] {e} -- emitting NO_DATA for {cam}")

        for gw in state.wagons:
            gw_id = gw.global_id
            t0 = time.time()
            try:
                if model is None:
                    write_per_wagon_json(feature_out, gw_id, _empty_door_payload(
                        gw_id, cam, side, C.NO_DATA, error=model_err))
                    summary[gw_id] = C.NO_DATA
                    continue
                summary[gw_id] = _process_wagon_camera_door(
                    model, cam_conf, cache_root, gw, cam, side,
                    feature_out, evidence_root, verbose)
            except Exception as e:
                write_per_wagon_json(feature_out, gw_id, _empty_door_payload(
                    gw_id, cam, side, C.STATUS_FAILED,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=2)))
                summary[gw_id] = C.STATUS_FAILED
                if verbose:
                    print(f"  [door/{cam}/{gw_id}] FAILED: {e}")
            finally:
                timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/door] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
