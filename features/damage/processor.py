"""Damage feature processor -- PRODUCTION top-damage behaviour on the wagon cache.

Milestone 1 reproduces the deployed production TOP-camera damage logic exactly,
on the GlobalTrain wagon cache. Production uses a 4-class top-damage model per
top camera (`top_left_damage.pt` for LEFT_UP_TOP, `right_top_damage.pt` for
RIGHT_UP_TOP) emitting:

    body_dmg, body_dmg_probable, floor_dmg, floor_dmg_probable

Production logic (Train-Inspection-Engine `_detect_top`):
  1. Per wagon, iterate interior frames (fixed 10-frame edge skip).
  2. Run the damage model at conf 0.70; collect detections per class.
  3. Band each class by frame proximity (gap_tolerance 5); each band keeps its
     highest-confidence frame (best_frame).
  4. Confirmed damage = any body_dmg OR floor_dmg band; probable damage = any
     `*_probable` band. `damage_detected = body∨floor` (confirmed).
  5. Evidence = each band's best frame, annotated (confirmed=red, probable=orange).

NOTE: production Train-Inspection-Engine does NOT apply a loaded-wagon
floor-damage filter (that was an old_system/v4 heuristic); it is intentionally
OMITTED for exact production behaviour. No Kalman/Hungarian damage tracking --
production used simple banding (same as doors).

The EXTERNAL CONTRACT is preserved: `run()` signature; per-camera output
`wagon_states/damage/<CAMERA>/GW_n.json` with keys `damage_status`
(== "DAMAGE"/"OK"/"NO_DATA"), `top_damage_details`, `tracks`, `frames_used`,
`supporting_cameras`, `frame_count`, `evidence`; plus additive production fields
(`body_dmg_detected`, `floor_dmg_detected`, `*_probable_detected`,
`damage_detected`, `probable_damage_detected`) so the production report/dashboard
data is retained. Evidence layout (`track_N.jpg` / `track_N_crop.jpg` /
`metadata.json` / `overlay.json`) is unchanged. Graceful `NO_DATA` when the
production model is absent. Cross-camera "any top camera DAMAGE" fusion is
Stage-4's job (unchanged).
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
    list_wagon_frames, write_per_wagon_json, empty_payload,
    FeatureTimer, feature_camera_dir, DEVICE, HALF, batched_detect,
)
from features._evidence import (
    atomic_camera_evidence, read_cached_frame,
    save_jpeg, safe_crop, write_metadata, draw_evidence_annotation,
)


FEATURE_NAME = "damage"

# Production top-damage 4-class model, thresholds (Train-Inspection-Engine top).
_DAMAGE_CONFIDENCE = 0.70
_EDGE_SKIP_FRAMES = 10
_BAND_GAP_TOLERANCE = 5

_CLASS_BODY = "body_dmg"
_CLASS_BODY_PROB = "body_dmg_probable"
_CLASS_FLOOR = "floor_dmg"
_CLASS_FLOOR_PROB = "floor_dmg_probable"
_TOP_CLASSES = (_CLASS_BODY, _CLASS_BODY_PROB, _CLASS_FLOOR, _CLASS_FLOOR_PROB)
_CONFIRMED_CLASSES = {_CLASS_BODY, _CLASS_FLOOR}
_PROBABLE_CLASSES = {_CLASS_BODY_PROB, _CLASS_FLOOR_PROB}

# Annotation colours (BGR): confirmed=red, probable=orange (production).
_COLOR_CONFIRMED = (0, 0, 255)
_COLOR_PROBABLE = (0, 140, 255)

# --- SIDE damage (production side_damage.pt `damage` class on side cameras) ---
# Same side model + per-camera confidence as the door pass (0.85/0.88); the
# `damage` class is production side damage. Annotated orange (production).
_CLASS_SIDE_DAMAGE = "damage"
_SIDE_DAMAGE_CONF = {C.CAMERA_RIGHT_UP: 0.85, C.CAMERA_LEFT_UP: 0.88}
_COLOR_SIDE_DAMAGE = (0, 140, 255)


def _color_for(cls_name: str) -> tuple:
    return _COLOR_PROBABLE if cls_name in _PROBABLE_CLASSES else _COLOR_CONFIRMED


def _parse_frame_index(path: str) -> int:
    try:
        return int(os.path.basename(path).split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def _interior_frames(paths: List[str]) -> List[str]:
    n = len(paths)
    if n <= 2 * _EDGE_SKIP_FRAMES:
        return []
    return paths[_EDGE_SKIP_FRAMES:n - _EDGE_SKIP_FRAMES]


def _analyze_bands(
    detections: List[Tuple[int, float, float, float, float, float]],
    gap_tolerance: int,
) -> List[Dict[str, Any]]:
    """Group ``(frame, conf, x1, y1, x2, y2)`` detections into bands
    (production ``_analyze_detection_bands``); best_frame = highest conf."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[0])
    bands: List[Dict[str, Any]] = []
    cur = {"band_id": 1, "start": dets[0][0], "end": dets[0][0],
           "frames": [dets[0][0]], "confs": [dets[0][1]], "dets": [dets[0]]}
    for d in dets[1:]:
        if d[0] - cur["end"] <= gap_tolerance + 1:
            cur["end"] = d[0]
            if d[0] not in cur["frames"]:
                cur["frames"].append(d[0])
            cur["confs"].append(d[1])
            cur["dets"].append(d)
        else:
            bands.append(cur)
            cur = {"band_id": len(bands) + 1, "start": d[0], "end": d[0],
                   "frames": [d[0]], "confs": [d[1]], "dets": [d]}
    bands.append(cur)
    for b in bands:
        best = max(b["dets"], key=lambda d: d[1])
        b["best_frame"] = int(best[0])
        b["best_conf"] = float(best[1])
        b["best_bbox"] = [float(best[2]), float(best[3]), float(best[4]), float(best[5])]
        b["frame_count"] = len(set(b["frames"]))
    return bands


def _collect_top_batched(model, interior, dets_by_class):
    """OPT-IN batched top-damage collection (FEATURE_BATCH_SIZE>1): decode in
    chunks, batched inference, cache decoded frames that carried a detection for
    evidence reuse.  Produces the SAME per-class (frame,conf,bbox) records as the
    per-frame path."""
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
        per = batched_detect(model, frames, confidence=_DAMAGE_CONFIDENCE)
        for fi, fr, dets in zip(fis, frames, per):
            hit = False
            for d in dets:
                cname = d["class_name"]
                if cname not in dets_by_class:
                    continue
                dets_by_class[cname].append(
                    (fi, d["confidence"], d["bbox"][0], d["bbox"][1],
                     d["bbox"][2], d["bbox"][3]))
                hit = True
            if hit:
                frame_cache[fi] = fr
    return used, frame_cache


def _collect_side_batched(model, cam_conf, interior, dmg_dets, side_class):
    """OPT-IN batched side-damage collection (FEATURE_BATCH_SIZE>1)."""
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
                if d["class_name"] == side_class:
                    dmg_dets.append((fi, d["confidence"], d["bbox"][0], d["bbox"][1],
                                     d["bbox"][2], d["bbox"][3]))
                    hit = True
            if hit:
                frame_cache[fi] = fr
    return used, frame_cache


def _process_wagon_camera_damage(
    model, cache_root: str, gw, camera_id: str, feature_out: str,
    evidence_root: Optional[str], verbose: bool,
) -> str:
    gw_id = gw.global_id

    # ENGINE / BRAKE_VAN gate (sealed GlobalTrainState).
    if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_OK,
            camera_id=camera_id, damage_status=C.NO_DATA,
            top_damage_details=[], supporting_cameras=[],
            skipped_reason=f"classification={gw.classification}"))
        return C.STATUS_OK

    paths = list_wagon_frames(cache_root, gw_id, camera_id)
    interior = _interior_frames(paths)
    used = 0
    frame_cache: Dict[int, Any] = {}
    names = getattr(model, "names", {}) or {}
    dets_by_class: Dict[str, List[Tuple[int, float, float, float, float, float]]] = {
        c: [] for c in _TOP_CLASSES
    }

    if CFG.FEATURE_BATCH_SIZE > 1:
        used, frame_cache = _collect_top_batched(model, interior, dets_by_class)
    else:
        # === DEFAULT per-frame path -- VERBATIM pre-optimization behaviour ===
        for p in interior:
            frame = cv2.imread(p)
            if frame is None:
                continue
            used += 1
            fi = _parse_frame_index(p)
            try:
                res = model(frame, verbose=False, half=HALF, device=DEVICE,
                            conf=_DAMAGE_CONFIDENCE)[0]
            except Exception:
                continue
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for bbox, conf, cid in zip(boxes, confs, clss):
                cname = str(names.get(int(cid), "")).lower()
                if cname not in dets_by_class:
                    continue
                dets_by_class[cname].append(
                    (fi, float(conf), float(bbox[0]), float(bbox[1]),
                     float(bbox[2]), float(bbox[3])))

    if used == 0:
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            camera_id=camera_id, damage_status=C.NO_DATA,
            top_damage_details=[], supporting_cameras=[]))
        return C.STATUS_NO_FRAMES

    bands_by_class = {c: _analyze_bands(dets_by_class[c], _BAND_GAP_TOLERANCE)
                      for c in _TOP_CLASSES}
    body = bool(bands_by_class[_CLASS_BODY])
    floor = bool(bands_by_class[_CLASS_FLOOR])
    body_prob = bool(bands_by_class[_CLASS_BODY_PROB])
    floor_prob = bool(bands_by_class[_CLASS_FLOOR_PROB])
    damage_detected = body or floor
    probable_detected = body_prob or floor_prob
    damage_status = C.DAMAGE_PRESENT if damage_detected else C.DAMAGE_OK

    # Flatten bands into detail records (confirmed + probable), one per band.
    details: List[Dict[str, Any]] = []
    for cls in _TOP_CLASSES:
        for b in bands_by_class[cls]:
            details.append({
                "camera_id": camera_id,
                "class_name": cls,
                "confidence": round(b["best_conf"], 4),
                "best_confidence": round(b["best_conf"], 4),
                "best_frame_idx": b["best_frame"],
                "bbox": b["best_bbox"],
                "first_frame": int(b["start"]),
                "last_frame": int(b["end"]),
                "total_hits": int(b["frame_count"]),
            })

    # ---- evidence: each band's best frame, annotated ----
    evidence_paths: Dict[str, str] = {}
    if evidence_root and details:
        final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
        track_meta: List[Dict[str, Any]] = []
        with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
            write_metadata(os.path.join(ev_tmp, "overlay.json"), {
                "global_id": gw_id, "feature": FEATURE_NAME,
                "camera_id": camera_id, "detections": details,
            })
            for i, d in enumerate(details, start=1):
                # Reuse the already-decoded frame from the batched path (identical
                # bytes); default path has an empty cache -> original re-read.
                frame = frame_cache.get(d["best_frame_idx"])
                if frame is None:
                    frame = read_cached_frame(cache_root, gw_id,
                                              C.CAMERA_FOLDER[camera_id], d["best_frame_idx"])
                if frame is None:
                    continue
                annotated = draw_evidence_annotation(
                    frame, d["bbox"],
                    label=f"{d['class_name']} {d['best_confidence']:.2f}",
                    color=_color_for(d["class_name"]),
                    gw_id=gw_id, camera_id=camera_id,
                    frame_idx=int(d["best_frame_idx"]))
                save_jpeg(os.path.join(ev_tmp, f"track_{i}.jpg"), annotated)
                crop_img = safe_crop(frame, d["bbox"], pad=10)
                if crop_img is not None:
                    save_jpeg(os.path.join(ev_tmp, f"track_{i}_crop.jpg"), crop_img)
                evidence_paths[f"track_{i}"] = os.path.join(final_dir, f"track_{i}.jpg")
                if crop_img is not None:
                    evidence_paths[f"track_{i}_crop"] = os.path.join(final_dir, f"track_{i}_crop.jpg")
                track_meta.append({"track_idx": i, **d})
            write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                "global_id": gw_id, "feature": FEATURE_NAME,
                "camera_id": camera_id, "damage_status": damage_status,
                "tracks": track_meta,
            })

    write_per_wagon_json(feature_out, gw_id, {
        "global_id": gw_id,
        "feature": FEATURE_NAME,
        "camera_id": camera_id,
        "status": C.STATUS_OK,
        "damage_status": damage_status,
        "top_damage_details": details,
        "tracks": details,
        # additive production fields (retain the production JSON data)
        "body_dmg_detected": body,
        "floor_dmg_detected": floor,
        "body_dmg_probable_detected": body_prob,
        "floor_dmg_probable_detected": floor_prob,
        "damage_detected": damage_detected,
        "probable_damage_detected": probable_detected,
        "frames_used": used,
        "supporting_cameras": [camera_id],
        "frame_count": used,
        "evidence": evidence_paths,
    })
    if verbose:
        print(f"  [damage/{camera_id}/{gw_id}] {damage_status}  "
              f"body={body} floor={floor} prob={probable_detected} "
              f"bands={len(details)} frames={used}")
    return C.STATUS_OK


def _process_wagon_camera_side_damage(
    model, cam_conf: float, cache_root: str, gw, camera_id: str,
    feature_out: str, evidence_root: Optional[str], verbose: bool,
) -> str:
    """Production SIDE damage: band side_damage.pt's `damage` class on one side
    camera. Writes damage/<SIDE_CAMERA>/<gw>.json with damage_status +
    side_damage_details (fusion reads these for UnifiedWagonState.side_damage)."""
    gw_id = gw.global_id
    if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_OK,
            camera_id=camera_id, damage_status=C.NO_DATA,
            side_damage_details=[], side=True, supporting_cameras=[],
            skipped_reason=f"classification={gw.classification}"))
        return C.STATUS_OK

    paths = list_wagon_frames(cache_root, gw_id, camera_id)
    interior = _interior_frames(paths)
    used = 0
    frame_cache: Dict[int, Any] = {}
    names = getattr(model, "names", {}) or {}
    dmg_dets: List[Tuple[int, float, float, float, float, float]] = []

    if CFG.FEATURE_BATCH_SIZE > 1:
        used, frame_cache = _collect_side_batched(
            model, cam_conf, interior, dmg_dets, _CLASS_SIDE_DAMAGE)
    else:
        # === DEFAULT per-frame path -- VERBATIM pre-optimization behaviour ===
        for p in interior:
            frame = cv2.imread(p)
            if frame is None:
                continue
            used += 1
            fi = _parse_frame_index(p)
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
                if str(names.get(int(cid), "")).lower() == _CLASS_SIDE_DAMAGE:
                    dmg_dets.append((fi, float(conf), float(bbox[0]), float(bbox[1]),
                                     float(bbox[2]), float(bbox[3])))

    if used == 0:
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            camera_id=camera_id, damage_status=C.NO_DATA,
            side_damage_details=[], side=True, supporting_cameras=[]))
        return C.STATUS_NO_FRAMES

    bands = _analyze_bands(dmg_dets, _BAND_GAP_TOLERANCE)
    damage_status = C.DAMAGE_PRESENT if bands else C.DAMAGE_OK
    details = [{
        "camera_id": camera_id, "class_name": _CLASS_SIDE_DAMAGE,
        "confidence": round(b["best_conf"], 4), "best_confidence": round(b["best_conf"], 4),
        "best_frame_idx": b["best_frame"], "bbox": b["best_bbox"],
        "first_frame": int(b["start"]), "last_frame": int(b["end"]),
        "total_hits": int(b["frame_count"]),
    } for b in bands]

    evidence_paths: Dict[str, str] = {}
    if evidence_root and details:
        final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
        track_meta: List[Dict[str, Any]] = []
        with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
            write_metadata(os.path.join(ev_tmp, "overlay.json"), {
                "global_id": gw_id, "feature": FEATURE_NAME, "camera_id": camera_id,
                "side": True, "detections": details})
            for i, d in enumerate(details, start=1):
                frame = frame_cache.get(d["best_frame_idx"])
                if frame is None:
                    frame = read_cached_frame(cache_root, gw_id, C.CAMERA_FOLDER[camera_id],
                                              d["best_frame_idx"])
                if frame is None:
                    continue
                annotated = draw_evidence_annotation(
                    frame, d["bbox"], label=f"DAMAGE {d['best_confidence']:.2f}",
                    color=_COLOR_SIDE_DAMAGE,
                    gw_id=gw_id, camera_id=camera_id,
                    frame_idx=int(d["best_frame_idx"]))
                save_jpeg(os.path.join(ev_tmp, f"track_{i}.jpg"), annotated)
                crop_img = safe_crop(frame, d["bbox"], pad=10)
                if crop_img is not None:
                    save_jpeg(os.path.join(ev_tmp, f"track_{i}_crop.jpg"), crop_img)
                evidence_paths[f"track_{i}"] = os.path.join(final_dir, f"track_{i}.jpg")
                if crop_img is not None:
                    evidence_paths[f"track_{i}_crop"] = os.path.join(final_dir, f"track_{i}_crop.jpg")
                track_meta.append({"track_idx": i, **d})
            write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                "global_id": gw_id, "feature": FEATURE_NAME, "camera_id": camera_id,
                "side": True, "damage_status": damage_status, "tracks": track_meta})

    write_per_wagon_json(feature_out, gw_id, {
        "global_id": gw_id, "feature": FEATURE_NAME, "camera_id": camera_id,
        "side": True, "status": C.STATUS_OK,
        "damage_status": damage_status,
        "side_damage_details": details,
        "side_damage_detected": bool(bands),
        "tracks": details,
        "frames_used": used, "supporting_cameras": [camera_id],
        "frame_count": used, "evidence": evidence_paths,
    })
    if verbose:
        print(f"  [damage/{camera_id}/{gw_id}] SIDE {damage_status}  "
              f"bands={len(details)} frames={used}")
    return C.STATUS_OK


def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    confidence: float = C.CONF_DAMAGE,
    verbose: bool = True,
    every_nth: int = 1,
    max_frames: int = 0,
    min_persistent_frames: int = 2,
) -> Dict[str, str]:
    """Run production damage per camera, writing wagon_states/damage/<CAMERA>/
    <gw>.json. TOP cameras -> 4-class top damage; SIDE cameras -> side_damage.pt
    `damage` class (feeds UnifiedWagonState.side_damage). Signature preserved for
    the orchestrator; `feature_models_dir`/`confidence`/`every_nth`/`max_frames`/
    `min_persistent_frames` are accepted but not used (production model +
    thresholds)."""
    del feature_models_dir, confidence, every_nth, max_frames, min_persistent_frames

    target_cams = [c for c in C.ALL_CAMERAS
                   if (cameras is None or c in cameras)
                   and (c in C.TOP_CAMERAS or c in C.SIDE_CAMERAS)]
    if not target_cams:
        return {}
    timer = FeatureTimer("damage")
    summary: Dict[str, str] = {}

    if verbose:
        print(f"[FEAT/damage] running on {len(state.wagons)} wagons, cameras={target_cams} "
              f"(TOP=4-class body/floor+probable conf={_DAMAGE_CONFIDENCE}; "
              f"SIDE=side_damage.pt 'damage' conf 0.85/0.88; band gap_tol=5; edge_skip=10)")

    for cam in target_cams:
        is_side = cam in C.SIDE_CAMERAS
        feature_out = feature_camera_dir(output_dir, FEATURE_NAME, cam)
        model = None
        model_err: Optional[str] = None
        try:
            model = PM.load_for(FEATURE_NAME, cam)
        except PM.MissingProductionModel as e:
            model_err = str(e)
            if verbose:
                print(f"[FEAT/damage] {e} -- emitting NO_DATA for {cam}")

        for gw in state.wagons:
            gw_id = gw.global_id
            t0 = time.time()
            try:
                if model is None:
                    extra = ({"side_damage_details": [], "side": True} if is_side
                             else {"top_damage_details": []})
                    write_per_wagon_json(feature_out, gw_id, empty_payload(
                        gw_id, FEATURE_NAME, C.NO_DATA,
                        camera_id=cam, damage_status=C.NO_DATA,
                        supporting_cameras=[], error=model_err, **extra))
                    summary[gw_id] = C.NO_DATA
                    continue
                if is_side:
                    summary[gw_id] = _process_wagon_camera_side_damage(
                        model, _SIDE_DAMAGE_CONF[cam], cache_root, gw, cam,
                        feature_out, evidence_root, verbose)
                else:
                    summary[gw_id] = _process_wagon_camera_damage(
                        model, cache_root, gw, cam, feature_out, evidence_root, verbose)
            except Exception as e:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_FAILED,
                    camera_id=cam, damage_status=C.NO_DATA,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=2)))
                summary[gw_id] = C.STATUS_FAILED
                if verbose:
                    print(f"  [damage/{cam}/{gw_id}] FAILED: {e}")
            finally:
                timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/damage] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
