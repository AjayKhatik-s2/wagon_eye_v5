"""Load feature processor -- PRODUCTION behaviour on the GlobalTrain wagon cache.

Milestone 1 reproduces the production load determination. Production has NO
dedicated load model: load status is derived from the TOP-camera CLASSIFICATION
model (`ltop.pt` for LEFT_UP_TOP, `top_classification.pt` for RIGHT_UP_TOP)
that emits `wagon_empty` / `wagon_loaded` (+ engine/brakevan/track), via the
production `classify_segment_type` majority vote. This processor runs that model
on the top-camera cache frames and votes LOADED vs EMPTY.

Production `classify_segment_type` parameters (segments.py):
  * frames sampled every OTHER frame (step 2),
  * a fixed 5-frame margin skipped at each end (safe_start = start+5,
    safe_end = end-5),
  * only predictions with confidence >= classification_confidence (0.80) vote,
  * the majority class wins.
Here the vote is restricted to the load-relevant classes: `wagon_loaded` /
`wagon_filled` -> LOADED, `wagon_empty` / `wagon` -> EMPTY. LOADED when the
loaded votes outnumber the empty votes, else EMPTY, else NO_DATA. Confidence is
the mean top-1 probability of the winning side.

ENGINE / BRAKE_VAN wagons (from the sealed GlobalTrainState) carry no payload
and are short-circuited. Cross-camera authority (RIGHT_UP_TOP primary,
LEFT_UP_TOP fallback) is applied by Stage-4 fusion, unchanged.

The EXTERNAL CONTRACT is preserved: `run()` signature, per-camera output
`wagon_states/load/<CAMERA>/GW_n.json` with keys `load_status`,
`load_confidence`, `loaded_count`, `empty_count`, `frames_used`, `loaded_ratio`,
`supporting_cameras`, `frame_count`, `evidence`; and the `best_frame` evidence
layout. Graceful `NO_DATA` when the production model is absent.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2

from core import constants as C
from core import production_models as PM
from core.global_state_loader import GlobalTrainState

from features._common import (
    run_classification, list_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer, feature_camera_dir,
)
from features._evidence import (
    BestFrameTracker, atomic_camera_evidence, save_jpeg, write_metadata,
)


FEATURE_NAME = "load"

# Production classify_segment_type parameters (segments.py).
_CLASSIFICATION_CONFIDENCE = 0.80
_CLASSIFY_EDGE_SKIP = 5
_CLASSIFY_STEP = 2

# Load-relevant class canonicalization (production class aliases:
# wagon_empty->wagon, wagon_filled->wagon_loaded; wagon == empty load).
_LOAD_LOADED_CLASSES = {"wagon_loaded", "wagon_filled", "loaded", "load", "full"}
_LOAD_EMPTY_CLASSES = {"wagon_empty", "wagon", "empty", "unload"}


def _canonical_load(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r in _LOAD_LOADED_CLASSES:
        return C.LOAD_LOADED
    if r in _LOAD_EMPTY_CLASSES:
        return C.LOAD_EMPTY
    return C.NO_DATA


def _classify_frames(paths: List[str]) -> List[str]:
    """Production classify_segment_type frame selection: skip a fixed 5-frame
    margin at each end, then take every other frame."""
    n = len(paths)
    if n <= 2 * _CLASSIFY_EDGE_SKIP:
        return []
    return paths[_CLASSIFY_EDGE_SKIP:n - _CLASSIFY_EDGE_SKIP:_CLASSIFY_STEP]


def _parse_frame_index(path: str) -> int:
    try:
        return int(os.path.basename(path).split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def _aggregate_camera(
    model, cache_root: str, gw_id: str, camera_id: str,
) -> Tuple[str, float, int, int, int, "BestFrameTracker", "BestFrameTracker"]:
    """Production majority vote of wagon_loaded vs wagon_empty for one camera.

    Returns (load_status, confidence, frames_used, loaded_count, empty_count,
             best_loaded_frame, best_empty_frame)."""
    paths = list_wagon_frames(cache_root, gw_id, camera_id)
    frames = _classify_frames(paths)

    loaded_confs: List[float] = []
    empty_confs: List[float] = []
    used = 0
    best_loaded = BestFrameTracker()
    best_empty = BestFrameTracker()

    for p in frames:
        frame = cv2.imread(p)
        if frame is None:
            continue
        used += 1
        cls, conf = run_classification(model, frame)
        # Production gate: only confident predictions vote.
        if conf < _CLASSIFICATION_CONFIDENCE:
            continue
        canon = _canonical_load(cls)
        fi = _parse_frame_index(p)
        if canon == C.LOAD_LOADED:
            loaded_confs.append(conf)
            best_loaded.update(score=float(conf), frame=frame, frame_idx=int(fi),
                               camera_id=camera_id, class_name=cls, confidence=float(conf))
        elif canon == C.LOAD_EMPTY:
            empty_confs.append(conf)
            best_empty.update(score=float(conf), frame=frame, frame_idx=int(fi),
                              camera_id=camera_id, class_name=cls, confidence=float(conf))

    n_loaded = len(loaded_confs)
    n_empty = len(empty_confs)
    if n_loaded == 0 and n_empty == 0:
        return C.NO_DATA, 0.0, used, 0, 0, best_loaded, best_empty
    # Production majority (classify_segment_type winner = max vote count).
    if n_loaded > n_empty:
        return (C.LOAD_LOADED, float(sum(loaded_confs) / n_loaded), used,
                n_loaded, n_empty, best_loaded, best_empty)
    return (C.LOAD_EMPTY, float(sum(empty_confs) / n_empty), used,
            n_loaded, n_empty, best_loaded, best_empty)


def _process_wagon_camera(
    model, cache_root: str, gw, camera_id: str, feature_out: str,
    evidence_root: Optional[str],
) -> str:
    gw_id = gw.global_id

    # ENGINE / BRAKE_VAN gate (from sealed GlobalTrainState): no payload.
    if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_OK,
            camera_id=camera_id, load_status=C.NO_DATA, load_confidence=0.0,
            supporting_cameras=[], skipped_reason=f"classification={gw.classification}",
        ))
        return C.STATUS_OK

    cls, conf, used, n_l, n_e, b_l, b_e = _aggregate_camera(
        model, cache_root, gw_id, camera_id)
    if used == 0:
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            camera_id=camera_id, load_status=C.NO_DATA, load_confidence=0.0,
            supporting_cameras=[],
        ))
        return C.STATUS_NO_FRAMES

    evidence_paths: Dict[str, str] = {}
    if evidence_root and cls != C.NO_DATA:
        winning = b_l if cls == C.LOAD_LOADED else b_e
        if winning.has_data():
            final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
            with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
                save_jpeg(os.path.join(ev_tmp, "best_frame.jpg"), winning.frame)
                write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                    "global_id": gw_id, "feature": FEATURE_NAME,
                    "camera_id": camera_id, "load_status": cls, "confidence": conf,
                    "best_frame_idx": winning.frame_idx,
                    "best_class": winning.meta.get("class_name"),
                    "best_confidence": winning.meta.get("confidence"),
                })
            evidence_paths["best_frame"] = os.path.join(final_dir, "best_frame.jpg")

    write_per_wagon_json(feature_out, gw_id, {
        "global_id": gw_id,
        "feature": FEATURE_NAME,
        "camera_id": camera_id,
        "status": C.STATUS_OK,
        "load_status": cls,
        "load_confidence": round(float(conf), 4),
        "loaded_count": n_l,
        "empty_count": n_e,
        "frames_used": used,
        "loaded_ratio": round(n_l / used, 4) if used else 0.0,
        "supporting_cameras": [camera_id],
        "frame_count": used,
        "evidence": evidence_paths,
    })
    return C.STATUS_OK


def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    every_nth: int = 2,
    max_frames: int = 0,
    verbose: bool = True,
) -> Dict[str, str]:
    """Run production load classification per top camera, writing
    wagon_states/load/<CAMERA>/<gw>.json. Signature preserved for the
    orchestrator; `feature_models_dir`/`every_nth`/`max_frames` are accepted but
    not used (the model is the PRODUCTION top-classification model and the frame
    sampling follows production classify_segment_type)."""
    del feature_models_dir, every_nth, max_frames

    target_cams = [c for c in C.TOP_CAMERAS if (cameras is None or c in cameras)]
    if not target_cams:
        return {}
    timer = FeatureTimer("load")
    summary: Dict[str, str] = {}

    if verbose:
        print(f"[FEAT/load] running on {len(state.wagons)} wagons, cameras={target_cams} "
              f"(PRODUCTION top classification; majority wagon_loaded vs wagon_empty; "
              f"conf>={_CLASSIFICATION_CONFIDENCE})")

    for cam in target_cams:
        feature_out = feature_camera_dir(output_dir, FEATURE_NAME, cam)
        model = None
        model_err: Optional[str] = None
        try:
            model = PM.load_for(FEATURE_NAME, cam, task="classify")
        except PM.MissingProductionModel as e:
            model_err = str(e)
            if verbose:
                print(f"[FEAT/load] {e} -- emitting NO_DATA for {cam}")

        for gw in state.wagons:
            gw_id = gw.global_id
            t0 = time.time()
            try:
                if model is None:
                    write_per_wagon_json(feature_out, gw_id, empty_payload(
                        gw_id, FEATURE_NAME, C.NO_DATA,
                        camera_id=cam, load_status=C.NO_DATA, load_confidence=0.0,
                        supporting_cameras=[], error=model_err))
                    summary[gw_id] = C.NO_DATA
                    continue
                summary[gw_id] = _process_wagon_camera(
                    model, cache_root, gw, cam, feature_out, evidence_root)
            except Exception as e:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_FAILED,
                    camera_id=cam, load_status=C.NO_DATA,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=2)))
                summary[gw_id] = C.STATUS_FAILED
                if verbose:
                    print(f"  [load/{cam}/{gw_id}] FAILED: {e}")
            finally:
                timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/load] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
