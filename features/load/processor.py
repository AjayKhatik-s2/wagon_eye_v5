"""Load feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Legacy logic from `RIGHT_UP_TOP/damage_processor.py:1038-1087` (per-wagon
frame-by-frame loaded/empty voting):
    is_loaded = (loaded_count / total_count) > 0.35

Plus three additional guards that came along for the ride in the
legacy code:
    * fall back to a YOLO-detection-style classifier when `loaded.pt`
      emits boxes instead of probs (some loaded variants do this)
    * default to EMPTY if the model can't be reached at all
    * skip ENGINE / BRAKE_VAN wagons -- they don't carry payload, and
      running the loaded model on them produces meaningless votes

Per-wagon JSON shape:
    {
        "global_id":   "GW_7",
        "feature":     "load",
        "status":      "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "load_status": "LOADED" | "EMPTY" | "NO_DATA",
        "load_confidence": 0.85,
        "per_camera": {
            "RIGHT_UP_TOP": {load_status, confidence, loaded_count,
                             empty_count, frames_used, loaded_ratio},
            "LEFT_UP_TOP":  {...}
        },
        "supporting_cameras": [...],
        "frame_count": ...,
    }

Authority rule (Stage 2 fusion respects this too): RIGHT_UP_TOP is
authoritative; LEFT_UP_TOP is supporting / fallback.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, run_classification, iter_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer, feature_camera_dir,
)
from features._evidence import (
    BestFrameTracker, atomic_camera_evidence,
    save_jpeg, write_metadata,
)


FEATURE_NAME = "load"


# Legacy threshold from RIGHT_UP_TOP/damage_processor.py:1047
_LOADED_RATIO_THRESHOLD = 0.35


def _canonical_load(raw: str) -> str:
    return C.LOAD_LABEL_TO_STATE.get((raw or "").strip().lower(), C.NO_DATA)


def _aggregate_camera(
    model, cache_root: str, gw_id: str, camera_id: str,
    *, every_nth: int, max_frames: int,
) -> Tuple[str, float, int, int, int, "BestFrameTracker", "BestFrameTracker"]:
    """Return (load_status, confidence, frames_used, loaded_count, empty_count,
               best_loaded_frame, best_empty_frame).

    Follows the legacy frame-by-frame voting rule:
        is_loaded = (loaded_count / total_count) > 0.35
    Confidence is the mean top-1 probability of frames that voted with
    the winning side.

    Best-frame trackers retain the highest-conf frame on each side for
    evidence persistence.
    """
    loaded_confs: List[float] = []
    empty_confs:  List[float] = []
    used = 0
    best_loaded = BestFrameTracker()
    best_empty  = BestFrameTracker()

    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id,
                                       every_nth=every_nth,
                                       max_frames=max_frames,
                                       trim_stable=True):
        cls, conf = run_classification(model, frame)
        cls_canon = _canonical_load(cls)
        used += 1
        if cls_canon == C.LOAD_LOADED:
            loaded_confs.append(conf)
            best_loaded.update(score=float(conf), frame=frame,
                               frame_idx=int(fi),
                               camera_id=camera_id, class_name=cls,
                               confidence=float(conf))
        elif cls_canon == C.LOAD_EMPTY:
            empty_confs.append(conf)
            best_empty.update(score=float(conf), frame=frame,
                              frame_idx=int(fi),
                              camera_id=camera_id, class_name=cls,
                              confidence=float(conf))

    if used == 0:
        return C.NO_DATA, 0.0, 0, 0, 0, best_loaded, best_empty

    n_loaded = len(loaded_confs)
    n_empty  = len(empty_confs)
    total    = max(1, used)
    loaded_ratio = n_loaded / total

    if loaded_ratio > _LOADED_RATIO_THRESHOLD and n_loaded > 0:
        mean_conf = sum(loaded_confs) / n_loaded
        return C.LOAD_LOADED, float(mean_conf), used, n_loaded, n_empty, best_loaded, best_empty
    if n_empty > 0:
        mean_conf = sum(empty_confs) / n_empty
        return C.LOAD_EMPTY, float(mean_conf), used, n_loaded, n_empty, best_loaded, best_empty
    return C.NO_DATA, 0.0, used, n_loaded, n_empty, best_loaded, best_empty


def _process_wagon_camera(
    model, cache_root: str, gw, camera_id: str, feature_out: str,
    evidence_root: Optional[str], every_nth: int, max_frames: int,
) -> str:
    """Aggregate ONE top camera for ONE wagon and write
    load/<CAMERA>/<gw>.json.  Returns the status string.

    This is a single-camera vote only -- cross-camera load authority
    (RIGHT_UP_TOP primary, LEFT_UP_TOP fallback) is applied in Stage 4."""
    gw_id = gw.global_id

    # ENGINE / BRAKE_VAN gate: no payload to classify.
    if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_OK,
            camera_id=camera_id, load_status=C.NO_DATA, load_confidence=0.0,
            supporting_cameras=[], skipped_reason=f"classification={gw.classification}",
        ))
        return C.STATUS_OK

    cls, conf, used, n_l, n_e, b_l, b_e = _aggregate_camera(
        model, cache_root, gw_id, camera_id,
        every_nth=every_nth, max_frames=max_frames,
    )
    if used == 0:
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            camera_id=camera_id, load_status=C.NO_DATA, load_confidence=0.0,
            supporting_cameras=[],
        ))
        return C.STATUS_NO_FRAMES

    # Evidence: strongest frame of the winning class for THIS camera only.
    evidence_paths: Dict[str, str] = {}
    if evidence_root and cls != C.NO_DATA:
        winning = b_l if cls == C.LOAD_LOADED else b_e
        if winning.has_data():
            final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
            with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME,
                                        camera_id) as ev_tmp:
                save_jpeg(os.path.join(ev_tmp, "best_frame.jpg"), winning.frame)
                write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                    "global_id": gw_id, "feature": FEATURE_NAME,
                    "camera_id": camera_id, "load_status": cls,
                    "confidence": conf, "best_frame_idx": winning.frame_idx,
                    "best_class": winning.meta.get("class_name"),
                    "best_confidence": winning.meta.get("confidence"),
                })
            evidence_paths["best_frame"] = os.path.join(final_dir, "best_frame.jpg")

    write_per_wagon_json(feature_out, gw_id, {
        "global_id": gw_id,
        "feature":   FEATURE_NAME,
        "camera_id": camera_id,
        "status":    C.STATUS_OK,
        "load_status":     cls,
        "load_confidence": round(float(conf), 4),
        "loaded_count": n_l,
        "empty_count":  n_e,
        "frames_used":  used,
        "loaded_ratio": round(n_l / used, 4) if used else 0.0,
        "supporting_cameras": [camera_id],
        "frame_count": used,
        "evidence":    evidence_paths,
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
    max_frames: int = 0,    # legacy walks the full window; 0 = unbounded
    verbose: bool = True,
) -> Dict[str, str]:
    """Run load classification per top camera, writing the per-camera layout
    wagon_states/load/<CAMERA>/<gw>.json.  Each camera is INDEPENDENT; Stage 4
    applies RIGHT_UP_TOP-primary / LEFT_UP_TOP-fallback authority."""
    model_path = os.path.join(feature_models_dir, C.MODEL_LOADED)
    model = load_yolo(model_path)

    target_cams = [c for c in C.TOP_CAMERAS if (cameras is None or c in cameras)]
    if not target_cams:
        return {}
    timer = FeatureTimer("load")
    summary: Dict[str, str] = {}

    if model is None and verbose:
        print(f"[FEAT/load] WARNING: {model_path} missing -- NO_DATA for all wagons.")
    if verbose:
        print(f"[FEAT/load] running on {len(state.wagons)} wagons, cameras={target_cams} "
              f"(per-camera voting, >{_LOADED_RATIO_THRESHOLD:.0%} -> LOADED)")

    for cam in target_cams:
        feature_out = feature_camera_dir(output_dir, FEATURE_NAME, cam)
        for gw in state.wagons:
            gw_id = gw.global_id
            t0 = time.time()
            try:
                if model is None:
                    write_per_wagon_json(feature_out, gw_id, empty_payload(
                        gw_id, FEATURE_NAME, C.NO_DATA,
                        camera_id=cam, load_status=C.NO_DATA, load_confidence=0.0,
                        supporting_cameras=[], error="loaded.pt not present",
                    ))
                    summary[gw_id] = C.NO_DATA
                    continue
                summary[gw_id] = _process_wagon_camera(
                    model, cache_root, gw, cam, feature_out,
                    evidence_root, every_nth, max_frames,
                )
            except Exception as e:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_FAILED,
                    camera_id=cam, load_status=C.NO_DATA,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(limit=2),
                ))
                summary[gw_id] = C.STATUS_FAILED
                if verbose:
                    print(f"  [load/{cam}/{gw_id}] FAILED: {e}")
            finally:
                timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/load] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
