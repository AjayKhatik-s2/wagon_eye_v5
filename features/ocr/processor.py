"""OCR feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Pipeline:
    1. YOLO `wagon_id_counting.pt` detects wagon-number bbox regions on
       RIGHT_UP frames (master / OCR authority).
    2. Each crop is fed through the legacy `WagonNumberOCR`:
           padding 10 -> 3x cubic upscale -> NLMeans denoise (h=8) ->
           CLAHE (clipLimit=3.5, tile 8x8) -> unsharp masking ->
           easyocr (allowlist='0123456789') -> digit extraction ->
           wagon-type confusion-map correction (first 2 digits in 10-39)
           -> WagonNumberValidator (length=11, structure check).
    3. Surviving candidates per frame are added to the legacy
       `WagonNumberAggregator` which performs:
           exact-string grouping with digit-level voting at each position
           min 2 frames + min OCR conf 0.3
    4. Best aggregated number is picked by (observations, mean conf).

Output JSON shape:
    {
        "global_id":  "GW_7",
        "feature":    "ocr",
        "status":     "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "wagon_identifier":  "32145678901",
        "wagon_identifier_confidence": 0.83,
        "candidates":  [...],
        "supporting_cameras": ["RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, run_detection, iter_wagon_frames, crop_bbox,
    write_per_wagon_json, empty_payload, FeatureTimer, feature_camera_dir,
    DEVICE, HALF,
)

# Mature intelligence ported from legacy
from features.inference_lib.wagon_number_ocr import WagonNumberOCR, WagonNumber
from features.inference_lib.wagon_number_aggregator import (
    WagonNumberAggregator, AggregatorConfig,
)
from features._evidence import (
    BestFrameTracker, atomic_camera_evidence,
    save_jpeg, safe_crop, write_metadata, draw_annotated_bbox,
)


FEATURE_NAME = "ocr"


# -----------------------------------------------------------------------------
# Per-process singleton (easyocr Reader is heavy; load once)
# -----------------------------------------------------------------------------

_OCR_SINGLETON: Optional[WagonNumberOCR] = None


def _get_ocr() -> Optional[WagonNumberOCR]:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    try:
        _OCR_SINGLETON = WagonNumberOCR(
            # CUDA if available, else CPU.  Was hardcoded True, which errors /
            # silently degrades on a CPU-only EC2 host; now matches the
            # resolved device (still GPU on a GPU box -- identical behaviour).
            use_gpu=(DEVICE == "cuda"),
            min_confidence=0.30,        # legacy default for cross-frame aggregation
            resize_factor=3.0,
        )
        if getattr(_OCR_SINGLETON, "reader", None) is None:
            _OCR_SINGLETON = None
    except Exception as e:
        print(f"[FEAT/ocr] WagonNumberOCR init failed: {e}")
        _OCR_SINGLETON = None
    return _OCR_SINGLETON


# -----------------------------------------------------------------------------
# Per-wagon driver
# -----------------------------------------------------------------------------

def _process_one_wagon(
    yolo_model,
    ocr: WagonNumberOCR,
    cache_root: str,
    gw_id: str,
    det_confidence: float,
) -> Dict[str, Any]:
    """Iterate cached RIGHT_UP frames, run YOLO + OCR, aggregate."""
    aggregator = WagonNumberAggregator(AggregatorConfig(
        min_frame_count=2,
        min_confidence=0.3,
        require_validation=True,
    ))

    used = 0
    raw_candidates: List[Dict[str, Any]] = []
    best = BestFrameTracker()    # remembers the highest-conf OCR snapshot

    for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP, trim_stable=True):
        used += 1

        # Stage A: YOLO detection -- locate wagon-number bbox regions
        try:
            results = yolo_model(frame, verbose=False, half=True)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # Stage B: per-detection OCR pipeline (preprocess + easyocr +
        # validate + reconstruct)
        for bbox, yolo_conf in zip(boxes, confs):
            if float(yolo_conf) < det_confidence:
                continue
            bbox_list = [float(b) for b in bbox]
            crop = crop_bbox(frame, bbox_list, pad=10)
            if crop is None or crop.size == 0:
                continue
            try:
                wagon_num = ocr.reconstruct_wagon_number(
                    crop, float(yolo_conf), debug=False,
                )
            except Exception:
                continue
            if wagon_num is None:
                continue
            aggregator.add_wagon_number(wagon_num, frame_idx=fi)

            full = getattr(wagon_num, "full_number", None)
            if full:
                ocr_conf = float(getattr(wagon_num, "ocr_confidence", 0.0))
                raw_candidates.append({
                    "frame_idx":       int(fi),
                    "full_number":     str(full),
                    "ocr_confidence":  ocr_conf,
                    "yolo_confidence": float(getattr(wagon_num, "yolo_confidence", 0.0)),
                    "bbox":            bbox_list,
                })
                # Track best snapshot:  prefer full-length (11-digit) numbers
                # and within that bucket, highest OCR confidence.
                is_full = int(len(str(full)) == C.WAGON_NUMBER_LENGTH)
                score = is_full * 10.0 + ocr_conf
                best.update(
                    score=score, frame=frame, bbox=bbox_list,
                    frame_idx=fi,
                    full_number=str(full),
                    ocr_confidence=ocr_conf,
                    yolo_confidence=float(yolo_conf),
                    is_full_length=bool(is_full),
                )

    # Stage C: pick the dominant aggregated wagon number
    aggregated = aggregator.get_aggregated_numbers()
    return {
        "frame_count": used,
        "aggregated":  aggregated,
        "raw":         raw_candidates,
        "best":        best,
    }


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    det_confidence: float = C.CONF_OCR_BOX,
    wagon_number_length: int = C.WAGON_NUMBER_LENGTH,
    every_nth: int = 1,
    max_frames: int = 0,
    verbose: bool = True,
) -> Dict[str, str]:
    """Run OCR on every wagon (RIGHT_UP only), writing the per-camera layout
    wagon_states/ocr/RIGHT_UP/<gw>.json and evidence/<gw>/ocr/RIGHT_UP/."""
    del every_nth, max_frames, wagon_number_length  # legacy code uses its own thresholds

    # OCR authority is RIGHT_UP only.  If a caller scopes to other cameras,
    # there is nothing for OCR to do.
    camera_id = C.CAMERA_RIGHT_UP
    if cameras is not None and camera_id not in cameras:
        return {}

    model_path = os.path.join(feature_models_dir, C.MODEL_WAGON_ID_COUNTING)
    yolo_model = load_yolo(model_path)
    ocr = _get_ocr()

    feature_out = feature_camera_dir(output_dir, FEATURE_NAME, camera_id)
    timer = FeatureTimer("ocr")
    summary: Dict[str, str] = {}

    if yolo_model is None and verbose:
        print(f"[FEAT/ocr] WARNING: {model_path} missing -- NO_DATA for all wagons.")
    if ocr is None and verbose:
        print(f"[FEAT/ocr] WARNING: easyocr unavailable -- NO_DATA for all wagons.")

    if verbose:
        print(f"[FEAT/ocr] running on {len(state.wagons)} wagons "
              f"(legacy WagonNumberOCR + WagonNumberAggregator, RIGHT_UP only)")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None or ocr is None:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[], supporting_cameras=[],
                    error="detector or OCR engine unavailable",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # ENGINE / BRAKE_VAN wagons rarely carry the standard 11-digit
            # wagon number; running OCR on them produces noise.  Skip but
            # still record the wagon entry.
            if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[C.CAMERA_RIGHT_UP],
                    skipped_reason=f"classification={gw.classification}",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            outcome = _process_one_wagon(
                yolo_model, ocr, cache_root, gw_id, det_confidence,
            )
            used = outcome["frame_count"]
            aggregated = outcome["aggregated"]

            if used == 0:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            # Build serialized candidate list from the aggregator's output
            candidates_out: List[Dict[str, Any]] = []
            for agg in aggregated:
                candidates_out.append({
                    "full_number":     str(getattr(agg, "wagon_number", "")),
                    "observations":    int(getattr(agg, "frame_count", 0)),
                    "mean_conf":       float(getattr(agg, "avg_confidence", 0.0)),
                    "yolo_conf":       float(getattr(agg, "avg_yolo_confidence",
                                              getattr(agg, "yolo_confidence", 0.0))),
                    "is_full_length":  len(str(getattr(agg, "wagon_number", "")))
                                       == C.WAGON_NUMBER_LENGTH,
                })

            # Aggregator already enforces min_frame_count + min_confidence.
            # The "best" candidate is the one with the highest combined
            # (observations, mean_conf) score.
            candidates_out.sort(
                key=lambda c: (
                    -int(c["is_full_length"]),
                    -c["observations"],
                    -c["mean_conf"],
                    c["full_number"],
                )
            )

            if candidates_out and candidates_out[0]["is_full_length"]:
                top = candidates_out[0]
                ident = top["full_number"]
                conf  = top["mean_conf"]
            else:
                ident = C.NO_DATA
                conf  = 0.0

            # Persist best-frame evidence:  full annotated frame +
            # tight crop of the wagon-number plate.
            evidence_paths: Dict[str, str] = {}
            best_obj = outcome.get("best")
            if evidence_root and best_obj is not None and best_obj.has_data():
                final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
                crop_img = safe_crop(best_obj.frame, best_obj.bbox, pad=4)
                with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME,
                                            camera_id) as ev_tmp:
                    annotated = draw_annotated_bbox(
                        best_obj.frame, best_obj.bbox,
                        label=f"OCR {best_obj.meta.get('full_number','?')} "
                              f"{best_obj.meta.get('ocr_confidence',0.0):.2f}",
                        color=(0, 255, 0),
                    )
                    save_jpeg(os.path.join(ev_tmp, "best_frame.jpg"), annotated)
                    if crop_img is not None:
                        save_jpeg(os.path.join(ev_tmp, "number_crop.jpg"), crop_img)
                    write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                        "global_id":       gw_id,
                        "feature":         FEATURE_NAME,
                        "camera_id":       camera_id,
                        "frame_idx":       best_obj.frame_idx,
                        "bbox":            best_obj.bbox,
                        "full_number":     best_obj.meta.get("full_number"),
                        "ocr_confidence":  best_obj.meta.get("ocr_confidence"),
                        "yolo_confidence": best_obj.meta.get("yolo_confidence"),
                        "is_full_length":  best_obj.meta.get("is_full_length"),
                        "aggregated_winner": ident,
                        "aggregated_confidence": conf,
                    })
                evidence_paths["best_frame"] = os.path.join(final_dir, "best_frame.jpg")
                if crop_img is not None:
                    evidence_paths["number_crop"] = os.path.join(final_dir, "number_crop.jpg")

            payload: Dict[str, Any] = {
                "global_id":   gw_id,
                "feature":     FEATURE_NAME,
                "camera_id":   camera_id,
                "status":      C.STATUS_OK,
                "wagon_identifier":            ident,
                "wagon_identifier_confidence": round(float(conf), 4),
                "candidates":  candidates_out[:8],
                "raw_candidates_first_8":      outcome["raw"][:8],
                "supporting_cameras": [C.CAMERA_RIGHT_UP],
                "frame_count": used,
                "evidence":    evidence_paths,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [ocr/{gw_id}] {ident} (conf={conf:.2f}, "
                      f"candidates={len(candidates_out)}, frames={used})")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                wagon_identifier=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [ocr/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/ocr] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
