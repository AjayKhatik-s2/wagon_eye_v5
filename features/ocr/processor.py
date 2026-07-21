"""OCR feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Milestone 1: uses the PRODUCTION wagon-number detector `wagon_number.pt` (via
core.production_models) with the production-lineage OCR engine below (the
6-step preprocessing + Indian-Railways confusion-map correction ARE production
behaviour). Loco-number (5-digit) OCR is a documented remaining item (needs the
loco-region detector wired) -- see the end-to-end summary.

Pipeline:
    1. YOLO `wagon_number.pt` detects wagon-number bbox regions on
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

import itertools
import os
import re
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from core import constants as C
from core import config as CFG
from core import production_models as PM
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, run_detection, iter_wagon_frames, crop_bbox, batched_detect,
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
# Loco-number OCR (production: 5-digit numbers on locomotives / ENGINE wagons)
# -----------------------------------------------------------------------------
# Production reads the 5-digit loco number from the gap+loco model's `locono`
# detections on RIGHT_UP. The reconstruction gap model (right_up_gap.pt) already
# emits `locono`; we run it on ENGINE wagons' RIGHT_UP frames and read the crop
# with the production-lineage OCR engine's preprocessing + reader.

_LOCO_NUMBER_LENGTH = 5
_LOCO_CONFIDENCE = 0.60                 # production LOCO_CONFIDENCE_THRESHOLD
_LOCO_CLASS = "locono"
_LOCO_DET_FILENAMES = ("right_up_gap.pt", "right_up_wagon_gap.pt")

_LOCO_DET_SINGLETON: Optional[Any] = None
_LOCO_DET_TRIED = False


def _get_loco_detector():
    """Cached loco-region detector (the reconstruction gap model, `locono`)."""
    global _LOCO_DET_SINGLETON, _LOCO_DET_TRIED
    if _LOCO_DET_TRIED:
        return _LOCO_DET_SINGLETON
    _LOCO_DET_TRIED = True
    for fn in _LOCO_DET_FILENAMES:
        m = load_yolo(os.path.join(CFG.RECON_MODELS_DIR, fn))
        if m is not None:
            _LOCO_DET_SINGLETON = m
            break
    return _LOCO_DET_SINGLETON


def _read_loco_number(ocr, crop) -> tuple[str, float]:
    """Read a 5-digit loco number from a `locono` crop using the production OCR
    engine's preprocessing + reader. Row-cluster digits top-to-bottom then
    left-to-right; return the number only when it is exactly 5 digits."""
    if ocr is None or getattr(ocr, "reader", None) is None or crop is None or crop.size == 0:
        return "", 0.0
    try:
        pre = ocr.preprocess_crop(crop)
        results = ocr.reader.readtext(pre, allowlist="0123456789")
    except Exception:
        return "", 0.0
    if not results:
        return "", 0.0

    def cy(r): return sum(p[1] for p in r[0]) / len(r[0])
    def cx(r): return sum(p[0] for p in r[0]) / len(r[0])
    ordered = sorted(results, key=lambda r: (round(cy(r) / 10.0), cx(r)))
    digits = "".join(re.sub(r"\D", "", str(r[1])) for r in ordered)
    confs = [float(r[2]) for r in ordered if r[2] is not None]
    conf = sum(confs) / len(confs) if confs else 0.0
    return (digits, conf) if len(digits) == _LOCO_NUMBER_LENGTH else ("", 0.0)


def _process_engine_loco(loco_model, ocr, cache_root: str, gw_id: str) -> Dict[str, Any]:
    """Detect `locono` on an ENGINE wagon's RIGHT_UP frames, OCR each crop, and
    vote for the dominant 5-digit loco number."""
    votes: Counter = Counter()
    conf_sum: Dict[str, float] = {}
    used = 0
    best = BestFrameTracker()
    for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP, trim_stable=True):
        used += 1
        try:
            res = loco_model(frame, verbose=False, half=HALF, device=DEVICE,
                             conf=_LOCO_CONFIDENCE)[0]
        except Exception:
            continue
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        names = getattr(loco_model, "names", {}) or {}
        for bbox, cid in zip(boxes, clss):
            if str(names.get(int(cid), "")).lower() != _LOCO_CLASS:
                continue
            bbox_list = [float(b) for b in bbox]
            crop = crop_bbox(frame, bbox_list, pad=6)
            num, ocr_conf = _read_loco_number(ocr, crop)
            if num:
                votes[num] += 1
                conf_sum[num] = conf_sum.get(num, 0.0) + ocr_conf
                if ocr_conf > best.score:
                    best.update(score=ocr_conf, frame=frame, bbox=bbox_list, frame_idx=fi,
                                loco_number=num, ocr_confidence=ocr_conf)
    if not votes:
        return {"used": used, "loco_number": "", "confidence": 0.0, "best": best, "votes": {}}
    winner = max(votes, key=lambda n: (votes[n], conf_sum[n] / votes[n]))
    return {"used": used, "loco_number": winner,
            "confidence": conf_sum[winner] / votes[winner],
            "best": best, "votes": dict(votes)}


# -----------------------------------------------------------------------------
# Per-wagon driver
# -----------------------------------------------------------------------------

def _ocr_one_detection(ocr, aggregator, raw_candidates, best, fi, frame,
                       bbox_list, yolo_conf):
    """Stage B per-detection OCR pipeline (preprocess + easyocr + validate +
    reconstruct + aggregate + best-frame).  Shared VERBATIM by the per-frame and
    batched-detection paths so EasyOCR recognition is byte-identical -- only the
    upstream YOLO *detection* is optionally batched."""
    crop = crop_bbox(frame, bbox_list, pad=10)
    if crop is None or crop.size == 0:
        return
    try:
        wagon_num = ocr.reconstruct_wagon_number(crop, float(yolo_conf), debug=False)
    except Exception:
        return
    if wagon_num is None:
        return
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
        # Track best snapshot: prefer full-length (11-digit) numbers, then conf.
        is_full = int(len(str(full)) == C.WAGON_NUMBER_LENGTH)
        score = is_full * 10.0 + ocr_conf
        best.update(
            score=score, frame=frame, bbox=bbox_list, frame_idx=fi,
            full_number=str(full), ocr_confidence=ocr_conf,
            yolo_confidence=float(yolo_conf), is_full_length=bool(is_full))


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

    if CFG.FEATURE_BATCH_SIZE > 1:
        # OPT-IN: batch ONLY the YOLO wagon-number DETECTION; EasyOCR recognition
        # stays per-detection (same crops, same order) -> recognition unchanged.
        gen = iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP, trim_stable=True)
        bs = CFG.FEATURE_BATCH_SIZE
        while True:
            chunk = list(itertools.islice(gen, bs))
            if not chunk:
                break
            used += len(chunk)
            frames = [fr for _, fr in chunk]
            # confidence=None -> model's own default conf (== per-frame model(frame));
            # the det_confidence (0.40) gate is applied below exactly as per-frame.
            per = batched_detect(yolo_model, frames, confidence=None)
            for (fi, frame), dets in zip(chunk, per):
                for d in dets:
                    yolo_conf = d["confidence"]
                    if yolo_conf < det_confidence:
                        continue
                    _ocr_one_detection(ocr, aggregator, raw_candidates, best,
                                       fi, frame, d["bbox"], float(yolo_conf))
    else:
        # === DEFAULT per-frame path -- VERBATIM pre-optimization behaviour ===
        for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP, trim_stable=True):
            used += 1
            # Stage A: YOLO detection -- locate wagon-number bbox regions
            try:
                results = yolo_model(frame, verbose=False, half=HALF, device=DEVICE)[0]
            except Exception:
                continue
            if results.boxes is None or len(results.boxes) == 0:
                continue
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            # Stage B: per-detection OCR (preprocess + easyocr + validate)
            for bbox, yolo_conf in zip(boxes, confs):
                if float(yolo_conf) < det_confidence:
                    continue
                bbox_list = [float(b) for b in bbox]
                _ocr_one_detection(ocr, aggregator, raw_candidates, best,
                                   fi, frame, bbox_list, float(yolo_conf))

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

    # PRODUCTION wagon-number bbox detector (wagon_number.pt) via the shared
    # registry. Absent -> clear error -> NO_DATA for every wagon (no fake read).
    del feature_models_dir  # model comes from core.production_models now
    model_err: Optional[str] = None
    try:
        yolo_model = PM.load_for(FEATURE_NAME, camera_id)
    except PM.MissingProductionModel as e:
        yolo_model = None
        model_err = str(e)
    ocr = _get_ocr()

    feature_out = feature_camera_dir(output_dir, FEATURE_NAME, camera_id)
    timer = FeatureTimer("ocr")
    summary: Dict[str, str] = {}

    if yolo_model is None and verbose:
        print(f"[FEAT/ocr] {model_err or 'wagon_number.pt missing'} -- NO_DATA for all wagons.")
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
                    error=model_err or "detector or OCR engine unavailable",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # BRAKE_VAN carries no standard number -> skip (record the entry).
            if gw.classification == C.CLASS_BRAKE_VAN:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
                    loco_number=C.NO_DATA, loco_number_confidence=0.0, is_valid_5_digit=False,
                    candidates=[], supporting_cameras=[C.CAMERA_RIGHT_UP],
                    skipped_reason="classification=BRAKE_VAN"))
                summary[gw_id] = C.STATUS_OK
                continue

            # ENGINE wagons -> production loco-number (5-digit) OCR.
            if gw.classification == C.CLASS_ENGINE:
                loco_model = _get_loco_detector()
                lo = (_process_engine_loco(loco_model, ocr, cache_root, gw_id)
                      if loco_model is not None else None)
                loco_num = (lo or {}).get("loco_number") or ""
                loco_conf = float((lo or {}).get("confidence", 0.0) or 0.0)
                is_valid5 = len(loco_num) == _LOCO_NUMBER_LENGTH
                loco_ev: Dict[str, str] = {}
                bo = (lo or {}).get("best")
                if evidence_root and is_valid5 and bo is not None and bo.has_data():
                    final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
                    crop_img = safe_crop(bo.frame, bo.bbox, pad=4)
                    with atomic_camera_evidence(evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
                        annotated = draw_annotated_bbox(
                            bo.frame, bo.bbox, label=f"LOCO {loco_num} {loco_conf:.2f}",
                            color=(0, 255, 0))
                        save_jpeg(os.path.join(ev_tmp, "loco_best.jpg"), annotated)
                        if crop_img is not None:
                            save_jpeg(os.path.join(ev_tmp, "loco_crop.jpg"), crop_img)
                        write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                            "global_id": gw_id, "feature": FEATURE_NAME, "camera_id": camera_id,
                            "loco_number": loco_num, "loco_confidence": round(loco_conf, 4),
                            "frame_idx": bo.frame_idx, "bbox": bo.bbox})
                    loco_ev["loco_best"] = os.path.join(final_dir, "loco_best.jpg")
                    if crop_img is not None:
                        loco_ev["loco_crop"] = os.path.join(final_dir, "loco_crop.jpg")
                write_per_wagon_json(feature_out, gw_id, {
                    "global_id": gw_id, "feature": FEATURE_NAME, "camera_id": camera_id,
                    "status": C.STATUS_OK,
                    "wagon_identifier": C.NO_DATA, "wagon_identifier_confidence": 0.0,
                    "loco_number": loco_num or C.NO_DATA,
                    "loco_number_confidence": round(loco_conf, 4),
                    "is_valid_5_digit": is_valid5,
                    "loco_vote_counts": (lo or {}).get("votes", {}),
                    "candidates": [], "supporting_cameras": [C.CAMERA_RIGHT_UP],
                    "skipped_reason": "classification=ENGINE (loco OCR)",
                    "frame_count": int((lo or {}).get("used", 0)),
                    "evidence": loco_ev,
                })
                summary[gw_id] = C.STATUS_OK
                if verbose:
                    print(f"  [ocr/{gw_id}] ENGINE loco={loco_num or '-'} ({loco_conf:.2f})")
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
