"""Shared low-level primitives for every feature processor.

Each processor (door, load, damage, ocr) is a small ~150-line module
that uses only these helpers + its own aggregation rule:

    1. `load_yolo(path)` -- lazy + cached YOLO loader (one load per
       process per .pt file).
    2. `iter_wagon_frames(...)` -- iterate frame_NNNNNN.jpg files in a
       wagon-cache folder.
    3. `run_detection(...)` / `run_classification(...)` -- one-line
       calls to ultralytics that return clean dicts.
    4. `majority_vote(...)` / `confidence_mean(...)` -- deterministic
       aggregation helpers.
    5. `write_per_wagon_json(...)` -- consistent JSON shape with status
       sentinels (`OK` / `NO_FRAMES` / `FAILED`).
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import threading
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from core import constants as C
from core import config as CFG


# -----------------------------------------------------------------------------
# Inference device -- resolved once for the whole process (CUDA if available,
# else CPU; WAGONEYE_DEVICE overrides).  Before the EC2 migration nothing
# branched on device: ultralytics guessed and half-precision was hardcoded ON,
# which is a footgun on a CPU-only host.  DEVICE + HALF make the choice explicit
# while preserving the exact GPU behaviour (DEVICE='cuda' => HALF=True).
# -----------------------------------------------------------------------------

DEVICE = CFG.resolve_device()
HALF = CFG.use_half_precision(DEVICE)


# -----------------------------------------------------------------------------
# YOLO loader cache
# -----------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def load_yolo(model_path: str):
    """Cached YOLO loader.  Returns None if the file is missing -- the
    caller is expected to short-circuit to NO_DATA in that case.

    Patches `torch.load` once on first call so .pt weights load on
    torch >= 2.6 (the same monkey-patch used by wagon_count).
    """
    # Transparent sync: pull the weight from the models bucket if it is missing
    # locally (no-op when already present).  Loading behaviour below is unchanged.
    if model_path:
        from core import model_sync
        model_sync.ensure_local(model_path)
    if not model_path or not os.path.isfile(model_path):
        return None

    abspath = os.path.abspath(model_path)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(abspath)
        if cached is not None:
            return cached

        # torch.load shim for newer torch versions
        import torch
        _orig_load = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _patched

        from ultralytics import YOLO
        model = YOLO(abspath)
        _MODEL_CACHE[abspath] = model
        return model


def model_class_names(model) -> Dict[int, str]:
    """Return YOLO's class-id -> name mapping for the loaded model."""
    if model is None:
        return {}
    return dict(getattr(model, "names", {}) or {})


# -----------------------------------------------------------------------------
# Frame iteration
# -----------------------------------------------------------------------------

def wagon_camera_dir(cache_root: str, gw_id: str, camera_id: str) -> str:
    """Path to wagon_cache/<gw>/<camera_folder>/."""
    return os.path.join(cache_root, gw_id, C.CAMERA_FOLDER[camera_id])


# -----------------------------------------------------------------------------
# Stable-interior trimming
# -----------------------------------------------------------------------------
# A wagon's first/last frames are the noisiest part of its pass (entering /
# leaving view, motion blur, partial occlusion at the gap).  For FEATURE
# INFERENCE we drop a margin from each end and use only the stable interior.
# The full span is still used for processed-video rendering and report
# continuity (those read the raw video / evidence, not this iterator).
#
# The trim is adaptive: 5% of the span, clamped to [3, 12] frames per side,
# so it scales with variable wagon durations (speed / visibility / geometry).

_STABLE_TRIM_FRACTION = 0.05
_STABLE_TRIM_MIN = 3
_STABLE_TRIM_MAX = 12


def stable_trim_count(span_length: int) -> int:
    """Frames to trim from EACH end of a wagon span for stable inference.

        trim_k = int(span_length * 0.05)   clamped to [3, 12]

    Returns 0 (no trim) when the span is too short to leave a usable interior
    (2*trim_k would consume the whole span) so short wagons still get inference
    instead of dropping to NO_DATA.
    """
    if span_length <= 0:
        return 0
    trim_k = int(span_length * _STABLE_TRIM_FRACTION)
    trim_k = max(_STABLE_TRIM_MIN, min(_STABLE_TRIM_MAX, trim_k))
    if span_length <= 2 * trim_k:
        return 0
    return trim_k


def stable_interior(paths: List[str]) -> List[str]:
    """Symmetric stable interior of a sorted frame-path list (see
    stable_trim_count).  Returns the list unchanged when the span is too short
    to trim."""
    k = stable_trim_count(len(paths))
    if k <= 0:
        return paths
    return paths[k:len(paths) - k]


# Directory-listing cache: the same wagon-cache dir is listed by several
# features (e.g. door + side-damage both read RIGHT_UP/<gw>/; load + top-damage
# both read the top dirs).  Caching the sorted glob per (dir, mtime_ns) turns
# those redundant scans into a single stat -- byte-identical output, and mtime
# invalidation means any materializer add/remove is picked up.  Thread-safe
# (features run in parallel threads).  Default-on; alters no outputs.
_LISTING_CACHE: Dict[str, Tuple[int, List[str]]] = {}
_LISTING_LOCK = threading.Lock()


def _cached_sorted_frames(d: str) -> List[str]:
    try:
        mtime = os.stat(d).st_mtime_ns
    except OSError:
        return []
    with _LISTING_LOCK:
        ent = _LISTING_CACHE.get(d)
        if ent is not None and ent[0] == mtime:
            return ent[1]
    paths = glob.glob(os.path.join(d, "frame_*.jpg"))
    paths.sort()
    with _LISTING_LOCK:
        _LISTING_CACHE[d] = (mtime, paths)
    return paths


def list_wagon_frames(
    cache_root: str, gw_id: str, camera_id: str,
    *, trim_stable: bool = False,
) -> List[str]:
    """Return sorted JPEG paths for one (gw, camera) pair.

    When ``trim_stable`` is True, returns only the adaptive stable interior
    (5% per side, clamped [3, 12]) used for feature inference.  Listings are
    cached per (dir, mtime) so repeat scans of the same wagon dir are avoided
    (identical result).
    """
    d = wagon_camera_dir(cache_root, gw_id, camera_id)
    if not os.path.isdir(d):
        return []
    paths = _cached_sorted_frames(d)
    if trim_stable:
        paths = stable_interior(paths)
    return paths


def iter_wagon_frames(
    cache_root: str, gw_id: str, camera_id: str,
    *, every_nth: int = 1, max_frames: Optional[int] = None,
    trim_stable: bool = False,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR ndarray) in monotonic order.

    Args:
        every_nth: yield 1 of every N frames (default 1 = all).
        max_frames: hard cap; useful when a wagon has hundreds of
            frames and we only need a sample.
        trim_stable: iterate only the adaptive stable interior (drop the
            noisy first/last 5% per side, clamped [3, 12]) -- used for
            feature inference, NOT for rendering.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=trim_stable)
    if every_nth > 1:
        paths = paths[::every_nth]
    if max_frames is not None and len(paths) > max_frames:
        # evenly-spaced subsample so first / last / middle are covered
        idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in idx]
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        # parse frame_NNNNNN.jpg
        try:
            fi = int(os.path.basename(p).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            fi = -1
        yield fi, frame


# -----------------------------------------------------------------------------
# YOLO calls
# -----------------------------------------------------------------------------

def run_detection(
    model, frame: np.ndarray,
    *, confidence: float = 0.4, half: Optional[bool] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a YOLO detection model on a frame; return clean dicts.

    Each dict: {class_id, class_name, confidence, bbox: [x1, y1, x2, y2]}.

    device/half default to the process-resolved DEVICE/HALF (CUDA => FP16,
    CPU => FP32) -- callers may override but the defaults preserve the
    pre-migration GPU behaviour exactly.
    """
    if model is None:
        return []
    dev = device if device is not None else DEVICE
    use_half = HALF if half is None else half
    res = model(frame, verbose=False, half=use_half, device=dev)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss  = res.boxes.cls.cpu().numpy().astype(int)
    names = getattr(model, "names", {}) or {}

    out: List[Dict[str, Any]] = []
    for bbox, conf, cls_id in zip(boxes, confs, clss):
        if float(conf) < confidence:
            continue
        out.append({
            "class_id": int(cls_id),
            "class_name": str(names.get(int(cls_id), "unknown")).lower(),
            "confidence": float(conf),
            "bbox": [float(bbox[0]), float(bbox[1]),
                     float(bbox[2]), float(bbox[3])],
        })
    return out


def _parse_detection_result(res, names, confidence: float) -> List[Dict[str, Any]]:
    """Parse ONE ultralytics Results into clean detection dicts (shared by the
    per-frame and batched paths so both yield identical records)."""
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    out: List[Dict[str, Any]] = []
    for bbox, conf, cls_id in zip(boxes, confs, clss):
        if float(conf) < confidence:
            continue
        out.append({
            "class_id": int(cls_id),
            "class_name": str(names.get(int(cls_id), "unknown")).lower(),
            "confidence": float(conf),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        })
    return out


# -----------------------------------------------------------------------------
# Adaptive batch sizing (default FEATURE_BATCH_SIZE=32).  On an out-of-memory
# condition the batch halves (32->16->8->4->2->1) and retries -- the pipeline
# NEVER fails on OOM.  The reduced size is remembered PROCESS-WIDE (sticky) so
# later wagons/cameras don't repeatedly hit OOM at the too-large size.
# -----------------------------------------------------------------------------

_effective_bs: Optional[int] = None
_bs_lock = threading.Lock()


def _is_oom(e: Exception) -> bool:
    if isinstance(e, MemoryError):
        return True
    return "out of memory" in str(e).lower()


def _free_cuda() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_bs(requested: Optional[int]) -> int:
    """Starting batch size.  Explicit `requested` (tests/callers) wins as-is;
    otherwise use the sticky process-wide effective size (seeded from
    CFG.FEATURE_BATCH_SIZE, only ever ratcheted DOWN by OOM)."""
    if requested is not None:
        return max(1, requested)
    global _effective_bs
    with _bs_lock:
        if _effective_bs is None:
            _effective_bs = max(1, CFG.FEATURE_BATCH_SIZE)
        return _effective_bs


def _ratchet_down(requested: Optional[int], cur: int) -> int:
    """Halve the current chunk after an OOM; on the adaptive (requested is None)
    path also lower the sticky process-wide size so future calls start smaller."""
    new = max(1, cur // 2)
    if requested is None:
        global _effective_bs
        with _bs_lock:
            if _effective_bs is None or new < _effective_bs:
                _effective_bs = new
    _free_cuda()
    return new


def effective_batch_size() -> int:
    """Current adaptive batch size (for processors' decode-chunking)."""
    return _resolve_bs(None)


def batched_detect(
    model, frames: List[np.ndarray], *, confidence: Optional[float] = None,
    batch_size: Optional[int] = None, half: Optional[bool] = None,
    device: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """Run a YOLO DETECTION model over `frames`, returning ONE detection list per
    frame IN INPUT ORDER.

    confidence:
      * a float -> passed to `model(..., conf=confidence)` (door/damage: their
        per-camera threshold) AND used as the parse floor.  Matches the
        per-frame processors that call `model(frame, conf=cam_conf)`.
      * None    -> the model runs at its OWN default conf and NO parse floor is
        applied; the caller filters (OCR: `model(frame)` then its own 0.40 gate).

    batch_size: None -> core.config.FEATURE_BATCH_SIZE (default 1).
      * <=1 -> EXACT per-frame path (byte-identical to the pre-opt processors).
      * >1  -> `model([chunk], ...)`; on CUDA OOM the chunk halves and retries
        down to 1 (never drops a frame).  Records are parsed by the SAME
        `_parse_detection_result` in both paths, so bs=1 == per-frame exactly.
    """
    if model is None or not frames:
        return [[] for _ in frames]
    bs = _resolve_bs(batch_size)
    dev = device if device is not None else DEVICE
    use_half = HALF if half is None else half
    names = getattr(model, "names", {}) or {}
    floor = 0.0 if confidence is None else confidence

    def _predict(imgs):
        kw = dict(verbose=False, half=use_half, device=dev)
        if confidence is not None:
            kw["conf"] = confidence
        return model(imgs, **kw)

    if bs <= 1:
        return [_parse_detection_result(_predict(fr)[0], names, floor) for fr in frames]

    out: List[List[Dict[str, Any]]] = []
    i, cur, n = 0, bs, len(frames)
    while i < n:
        chunk = frames[i:i + cur]
        try:
            for res in _predict(chunk):
                out.append(_parse_detection_result(res, names, floor))
            i += len(chunk)
        except (RuntimeError, MemoryError) as e:
            if cur > 1 and _is_oom(e):
                cur = _ratchet_down(batch_size, cur)      # 32->16->8->4->2->1
                continue
            raise
    return out


def _parse_classification_result(res, names) -> Tuple[str, float]:
    """Parse ONE ultralytics classification Results into (top1_name, conf) --
    identical logic to run_classification (incl. the boxes fallback)."""
    if getattr(res, "probs", None) is None:
        if res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            i = int(np.argmax(confs))
            return str(names.get(int(clss[i]), "unknown")).lower(), float(confs[i])
        return "", 0.0
    top1 = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    return str(names.get(top1, "unknown")).lower(), conf


def batched_classify(
    model, frames: List[np.ndarray], *, batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Run a YOLO CLASSIFICATION model over `frames`, returning (class, conf) per
    frame IN INPUT ORDER.  bs<=1 -> `model(frame)` per frame (== run_classification
    exactly); bs>1 -> batched with CUDA-OOM halving.  No `half`/`conf` passed
    (mirrors run_classification)."""
    if model is None or not frames:
        return [("", 0.0) for _ in frames]
    bs = _resolve_bs(batch_size)
    dev = device if device is not None else DEVICE
    names = getattr(model, "names", {}) or {}
    if bs <= 1:
        return [_parse_classification_result(model(fr, verbose=False, device=dev)[0], names)
                for fr in frames]
    out: List[Tuple[str, float]] = []
    i, cur, n = 0, bs, len(frames)
    while i < n:
        chunk = frames[i:i + cur]
        try:
            for res in model(chunk, verbose=False, device=dev):
                out.append(_parse_classification_result(res, names))
            i += len(chunk)
        except (RuntimeError, MemoryError) as e:
            if cur > 1 and _is_oom(e):
                cur = _ratchet_down(batch_size, cur)      # 32->16->8->4->2->1
                continue
            raise
    return out


def run_classification(model, frame: np.ndarray,
                       *, device: Optional[str] = None) -> Tuple[str, float]:
    """Run a YOLO classification model. Returns (top1_class_name, conf).

    device defaults to the process-resolved DEVICE (CUDA if available, else
    CPU) so behaviour no longer depends on ultralytics' internal guess.
    """
    if model is None:
        return "", 0.0
    dev = device if device is not None else DEVICE
    res = model(frame, verbose=False, device=dev)[0]
    if getattr(res, "probs", None) is None:
        # Some "classification" models still emit boxes; pick the
        # highest-conf detection's class as a fallback.
        if res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy().astype(int)
            i = int(np.argmax(confs))
            names = getattr(model, "names", {}) or {}
            return str(names.get(int(clss[i]), "unknown")).lower(), float(confs[i])
        return "", 0.0
    top1 = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    names = getattr(model, "names", {}) or {}
    return str(names.get(top1, "unknown")).lower(), conf


def crop_bbox(frame: np.ndarray, bbox: List[float], pad: int = 0) -> Optional[np.ndarray]:
    """Crop a frame to a bbox with optional pixel padding."""
    if frame is None or bbox is None or len(bbox) != 4:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------

def majority_vote(
    items: List[Dict[str, Any]],
    *, key: str = "class_name", conf_key: str = "confidence",
) -> Tuple[Optional[str], float, int]:
    """Confidence-weighted majority vote.

    Returns (winning_value, mean_conf_of_winner, vote_count).
    Ties broken by:  more votes  >  higher mean conf  >  alphabetical.
    """
    if not items:
        return None, 0.0, 0
    votes: Counter = Counter()
    confs: Dict[str, List[float]] = {}
    for it in items:
        v = it.get(key)
        if v is None:
            continue
        v = str(v).lower()
        votes[v] += 1
        confs.setdefault(v, []).append(float(it.get(conf_key, 0.0) or 0.0))
    if not votes:
        return None, 0.0, 0

    def sort_key(item):
        cls, n = item
        mean_c = sum(confs[cls]) / len(confs[cls]) if confs[cls] else 0.0
        return (-n, -mean_c, cls)

    best, n = sorted(votes.items(), key=sort_key)[0]
    mean_c = sum(confs[best]) / len(confs[best]) if confs[best] else 0.0
    return best, mean_c, n


def fraction_with(items: List[Dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if predicate(it)) / float(len(items))


# -----------------------------------------------------------------------------
# Per-wagon JSON I/O
# -----------------------------------------------------------------------------

def write_per_wagon_json(
    output_dir: str, gw_id: str, payload: Dict[str, Any],
) -> str:
    """Atomically write <output_dir>/<gw_id>.json (temp file + os.replace)."""
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, f"{gw_id}.json")
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=f".{gw_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return p


def feature_camera_dir(output_dir: str, feature: str, camera_id: str) -> str:
    """Per-camera feature output dir: <output_dir>/<feature>/<CAMERA>/.

    Each camera writes ONLY inside its own namespace so a late camera can never
    overwrite another camera's per-wagon feature files."""
    d = os.path.join(output_dir, feature, camera_id)
    os.makedirs(d, exist_ok=True)
    return d


def empty_payload(gw_id: str, feature: str, status: str, **extra) -> Dict[str, Any]:
    payload = {
        "global_id": gw_id,
        "feature": feature,
        "status": status,
        "frame_count": 0,
    }
    payload.update(extra)
    return payload


# -----------------------------------------------------------------------------
# Lightweight per-feature timing helper
# -----------------------------------------------------------------------------

class FeatureTimer:
    """Track per-wagon timings inside a processor."""
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()
        self.per_wagon: Dict[str, float] = {}

    def stamp(self, gw_id: str, t0: float) -> float:
        dt = time.time() - t0
        self.per_wagon[gw_id] = round(dt, 3)
        return dt

    def total(self) -> float:
        return time.time() - self.start
