"""delivery/dashboard_ingest.py -- Stage-6-only, read-only legacy dashboard adapter.

Purpose
-------
The pre-migration ("old") pipeline fed a per-camera dashboard by POSTing one
``*_inspection.json`` (schema ``{camera_id, version, inspection_data}``) per
camera angle to an S3 bucket and then calling a ``cctv-receiver/inspections/ingest``
API.  The train-state-native v4 pipeline does NOT produce that per-camera feed --
it emits one combined ``combined_train_report.json`` per train.

This module RE-DERIVES the legacy per-camera dashboard payload from finalized v4
artifacts so the existing dashboard keeps working, WITHOUT changing anything
about how the new system computes results.

Hard guarantees (by construction)
---------------------------------
* **Read-only w.r.t. the pipeline.**  It reads finalized artifacts only:
    <batch_root>/reports/combined_train_report.json
    <batch_root>/evidence/<GW>/<feature>/<CAMERA>/{metadata.json,*.jpg}
    <batch_root>/delivery/finalization.json
  It NEVER imports or mutates GlobalTrainState, feature processors, fusion, or
  the report builders, and it NEVER loads a model or opens a video.
* **Writes only under <batch_root>/delivery/.**  Generated JSON goes to
  ``delivery/dashboard/<CAMERA>_inspection.json``; ingest status is merged into
  ``delivery/finalization.json``.  Nothing else on disk is touched.
* **Enabled by default.**  ``WAGONEYE_DASHBOARD_INGEST_ENABLED`` defaults to
  ``true`` -- every finalized batch posts to the live ingest API (version v1).
  Set it to ``false`` to make ``run()`` a no-op (staging / shadow runs).
* **Failure-isolating.**  ``run()`` never raises; any error is logged and
  recorded.  It cannot corrupt the final report or the sealed batch state.
* **Idempotent across restarts.**  Per-camera ingest status (keyed by the
  generated JSON's sha256 + report revision) is persisted; an already-ingested
  camera is skipped on re-entry -- no duplicate uploads, no duplicate ingest.

Degraded fields (documented, never invented)
--------------------------------------------
* ``direction``            -> "unknown" (optical-flow direction is not in any
                              finalized artifact; recompute is out of scope for a
                              read-only adapter).
* ``rake_status``          -> derived from FUSED load results (Loaded/Empty),
                              a measured proxy for the old direction heuristic.
* ``loco_frames`` /
  ``loco_number_results`` /
  ``total_loco_frames``    -> empty (v4 has no loco-specific frame/OCR feed).
* ``wagon_frames`` gallery -> synthesized from whatever per-camera evidence JPEGs
                              exist; only files that are actually present are
                              referenced (never fabricated).

The degraded set for each payload is echoed under ``inspection_data._adapter`` so
a consumer can see exactly what was and was not faithfully reproduced.

NOTE: this posts to the LIVE dashboard on every run.  Confirm with the dashboard
team that (a) reused ``train_batch/.../evidence/...`` HTTPS URLs are accepted and
(b) the degraded loco/direction/gallery fields are acceptable.  Set
``WAGONEYE_DASHBOARD_INGEST_ENABLED=false`` to disable without a code change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger
from delivery import finalization as FIN

log = get_logger("delivery.dashboard")

_IST = timezone(timedelta(hours=5, minutes=30))
_TS_RE = re.compile(r"(\d{8})_(\d{6})")
_DATE_RE = re.compile(r"(\d{8})")

# Local (delivery/) scratch subdir for generated per-camera JSON.
_LOCAL_SUBDIR = os.path.join("delivery", "dashboard")


# -----------------------------------------------------------------------------
# Configuration (all self-contained here; nothing shared is modified).
# Every value defaults to the pre-migration production value and is
# env-overridable so a staging deployment needs no source edit.
# -----------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_json_map(name: str, default: Dict[str, str]) -> Dict[str, str]:
    """Merge a JSON-object env override over `default` (override wins)."""
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            merged = dict(default)
            merged.update({str(k): str(v) for k, v in parsed.items()})
            return merged
    except (ValueError, TypeError):
        log.warning("[DASHBOARD] ignoring malformed %s (not a JSON object)", name)
    return dict(default)


# Full CCTV camera ids (dashboard primary key).  Defaults taken from the old
# per-camera run_service.py INPUT_BUCKET suffixes.
_DEFAULT_FULL_IDS = {
    C.CAMERA_RIGHT_UP:     "camera_CCTV_HZBN_DHN_2_RIGHT_UP",
    C.CAMERA_LEFT_UP:      "camera_CCTV_HZBN_DHN_1_LEFT_UP",
    C.CAMERA_RIGHT_UP_TOP: "camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
    C.CAMERA_LEFT_UP_TOP:  "camera_CCTV_HZBN_DHN_6_LEFT_TOP",
}

# Legacy dashboard S3 folder (prefix) per camera.  RIGHT_UP="Right_up" is the
# only one confirmed from the old env; the others follow the same convention and
# MUST be confirmed with the dashboard team before enabling.
_DEFAULT_FOLDERS = {
    C.CAMERA_RIGHT_UP:     "Right_up",
    C.CAMERA_LEFT_UP:      "Left_up",
    C.CAMERA_RIGHT_UP_TOP: "Right_up_top",
    C.CAMERA_LEFT_UP_TOP:  "Left_up_top",
}


def is_enabled() -> bool:
    # ON by default: every finalized batch posts the legacy per-camera feed to
    # the dashboard ingest API (version v1).  Set WAGONEYE_DASHBOARD_INGEST_ENABLED=false
    # to turn it off (e.g. staging / shadow runs).
    return _env_bool("WAGONEYE_DASHBOARD_INGEST_ENABLED", True)


def _inspection_bucket() -> str:
    return _env("WAGONEYE_INSPECTION_JSON_BUCKET", "ankit-version-1-prod")


def _ingest_api_url() -> str:
    return _env(
        "WAGONEYE_INSPECTION_INGEST_API_URL",
        "https://ms-pnr-location-notification-api.suvidhaen.com/"
        "cctv-receiver/inspections/ingest",
    )


def _version() -> str:
    # The dashboard chooses its tab from this value: version "v1" -> V1 tab.
    # Override with WAGONEYE_INSPECTION_VERSION (e.g. v2/v3/v4) if needed.
    return _env("WAGONEYE_INSPECTION_VERSION", "v1")


def _model_id() -> str:
    return _env("WAGONEYE_INSPECTION_MODEL_ID", "model-v3")


def _reuse_evidence_urls() -> bool:
    return _env_bool("WAGONEYE_DASHBOARD_REUSE_EVIDENCE_URLS", True)


def full_camera_id(camera: str) -> str:
    return _env_json_map("WAGONEYE_INSPECTION_CAMERA_FULL_IDS",
                         _DEFAULT_FULL_IDS).get(camera, camera)


def folder_for(camera: str) -> str:
    return _env_json_map("WAGONEYE_INSPECTION_FOLDERS",
                         _DEFAULT_FOLDERS).get(camera, C.CAMERA_FOLDER.get(camera, camera))


# -----------------------------------------------------------------------------
# Pure helpers (timestamp / date-folder / URLs) -- fully unit-testable
# -----------------------------------------------------------------------------

def extract_train_timestamp(*texts: Optional[str]) -> Optional[datetime]:
    """First ``YYYYMMDD_HHMMSS`` (or ``YYYYMMDD``) token across `texts`.

    Returns a naive datetime (interpreted as train local/IST wall-clock, exactly
    as the old pipeline treated the filename timestamp)."""
    for t in texts:
        if not t:
            continue
        m = _TS_RE.search(t)
        if m:
            try:
                return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        m = _DATE_RE.search(t)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d")
            except ValueError:
                pass
    return None


def date_folder(dt: Optional[datetime]) -> str:
    """Operational-day folder with the old 05:00 IST boundary: a train recorded
    before 05:00 lands in the PREVIOUS calendar day's folder."""
    if dt is None:
        dt = datetime.now(_IST)
    shifted = (dt - timedelta(days=1)) if dt.hour < 5 else dt
    return shifted.strftime("%Y-%m-%d")


def evidence_url(output_bucket: str, region: str, batch_key: str,
                 gw: str, feature: str, camera: str, filename: str) -> str:
    """Deterministic HTTPS URL for an evidence JPEG already mirrored to S3 by the
    Stage-6 tree upload (``train_batch/<key>/evidence/...``)."""
    key = (f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/evidence/"
           f"{gw}/{feature}/{camera}/{filename}")
    return f"https://{output_bucket}.s3.{region}.amazonaws.com/{key}"


def _seg_type(classification: Optional[str]) -> str:
    return {
        C.CLASS_ENGINE:    "engine",
        C.CLASS_WAGON:     "wagon",
        C.CLASS_BRAKE_VAN: "brake_van",
    }.get(classification or "", "wagon")


def _door_side(camera: str) -> Optional[str]:
    if camera == C.CAMERA_RIGHT_UP:
        return "right"
    if camera == C.CAMERA_LEFT_UP:
        return "left"
    return None


# -----------------------------------------------------------------------------
# Evidence reads (read-only)
# -----------------------------------------------------------------------------

def _read_meta(evidence_root: str, gw: str, feature: str, camera: str) -> Dict[str, Any]:
    p = os.path.join(evidence_root, gw, feature, camera, "metadata.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _evidence_file(evidence_root: str, gw: str, feature: str, camera: str,
                   filename: str) -> Optional[str]:
    p = os.path.join(evidence_root, gw, feature, camera, filename)
    return p if os.path.isfile(p) else None


class _UrlMaker:
    """Turn a local evidence JPEG into a dashboard-usable HTTPS URL.

    Default (reuse=True): reference the already-uploaded train_batch evidence URL
    -- no extra upload.  reuse=False: copy ONLY the referenced JPEG into the
    legacy inspection bucket and return that URL."""

    def __init__(self, *, s3_client, output_bucket: str, region: str,
                 inspection_bucket: str, batch_key: str, folder: str,
                 date_folder_str: str, reuse: bool, skip_upload: bool):
        self.s3 = s3_client
        self.output_bucket = output_bucket
        self.region = region
        self.inspection_bucket = inspection_bucket
        self.batch_key = batch_key
        self.folder = folder
        self.date_folder = date_folder_str
        self.reuse = reuse
        self.skip_upload = skip_upload

    def url(self, evidence_root: str, gw: str, feature: str, camera: str,
            filename: str) -> Optional[str]:
        local = _evidence_file(evidence_root, gw, feature, camera, filename)
        if local is None:
            return None
        if self.reuse:
            return evidence_url(self.output_bucket, self.region, self.batch_key,
                                gw, feature, camera, filename)
        # copy-only mode: upload just this JPEG to the legacy bucket
        key = (f"{self.folder}/{self.date_folder}/evidence/"
               f"{gw}/{feature}/{camera}/{filename}")
        if self.skip_upload or self.s3 is None:
            return f"https://{self.inspection_bucket}.s3.{self.region}.amazonaws.com/{key}"
        try:
            self.s3.upload_file(local, self.inspection_bucket, key,
                                ExtraArgs={"ContentType": "image/jpeg"})
        except Exception as e:  # pragma: no cover - network path
            log.warning("[DASHBOARD] evidence copy failed %s: %s", key, e)
            return None
        return f"https://{self.inspection_bucket}.s3.{self.region}.amazonaws.com/{key}"


# -----------------------------------------------------------------------------
# Per-camera legacy payload builder (PURE: no I/O beyond reading evidence)
# -----------------------------------------------------------------------------

# Ordered gallery candidates per camera role.
_SIDE_GALLERY = ("door/{side}_best.jpg", "door/{side}_crop.jpg",
                 "ocr/best_frame.jpg", "ocr/number_crop.jpg")
_TOP_GALLERY = ("load/best_frame.jpg", "damage/track_1.jpg",
                "damage/track_2.jpg", "damage/track_3.jpg")
_POSITIONS = ("start", "mid1", "mid2", "end")


def build_inspection_json(*, camera: str, report_doc: Dict[str, Any],
                          evidence_root: str, url_maker: "_UrlMaker") -> Dict[str, Any]:
    """Build ONE legacy ``{camera_id, version, inspection_data}`` document for
    `camera` from the finalized combined report + this camera's evidence.

    Pure w.r.t. the pipeline: reads only `report_doc` (already loaded) and files
    under `evidence_root`.  Never invents image URLs or numbers."""
    wagons = report_doc.get("wagons", []) or []
    summary = report_doc.get("summary", {}) or {}
    train_meta = report_doc.get("train_metadata", {}) or {}
    report_meta = report_doc.get("report_meta", {}) or {}
    batch_key = report_doc.get("batch_key", "")
    source_urls = train_meta.get("source_video_urls", {}) or {}
    processed_urls = train_meta.get("processed_video_urls", {}) or {}

    side = _door_side(camera)
    is_top = camera in C.TOP_CAMERAS

    src_url = source_urls.get(camera, "")
    raw_video_name = os.path.basename(src_url) if src_url else \
        f"{batch_key}_{C.CAMERA_FOLDER.get(camera, camera)}.mp4"
    ts = extract_train_timestamp(raw_video_name, batch_key)
    upload_ts = (ts or datetime.now(_IST)).strftime("%Y-%m-%dT%H:%M:%S")
    upload_ts_readable = (ts or datetime.now(_IST)).strftime("%d-%m-%Y %H:%M:%S") + " IST"

    # Train-level counts.  total_wagons uses the GLOBAL fused count (authoritative
    # across cameras); door/damage counts are scoped to THIS camera's authority.
    total_wagons = int(summary.get("total_wagons", len(wagons)))
    num_engines = int(summary.get("engine_count", 0))
    loaded = int(summary.get("loaded", 0))
    empty = int(summary.get("empty", 0))
    if loaded == 0 and empty == 0:
        rake_status = "Unknown"
    else:
        rake_status = "Loaded" if loaded >= empty else "Empty"

    doors_open = doors_closed = 0
    if side:
        state_key = f"{side}_door"
        doors_open = sum(1 for w in wagons if w.get(state_key) == C.DOOR_OPEN)
        doors_closed = sum(1 for w in wagons if w.get(state_key) == C.DOOR_CLOSED)

    wagon_number_results: Dict[str, Any] = {}
    segment_type_map: Dict[str, Any] = {}
    wagon_segments: List[Dict[str, Any]] = []
    problem_frames: List[Dict[str, Any]] = []
    pf_type_counts: Dict[str, int] = {}
    damaged_wagons: set = set()

    def _bump(t: str) -> None:
        pf_type_counts[t] = pf_type_counts.get(t, 0) + 1

    for w in wagons:
        gw = w.get("global_id", "")
        idx = w.get("wagon_index", 0)
        ident = w.get("wagon_identifier") or C.NO_DATA
        digits = re.sub(r"[^0-9]", "", str(ident)) if ident != C.NO_DATA else ""
        is_valid = len(digits) == C.WAGON_NUMBER_LENGTH
        wagon_number_results[str(idx)] = {
            "is_valid_11_digit": bool(is_valid),
            "display_number": digits if digits else "-",
        }
        segment_type_map[str(idx)] = {"type": _seg_type(w.get("classification")),
                                      "number": idx}

        # ---- wagon gallery (synthesized from EXISTING evidence only) ----
        templates = (_TOP_GALLERY if is_top
                     else tuple(t.format(side=side) for t in _SIDE_GALLERY))
        frames: List[Dict[str, Any]] = []
        for rel in templates:
            feat, fn = rel.split("/", 1)
            u = url_maker.url(evidence_root, gw, feat, camera, fn)
            if u:
                frames.append({"position": _POSITIONS[min(len(frames), 3)],
                               "s3_url": u})
            if len(frames) >= 4:
                break

        door_status = "open" if (side and w.get(f"{side}_door") == C.DOOR_OPEN) \
            else ("closed" if side else "N/A")
        seg: Dict[str, Any] = {
            "segment_id": idx,
            "segment_type": _seg_type(w.get("classification")),
            "wagon_count": idx,
            "is_valid_wagon_id": bool(is_valid),
            "door_status": door_status,
            "damage_detected": False,
            "wagon_frames": frames,
        }
        if is_valid:
            seg["wagon_number"] = digits

        # ---- problem frames scoped to this camera's authority ----
        if side:
            meta = _read_meta(evidence_root, gw, "door", camera)
            side_meta = (meta.get("sides") or {}).get(side, {})
            dstate = w.get(f"{side}_door")
            if dstate == C.DOOR_OPEN:
                _bump("door_open")
                problem_frames.append(_problem_frame(
                    idx=idx, gw=gw, camera=camera, evidence_root=evidence_root,
                    url_maker=url_maker, feature="door", img=f"{side}_best.jpg",
                    problem_type="door_open", class_name="door_open",
                    bbox=side_meta.get("bbox"), conf=side_meta.get("confidence"),
                    door_status="open", damage=False))
            elif dstate == C.DOOR_DAMAGED:
                _bump("side_damage")
                damaged_wagons.add(idx)
                seg["damage_detected"] = True
                problem_frames.append(_problem_frame(
                    idx=idx, gw=gw, camera=camera, evidence_root=evidence_root,
                    url_maker=url_maker, feature="door", img=f"{side}_best.jpg",
                    problem_type="side_damage", class_name="damage",
                    bbox=side_meta.get("bbox"), conf=side_meta.get("confidence"),
                    door_status="N/A", damage=True))

        if is_top:
            dmeta = _read_meta(evidence_root, gw, "damage", camera)
            for tr in (dmeta.get("tracks") or []):
                ti = tr.get("track_idx", 1)
                cls = tr.get("class_name", "damage")
                _bump(cls)
                damaged_wagons.add(idx)
                seg["damage_detected"] = True
                problem_frames.append(_problem_frame(
                    idx=idx, gw=gw, camera=camera, evidence_root=evidence_root,
                    url_maker=url_maker, feature="damage", img=f"track_{ti}.jpg",
                    problem_type=cls, class_name=cls,
                    bbox=tr.get("bbox"),
                    conf=tr.get("best_confidence", tr.get("confidence")),
                    door_status="N/A", damage=True))

        wagon_segments.append(seg)

    if is_top:
        damaged_count = len(damaged_wagons)
    elif side:
        damaged_count = len(damaged_wagons)
    else:
        damaged_count = 0

    # Loco numbers (5-digit): production emitted them on RIGHT_UP only, keyed by
    # loco_id. Re-derived here from the fused ENGINE-wagon loco_number values.
    loco_number_results: Dict[str, Any] = {}
    if camera == C.CAMERA_RIGHT_UP:
        _loco_id = 0
        for w in wagons:
            _ln = w.get("loco_number") or C.NO_DATA
            if _ln not in (None, "", C.NO_DATA):
                _loco_id += 1
                _digits = re.sub(r"[^0-9]", "", str(_ln))
                loco_number_results[str(_loco_id)] = {
                    "is_valid_5_digit": len(_digits) == 5,
                    "display_number": str(_ln),
                    "raw_number": str(_ln),
                    "confidence": float(w.get("loco_number_confidence", 0.0) or 0.0),
                }

    degraded = ["direction", "loco_frames", "total_loco_frames"]
    if not loco_number_results:
        degraded.append("loco_number_results")
    if not is_top and not side:
        degraded.append("doors_open/doors_closed")

    inspection_data = {
        "raw_video_name": raw_video_name,
        "identified_by": _model_id(),
        "upload_timestamp": upload_ts,
        "upload_timestamp_readable": upload_ts_readable,
        "direction": "unknown",                       # DEGRADED (see module docstring)
        "rake_status": rake_status,                   # DEGRADED: fused load proxy
        "pdf_report_url": _pdf_url(report_meta, camera),
        "trimmed_video_url": src_url,
        "detected_video_url": processed_urls.get(camera, ""),
        "raw_video_urls": [src_url] if src_url else [],
        "total_wagons": total_wagons,
        "doors_open": doors_open,
        "doors_closed": doors_closed,
        "damaged_wagons": damaged_count,
        "num_engines": num_engines,
        "total_loco_frames": 0,                       # DEGRADED: no loco feed in v4
        "total_problem_frames": len(problem_frames),
        "problem_frames_by_type": pf_type_counts,
        "wagon_number_results": wagon_number_results,
        "loco_number_results": loco_number_results,   # RIGHT_UP: 5-digit loco numbers
        "segment_type_map": segment_type_map,
        "wagon_segments": wagon_segments,
        "loco_frames": [],                            # DEGRADED
        "problem_frames": problem_frames,
        "_adapter": {
            "generated_by": "wagon_eye_v4 delivery.dashboard_ingest",
            "source": "combined_train_report.json",
            "report_revision": report_meta.get("report_revision", 0),
            "report_status": report_meta.get("report_status", ""),
            "global_state_version":
                report_meta.get("generated_from_global_state_version", ""),
            "camera_authority": ("top:load+damage" if is_top
                                 else (f"side:{side}_door+ocr" if side else "none")),
            "degraded_fields": degraded,
        },
    }
    return {
        "camera_id": full_camera_id(camera),
        "version": _version(),
        "inspection_data": inspection_data,
    }


def _problem_frame(*, idx, gw, camera, evidence_root, url_maker, feature, img,
                   problem_type, class_name, bbox, conf, door_status, damage):
    u = url_maker.url(evidence_root, gw, feature, camera, img)
    coords = list(bbox)[:4] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 \
        else [0, 0, 0, 0]
    return {
        "wagon_count": idx, "segment_type": "wagon", "segment_number": idx,
        "problem_type": problem_type, "frame_number": 0,
        "filename": f"{gw}_{camera}_{img}",
        "s3_url": u,
        "is_annotated": True,
        "annotated_image_url": u,
        "bounding_box": {
            "bounding_box_coordinates": coords,
            "confidence": round(float(conf), 3) if conf is not None else 0.0,
            "class_name": class_name,
        },
        "door_status": door_status,
        "door_close_detected": False,
        "damage_detected": bool(damage),
    }


def _pdf_url(report_meta: Dict[str, Any], camera: str) -> str:
    # Prefer a per-camera PDF url if the finalization marker carried one; the
    # caller injects finalization upload_urls into report_meta before building.
    urls = report_meta.get("_upload_urls", {}) or {}
    return urls.get(f"camera_{camera}") or urls.get("pdf") or ""


# -----------------------------------------------------------------------------
# Ingest (HTTP) with retries -- mirrors the old ingest loop
# -----------------------------------------------------------------------------

def ingest_idempotency_key(batch_key: str, camera: str, report_revision: int,
                           json_sha256: str) -> str:
    raw = f"{batch_key}|{camera}|{report_revision}|{json_sha256}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _post_ingest(*, api_url: str, payload: Dict[str, Any], idem_key: str,
                 max_retries: int = 3, base_delay: float = 15.0,
                 requests_mod=None) -> Dict[str, Any]:
    """POST once (with retries).  Returns {ok, status_code, run_id, error}.

    Retries only on >=500 (transient); 422 is treated as a permanent validation
    failure (no retry).  Never raises."""
    if requests_mod is None:  # pragma: no cover - exercised via injection in tests
        import requests as requests_mod  # type: ignore
    headers = {"Idempotency-Key": idem_key}
    body = dict(payload, idempotency_key=idem_key)
    delay = base_delay
    last: Dict[str, Any] = {"ok": False, "status_code": None, "run_id": None,
                            "error": "not_attempted"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests_mod.post(api_url, json=body, headers=headers, timeout=60)
            code = getattr(resp, "status_code", None)
            if code == 200:
                data = {}
                try:
                    data = resp.json()
                except Exception:
                    pass
                return {"ok": True, "status_code": 200,
                        "run_id": data.get("run_id"), "error": None}
            if code == 422:
                txt = ""
                try:
                    txt = resp.text[:300]
                except Exception:
                    pass
                return {"ok": False, "status_code": 422, "run_id": None,
                        "error": f"validation: {txt}"}
            last = {"ok": False, "status_code": code, "run_id": None,
                    "error": f"http_{code}"}
            if code is not None and code < 500:
                return last  # non-retryable client error
        except Exception as e:  # network/timeout -> retryable
            last = {"ok": False, "status_code": None, "run_id": None,
                    "error": str(e)}
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2
    return last


# -----------------------------------------------------------------------------
# finalization.json per-camera status (idempotency ledger)
# -----------------------------------------------------------------------------

_DASH_KEY = "dashboard_ingested"


def _load_status(batch_root: str) -> Dict[str, Any]:
    marker = FIN.load(batch_root) or {}
    return dict(marker.get(_DASH_KEY) or {})


def _record_status(batch_root: str, camera: str, entry: Dict[str, Any]) -> None:
    marker = FIN.load(batch_root) or {}
    block = dict(marker.get(_DASH_KEY) or {})
    block[camera] = entry
    marker[_DASH_KEY] = block
    FIN.write(batch_root, marker)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(*, batch_root: str, s3_client=None, skip_upload: bool = False,
        skip_ingest: Optional[bool] = None, requests_mod=None) -> Dict[str, Any]:
    """Generate + (optionally) deliver the legacy per-camera dashboard feed.

    NEVER raises.  Returns a summary dict.  A no-op (returns {'enabled': False})
    unless WAGONEYE_DASHBOARD_INGEST_ENABLED is truthy.

    skip_upload=True (shadow/dry-run) -> build + record locally, do NOT upload
    JSON and do NOT POST ingest.  skip_ingest defaults to skip_upload."""
    result: Dict[str, Any] = {"enabled": is_enabled(), "cameras": {}}
    if not is_enabled():
        return result
    if skip_ingest is None:
        skip_ingest = skip_upload
    try:
        return _run_inner(batch_root=batch_root, s3_client=s3_client,
                          skip_upload=skip_upload, skip_ingest=skip_ingest,
                          requests_mod=requests_mod, result=result)
    except Exception as e:  # absolute isolation: never propagate
        log.error("[DASHBOARD] ingest aborted (non-fatal): %s", e)
        result["error"] = str(e)
        return result


def _run_inner(*, batch_root, s3_client, skip_upload, skip_ingest,
               requests_mod, result) -> Dict[str, Any]:
    # reports/ is the fixed finalized-artifact location (core.config.DIR_REPORTS);
    # hardcoded here to keep the adapter decoupled from config internals.
    report_path = os.path.join(batch_root, "reports", "combined_train_report.json")
    if not os.path.isfile(report_path):
        log.warning("[DASHBOARD] no combined_train_report.json -- nothing to ingest")
        result["error"] = "no_report"
        return result
    with open(report_path, "r", encoding="utf-8") as f:
        report_doc = json.load(f)

    report_meta = report_doc.get("report_meta", {}) or {}
    # inject finalization upload_urls so per-camera pdf urls resolve
    fin_marker = FIN.load(batch_root) or {}
    report_meta = dict(report_meta)
    report_meta["_upload_urls"] = fin_marker.get("upload_urls", {}) or {}
    report_doc = dict(report_doc, report_meta=report_meta)
    report_revision = int(report_meta.get("report_revision", 0))

    present = report_meta.get("cameras_present") or [
        c for c in C.ALL_CAMERAS
        if c in {w0 for w in report_doc.get("wagons", [])
                 for w0 in (w.get("supporting_cameras") or [])}
    ]
    present = [c for c in C.ALL_CAMERAS if c in present]  # canonical order

    evidence_root = os.path.join(batch_root, "evidence")
    local_dir = os.path.join(batch_root, _LOCAL_SUBDIR)
    os.makedirs(local_dir, exist_ok=True)

    output_bucket = C.S3_OUTPUT_BUCKET
    region = C.S3_REGION
    inspection_bucket = _inspection_bucket()
    api_url = _ingest_api_url()
    reuse = _reuse_evidence_urls()

    batch_key = report_doc.get("batch_key", "")
    ts = extract_train_timestamp(batch_key)
    df = date_folder(ts)

    prior = _load_status(batch_root)

    for camera in present:
        url_maker = _UrlMaker(
            s3_client=s3_client, output_bucket=output_bucket, region=region,
            inspection_bucket=inspection_bucket, batch_key=batch_key,
            folder=folder_for(camera), date_folder_str=df,
            reuse=reuse, skip_upload=skip_upload)
        try:
            doc = build_inspection_json(camera=camera, report_doc=report_doc,
                                        evidence_root=evidence_root,
                                        url_maker=url_maker)
        except Exception as e:
            log.error("[DASHBOARD] build failed for %s: %s", camera, e)
            result["cameras"][camera] = {"status": "build_failed", "error": str(e)}
            continue

        raw_video_name = doc["inspection_data"]["raw_video_name"]
        json_name = f"{os.path.splitext(raw_video_name)[0]}_inspection.json"
        text = json.dumps(doc, indent=2, default=str)
        json_sha = _sha256_text(text)
        idem = ingest_idempotency_key(batch_key, camera, report_revision, json_sha)

        # ---- idempotency: already ingested this exact payload? ----
        pj = prior.get(camera) or {}
        if pj.get("status") == "ingested" and pj.get("json_sha256") == json_sha:
            log.info("[DASHBOARD] %s already ingested (rev=%s) -- skip",
                     camera, report_revision)
            result["cameras"][camera] = {"status": "already_ingested",
                                         "run_id": pj.get("run_id")}
            continue

        # ---- write local JSON (delivery/ only) ----
        local_json = os.path.join(local_dir, json_name)
        with open(local_json, "w", encoding="utf-8") as f:
            f.write(text)

        folder = folder_for(camera)
        s3_key = f"{folder}/{df}/{json_name}"
        s3_uri = f"s3://{inspection_bucket}/{s3_key}"

        entry = {
            "camera_id": full_camera_id(camera),
            "json_sha256": json_sha,
            "idempotency_key": idem,
            "report_revision": report_revision,
            "s3_uri": s3_uri,
            "run_id": None,
            "status": "prepared",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        # ---- upload JSON ----
        if skip_upload or s3_client is None:
            entry["status"] = "prepared_local_only"
        else:
            try:
                s3_client.upload_file(local_json, inspection_bucket, s3_key,
                                      ExtraArgs={"ContentType": "application/json"})
                entry["status"] = "uploaded"
            except Exception as e:
                log.error("[DASHBOARD] JSON upload failed %s: %s", s3_uri, e)
                entry["status"] = "upload_failed"
                entry["error"] = str(e)
                _record_status(batch_root, camera, entry)
                result["cameras"][camera] = {"status": entry["status"]}
                continue

        # ---- ingest POST ----
        if skip_ingest:
            entry["status"] = "prepared" if entry["status"] == "prepared_local_only" \
                else entry["status"]
            _record_status(batch_root, camera, entry)
            result["cameras"][camera] = {"status": entry["status"], "dry_run": True}
            continue

        payload = {"camera_id": full_camera_id(camera),
                   "inspection_s3_uri": s3_uri, "version": _version()}
        res = _post_ingest(api_url=api_url, payload=payload, idem_key=idem,
                           requests_mod=requests_mod)
        if res["ok"]:
            entry["status"] = "ingested"
            entry["run_id"] = res.get("run_id")
        else:
            entry["status"] = "ingest_failed"
            entry["error"] = res.get("error")
            entry["last_status_code"] = res.get("status_code")
        _record_status(batch_root, camera, entry)
        result["cameras"][camera] = {"status": entry["status"],
                                     "run_id": entry.get("run_id")}

    return result
