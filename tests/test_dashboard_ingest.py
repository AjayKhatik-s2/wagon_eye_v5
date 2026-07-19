"""Tests for delivery/dashboard_ingest.py -- the read-only legacy dashboard adapter.

Covers: per-camera authority, legacy wrapper/schema, operational-day date-folder,
missing evidence, idempotent restart, retry behaviour, disabled-by-default, and
the "no writes outside delivery/" guarantee.

No models, no network, no real S3 -- a fake s3 client and a fake requests module
are injected.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delivery import dashboard_ingest as DI
from delivery import finalization as FIN
from core import constants as C


# -----------------------------------------------------------------------------
# fakes
# -----------------------------------------------------------------------------

class FakeS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        self.uploads.append((bucket, key))


class FakeResp:
    def __init__(self, code, body=None, text=""):
        self.status_code = code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeRequests:
    """Returns queued responses in order; records calls."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._responses.pop(0) if self._responses else FakeResp(200, {"run_id": "R"})


# -----------------------------------------------------------------------------
# fixture batch
# -----------------------------------------------------------------------------

def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _touch_jpg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")  # tiny fake JPEG


def make_batch(root):
    """Build a finalized-artifact batch with 2 wagons and camera-scoped evidence."""
    report = {
        "schema": "wagon_eye.combined_report/1",
        "batch_key": "20260408_032134",
        "train_metadata": {
            "master_camera": "RIGHT_UP",
            "source_video_urls": {
                "RIGHT_UP": "s3://in/right_up_20260408_032134.mp4",
                "LEFT_UP": "s3://in/left_up_20260408_032134.mp4",
                "RIGHT_UP_TOP": "s3://in/right_up_top_20260408_032134.mp4",
            },
            "processed_video_urls": {"RIGHT_UP": "https://x/right_processed.mp4"},
        },
        "summary": {"total_wagons": 2, "engine_count": 1, "wagon_count": 1,
                    "loaded": 1, "empty": 1},
        "wagons": [
            {"global_id": "GW_1", "wagon_index": 1, "classification": C.CLASS_WAGON,
             "wagon_identifier": "12345678901",
             "right_door": C.DOOR_OPEN, "left_door": C.DOOR_CLOSED,
             "load_status": C.LOAD_LOADED, "top_damage": C.DAMAGE_PRESENT,
             "supporting_cameras": ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"]},
            {"global_id": "GW_2", "wagon_index": 2, "classification": C.CLASS_ENGINE,
             "wagon_identifier": C.NO_DATA,
             "right_door": C.DOOR_CLOSED, "left_door": C.DOOR_CLOSED,
             "load_status": C.LOAD_EMPTY, "top_damage": C.DAMAGE_OK,
             "supporting_cameras": ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"]},
        ],
        "report_meta": {"report_revision": 0, "report_status": "FINAL",
                        "cameras_present": ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"],
                        "cameras_missing_final": [],
                        "generated_from_global_state_version": "deadbeef"},
    }
    _write(os.path.join(root, "reports", "combined_train_report.json"), report)

    ev = os.path.join(root, "evidence")
    # GW_1 RIGHT_UP door (open) + ocr
    _write(os.path.join(ev, "GW_1", "door", "RIGHT_UP", "metadata.json"),
           {"sides": {"right": {"bbox": [10, 20, 110, 220], "state": "OPEN",
                                "confidence": 0.83}}})
    _touch_jpg(os.path.join(ev, "GW_1", "door", "RIGHT_UP", "right_best.jpg"))
    _write(os.path.join(ev, "GW_1", "ocr", "RIGHT_UP", "metadata.json"),
           {"full_number": "12345678901", "ocr_confidence": 0.9})
    _touch_jpg(os.path.join(ev, "GW_1", "ocr", "RIGHT_UP", "best_frame.jpg"))
    # GW_1 LEFT_UP door (closed)
    _write(os.path.join(ev, "GW_1", "door", "LEFT_UP", "metadata.json"),
           {"sides": {"left": {"bbox": [1, 2, 3, 4], "state": "CLOSED",
                               "confidence": 0.7}}})
    # GW_1 TOP load + damage
    _write(os.path.join(ev, "GW_1", "load", "RIGHT_UP_TOP", "metadata.json"),
           {"load_status": "LOADED", "confidence": 0.77})
    _touch_jpg(os.path.join(ev, "GW_1", "load", "RIGHT_UP_TOP", "best_frame.jpg"))
    _write(os.path.join(ev, "GW_1", "damage", "RIGHT_UP_TOP", "metadata.json"),
           {"damage_status": "DAMAGE",
            "tracks": [{"track_idx": 1, "class_name": "floor_damage",
                        "bbox": [5, 6, 55, 66], "best_confidence": 0.61}]})
    _touch_jpg(os.path.join(ev, "GW_1", "damage", "RIGHT_UP_TOP", "track_1.jpg"))

    # finalization marker already written by stage_finalize
    FIN.write(root, {"batch_key": "20260408_032134", "report_revision": 0,
                     "uploaded": True, "email_sent": True,
                     "upload_urls": {"pdf": "https://x/combined.pdf",
                                     "camera_RIGHT_UP": "https://x/right.pdf"}})
    return report


def _all_files(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = os.path.getmtime(p)
    return out


# -----------------------------------------------------------------------------
# date-folder (operational-day 05:00 IST rule)
# -----------------------------------------------------------------------------

def test_date_folder_operational_day_rule():
    assert DI.extract_train_timestamp("x_20260408_032134.mp4") == datetime(2026, 4, 8, 3, 21, 34)
    # 03:21 is before 05:00 -> previous day
    assert DI.date_folder(datetime(2026, 4, 8, 3, 21, 34)) == "2026-04-07"
    # 13:34 is after 05:00 -> same day
    assert DI.date_folder(datetime(2026, 4, 8, 13, 34, 56)) == "2026-04-08"
    # exactly 05:00 -> same day
    assert DI.date_folder(datetime(2026, 4, 8, 5, 0, 0)) == "2026-04-08"


# -----------------------------------------------------------------------------
# per-camera authority + legacy wrapper/schema
# -----------------------------------------------------------------------------

def _url_maker(root, camera):
    return DI._UrlMaker(
        s3_client=None, output_bucket=C.S3_OUTPUT_BUCKET, region=C.S3_REGION,
        inspection_bucket=C.S3_OUTPUT_BUCKET, batch_key="20260408_032134",
        folder=DI.folder_for(camera), date_folder_str="2026-04-07",
        reuse=True, skip_upload=True)


def test_legacy_wrapper_and_schema(tmp_path):
    root = str(tmp_path)
    report = make_batch(root)
    ev = os.path.join(root, "evidence")
    doc = DI.build_inspection_json(camera="RIGHT_UP", report_doc=report,
                                   evidence_root=ev, url_maker=_url_maker(root, "RIGHT_UP"))
    assert set(doc.keys()) == {"camera_id", "version", "inspection_data"}
    assert doc["camera_id"] == "camera_CCTV_HZBN_DHN_2_RIGHT_UP"
    assert doc["version"] == "v1"
    d = doc["inspection_data"]
    for k in ("raw_video_name", "direction", "rake_status", "total_wagons",
              "doors_open", "doors_closed", "damaged_wagons", "wagon_number_results",
              "segment_type_map", "wagon_segments", "problem_frames",
              "loco_frames", "loco_number_results"):
        assert k in d, f"missing legacy key {k}"


def test_right_up_authority_doors_and_ocr(tmp_path):
    root = str(tmp_path)
    report = make_batch(root)
    ev = os.path.join(root, "evidence")
    d = DI.build_inspection_json(camera="RIGHT_UP", report_doc=report,
                                 evidence_root=ev,
                                 url_maker=_url_maker(root, "RIGHT_UP"))["inspection_data"]
    assert d["total_wagons"] == 2               # global authoritative count
    assert d["doors_open"] == 1                 # GW_1 right OPEN
    assert d["wagon_number_results"]["1"] == {"is_valid_11_digit": True,
                                              "display_number": "12345678901"}
    assert d["wagon_number_results"]["2"]["is_valid_11_digit"] is False
    # one door_open problem frame with bbox+class from door metadata
    pf = [p for p in d["problem_frames"] if p["problem_type"] == "door_open"]
    assert len(pf) == 1
    assert pf[0]["bounding_box"]["class_name"] == "door_open"
    assert pf[0]["bounding_box"]["bounding_box_coordinates"] == [10, 20, 110, 220]
    assert pf[0]["s3_url"] and "evidence/GW_1/door/RIGHT_UP/right_best.jpg" in pf[0]["s3_url"]
    assert d["rake_status"] == "Loaded"         # fused load: loaded>=empty
    assert d["_adapter"]["camera_authority"] == "side:right_door+ocr"


def test_left_up_authority_isolated(tmp_path):
    root = str(tmp_path)
    report = make_batch(root)
    ev = os.path.join(root, "evidence")
    d = DI.build_inspection_json(camera="LEFT_UP", report_doc=report,
                                 evidence_root=ev,
                                 url_maker=_url_maker(root, "LEFT_UP"))["inspection_data"]
    assert d["doors_open"] == 0                 # both left doors CLOSED
    assert d["problem_frames"] == []
    assert d["_adapter"]["camera_authority"] == "side:left_door+ocr"


def test_top_authority_load_and_damage(tmp_path):
    root = str(tmp_path)
    report = make_batch(root)
    ev = os.path.join(root, "evidence")
    d = DI.build_inspection_json(camera="RIGHT_UP_TOP", report_doc=report,
                                 evidence_root=ev,
                                 url_maker=_url_maker(root, "RIGHT_UP_TOP"))["inspection_data"]
    assert d["doors_open"] == 0 and d["doors_closed"] == 0
    assert d["damaged_wagons"] == 1
    pf = [p for p in d["problem_frames"] if p["damage_detected"]]
    assert len(pf) == 1 and pf[0]["bounding_box"]["class_name"] == "floor_damage"
    assert pf[0]["bounding_box"]["bounding_box_coordinates"] == [5, 6, 55, 66]
    assert d["_adapter"]["camera_authority"] == "top:load+damage"


def test_degraded_fields_not_invented(tmp_path):
    root = str(tmp_path)
    report = make_batch(root)
    ev = os.path.join(root, "evidence")
    d = DI.build_inspection_json(camera="RIGHT_UP", report_doc=report,
                                 evidence_root=ev,
                                 url_maker=_url_maker(root, "RIGHT_UP"))["inspection_data"]
    assert d["direction"] == "unknown"
    assert d["loco_frames"] == [] and d["loco_number_results"] == {}
    assert d["total_loco_frames"] == 0
    assert "direction" in d["_adapter"]["degraded_fields"]


# -----------------------------------------------------------------------------
# missing evidence -> graceful (no crash, no fabricated urls)
# -----------------------------------------------------------------------------

def test_missing_evidence_graceful(tmp_path):
    root = str(tmp_path)
    report = make_batch(root)
    # wipe evidence entirely
    import shutil
    shutil.rmtree(os.path.join(root, "evidence"))
    d = DI.build_inspection_json(camera="RIGHT_UP", report_doc=report,
                                 evidence_root=os.path.join(root, "evidence"),
                                 url_maker=_url_maker(root, "RIGHT_UP"))["inspection_data"]
    # still counts doors from the report, but no gallery/problem-frame images
    assert d["doors_open"] == 1
    for seg in d["wagon_segments"]:
        assert seg["wagon_frames"] == []       # no invented urls
    # door_open PF still emitted but with zeroed bbox (metadata absent)
    pf = [p for p in d["problem_frames"] if p["problem_type"] == "door_open"]
    assert pf and pf[0]["bounding_box"]["bounding_box_coordinates"] == [0, 0, 0, 0]
    assert pf[0]["s3_url"] is None


# -----------------------------------------------------------------------------
# disabled-by-default
# -----------------------------------------------------------------------------

def test_enabled_by_default(monkeypatch):
    # default is now ON
    monkeypatch.delenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", raising=False)
    assert DI.is_enabled() is True


def test_disable_via_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "false")
    root = str(tmp_path)
    make_batch(root)
    before = _all_files(root)
    res = DI.run(batch_root=root, s3_client=FakeS3())
    assert res == {"enabled": False, "cameras": {}}
    assert _all_files(root) == before          # nothing written


# -----------------------------------------------------------------------------
# full run + idempotent restart
# -----------------------------------------------------------------------------

def test_run_ingests_each_present_camera_then_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    make_batch(root)
    s3 = FakeS3()
    rq = FakeRequests([FakeResp(200, {"run_id": "R1"}),
                       FakeResp(200, {"run_id": "R2"}),
                       FakeResp(200, {"run_id": "R3"})])
    res = DI.run(batch_root=root, s3_client=s3, requests_mod=rq)
    assert res["enabled"] is True
    assert set(res["cameras"]) == {"RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"}
    assert all(v["status"] == "ingested" for v in res["cameras"].values())
    assert len(rq.calls) == 3
    # 3 JSON uploads to the inspection bucket
    assert sum(1 for b, k in s3.uploads if b == C.S3_OUTPUT_BUCKET) == 3

    # ---- restart: same artifacts -> no re-upload, no re-POST ----
    rq2 = FakeRequests([])
    s3b = FakeS3()
    res2 = DI.run(batch_root=root, s3_client=s3b, requests_mod=rq2)
    assert all(v["status"] == "already_ingested" for v in res2["cameras"].values())
    assert len(rq2.calls) == 0
    assert s3b.uploads == []

    # marker recorded per-camera dashboard status
    marker = FIN.load(root)
    assert set(marker["dashboard_ingested"]) == {"RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"}
    assert marker["dashboard_ingested"]["RIGHT_UP"]["run_id"] == "R1"
    # existing finalization fields preserved
    assert marker["email_sent"] is True and marker["uploaded"] is True


def test_new_revision_reingests(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    report = make_batch(root)
    DI.run(batch_root=root, s3_client=FakeS3(),
           requests_mod=FakeRequests([FakeResp(200, {"run_id": "R1"})] * 3))
    # bump revision -> different json hash -> re-ingest
    report["report_meta"]["report_revision"] = 1
    _write(os.path.join(root, "reports", "combined_train_report.json"), report)
    rq = FakeRequests([FakeResp(200, {"run_id": "R9"})] * 3)
    res = DI.run(batch_root=root, s3_client=FakeS3(), requests_mod=rq)
    assert all(v["status"] == "ingested" for v in res["cameras"].values())
    assert len(rq.calls) == 3


# -----------------------------------------------------------------------------
# retry behaviour
# -----------------------------------------------------------------------------

def test_post_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    rq = FakeRequests([FakeResp(503), FakeResp(200, {"run_id": "OK"})])
    out = DI._post_ingest(api_url="http://x", payload={}, idem_key="k",
                          requests_mod=rq, base_delay=0.0)
    assert out["ok"] is True and out["run_id"] == "OK"
    assert len(rq.calls) == 2


def test_post_422_is_permanent(monkeypatch):
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    rq = FakeRequests([FakeResp(422, text="bad")])
    out = DI._post_ingest(api_url="http://x", payload={}, idem_key="k",
                          requests_mod=rq, base_delay=0.0)
    assert out["ok"] is False and out["status_code"] == 422
    assert len(rq.calls) == 1


def test_post_exhausts_retries(monkeypatch):
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    rq = FakeRequests([FakeResp(500), FakeResp(500), FakeResp(500)])
    out = DI._post_ingest(api_url="http://x", payload={}, idem_key="k",
                          requests_mod=rq, base_delay=0.0)
    assert out["ok"] is False
    assert len(rq.calls) == 3


def test_ingest_failure_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    root = str(tmp_path)
    make_batch(root)
    rq = FakeRequests([FakeResp(500)] * 9)     # all cameras fail
    res = DI.run(batch_root=root, s3_client=FakeS3(), requests_mod=rq)
    assert all(v["status"] == "ingest_failed" for v in res["cameras"].values())
    marker = FIN.load(root)
    assert marker["dashboard_ingested"]["RIGHT_UP"]["status"] == "ingest_failed"


# -----------------------------------------------------------------------------
# no writes outside delivery/
# -----------------------------------------------------------------------------

def test_no_writes_outside_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    make_batch(root)
    before = _all_files(root)
    # dry-run: build + record, no upload/POST
    DI.run(batch_root=root, s3_client=FakeS3(), skip_upload=True)
    after = _all_files(root)
    changed = {p for p in after if p not in before or after[p] != before.get(p)}
    assert changed, "adapter should have written something"
    for p in changed:
        assert p.startswith("delivery/"), f"wrote outside delivery/: {p}"


def test_dry_run_does_not_post(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    make_batch(root)
    rq = FakeRequests([FakeResp(200, {"run_id": "X"})] * 3)
    s3 = FakeS3()
    res = DI.run(batch_root=root, s3_client=s3, skip_upload=True, requests_mod=rq)
    assert len(rq.calls) == 0
    assert s3.uploads == []
    assert all(v.get("dry_run") for v in res["cameras"].values())
