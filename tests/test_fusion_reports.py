"""Commit 5 tests: per-camera fusion adapter + evidence scoping + report metadata.

No .pt models required -- these assert the fusion authority, provenance,
pending/missing distinction, disabled handling, legacy-flat vs new-schema
layout selection, atomic rewrite, camera-scoped evidence lookup, render
subsetting, and report metadata.

Run:  python tests/test_fusion_reports.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.lifecycle import ArrivalState, ResultState
from core.unified_wagon_state import UnifiedWagonState
from features._common import write_per_wagon_json as _w
from fusion import wagon_state_builder as WSB


def _state(n=1):
    return GlobalTrainState(total_wagons=n, master_fps=10.0,
                            wagons=[GlobalWagon(f"GW_{i}", i, 0, 10, 0.0, 1.0, "WAGON")
                                    for i in range(1, n + 1)])


def _cam(states, feature, camera, gw, payload):
    payload.setdefault("status", "OK")
    _w(os.path.join(states, feature, camera), gw, payload)


_ALL_PRESENT = {c: ArrivalState.PRESENT for c in C.ALL_CAMERAS}


def _fuse(states, arrival=None, disabled=None):
    return WSB.build(state=_state(1), wagon_states_root=states,
                     camera_arrival=arrival or _ALL_PRESENT,
                     disabled_features=disabled, global_state_version="H1",
                     fusion_revision=1, verbose=False)["GW_1"]


# -----------------------------------------------------------------------------
# authority + independence + fallback + damage-any-wins
# -----------------------------------------------------------------------------

def test_fusion_authority_and_independence():
    s = tempfile.mkdtemp()
    _cam(s, "door", "RIGHT_UP", "GW_1", {"right_door": "CLOSED", "right_door_confidence": 0.8})
    _cam(s, "door", "LEFT_UP", "GW_1", {"left_door": "OPEN", "left_door_confidence": 0.9})
    _cam(s, "ocr", "RIGHT_UP", "GW_1", {"wagon_identifier": "32145678901",
                                        "wagon_identifier_confidence": 0.7})
    _cam(s, "load", "RIGHT_UP_TOP", "GW_1", {"load_status": "LOADED", "load_confidence": 0.9})
    _cam(s, "load", "LEFT_UP_TOP", "GW_1", {"load_status": "EMPTY", "load_confidence": 0.5})
    _cam(s, "damage", "RIGHT_UP_TOP", "GW_1", {"damage_status": "OK", "top_damage_details": []})
    _cam(s, "damage", "LEFT_UP_TOP", "GW_1", {"damage_status": "DAMAGE",
                                              "top_damage_details": [{"class_name": "floor_damage"}]})
    u = _fuse(s)
    assert u.right_door == "CLOSED" and u.field_sources["right_door"] == "RIGHT_UP"
    assert u.left_door == "OPEN" and u.field_sources["left_door"] == "LEFT_UP"
    assert u.wagon_identifier == "32145678901"
    # RIGHT_UP_TOP wins load authority
    assert u.load_status == "LOADED" and u.field_sources["load_status"] == "RIGHT_UP_TOP"
    # any top camera DAMAGE wins
    assert u.top_damage == C.DAMAGE_PRESENT and u.field_sources["top_damage"] == "LEFT_UP_TOP"
    assert "LEFT_DOOR_OPEN" in u.anomalies and "TOP_DAMAGE" in u.anomalies
    assert u.result_state == ResultState.COMPLETE_WITH_ANOMALY


def test_load_fallback_only_when_primary_invalid():
    s = tempfile.mkdtemp()
    # primary present but NO_FRAMES -> fall back to LEFT_UP_TOP
    _cam(s, "load", "RIGHT_UP_TOP", "GW_1", {"status": "NO_FRAMES", "load_status": C.NO_DATA})
    _cam(s, "load", "LEFT_UP_TOP", "GW_1", {"load_status": "LOADED", "load_confidence": 0.6})
    u = _fuse(s)
    assert u.load_status == "LOADED" and u.field_sources["load_status"] == "LEFT_UP_TOP"

    # primary valid -> primary wins even though support differs
    s2 = tempfile.mkdtemp()
    _cam(s2, "load", "RIGHT_UP_TOP", "GW_1", {"load_status": "EMPTY", "load_confidence": 0.7})
    _cam(s2, "load", "LEFT_UP_TOP", "GW_1", {"load_status": "LOADED", "load_confidence": 0.9})
    u2 = _fuse(s2)
    assert u2.load_status == "EMPTY" and u2.field_sources["load_status"] == "RIGHT_UP_TOP"


# -----------------------------------------------------------------------------
# missing / pending never becomes OK
# -----------------------------------------------------------------------------

def test_missing_and_pending_never_ok():
    s = tempfile.mkdtemp()
    _cam(s, "door", "RIGHT_UP", "GW_1", {"right_door": "CLOSED", "right_door_confidence": 0.8})
    _cam(s, "ocr", "RIGHT_UP", "GW_1", {"wagon_identifier": "32145678901",
                                        "wagon_identifier_confidence": 0.7})
    # tops not arrived -> PENDING pre-closure
    arrival = {C.CAMERA_RIGHT_UP: ArrivalState.PRESENT,
               C.CAMERA_LEFT_UP: ArrivalState.PENDING_CAMERA,
               C.CAMERA_RIGHT_UP_TOP: ArrivalState.PENDING_CAMERA,
               C.CAMERA_LEFT_UP_TOP: ArrivalState.PENDING_CAMERA}
    u = _fuse(s, arrival)
    assert u.load_status == C.NO_DATA and u.field_status["load_status"] == ResultState.PENDING_CAMERA
    assert u.top_damage == C.NO_DATA and u.field_status["top_damage"] == ResultState.PENDING_CAMERA
    assert u.left_door == C.NO_DATA and u.field_status["left_door"] == ResultState.PENDING_CAMERA
    assert "TOP_DAMAGE" not in u.anomalies and u.result_state == "PENDING"

    # at closure the missing tops become CAMERA_MISSING_FINAL (still not OK)
    closure = dict(arrival)
    closure[C.CAMERA_LEFT_UP] = ArrivalState.CAMERA_MISSING_FINAL
    closure[C.CAMERA_RIGHT_UP_TOP] = ArrivalState.CAMERA_MISSING_FINAL
    closure[C.CAMERA_LEFT_UP_TOP] = ArrivalState.CAMERA_MISSING_FINAL
    u2 = _fuse(s, closure)
    assert u2.field_status["load_status"] == ResultState.CAMERA_MISSING_FINAL
    assert u2.load_status == C.NO_DATA


# -----------------------------------------------------------------------------
# disabled feature: DISABLED_BY_USER, no anomaly
# -----------------------------------------------------------------------------

def test_disabled_feature_no_anomaly():
    s = tempfile.mkdtemp()
    _cam(s, "door", "LEFT_UP", "GW_1", {"left_door": "OPEN", "left_door_confidence": 0.9})
    u = _fuse(s, disabled={"door"})
    assert u.left_door == C.DISABLED_DISPLAY
    assert u.field_status["left_door"] == ResultState.DISABLED_BY_USER
    assert "LEFT_DOOR_OPEN" not in u.anomalies


# -----------------------------------------------------------------------------
# legacy flat vs new-schema selection
# -----------------------------------------------------------------------------

def test_legacy_flat_readable():
    s = tempfile.mkdtemp()
    _w(os.path.join(s, "door"), "GW_1", {"status": "OK", "left_door": "OPEN",
                                         "right_door": "CLOSED",
                                         "left_door_confidence": 0.9})
    assert WSB.detect_layout(s) == "flat"
    u = _fuse(s)
    assert u.left_door == "OPEN" and u.right_door == "CLOSED"


def test_new_schema_never_reads_flat():
    s = tempfile.mkdtemp()
    # per-camera dir marks the batch as new-schema
    _cam(s, "door", "RIGHT_UP", "GW_1", {"right_door": "CLOSED", "right_door_confidence": 0.8})
    # a stale flat file with a DIFFERENT left/right must be IGNORED
    _w(os.path.join(s, "door"), "GW_1", {"status": "OK", "left_door": "OPEN",
                                         "right_door": "OPEN"})
    assert WSB.detect_layout(s) == "camera"
    u = _fuse(s)
    assert u.right_door == "CLOSED"          # from per-camera, not the flat OPEN
    assert u.left_door == C.NO_DATA          # LEFT_UP file absent -> NOT read from flat


# -----------------------------------------------------------------------------
# atomic rewrite preserves previous valid unified JSON on failure
# -----------------------------------------------------------------------------

def test_atomic_fusion_preserves_on_failure():
    s = tempfile.mkdtemp()
    _cam(s, "door", "RIGHT_UP", "GW_1", {"right_door": "CLOSED", "right_door_confidence": 0.8})
    _fuse(s)
    p = os.path.join(s, "unified", "GW_1.json")
    before = open(p, "rb").read()

    orig = WSB._fuse_camera_scoped
    WSB._fuse_camera_scoped = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        WSB.build(state=_state(1), wagon_states_root=s, camera_arrival=_ALL_PRESENT,
                  global_state_version="H1", fusion_revision=2, verbose=False)
    finally:
        WSB._fuse_camera_scoped = orig
    assert open(p, "rb").read() == before  # previous unified JSON intact


# -----------------------------------------------------------------------------
# evidence lookup is camera-scoped
# -----------------------------------------------------------------------------

def test_evidence_lookup_camera_scoped():
    from reporting import _evidence_lookup as EL
    ev = tempfile.mkdtemp()
    for cam, cls in (("RIGHT_UP_TOP", "inner_wall_damage"), ("LEFT_UP_TOP", "floor_damage")):
        d = os.path.join(ev, "GW_1", "damage", cam)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "track_1.jpg"), "w").write(cam)
        json.dump({"tracks": [{"track_idx": 1, "class_name": cls, "best_confidence": 0.9}]},
                  open(os.path.join(d, "metadata.json"), "w"))
    r = EL.damage_track_snapshots(ev, "GW_1", camera_id="RIGHT_UP_TOP")
    assert len(r) == 1 and r[0][1]["class_name"] == "inner_wall_damage"
    l = EL.damage_track_snapshots(ev, "GW_1", camera_id="LEFT_UP_TOP")
    assert len(l) == 1 and l[0][1]["class_name"] == "floor_damage"
    # snapshot resolves within the requested camera only
    assert EL.evidence_snapshot(ev, "GW_1", "damage", "track_1",
                                camera_id="RIGHT_UP_TOP").endswith(
        os.path.join("RIGHT_UP_TOP", "track_1.jpg"))


# -----------------------------------------------------------------------------
# rendering regenerates only the requested camera's mp4
# -----------------------------------------------------------------------------

def test_render_subset_only_target_camera():
    from rendering import feature_overlay_renderer as R
    out = tempfile.mkdtemp()
    orig = R._render_one_camera

    def fake_one(*, camera_id, output_path, **kw):
        open(output_path, "w").write("mp4")
        return output_path
    R._render_one_camera = fake_one
    try:
        video_paths = {c: f"/fake/{c}.mp4" for c in C.ALL_CAMERAS}
        res = R.render_all_cameras(
            state=_state(1), unified={}, evidence_root=out,
            video_paths=video_paths, per_camera_tracking_path="",
            output_dir=out, cameras=[C.CAMERA_LEFT_UP], verbose=False)
    finally:
        R._render_one_camera = orig
    assert set(res.keys()) == {C.CAMERA_LEFT_UP}
    assert os.path.isfile(os.path.join(out, "LEFT_UP_processed.mp4"))
    assert not os.path.isfile(os.path.join(out, "RIGHT_UP_processed.mp4"))


# -----------------------------------------------------------------------------
# report metadata: INTERIM -> FINAL / FINAL_PARTIAL; revision increments
# -----------------------------------------------------------------------------

def test_report_meta_status_transitions():
    from orchestrator import lifecycle_runner as LR
    from orchestrator import batch_manifest as BM
    m = BM.BatchManifest.new(batch_key="B", train_timestamp="B")
    m.global_state_version = "HASH1"
    for cam in C.ALL_CAMERAS:
        m.set_camera(BM.CameraSlot(camera_id=cam, s3_key=cam, etag="e"))
    assert LR._report_meta(m, final=False)["report_status"] == "INTERIM"
    assert LR._report_meta(m, final=True)["report_status"] == "FINAL"

    # a permanently missing camera -> FINAL_PARTIAL
    m2 = BM.BatchManifest.new(batch_key="B2", train_timestamp="B2")
    m2.global_state_version = "HASH1"
    m2.set_camera(BM.CameraSlot(camera_id=C.CAMERA_RIGHT_UP, s3_key="r", etag="e"))
    m2.cameras[C.CAMERA_LEFT_UP_TOP] = BM.CameraSlot(
        camera_id=C.CAMERA_LEFT_UP_TOP, arrival_state=ArrivalState.CAMERA_MISSING_FINAL)
    meta = LR._report_meta(m2, final=True)
    assert meta["report_status"] == "FINAL_PARTIAL"
    assert C.CAMERA_LEFT_UP_TOP in meta["cameras_missing_final"]
    assert meta["generated_from_global_state_hash"] == "HASH1"
    assert meta["partial_reason"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
