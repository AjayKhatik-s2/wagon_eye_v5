"""Commit 4 tests: per-camera feature-output + evidence isolation.

These prove that late-camera processing can NEVER overwrite another camera's
results or evidence.  They run WITHOUT .pt models (missing model -> the
processor writes NO_DATA per-camera payloads), so they assert the LAYOUT,
ISOLATION, MARKER, and ORDERING guarantees -- not inference quality.

Run:  python tests/test_camera_isolation.py   (or via pytest)
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


def _state(n=1):
    wagons = [GlobalWagon(f"GW_{i}", i, 0, 10, 0.0, 1.0, "WAGON")
              for i in range(1, n + 1)]
    return GlobalTrainState(total_wagons=n, wagons=wagons,
                            master_fps=10.0, master_total_frames=10)


def _dirs():
    out = tempfile.mkdtemp()
    return (os.path.join(out, "states"), os.path.join(out, "cache"),
            os.path.join(out, "evidence"), os.path.join(out, "models"))


# -----------------------------------------------------------------------------
# door: per-camera isolation + no cross-side output
# -----------------------------------------------------------------------------

def test_door_right_unchanged_after_late_left():
    from features.door import processor as door
    states, cache, ev, models = _dirs()
    st = _state(1)
    common = dict(state=st, cache_root=cache, feature_models_dir=models,
                  output_dir=states, evidence_root=ev, verbose=False)

    door.run(cameras=[C.CAMERA_RIGHT_UP], **common)
    right_path = os.path.join(states, "door", "RIGHT_UP", "GW_1.json")
    assert os.path.isfile(right_path)
    right_before = open(right_path, "rb").read()

    # late LEFT_UP
    door.run(cameras=[C.CAMERA_LEFT_UP], **common)
    left_path = os.path.join(states, "door", "LEFT_UP", "GW_1.json")
    assert os.path.isfile(left_path)

    # RIGHT_UP result byte-for-byte unchanged
    assert open(right_path, "rb").read() == right_before

    r = json.load(open(right_path)); l = json.load(open(left_path))
    assert r["camera_id"] == "RIGHT_UP" and r["side"] == "right"
    assert l["camera_id"] == "LEFT_UP" and l["side"] == "left"
    # LEFT_UP file carries NO right-door result
    assert "right_door" not in l
    assert "left_door" not in r


# -----------------------------------------------------------------------------
# load + damage: independent per-top-camera files
# -----------------------------------------------------------------------------

def test_top_cameras_independent_files():
    from features.load import processor as load
    from features.damage import processor as damage
    states, cache, ev, models = _dirs()
    st = _state(1)
    common = dict(state=st, cache_root=cache, feature_models_dir=models,
                  output_dir=states, evidence_root=ev, verbose=False)

    load.run(cameras=[C.CAMERA_RIGHT_UP_TOP], **common)
    load.run(cameras=[C.CAMERA_LEFT_UP_TOP], **common)
    damage.run(cameras=[C.CAMERA_RIGHT_UP_TOP], **common)
    damage.run(cameras=[C.CAMERA_LEFT_UP_TOP], **common)

    for feat in ("load", "damage"):
        for cam in ("RIGHT_UP_TOP", "LEFT_UP_TOP"):
            p = os.path.join(states, feat, cam, "GW_1.json")
            assert os.path.isfile(p), p
            assert json.load(open(p))["camera_id"] == cam


# -----------------------------------------------------------------------------
# evidence isolation: one camera never deletes another's evidence
# -----------------------------------------------------------------------------

def test_evidence_isolation():
    from features.door import processor as door
    states, cache, ev, models = _dirs()
    st = _state(1)
    # pre-seed RIGHT_UP evidence
    ru_ev = os.path.join(ev, "GW_1", "door", "RIGHT_UP")
    os.makedirs(ru_ev, exist_ok=True)
    open(os.path.join(ru_ev, "keep.jpg"), "w").write("x")

    door.run(cameras=[C.CAMERA_LEFT_UP], state=st, cache_root=cache,
             feature_models_dir=models, output_dir=states, evidence_root=ev,
             verbose=False)
    assert os.path.isfile(os.path.join(ru_ev, "keep.jpg"))


# -----------------------------------------------------------------------------
# atomic evidence: failed build preserves previous, no temp left behind
# -----------------------------------------------------------------------------

def test_atomic_evidence_failed_build_preserves():
    from features._evidence import atomic_camera_evidence
    ev = tempfile.mkdtemp()
    final = os.path.join(ev, "GW_1", "damage", "RIGHT_UP_TOP")

    with atomic_camera_evidence(ev, "GW_1", "damage", "RIGHT_UP_TOP") as tmp:
        open(os.path.join(tmp, "track_1.jpg"), "w").write("good")
    assert os.path.isfile(os.path.join(final, "track_1.jpg"))

    try:
        with atomic_camera_evidence(ev, "GW_1", "damage", "RIGHT_UP_TOP") as tmp:
            open(os.path.join(tmp, "track_1.jpg"), "w").write("BAD")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # previous preserved, temp cleaned
    assert open(os.path.join(final, "track_1.jpg")).read() == "good"
    assert not os.path.isdir(final + ".tmp_build")


# -----------------------------------------------------------------------------
# damage reads THIS camera's own load result (per-camera dependency)
# -----------------------------------------------------------------------------

def test_damage_reads_same_camera_load():
    from features.damage.processor import _load_status_for_camera
    from features._common import write_per_wagon_json
    states = tempfile.mkdtemp()
    write_per_wagon_json(os.path.join(states, "load", "RIGHT_UP_TOP"), "GW_1",
                         {"status": C.STATUS_OK, "load_status": C.LOAD_LOADED})
    assert _load_status_for_camera(states, "GW_1", "RIGHT_UP_TOP") == C.LOAD_LOADED
    # a different top camera has no load file -> None (never reads across cameras)
    assert _load_status_for_camera(states, "GW_1", "LEFT_UP_TOP") is None


# -----------------------------------------------------------------------------
# markers: model/threshold change invalidates; identical skips
# -----------------------------------------------------------------------------

def test_marker_identity_and_invalidation():
    from orchestrator import feature_markers as FM
    states = tempfile.mkdtemp()
    models = tempfile.mkdtemp()
    model_path = os.path.join(models, C.MODEL_DOOR_STATE)
    open(model_path, "wb").write(b"v1-weights")

    ident = FM.compute_identity(camera_id="RIGHT_UP", feature="door",
                                source_key="k", etag="e1",
                                global_state_version="v1", feat_models_dir=models)
    FM.write_marker(states, ident, status="OK", wagons_completed=1)
    assert FM.is_up_to_date(states, ident)

    # ETag change invalidates
    ident2 = dict(ident); ident2["etag"] = "e2"
    assert not FM.is_up_to_date(states, ident2)

    # model content change invalidates (different sha)
    open(model_path, "wb").write(b"v2-weights-different-length")
    ident3 = FM.compute_identity(camera_id="RIGHT_UP", feature="door",
                                 source_key="k", etag="e1",
                                 global_state_version="v1", feat_models_dir=models)
    assert ident3["model_sha256"] != ident["model_sha256"]
    assert not FM.is_up_to_date(states, ident3)


# -----------------------------------------------------------------------------
# lifecycle: late camera runs ONLY its features; markers skip on re-run;
# load runs before damage for a top camera
# -----------------------------------------------------------------------------

def _patch_recording(monkey_calls):
    """Replace the 4 processor run()s with recorders; return a restore fn."""
    import features.door.processor as dp
    import features.ocr.processor as op
    import features.load.processor as lp
    import features.damage.processor as mp
    originals = {"door": dp.run, "ocr": op.run, "load": lp.run, "damage": mp.run}

    def mk(feat):
        def fake(*, cameras=None, **kw):
            monkey_calls.append((feat, tuple(cameras or [])))
            return {}
        return fake
    dp.run, op.run, lp.run, mp.run = mk("door"), mk("ocr"), mk("load"), mk("damage")

    def restore():
        dp.run, op.run, lp.run, mp.run = (originals["door"], originals["ocr"],
                                          originals["load"], originals["damage"])
    return restore


def test_lifecycle_late_camera_and_ordering_and_skip():
    from orchestrator import lifecycle_runner as LR
    from orchestrator import batch_manifest as BM
    states, cache, ev, models = _dirs()
    os.makedirs(states, exist_ok=True)
    st = _state(1)
    ctx = LR.RunContext(workspace_root=tempfile.mkdtemp(), recon_models_dir="x",
                        feat_models_dir=models, s3_client=None, verbose=False)
    m = BM.BatchManifest.new(batch_key="B", train_timestamp="B")
    m.global_state_version = "v1"
    for cam in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP):
        m.set_camera(BM.CameraSlot(camera_id=cam, s3_key=cam, etag="e", bucket="__local__"))

    calls = []
    restore = _patch_recording(calls)
    try:
        # late LEFT_UP -> ONLY door(left); no ocr/load/damage
        LR._run_camera_features(m, ctx, cameras=[C.CAMERA_LEFT_UP], state=st,
                                cache_root=cache, states_root=states, evidence_root=ev)
        assert calls == [("door", (C.CAMERA_LEFT_UP,))], calls

        # top camera -> load BEFORE damage (deterministic ordering)
        calls.clear()
        LR._run_camera_features(m, ctx, cameras=[C.CAMERA_RIGHT_UP_TOP], state=st,
                                cache_root=cache, states_root=states, evidence_root=ev)
        assert calls == [("load", (C.CAMERA_RIGHT_UP_TOP,)),
                         ("damage", (C.CAMERA_RIGHT_UP_TOP,))], calls

        # re-run (restart) -> markers skip everything
        calls.clear()
        LR._run_camera_features(m, ctx, cameras=[C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP],
                                state=st, cache_root=cache, states_root=states,
                                evidence_root=ev)
        assert calls == [], calls
    finally:
        restore()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
