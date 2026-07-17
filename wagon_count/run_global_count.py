"""
run_global_count.py  --  Wagon Eye Phase-1 (standalone)
========================================================

Self-contained CLI entry point for EC2 / any Linux server deployment.

CONVENTIONS (just-drop-files-and-run)
-------------------------------------
Place the 4 trimmed train videos in ./inputs/ with these exact names:
    inputs/right_up.mp4
    inputs/left_up.mp4
    inputs/right_up_top.mp4
    inputs/left_up_top.mp4

Place the 4 YOLO model weights in ./models/ with these exact names:
    models/right_up_wagon_gap.pt     (used by RIGHT_UP -- master)
    models/left_up_wagon_gap.pt      (used by LEFT_UP)
    models/top_gap.pt                (used by RIGHT_UP_TOP and LEFT_UP_TOP)
    models/side_classification.pt    (used by RIGHT_UP for ENGINE/WAGON/BRAKE_VAN)

Then run:
    python run_global_count.py

Outputs land in ./results/ (configurable with --output).

OVERRIDES
---------
You can override any path explicitly:
    python run_global_count.py \
        --right_up      /abs/path/cam_right_up.mp4 \
        --left_up       /abs/path/cam_left_up.mp4 \
        --right_up_top  /abs/path/cam_right_up_top.mp4 \
        --left_up_top   /abs/path/cam_left_up_top.mp4 \
        --models-dir    /abs/path/models \
        --output        /abs/path/results

WHAT THIS PRODUCES
------------------
    results/
        global_train_state.json          <-- canonical Phase-1 output
        per_camera_tracking.json
        processed_videos/
            RIGHT_UP_processed.mp4
            LEFT_UP_processed.mp4
            RIGHT_UP_TOP_processed.mp4
            LEFT_UP_TOP_processed.mp4
        frames/
            RIGHT_UP/
                GW_1/  frame_000000.jpg, frame_000001.jpg, ...
                GW_2/  ...
            LEFT_UP/   ...
            RIGHT_UP_TOP/  ...
            LEFT_UP_TOP/   ...

WHAT THIS DOES NOT DO
---------------------
No door / damage / OCR detection.  No PDF or email.  No S3 upload.
Phase-1 is gap counting + global synchronization + classification only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import List, Dict, Optional

from global_train_state import (
    GlobalTrainState,
    LocalCameraTracks,
    SegmentClass,
    MASTER_CAMERA,
    ALL_CAMERAS,
    CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP,
    CAMERA_LEFT_UP_TOP,
    summarize_state,
)
from tracker_engine import GapTracker, MasterClassifier, segments_from_gaps
import global_alignment as ga
import video_segmenter as vs


# =============================================================================
# Auto-discovery: default file conventions
# =============================================================================

DEFAULT_INPUT_FILENAMES = {
    CAMERA_RIGHT_UP:     "right_up.mp4",
    CAMERA_LEFT_UP:      "left_up.mp4",
    CAMERA_RIGHT_UP_TOP: "right_up_top.mp4",
    CAMERA_LEFT_UP_TOP:  "left_up_top.mp4",
}

# Some users may use these alternative names; we'll fall back to them.
_INPUT_FALLBACK_PATTERNS = {
    CAMERA_RIGHT_UP:     ["right_up.mp4", "RIGHT_UP.mp4", "cam_right_up.mp4"],
    CAMERA_LEFT_UP:      ["left_up.mp4", "LEFT_UP.mp4", "cam_left_up.mp4"],
    CAMERA_RIGHT_UP_TOP: ["right_up_top.mp4", "RIGHT_UP_TOP.mp4", "cam_right_up_top.mp4"],
    CAMERA_LEFT_UP_TOP:  ["left_up_top.mp4", "LEFT_UP_TOP.mp4", "cam_left_up_top.mp4"],
}


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _resolve_input(explicit: Optional[str], inputs_dir: str, camera_id: str) -> str:
    """Return a path to the camera's video.

    Search order:
        1. explicit (if provided)
        2. inputs_dir/<filename>  for each fallback name
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--{camera_id.lower()} path does not exist: {explicit}")
        return os.path.abspath(explicit)

    for name in _INPUT_FALLBACK_PATTERNS[camera_id]:
        p = os.path.join(inputs_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        f"No input video found for {camera_id}. "
        f"Looked in {inputs_dir} for: "
        f"{_INPUT_FALLBACK_PATTERNS[camera_id]}. "
        f"Either drop the file there or pass --{camera_id.lower()} <path>."
    )


def _resolve_optional_input(explicit: Optional[str], inputs_dir: str,
                            camera_id: str) -> Optional[str]:
    """Like _resolve_input but returns None (instead of raising) when the
    camera's video is absent -- used by the master-first subset flow where
    only the currently-arrived cameras are processed."""
    if explicit:
        return os.path.abspath(explicit) if os.path.exists(explicit) else None
    for name in _INPUT_FALLBACK_PATTERNS[camera_id]:
        p = os.path.join(inputs_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# Phase-2/v4 alias map: the new package prefers shorter model names.
# When resolving the canonical wagon_count names, we ALSO accept the
# shorter aliases (right_up_gap.pt / left_up_gap.pt) so the new
# `models/reconstruction/` directory can use either convention.
_MODEL_ALIASES = {
    "right_up_wagon_gap.pt": ("right_up_gap.pt",),
    "left_up_wagon_gap.pt":  ("left_up_gap.pt",),
}


def _resolve_model(name: str, models_dir: str) -> str:
    # 1) try the canonical name first
    p = os.path.join(models_dir, name)
    if os.path.exists(p):
        return os.path.abspath(p)
    # 2) try any registered short-name aliases
    for alias in _MODEL_ALIASES.get(name, ()):
        ap = os.path.join(models_dir, alias)
        if os.path.exists(ap):
            return os.path.abspath(ap)
    aliases = _MODEL_ALIASES.get(name, ())
    looked = [p] + [os.path.join(models_dir, a) for a in aliases]
    raise FileNotFoundError(
        f"Model not found: {name}. Looked at: {looked}. "
        f"Drop the .pt file in {models_dir} or pass --models-dir <path>."
    )


# =============================================================================
# Per-camera processing
# =============================================================================

def _process_side_camera(
    camera_id: str, video_path: str, gap_model_path: str,
    confidence: float, min_height_ratio: float,
    keep_raw_detections: bool, verbose: bool,
) -> LocalCameraTracks:
    tracker = GapTracker(
        camera_id=camera_id, model_path=gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    return tracker.process_video(video_path, keep_raw_detections=keep_raw_detections)


def _process_top_camera(
    camera_id: str, video_path: str, top_gap_model_path: str,
    confidence: float, min_height_ratio: float,
    keep_raw_detections: bool, verbose: bool,
) -> LocalCameraTracks:
    tracker = GapTracker(
        camera_id=camera_id, model_path=top_gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    return tracker.process_video(video_path, keep_raw_detections=keep_raw_detections)


def _classify_master_pre_fusion(
    master_tracks: LocalCameraTracks,
    side_classification_model_path: str,
    num_samples: int,
    verbose: bool,
):
    pre_segments = segments_from_gaps(master_tracks.gaps, master_tracks.total_frames)
    if not pre_segments:
        if verbose:
            print("[CLASSIFY] no pre-fusion segments to classify")
        return []
    if verbose:
        print(f"[CLASSIFY] classifying {len(pre_segments)} pre-fusion segments on "
              f"{os.path.basename(master_tracks.video_path)}")
    clf = MasterClassifier(side_classification_model_path, num_samples=num_samples, verbose=verbose)
    return clf.classify_segments(master_tracks.video_path, pre_segments)


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    here = _here()
    default_inputs = os.path.join(here, "inputs")
    default_models = os.path.join(here, "models")
    default_output = os.path.join(here, "results")

    p = argparse.ArgumentParser(
        prog="run_global_count.py",
        description="Phase-1 global wagon counting + classification (standalone).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default file conventions:\n"
            "  inputs/{right_up,left_up,right_up_top,left_up_top}.mp4\n"
            "  models/right_up_wagon_gap.pt   (RIGHT_UP -- master)\n"
            "  models/left_up_wagon_gap.pt    (LEFT_UP)\n"
            "  models/top_gap.pt              (RIGHT_UP_TOP + LEFT_UP_TOP)\n"
            "  models/side_classification.pt  (RIGHT_UP classification)\n"
            "Drop the files in ./inputs and ./models, then run with no args."
        ),
    )

    p.add_argument("--right_up",     default=None, help="Override path to RIGHT_UP video (master)")
    p.add_argument("--left_up",      default=None, help="Override path to LEFT_UP video")
    p.add_argument("--right_up_top", default=None, help="Override path to RIGHT_UP_TOP video")
    p.add_argument("--left_up_top",  default=None, help="Override path to LEFT_UP_TOP video")

    # Master-first incremental reconstruction: only the cameras actually
    # present are processed; the master defaults to RIGHT_UP.  A non-RIGHT_UP
    # master (LEFT_UP fallback) is UNVALIDATED and must be opted into.
    p.add_argument("--master-camera", default=CAMERA_RIGHT_UP,
                   choices=list(ALL_CAMERAS),
                   help="Camera to use as the master timeline (default RIGHT_UP)")
    p.add_argument("--allow-fallback-master", action="store_true",
                   help="Permit a non-RIGHT_UP master (LEFT_UP fallback). "
                        "OFF by default -- side_classification.pt is unvalidated "
                        "on LEFT_UP.")

    p.add_argument("--inputs-dir",   default=default_inputs,
                   help=f"Directory containing the 4 input videos (default: {default_inputs})")
    p.add_argument("--models-dir",   default=default_models,
                   help=f"Directory containing the 4 .pt models (default: {default_models})")
    p.add_argument("--output", "-o", default=default_output,
                   help=f"Output root directory (default: {default_output})")

    p.add_argument("--side-confidence", type=float, default=0.4,
                   help="Confidence threshold for the side gap models "
                        "right_up_wagon_gap.pt and left_up_wagon_gap.pt "
                        "(default: 0.4)")
    p.add_argument("--top-confidence",  type=float, default=0.4,
                   help="Confidence threshold for top_gap.pt (default: 0.4)")
    p.add_argument("--side-min-height-ratio", type=float, default=0.35,
                   help="Min bbox height / frame height for SIDE gap detections "
                        "(default: 0.35). Tall gaps are typical on side cameras.")
    p.add_argument("--top-min-height-ratio",  type=float, default=0.05,
                   help="Min bbox height / frame height for TOP gap detections "
                        "(default: 0.05). Top-camera gaps are thin horizontal "
                        "strips, so this MUST be much smaller than the side ratio.")
    p.add_argument("--classification-samples", type=int, default=5,
                   help="Frames per segment for side_classification.pt vote (default: 5)")

    p.add_argument("--fuse-min-support", type=int, default=2,
                   help="Min supporting cameras for inserting a missed gap (default: 2)")
    p.add_argument("--fuse-max-spread",  type=float, default=1.5,
                   help="Max time spread within a fusion cluster (default: 1.5s)")
    p.add_argument("--fuse-min-conf",    type=float, default=0.4,
                   help="Min mean confidence to insert a fused gap (default: 0.4)")

    p.add_argument("--no-videos", action="store_true", help="Skip overlay video rendering")
    p.add_argument("--no-frames", action="store_true", help="Skip per-wagon frame extraction")
    p.add_argument("--every-nth-frame", type=int, default=1,
                   help="Keep 1 of every N frames during extraction (default: 1)")
    p.add_argument("--no-raw-detections", action="store_true",
                   help="Don't keep raw per-frame detections in memory (saves RAM)")
    p.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    verbose = not args.quiet

    t_start = time.time()
    print("=" * 70)
    print("  WAGON EYE - PHASE 1 GLOBAL TRAIN RECONSTRUCTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Resolve inputs (present cameras only) + master selection
    # ------------------------------------------------------------------
    _explicit = {
        CAMERA_RIGHT_UP:     args.right_up,
        CAMERA_LEFT_UP:      args.left_up,
        CAMERA_RIGHT_UP_TOP: args.right_up_top,
        CAMERA_LEFT_UP_TOP:  args.left_up_top,
    }
    present_videos: Dict[str, str] = {}
    for cam in ALL_CAMERAS:
        p = _resolve_optional_input(_explicit[cam], args.inputs_dir, cam)
        if p:
            present_videos[cam] = p

    master_cam = args.master_camera
    if master_cam not in present_videos:
        print(f"ERROR: master camera {master_cam} video is not present; cannot "
              f"reconstruct (present: {sorted(present_videos)})", file=sys.stderr)
        return 4
    if master_cam != CAMERA_RIGHT_UP and not args.allow_fallback_master:
        print(f"ERROR: master {master_cam} != RIGHT_UP requires "
              f"--allow-fallback-master (LEFT_UP classification is unvalidated)",
              file=sys.stderr)
        return 4

    # Resolve only the models the present cameras need.
    _SIDE_GAP_MODEL = {
        CAMERA_RIGHT_UP: "right_up_wagon_gap.pt",
        CAMERA_LEFT_UP:  "left_up_wagon_gap.pt",
    }
    try:
        gap_model: Dict[str, str] = {}
        need_top = any(c in present_videos for c in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP))
        top_gap_path = _resolve_model("top_gap.pt", args.models_dir) if need_top else None
        for cam in present_videos:
            if cam in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
                gap_model[cam] = top_gap_path
            else:
                gap_model[cam] = _resolve_model(_SIDE_GAP_MODEL[cam], args.models_dir)
        # classification runs on the master video
        side_cls_path = _resolve_model("side_classification.pt", args.models_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    missing_at_reconstruction = [c for c in ALL_CAMERAS if c not in present_videos]
    print(f"  master camera            : {master_cam}"
          f"{'  (FALLBACK)' if master_cam != CAMERA_RIGHT_UP else ''}")
    for cam in ALL_CAMERAS:
        tag = present_videos.get(cam, "<absent>")
        print(f"  {cam:<24} : {tag}")
    print(f"  output root              : {args.output}")
    print()

    os.makedirs(args.output, exist_ok=True)
    processed_videos_dir = os.path.join(args.output, "processed_videos")
    frames_root = os.path.join(args.output, "frames")
    os.makedirs(processed_videos_dir, exist_ok=True)
    os.makedirs(frames_root, exist_ok=True)

    keep_raw = not args.no_raw_detections

    # ------------------------------------------------------------------
    # STEP 1 -- per-camera gap tracking (present cameras only)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1  Per-camera gap tracking")
    print("-" * 70)
    tracks: Dict[str, LocalCameraTracks] = {}
    try:
        for cam in ALL_CAMERAS:
            if cam not in present_videos:
                continue
            if cam in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
                tracks[cam] = _process_top_camera(
                    cam, present_videos[cam], gap_model[cam],
                    confidence=args.top_confidence,
                    min_height_ratio=args.top_min_height_ratio,
                    keep_raw_detections=keep_raw, verbose=verbose,
                )
            else:
                tracks[cam] = _process_side_camera(
                    cam, present_videos[cam], gap_model[cam],
                    confidence=args.side_confidence,
                    min_height_ratio=args.side_min_height_ratio,
                    keep_raw_detections=keep_raw, verbose=verbose,
                )
    except Exception as e:
        print(f"ERROR: per-camera tracking failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3

    print()
    print("  Local counts after Step 1:")
    for cam in ALL_CAMERAS:
        if cam not in tracks:
            continue
        t = tracks[cam]
        print(f"    {cam:<14}  wagons={t.local_wagon_count:>3}   gaps={len(t.gaps):>3}   "
              f"fps={t.fps:.2f}   frames={t.total_frames}")
    print()

    # ------------------------------------------------------------------
    # STEP 2 -- master classification (on the chosen master video)
    # ------------------------------------------------------------------
    print("-" * 70)
    print(f"  STEP 2  {master_cam} master classification (ENGINE / WAGON / BRAKE_VAN)")
    print("-" * 70)
    master = tracks[master_cam]
    try:
        initial_classifications = _classify_master_pre_fusion(
            master, side_cls_path,
            num_samples=args.classification_samples, verbose=verbose,
        )
    except Exception as e:
        print(f"WARNING: master classification failed: {e}", file=sys.stderr)
        traceback.print_exc()
        initial_classifications = []

    # ------------------------------------------------------------------
    # STEP 3 -- cross-camera fusion (support = present non-master cameras)
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  STEP 3  Cross-camera gap fusion")
    print("-" * 70)
    support = [tracks[c] for c in ALL_CAMERAS if c in tracks and c != master_cam]
    support_present = [c for c in ALL_CAMERAS if c in tracks and c != master_cam]
    fuse_cfg = dict(ga.PHASE1_DEFAULTS)
    fuse_cfg.update({
        "insert_min_support": int(args.fuse_min_support),
        "insert_max_spread_sec": float(args.fuse_max_spread),
        "insert_min_confidence": float(args.fuse_min_conf),
    })

    state: GlobalTrainState = ga.assemble_global_train_state(
        master_tracks=master,
        support_tracks=support,
        initial_classifications=initial_classifications,
        config=fuse_cfg,
        verbose=verbose,
    )

    # ---- Master-first reconstruction provenance ----
    # A gap is only RECOVERED when >=2 support cameras agree (insert_min_support),
    # emitted as a GapCorrection.  Support present != support used.
    recoveries = len(state.corrections_applied)
    if not support_present:
        recon_mode = "MASTER_ONLY"
    elif recoveries > 0:
        recon_mode = "MASTER_WITH_FUSED_SUPPORT"
    else:
        recon_mode = "MASTER_WITH_SUPPORT_AVAILABLE"
    state.participating_cameras = [c for c in ALL_CAMERAS if c in present_videos]
    state.missing_at_reconstruction = missing_at_reconstruction
    state.reconstruction_mode = recon_mode
    state.support_cameras_present = support_present
    state.support_gap_recoveries = recoveries
    state.support_fusion_used = recoveries > 0
    state.fallback_master_used = (master_cam != CAMERA_RIGHT_UP)
    state.reconstruction_confidence = 1.0 if master_cam == CAMERA_RIGHT_UP else 0.6
    state.sealed_at = _utc_now_iso()
    state.sealing_reason = (
        f"reconstructed(master={master_cam}, support_present={support_present}, "
        f"recoveries={recoveries}, mode={recon_mode})"
    )

    # ------------------------------------------------------------------
    # STEP 4 -- write JSON
    # ------------------------------------------------------------------
    state_json_path = os.path.join(args.output, "global_train_state.json")
    with open(state_json_path, "w", encoding="utf-8") as f:
        f.write(state.to_json())
    print()
    print(f"[OUTPUT] wrote {state_json_path}")

    tracking_dump = {
        cam: tracks[cam].to_dict(include_classifications=(cam == master_cam))
        for cam in ALL_CAMERAS if cam in tracks
    }
    if initial_classifications:
        tracking_dump[master_cam]["pre_fusion_classifications"] = [
            c.to_dict() for c in initial_classifications
        ]
    tracking_path = os.path.join(args.output, "per_camera_tracking.json")
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_dump, f, indent=2)
    print(f"[OUTPUT] wrote {tracking_path}")

    # ------------------------------------------------------------------
    # STEP 5 -- overlay videos
    # ------------------------------------------------------------------
    if not args.no_videos:
        print()
        print("-" * 70)
        print("  STEP 5  Overlay videos")
        print("-" * 70)
        for cam in ALL_CAMERAS:
            if cam not in tracks:
                continue
            try:
                out_mp4 = os.path.join(processed_videos_dir, f"{cam}_processed.mp4")
                vs.render_processed_video(
                    local_tracks=tracks[cam],
                    state=state,
                    output_path=out_mp4,
                    draw_raw_detections=keep_raw,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"WARNING: render failed for {cam}: {e}", file=sys.stderr)
                state.add_note(f"render_failed:{cam}:{e}")

    # ------------------------------------------------------------------
    # STEP 6 -- wagon-wise frame extraction
    # ------------------------------------------------------------------
    if not args.no_frames:
        print()
        print("-" * 70)
        print("  STEP 6  Per-wagon frame extraction")
        print("-" * 70)
        for cam in ALL_CAMERAS:
            if cam not in tracks:
                continue
            try:
                vs.extract_wagon_frames(
                    local_tracks=tracks[cam],
                    state=state,
                    output_root=frames_root,
                    every_nth_frame=args.every_nth_frame,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"WARNING: frame extraction failed for {cam}: {e}", file=sys.stderr)
                state.add_note(f"frame_extraction_failed:{cam}:{e}")

    # ------------------------------------------------------------------
    # STEP 7 -- final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print()
    print(summarize_state(state))
    print(f"  total elapsed: {elapsed:.1f}s")
    print(f"  output root  : {os.path.abspath(args.output)}")
    print()

    # Re-write JSON so any added notes are persisted
    with open(state_json_path, "w", encoding="utf-8") as f:
        f.write(state.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
