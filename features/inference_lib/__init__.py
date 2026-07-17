"""Ported feature-level intelligence from the legacy Wagon Eye system.

These modules are the mature inference building blocks (tracking,
identity merging, snapshot selection, illumination quality scoring,
geometric priors, OCR preprocessing, OCR cross-frame aggregation,
damage tracking).  They are intentionally extracted from the legacy
camera-centric process_video() wrappers and stripped of any
wagon_timeline / video-iteration dependencies so each can be driven
purely from a frame iterator inside a v4 feature processor.
"""
