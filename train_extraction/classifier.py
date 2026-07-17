"""YOLO classification model wrapper for train/empty-track detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from ultralytics import YOLO


# Class aliases applied to whatever the model emits so downstream code can
# assume a stable vocabulary. Maps source-model class names → canonical names.
#
# The "track" → "empty_track" alias matters: top-view classification models in
# this codebase have historically emitted "track" while side-view ones use the
# more explicit "empty_track". Without this alias, the top pipelines never see
# a no-train frame and every clip is mis-classified as an incomplete train.
DEFAULT_CLASS_ALIASES: Mapping[str, str] = {
    "wagon_empty": "wagon",
    "wagon_filled": "wagon_loaded",
    "track": "empty_track",
}


@dataclass
class FrameClassification:
    is_train: bool
    predicted_class: str
    confidence: float
    # Spatial info (None for pure classifiers — only kept for compatibility)
    zone: Optional[str] = None
    # True when the predicted class is in ``ignore_class_names`` (e.g. a parallel
    # train on the "second_track"). Such frames are non-train AND must be skipped
    # by the segment finder so they don't count toward the empty-track end streak.
    ignored: bool = False


class FrameClassifier:
    """Detects train presence per-frame using a YOLO image-classification model."""

    def __init__(
        self,
        model: YOLO,
        track_class_name: str = "empty_track",
        class_aliases: Optional[Mapping[str, str]] = None,
        ignore_class_names: Optional[Iterable[str]] = None,
    ):
        self.model = model
        self.track_class_name = track_class_name
        self.class_aliases = dict(class_aliases or DEFAULT_CLASS_ALIASES)
        # Classes to ignore entirely during extraction (see FrameClassification).
        self.ignore_class_names = set(ignore_class_names or ())

    def classify(self, frame) -> FrameClassification:
        results = self.model(frame, verbose=False)
        if not results or results[0].probs is None:
            return FrameClassification(False, self.track_class_name, 0.0)

        probs = results[0].probs
        top1 = int(probs.top1)
        conf = float(probs.top1conf)
        label = results[0].names[top1]
        label = self.class_aliases.get(label, label)
        ignored = label in self.ignore_class_names
        # Ignored classes are never a train; the segment finder skips them so
        # they neither start/extend a train nor add to the end-of-train streak.
        is_train = (not ignored) and (label != self.track_class_name)
        return FrameClassification(is_train, label, conf, ignored=ignored)
