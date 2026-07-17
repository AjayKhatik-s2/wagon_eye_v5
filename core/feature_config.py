"""Stage-3 feature registry + per-run enable/disable configuration.

A single source of truth for the set of feature processors the pipeline
knows about.  Adding a feature here makes it automatically appear in the
interactive prompt, in the disabled-feature sentinel JSON, and in the
reports -- nothing else needs editing for a new feature to be toggleable.

NOTHING in this module loads a model, reads a frame, or touches
GlobalTrainState.  It is pure configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FeatureSpec:
    """Static description of one Stage-3 feature.

    key            stable identifier == the wagon_states/<key>/ folder name
                   and the processor FEATURE_NAME (door/ocr/load/damage).
    display_name   human label used in the prompt + reports.
    owned_fields   UnifiedWagonState attribute names this feature populates.
                   When the feature is DISABLED those fields carry the
                   DISABLED_DISPLAY sentinel instead of NO_DATA / OK.
    """
    key: str
    display_name: str
    owned_fields: Tuple[str, ...]


# Registry order is the canonical order everywhere (prompt numbering, etc.).
# Adding a FeatureSpec here makes the feature appear automatically in the
# startup prompt, the disabled-feature sentinel, and the reports.
FEATURE_REGISTRY: Tuple[FeatureSpec, ...] = (
    FeatureSpec("door",   "Door",   ("left_door", "right_door")),
    FeatureSpec("ocr",    "OCR",    ("wagon_identifier",)),
    FeatureSpec("load",   "Load",   ("load_status",)),
    FeatureSpec("damage", "Damage", ("top_damage", "side_damage")),
)

FEATURE_KEYS: Tuple[str, ...] = tuple(f.key for f in FEATURE_REGISTRY)
_BY_KEY: Dict[str, FeatureSpec] = {f.key: f for f in FEATURE_REGISTRY}


def get_spec(key: str) -> Optional[FeatureSpec]:
    return _BY_KEY.get(key)


# Map an owned UnifiedWagonState field -> its feature key (reverse lookup used
# by the reporting layer to know which fields a disabled feature owns).
FIELD_TO_FEATURE: Dict[str, str] = {
    fld: f.key for f in FEATURE_REGISTRY for fld in f.owned_fields
}


@dataclass
class FeatureConfig:
    """Holds the enabled/disabled set for one pipeline run."""
    enabled: Dict[str, bool] = field(
        default_factory=lambda: {k: True for k in FEATURE_KEYS}
    )

    # ---- queries ----
    def is_enabled(self, key: str) -> bool:
        return self.enabled.get(key, True)

    def disabled_keys(self) -> List[str]:
        return [k for k in FEATURE_KEYS if not self.enabled.get(k, True)]

    def enabled_keys(self) -> List[str]:
        return [k for k in FEATURE_KEYS if self.enabled.get(k, True)]

    # ---- mutation ----
    def disable(self, key: str) -> None:
        if key in self.enabled:
            self.enabled[key] = False

    def disable_many(self, keys: Sequence[str]) -> None:
        for k in keys:
            self.disable(k)

    # ---- factories ----
    @classmethod
    def all_on(cls) -> "FeatureConfig":
        return cls()

    @classmethod
    def from_disabled(cls, keys: Sequence[str]) -> "FeatureConfig":
        cfg = cls()
        cfg.disable_many([k for k in keys if k in _BY_KEY])
        return cfg

    def to_dict(self) -> Dict[str, bool]:
        return dict(self.enabled)


def parse_disable_arg(raw: Optional[str]) -> List[str]:
    """Parse a CLI value like 'door,ocr' -> ['door', 'ocr'].

    Unknown keys are dropped (case-insensitive match against FEATURE_KEYS).
    """
    if not raw:
        return []
    out: List[str] = []
    for tok in str(raw).replace(";", ",").split(","):
        k = tok.strip().lower()
        if k in _BY_KEY and k not in out:
            out.append(k)
    return out
