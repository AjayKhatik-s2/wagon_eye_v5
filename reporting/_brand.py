"""WagonEye legacy report brand identity -- palette, paragraph styles,
shared widgets, anomaly + status helpers.

Ported (constants verbatim, helpers structural mirrors) from
old_system/RIGHT_UP/combined_report_generator.py and the per-feature
generators.  This module is pure presentation logic -- it does NOT
import any old_system module and does NOT touch any v4 backend state.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import Paragraph, Table, TableStyle


# -----------------------------------------------------------------------------
# Palette (verbatim from legacy combined_report_generator.py:39-57)
# -----------------------------------------------------------------------------

LIGHT_RED       = colors.Color(1.0, 0.85, 0.85)
LIGHT_GREEN     = colors.Color(0.85, 0.96, 0.85)
HEADER_GRAY     = colors.Color(0.92, 0.92, 0.92)
HEADER_BLUE     = colors.Color(0.20, 0.40, 0.60)

NAVY_DARK       = colors.HexColor("#0B1D3A")
NAVY_MID        = colors.HexColor("#162D50")
TEAL_ACCENT     = colors.HexColor("#1A8A7D")
SLATE_BG        = colors.HexColor("#F4F6F9")
SLATE_BORDER    = colors.HexColor("#C8CED8")
SLATE_LIGHT     = colors.HexColor("#E8EAF0")
WHITE           = colors.HexColor("#FFFFFF")
LINK_BLUE       = colors.HexColor("#1565C0")
NO_FEED_RED     = colors.HexColor("#C62828")
SECTION_TEAL_BG = colors.HexColor("#E0F2F1")
WARN_BG         = colors.HexColor("#FFF3E0")
WARN_BORDER     = colors.HexColor("#E65100")

ISSUE_BG        = colors.HexColor("#FFEBEE")
OK_BG           = colors.HexColor("#E8F5E9")
LOADED_BG       = colors.HexColor("#E3F2FD")
EMPTY_BG        = colors.HexColor("#FFF3E0")
NA_BG           = colors.HexColor("#ECEFF1")

TEXT_DARK       = colors.HexColor("#1A1A2E")
TEXT_BODY       = colors.HexColor("#263238")
TEXT_MUTED      = colors.HexColor("#546E7A")
TEXT_LIGHT      = colors.HexColor("#9E9E9E")
SUBTITLE_GRAY   = colors.HexColor("#B0BEC5")

# Status text colors
COLOR_OK_GREEN  = colors.HexColor("#2E7D32")
COLOR_NOT_OK    = colors.HexColor("#C62828")

# Wagon-type accents
COLOR_LOADED    = colors.HexColor("#1565C0")
COLOR_EMPTY     = colors.HexColor("#E65100")


# -----------------------------------------------------------------------------
# Logo handling -- per-page draw callback (legacy combined:104-120)
# -----------------------------------------------------------------------------

def make_logo_callback(logo_path: Optional[str]):
    """Return a `(canvas, doc) -> None` callback that draws the logo at the
    top-left of every page.  Safe no-op when logo_path is None or missing."""

    def _on_page(canvas, doc):
        if not logo_path or not os.path.exists(logo_path):
            return
        try:
            canvas.saveState()
            canvas.drawImage(
                logo_path,
                doc.leftMargin,
                doc.height + doc.topMargin + 0.05 * inch,
                width=1.0 * inch,
                height=0.4 * inch,
                preserveAspectRatio=True,
                mask="auto",
            )
            canvas.restoreState()
        except Exception:
            pass

    return _on_page


# -----------------------------------------------------------------------------
# Paragraph style factory -- shared by combined + per-feature reports
# -----------------------------------------------------------------------------

def build_styles() -> Dict[str, ParagraphStyle]:
    """Return the legacy WagonEye paragraph styles as a name->style dict.

    Verbatim port of CombinedReportGenerator._setup_custom_styles (legacy
    :122-267) plus the per-feature `Title_Custom`, `Heading_Custom`,
    `StateOpen`, `StateClosed`, `StateOther` from
    RIGHT_UP/report_generator.py:86-124.
    """
    base = getSampleStyleSheet()
    out: Dict[str, ParagraphStyle] = {}

    def _add(style: ParagraphStyle) -> None:
        out[style.name] = style

    # Combined report styles
    _add(ParagraphStyle("ReportTitle",   parent=base["Heading1"],
        fontSize=18, alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
        textColor=WHITE, fontName="Helvetica-Bold", leading=24))
    _add(ParagraphStyle("ReportSubtitle", parent=base["Normal"],
        fontSize=10, alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
        textColor=SUBTITLE_GRAY, fontName="Helvetica", leading=14))
    _add(ParagraphStyle("SectionHeader", parent=base["Heading2"],
        fontSize=12, alignment=TA_CENTER, spaceBefore=12, spaceAfter=6,
        textColor=NAVY_DARK, fontName="Helvetica-Bold"))
    _add(ParagraphStyle("TableHeader", parent=base["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=colors.black,
        fontName="Helvetica-Bold"))
    _add(ParagraphStyle("TableCell", parent=base["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=colors.black,
        fontName="Helvetica"))
    _add(ParagraphStyle("SmallNote", parent=base["Normal"],
        fontSize=8, alignment=TA_LEFT, textColor=colors.gray,
        fontName="Helvetica-Oblique"))
    _add(ParagraphStyle("SmallNoteRight", parent=base["Normal"],
        fontSize=8, alignment=TA_RIGHT, textColor=colors.gray,
        fontName="Helvetica-Oblique"))
    _add(ParagraphStyle("SectionLabel", parent=base["Normal"],
        fontSize=10, alignment=TA_LEFT, textColor=NAVY_DARK,
        fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=6))
    _add(ParagraphStyle("LinkCell", parent=base["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=colors.black,
        fontName="Helvetica"))
    _add(ParagraphStyle("BannerTitle", parent=base["Normal"],
        fontSize=18, alignment=TA_CENTER, textColor=WHITE,
        fontName="Helvetica-Bold", leading=24))
    _add(ParagraphStyle("BannerDate", parent=base["Normal"],
        fontSize=10, alignment=TA_CENTER, textColor=SUBTITLE_GRAY,
        fontName="Helvetica", leading=14))
    _add(ParagraphStyle("SectionTitleWhite", parent=base["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=WHITE,
        fontName="Helvetica-Bold", leading=13))
    _add(ParagraphStyle("CameraLabel", parent=base["Normal"],
        fontSize=8, alignment=TA_CENTER, textColor=TEXT_MUTED,
        fontName="Helvetica-Bold", leading=11))
    _add(ParagraphStyle("LinkCellPro", parent=base["Normal"],
        fontSize=9, alignment=TA_CENTER, textColor=LINK_BLUE,
        fontName="Helvetica-Bold"))
    _add(ParagraphStyle("NoFeedCell", parent=base["Normal"],
        fontSize=8, alignment=TA_CENTER, textColor=NO_FEED_RED,
        fontName="Helvetica-Oblique"))
    _add(ParagraphStyle("WarningText", parent=base["Normal"],
        fontSize=10, alignment=TA_CENTER, textColor=WARN_BORDER,
        fontName="Helvetica-Bold", leading=14))

    # Per-feature report extras (legacy report_generator.py:86-124)
    _add(ParagraphStyle("Title_Custom", parent=base["Title"],
        fontSize=24, spaceAfter=30))
    _add(ParagraphStyle("Heading_Custom", parent=base["Heading1"],
        fontSize=18, spaceAfter=12))
    _add(ParagraphStyle("StateOpen", parent=base["Normal"],
        fontSize=14, textColor=colors.red, fontName="Helvetica-Bold"))
    _add(ParagraphStyle("StateClosed", parent=base["Normal"],
        fontSize=14, textColor=colors.green, fontName="Helvetica-Bold"))
    _add(ParagraphStyle("StateOther", parent=base["Normal"],
        fontSize=14, textColor=colors.orange, fontName="Helvetica-Bold"))

    return out


# -----------------------------------------------------------------------------
# State formatters (verbatim from legacy combined:712-733 + 749-775 + 777-795)
# -----------------------------------------------------------------------------

_DOOR_STATE_MAP = {
    "OPEN": "OPEN",
    "CLOSED": "CLOSED",
    "DAMAGE": "DAMAGE",
    "DAMAGED": "DAMAGE",
    "PARTIAL CLOSED": "PARTIAL CLOSED",
    "PARTIALLY CLOSED": "PARTIAL CLOSED",
    "PARTIAL": "PARTIAL CLOSED",
    "FLOOR DAMAGE": "DAMAGE",
    "INNER WALL DAMAGE": "DAMAGE",
    "OUTER WALL DAMAGE": "DAMAGE",
    "SIDE DAMAGE": "DAMAGE",
    "FLOOR DMG": "DAMAGE",
    "FLOOR DMG PROBABLE": "DAMAGE",
    "BODY DMG": "DAMAGE",
    "BODY DMG PROBABLE": "DAMAGE",
}


def format_door_status(state: Optional[str]) -> str:
    if not state:
        return "UNKNOWN"
    s = str(state).upper().replace("_", " ")
    return _DOOR_STATE_MAP.get(s, s)


# Negative (non-damage) top-camera states.  Includes the v4 damage-feature
# sentinels "OK" (DAMAGE_OK) -- a wagon that was inspected and had no damage --
# alongside the legacy raw-class negatives.  "NO_DATA" is handled separately so
# it maps to NO_DATA rather than OK.
_DAMAGE_NEG = {"NO_DAMAGE", "CLOSED", "LOADED", "OK"}


def format_damage_status(state: Optional[str]) -> str:
    """Normalize a top-camera damage state string to {DAMAGE, OK, NO_DATA}.

    The v4 damage feature emits "DAMAGE" / "OK" / "NO_DATA" sentinels (not raw
    class names), so those must be recognised explicitly -- otherwise a clean
    "OK" wagon would fall through to "DAMAGE".
    """
    if state is None:
        return "NO_DATA"
    s = str(state).upper().replace("_", " ").strip()
    if not s:
        return "NO_DATA"
    key = s.replace(" ", "_")
    if key == "NO_DATA":
        return "NO_DATA"
    if key == C.STATUS_DISABLED or s == C.DISABLED_DISPLAY:
        return C.DISABLED_DISPLAY
    if key in _DAMAGE_NEG:
        return "OK"
    return "DAMAGE"


def is_side_anomaly(door_state: Optional[str]) -> bool:
    """Verbatim of legacy `_has_open_door` rule (combined:777-795).

    Returns True if the door state denotes DAMAGE or OPEN (excluding PARTIAL).
    """
    if not door_state:
        return False
    s = str(door_state).lower()
    if "damage" in s:
        return True
    if "open" in s and "partial" not in s:
        return True
    return False


def is_top_anomaly(state: Optional[str]) -> bool:
    """Top-camera damage present iff `state` is a positive damage indicator.

    Mirrors the legacy rule (combined:1073-1103) but also excludes the v4
    damage-feature sentinels "ok" (inspected, no damage) and "no_data"
    (engine / brake-van / no frames) -- otherwise EVERY wagon would be flagged
    as damaged, since a clean WAGON reports top_damage == "OK".
    """
    if state is None:
        return False
    return str(state).lower() not in (
        "no_damage", "closed", "", "loaded", "ok", "no_data",
        "disabled by user", "disabled_by_user")


def build_doors_text(left_state: Optional[str], right_state: Optional[str]) -> str:
    """Mirror of `_build_doors_text` (combined:749-775) but for a single
    side: returns "DOOR 1 STATUS / DOOR 2 STATUS" when both sides are
    present; falls back to a single "DOOR 1 STATUS"; "NO DOOR DETECTED"
    if both sides are missing.

    NOTE the legacy semantics: door_number 1 and 2 are independent doors
    in ONE camera view (not L/R sides).  In the v4 train-state-native
    world each camera sees exactly one logical "door" per side; the
    second-door slot is only populated if the v4 layer reports two
    distinct door tracks.  This helper accepts the per-side state we
    already have and prints it as "DOOR 1 <state>".  If a caller has a
    second door state available they can pass it as `right_state` to
    get the "/ DOOR 2 ..." segment.
    """
    has_left = left_state and str(left_state).upper() != "NO_DATA"
    has_right = right_state and str(right_state).upper() != "NO_DATA"
    if not has_left and not has_right:
        return "NO DOOR DETECTED"
    if has_left and has_right:
        return f"DOOR 1 {format_door_status(left_state)} / DOOR 2 {format_door_status(right_state)}"
    only = left_state if has_left else right_state
    return f"DOOR 1 {format_door_status(only)}"


# -----------------------------------------------------------------------------
# Reusable widgets
# -----------------------------------------------------------------------------

def make_warning_banner(missing_cameras: Sequence[str], styles: Dict[str, ParagraphStyle]) -> Optional[Table]:
    """PARTIAL REPORT amber banner (legacy combined:460-478). Returns None
    if no cameras are missing."""
    if not missing_cameras:
        return None
    missing_str = ", ".join(missing_cameras)
    warn_data = [[
        Paragraph(
            f"<b>⚠  PARTIAL REPORT</b> — No feed received from: "
            f"<b>{missing_str}</b>",
            styles["WarningText"],
        )
    ]]
    t = Table(warn_data, colWidths=[10.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
        ("BOX",        (0, 0), (-1, -1), 1.2, WARN_BORDER),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def make_camera_link(
    url: Optional[str],
    label: str,
    camera_name: str,
    missing_cameras: Sequence[str],
    styles: Dict[str, ParagraphStyle],
) -> Paragraph:
    """Clickable link / NO FEED text for a single camera cell
    (legacy combined:269-303)."""
    if camera_name in missing_cameras:
        return Paragraph(
            f'<font color="#C62828"><i>NO FEED</i></font>',
            styles["NoFeedCell"],
        )
    if url:
        return Paragraph(
            f'<a href="{url}" color="#1565C0"><b><u>{label}</u></b></a>',
            styles["LinkCellPro"],
        )
    return Paragraph(
        f'<font color="#78909C">{label}</font>',
        styles["LinkCell"],
    )


# -----------------------------------------------------------------------------
# State-tinted info-box helper (legacy report_generator.py:732-799)
# -----------------------------------------------------------------------------

def state_colors(state: Optional[str]):
    """Return (text_color, bg_color, border_color) for a door / damage
    state.  Mirrors the legacy state-color mapping."""
    s = str(state or "").lower()
    if "open" in s or "damage" in s:
        return (
            colors.Color(1, 0.2, 0.2),     # text  -- red
            colors.Color(1, 0.9, 0.9),     # bg    -- light red
            colors.Color(1, 0.2, 0.2),     # border-- red
        )
    if "closed" in s and "partial" not in s:
        return (
            colors.Color(0, 0.6, 0),
            colors.Color(0.9, 1, 0.9),
            colors.Color(0, 0.6, 0),
        )
    return (
        colors.Color(0.9, 0.6, 0),
        colors.Color(1, 0.95, 0.85),
        colors.Color(0.9, 0.6, 0),
    )


# -----------------------------------------------------------------------------
# Camera priority used everywhere in the evidence section (combined:1136-1150)
# -----------------------------------------------------------------------------

CAMERA_PRIORITY = {
    "Left":      0,
    "Right":     1,
    "Left-Side": 2,
    "Right-Side":3,
    "Left-Top":  4,
    "Right-Top": 5,
}

CAMERA_LABELS = {
    "Left":      "Side Camera (Left) – Open Door",
    "Right":     "Side Camera (Right) – Open Door",
    "Left-Side": "Side Camera (Left) – Side Damage",
    "Right-Side":"Side Camera (Right) – Side Damage",
    "Left-Top":  "Top Camera (Left) – Damage",
    "Right-Top": "Top Camera (Right) – Damage",
}


# -----------------------------------------------------------------------------
# Constants every page will use
# -----------------------------------------------------------------------------

PAGE_BODY_WIDTH = 10.0 * inch       # 11.0in landscape A4 - 2*0.5in margins
EVIDENCE_IMG_MAX_W = 4.2 * inch
EVIDENCE_IMG_MAX_H = 2.8 * inch
WAGON_IMG_W = 9.0 * inch            # per-wagon detail page large image
WAGON_IMG_H = 4.5 * inch

# Legacy report tags for the "Damaged Wagon Report" classifier
CAMERA_TAGS_SIDE = {"LEFT_UP": "Left", "RIGHT_UP": "Right"}
CAMERA_TAGS_TOP  = {"LEFT_UP_TOP": "Left-Top", "RIGHT_UP_TOP": "Right-Top"}
