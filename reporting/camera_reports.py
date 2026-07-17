"""Stage 5a -- CAMERA-WISE reports (legacy WagonEye reporting hierarchy).

The legacy system produced one report PER CAMERA, each showing only the
information that camera was authoritative for, then merged the four into
the combined train report.  This module reproduces that hierarchy on the
v4 train-state-native backend:

    reports/right_up_report.pdf       authority: right door, OCR, classification
    reports/left_up_report.pdf        authority: left door
    reports/right_up_top_report.pdf   authority: load, top damage
    reports/left_up_top_report.pdf    authority: top-damage support / validation

Each camera report contains, in order:
    1. Camera Summary page  (camera name, visible wagons, anomalies,
       processing confidence, coverage %, Detection Summary table)
    2. Per-wagon pages      (camera-authoritative detections only,
       snapshots from THAT camera only; anomalous wagons first)
    3. Camera Anomaly Summary (grouped by severity)
    4. Camera Evidence pages  (legacy snapshot grid)

Inputs are READ-ONLY: GlobalTrainState, UnifiedWagonState,
wagon_states/<feature>/<gw>.json, evidence/<gw>/<feature>/, and
wagon_cache/<gw>/<cam_folder>/ frames.  No detector / model loads.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState

from . import _brand
from . import _pages
from . import _evidence_lookup as ev


# -----------------------------------------------------------------------------
# Camera authority specification
# -----------------------------------------------------------------------------

CAMERA_TITLES = {
    C.CAMERA_RIGHT_UP:     "RIGHT_UP Camera Report",
    C.CAMERA_LEFT_UP:      "LEFT_UP Camera Report",
    C.CAMERA_RIGHT_UP_TOP: "RIGHT_UP_TOP Camera Report",
    C.CAMERA_LEFT_UP_TOP:  "LEFT_UP_TOP Camera Report",
}

CAMERA_AUTHORITY = {
    C.CAMERA_RIGHT_UP:     ("right door", "OCR", "classification"),
    C.CAMERA_LEFT_UP:      ("left door",),
    C.CAMERA_RIGHT_UP_TOP: ("load", "top damage"),
    C.CAMERA_LEFT_UP_TOP:  ("top-damage support",),
}

CAMERA_FILE = {
    C.CAMERA_RIGHT_UP:     "right_up_report.pdf",
    C.CAMERA_LEFT_UP:      "left_up_report.pdf",
    C.CAMERA_RIGHT_UP_TOP: "right_up_top_report.pdf",
    C.CAMERA_LEFT_UP_TOP:  "left_up_top_report.pdf",
}

# Anomaly severity ordering (high -> low) for the camera anomaly summary.
_SEVERITY_ORDER = ["HIGH", "MEDIUM", "LOW"]
_SEVERITY_COLOR = {
    "HIGH":   "#C62828",
    "MEDIUM": "#E65100",
    "LOW":    "#9E9E9E",
}


def _now_ist() -> datetime:
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


# -----------------------------------------------------------------------------
# Per-camera wagon item extraction
# -----------------------------------------------------------------------------

def _wagon_covered(cache_root: Optional[str], gw_id: str, camera_id: str) -> bool:
    """A wagon is 'visible' to a camera if it has >=1 cached frame for it."""
    if not cache_root:
        return False
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    d = os.path.join(cache_root, gw_id, folder)
    if not os.path.isdir(d):
        return False
    try:
        for fn in os.listdir(d):
            if fn.endswith(".jpg"):
                return True
    except OSError:
        pass
    return False


def _build_camera_items(
    *, camera_id: str, state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: Optional[str], wagon_states_root: Optional[str],
    cache_root: Optional[str],
) -> List[Dict[str, Any]]:
    """Return per-wagon camera-authoritative items in train order.

    Each item:
        {sr, gw_id, classification, classification_conf, visible,
         detections: [(label, state, confidence, snapshot_path), ...],
         anomalies: [(severity, text), ...],
         primary_confidence: float}
    """
    items: List[Dict[str, Any]] = []
    for idx, gw in enumerate(state.wagons, start=1):
        u = unified.get(gw.global_id)
        if u is None:
            # Never skip a wagon: synthesize a NO_DATA state carrying the
            # authoritative Stage-1 classification so GW_1..GW_n stay contiguous.
            u = UnifiedWagonState(
                global_id=gw.global_id, wagon_index=gw.wagon_index,
                classification=gw.classification,
                classification_confidence=gw.classification_confidence,
            )
        visible = _wagon_covered(cache_root, gw.global_id, camera_id)
        detections: List[Tuple[str, Optional[str], float, Optional[str]]] = []
        anomalies: List[Tuple[str, str]] = []
        confidences: List[float] = []

        if camera_id == C.CAMERA_RIGHT_UP:
            # Right door
            snap = ev.evidence_snapshot(evidence_root, gw.global_id, "door", "right_best")
            detections.append(("Right Door", u.right_door,
                               u.right_door_confidence, snap))
            confidences.append(u.right_door_confidence)
            if _brand.is_side_anomaly(u.right_door):
                sev = "HIGH"
                anomalies.append((sev, f"RIGHT_DOOR {_brand.format_door_status(u.right_door)}"))
            # OCR
            ocr_snap = ev.evidence_snapshot(evidence_root, gw.global_id, "ocr", "best_frame")
            ocr_val = (u.wagon_identifier
                       if u.wagon_identifier not in (None, "", C.NO_DATA) else "MISSING")
            detections.append(("Wagon Number (OCR)", ocr_val,
                               u.wagon_identifier_confidence, ocr_snap))
            confidences.append(u.wagon_identifier_confidence)
            if (u.wagon_identifier in (None, "", C.NO_DATA)
                    and u.classification == C.CLASS_WAGON):
                anomalies.append(("LOW", "OCR_MISSING"))
            # Classification
            confidences.append(gw.classification_confidence)

        elif camera_id == C.CAMERA_LEFT_UP:
            snap = ev.evidence_snapshot(evidence_root, gw.global_id, "door", "left_best")
            detections.append(("Left Door", u.left_door,
                               u.left_door_confidence, snap))
            confidences.append(u.left_door_confidence)
            if _brand.is_side_anomaly(u.left_door):
                anomalies.append(("HIGH", f"LEFT_DOOR {_brand.format_door_status(u.left_door)}"))
            elif u.left_door == C.DOOR_PARTIAL:
                anomalies.append(("MEDIUM", "LEFT_DOOR PARTIAL CLOSED"))

        elif camera_id == C.CAMERA_RIGHT_UP_TOP:
            # Load (authoritative)
            load_snap = ev.evidence_snapshot(evidence_root, gw.global_id, "load", "best_frame")
            detections.append(("Load Status", u.load_status,
                               u.load_confidence, load_snap))
            confidences.append(u.load_confidence)
            if u.load_status == C.NO_DATA and u.classification == C.CLASS_WAGON:
                anomalies.append(("LOW", "LOAD_NO_DATA"))
            # Damage (right top tracks)
            for path, tr in _camera_damage_tracks(evidence_root, gw.global_id,
                                                   C.CAMERA_RIGHT_UP_TOP):
                detections.append((f"Top Damage ({tr.get('class_name','damage')})",
                                   "DAMAGE", float(tr.get("best_confidence") or 0.0), path))
                confidences.append(float(tr.get("best_confidence") or 0.0))
            if u.top_damage == C.DAMAGE_PRESENT:
                anomalies.append(("HIGH", "TOP_DAMAGE"))

        elif camera_id == C.CAMERA_LEFT_UP_TOP:
            # Top-damage support / validation (left top tracks)
            tracks = _camera_damage_tracks(evidence_root, gw.global_id,
                                           C.CAMERA_LEFT_UP_TOP)
            for path, tr in tracks:
                detections.append((f"Top Damage Support ({tr.get('class_name','damage')})",
                                   "DAMAGE", float(tr.get("best_confidence") or 0.0), path))
                confidences.append(float(tr.get("best_confidence") or 0.0))
            if tracks:
                anomalies.append(("HIGH", "TOP_DAMAGE (support)"))
            # Validation note for load (left top corroborates right top)
            raw_load = ev.read_wagon_feature_json(wagon_states_root, "load", gw.global_id)
            per_cam = (raw_load.get("per_camera") or {}) if isinstance(raw_load, dict) else {}
            lt = per_cam.get(C.CAMERA_LEFT_UP_TOP) or {}
            if lt:
                detections.append(("Load Validation",
                                   lt.get("load_status", "-"),
                                   float(lt.get("confidence") or 0.0), None))

        primary_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        items.append({
            "sr": idx,
            "gw_id": gw.global_id,
            "classification": u.classification,
            "classification_conf": gw.classification_confidence,
            "visible": visible,
            "detections": detections,
            "anomalies": anomalies,
            "primary_confidence": primary_conf,
            "start_time": gw.start_time,
            "end_time": gw.end_time,
            "ocr": (u.wagon_identifier
                    if u.wagon_identifier not in (None, "", C.NO_DATA) else "-"),
        })
    return items


def _camera_damage_tracks(evidence_root, gw_id, camera_id):
    """Return [(snapshot_path, track_meta), ...] for one top camera."""
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    tracks = md.get("tracks") or []
    out = []
    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        if tr.get("camera_id") != camera_id:
            continue
        idx = tr.get("track_idx")
        path = ev.evidence_snapshot(evidence_root, gw_id, "damage", f"track_{int(idx)}") if idx else None
        if path:
            out.append((path, tr))
    out.sort(key=lambda x: float(x[1].get("best_confidence") or 0.0), reverse=True)
    return out


# -----------------------------------------------------------------------------
# Camera Summary page
# -----------------------------------------------------------------------------

def _camera_summary_page(
    *, camera_id: str, items: List[Dict[str, Any]],
    total_wagons: int, styles,
):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak

    visible = sum(1 for it in items if it["visible"])
    n_anom = sum(1 for it in items if it["anomalies"])
    coverage = (100.0 * visible / total_wagons) if total_wagons else 0.0
    confs = [it["primary_confidence"] for it in items if it["primary_confidence"] > 0]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    authority = ", ".join(CAMERA_AUTHORITY.get(camera_id, ()))

    elements: List[Any] = []
    # Navy banner title
    banner = Table([[Paragraph(
        f"{camera_id} CAMERA REPORT", styles["BannerTitle"])],
        [Paragraph(_now_ist().strftime("%d-%m-%Y  |  %H:%M IST"),
                   styles["BannerDate"])]],
        colWidths=[_brand.PAGE_BODY_WIDTH])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _brand.NAVY_DARK),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (0, 0), 14),
        ("BOTTOMPADDING", (0, 1), (0, 1), 12),
        ("BOX", (0, 0), (-1, -1), 1.5, _brand.NAVY_DARK),
    ]))
    elements.append(banner)
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(
        f"<b>Authoritative for:</b> {authority}",
        ParagraphStyle("AuthLine", fontSize=11, alignment=TA_CENTER,
                       textColor=_brand.TEAL_ACCENT, fontName="Helvetica-Bold"),
    ))
    elements.append(Spacer(1, 0.15 * inch))

    # KPI cards row
    kpi_cells = [
        ("VISIBLE WAGONS", f"{visible} / {total_wagons}", _brand.NAVY_MID),
        ("DETECTED ANOMALIES", str(n_anom),
         _brand.COLOR_NOT_OK if n_anom else _brand.COLOR_OK_GREEN),
        ("PROCESSING CONFIDENCE", f"{avg_conf:.0%}", _brand.NAVY_MID),
        ("COVERAGE", f"{coverage:.0f}%", _brand.NAVY_MID),
    ]
    hdr_p = ParagraphStyle("KpiHdr", fontSize=8, alignment=TA_CENTER,
                           textColor=_brand.WHITE, fontName="Helvetica-Bold")
    val_p = ParagraphStyle("KpiVal", fontSize=16, alignment=TA_CENTER,
                           textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold")
    header_row = [Paragraph(h, hdr_p) for h, _, _ in kpi_cells]
    value_row = [Paragraph(v, val_p) for _, v, _ in kpi_cells]
    kpi_t = Table([header_row, value_row], colWidths=[2.5 * inch] * 4)
    kstyle = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("TOPPADDING", (0, 1), (-1, 1), 12),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.WHITE),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, _brand.SLATE_BORDER),
    ]
    for col, (_, _, bgc) in enumerate(kpi_cells):
        kstyle.append(("BACKGROUND", (col, 0), (col, 0), bgc))
    kpi_t.setStyle(TableStyle(kstyle))
    elements.append(kpi_t)
    elements.append(Spacer(1, 0.2 * inch))

    # Detection Summary table (authority-specific)
    elements.append(Paragraph("Detection Summary", styles["Heading_Custom"]))
    rows = _camera_detection_summary_rows(camera_id, items)
    elements.append(_pages.detection_summary_table(rows))
    elements.append(PageBreak())
    return elements


def _camera_detection_summary_rows(camera_id, items) -> List[List[str]]:
    rows = [["Metric", "Count"]]
    if camera_id == C.CAMERA_RIGHT_UP:
        open_d = sum(1 for it in items
                     for (lbl, st, _, _) in it["detections"]
                     if lbl == "Right Door" and _brand.is_side_anomaly(st))
        captured = sum(1 for it in items if it["ocr"] != "-")
        missing = sum(1 for it in items
                      if it["ocr"] == "-" and it["classification"] == C.CLASS_WAGON)
        rows += [["RIGHT DOORS OPEN/DAMAGE", str(open_d)],
                 ["OCR CAPTURED", str(captured)],
                 ["OCR MISSING", str(missing)],
                 ["VISIBLE WAGONS", str(sum(1 for it in items if it['visible']))]]
    elif camera_id == C.CAMERA_LEFT_UP:
        open_d = sum(1 for it in items
                     for (lbl, st, _, _) in it["detections"]
                     if lbl == "Left Door" and _brand.is_side_anomaly(st))
        partial = sum(1 for it in items
                      for (lbl, st, _, _) in it["detections"]
                      if lbl == "Left Door" and st == C.DOOR_PARTIAL)
        rows += [["LEFT DOORS OPEN/DAMAGE", str(open_d)],
                 ["LEFT DOORS PARTIAL", str(partial)],
                 ["VISIBLE WAGONS", str(sum(1 for it in items if it['visible']))]]
    elif camera_id == C.CAMERA_RIGHT_UP_TOP:
        loaded = sum(1 for it in items
                     for (lbl, st, _, _) in it["detections"]
                     if lbl == "Load Status" and st == C.LOAD_LOADED)
        empty = sum(1 for it in items
                    for (lbl, st, _, _) in it["detections"]
                    if lbl == "Load Status" and st == C.LOAD_EMPTY)
        damaged = sum(1 for it in items
                      if any(a[1].startswith("TOP_DAMAGE") for a in it["anomalies"]))
        rows += [["LOADED WAGONS", str(loaded)],
                 ["EMPTY WAGONS", str(empty)],
                 ["TOP DAMAGED WAGONS", str(damaged)],
                 ["VISIBLE WAGONS", str(sum(1 for it in items if it['visible']))]]
    else:  # LEFT_UP_TOP
        damaged = sum(1 for it in items
                      if any("TOP_DAMAGE" in a[1] for a in it["anomalies"]))
        rows += [["TOP DAMAGE (SUPPORT)", str(damaged)],
                 ["VISIBLE WAGONS", str(sum(1 for it in items if it['visible']))]]
    return rows


# -----------------------------------------------------------------------------
# Camera anomaly summary (grouped by severity)
# -----------------------------------------------------------------------------

def _camera_anomaly_summary(*, camera_id, items, styles):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak

    # Group: severity -> [(sr, gw, text)]
    grouped: Dict[str, List[Tuple[int, str, str]]] = {s: [] for s in _SEVERITY_ORDER}
    for it in items:
        for sev, text in it["anomalies"]:
            grouped.setdefault(sev, []).append((it["sr"], it["gw_id"], text))

    if not any(grouped.values()):
        return []

    elements: List[Any] = [PageBreak()]
    elements.append(Paragraph(
        f"{camera_id} — Anomaly Summary",
        ParagraphStyle("AnomTitle", fontSize=20, alignment=TA_CENTER,
                       textColor=_brand.NAVY_DARK, fontName="Helvetica-Bold"),
    ))
    elements.append(Spacer(1, 0.15 * inch))

    for sev in _SEVERITY_ORDER:
        entries = grouped.get(sev) or []
        if not entries:
            continue
        elements.append(Paragraph(
            f"<font color='{_SEVERITY_COLOR[sev]}'><b>{sev} SEVERITY ({len(entries)})</b></font>",
            ParagraphStyle("SevHdr", fontSize=13, alignment=TA_LEFT,
                           fontName="Helvetica-Bold"),
        ))
        table_rows = [["SR.NO", "GW ID", "ANOMALY"]]
        for sr, gw, text in sorted(entries):
            table_rows.append([str(sr), gw, text])
        t = Table(table_rows, colWidths=[1.0 * inch, 1.5 * inch, 7.0 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "LEFT"),
            ("BOX", (0, 0), (-1, -1), 0.5, _brand.SLATE_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, _brand.SLATE_BORDER),
            ("BACKGROUND", (0, 1), (-1, -1),
             colors.HexColor("#FDF5F5") if sev == "HIGH" else _brand.WHITE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.15 * inch))
    return elements


# -----------------------------------------------------------------------------
# Camera evidence pages (legacy snapshot grid for anomalous wagons)
# -----------------------------------------------------------------------------

def _camera_evidence_pages(*, camera_id, items, styles):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak,
    )

    # Collect anomalous snapshots for this camera
    blocks_data: List[Tuple[int, str, List[Tuple[str, str]]]] = []
    for it in items:
        if not it["anomalies"]:
            continue
        snaps: List[Tuple[str, str]] = []
        for (lbl, st, conf, path) in it["detections"]:
            if path and os.path.isfile(path) and (
                _brand.is_side_anomaly(st) or st == "DAMAGE"
                or (lbl.startswith("Wagon Number") and st == "MISSING")
            ):
                snaps.append((lbl, path))
        if snaps:
            blocks_data.append((it["sr"], it["gw_id"], snaps))

    if not blocks_data:
        return []

    elements: List[Any] = [PageBreak()]
    elements.append(Paragraph("<b>Camera Evidence Report</b>", styles["ReportTitle"]))
    elements.append(Spacer(1, 0.05 * inch))
    elements.append(Paragraph(
        f"<b>{camera_id} — Flagged Wagons: {len(blocks_data)}</b>",
        styles["ReportSubtitle"]))
    elements.append(Spacer(1, 0.2 * inch))

    label_style = ParagraphStyle(
        "EvLabel", fontSize=8, leading=10, alignment=1,
        textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold")
    timestamp = _now_ist().strftime("%d-%m-%Y %H:%M:%S IST")

    for sr, gw, snaps in blocks_data:
        # info row
        header_row = [
            Paragraph("<b>SR</b>", styles["TableHeader"]),
            Paragraph("<b>GW ID</b>", styles["TableHeader"]),
            Paragraph("<b>Camera</b>", styles["TableHeader"]),
            Paragraph("<b>Issues</b>", styles["TableHeader"]),
            Paragraph("<b>Date &amp; Time</b>", styles["TableHeader"]),
        ]
        data_row = [
            Paragraph(f"{sr}.", styles["TableCell"]),
            Paragraph(gw, styles["TableCell"]),
            Paragraph(f"<b>{camera_id}</b>", styles["TableCell"]),
            Paragraph(f"<b>{len(snaps)}</b>", styles["TableCell"]),
            Paragraph(timestamp, styles["TableCell"]),
        ]
        info_t = Table([header_row, data_row],
                       colWidths=[0.6*inch, 1.4*inch, 2.0*inch, 0.8*inch, 2.6*inch])
        info_t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), _brand.HEADER_GRAY),
        ]))

        cells: List[List[Any]] = []
        for lbl, path in snaps:
            try:
                img = Image(path)
                w, h = img.drawWidth, img.drawHeight
                if w > _brand.EVIDENCE_IMG_MAX_W:
                    s = _brand.EVIDENCE_IMG_MAX_W / w; w *= s; h *= s
                if h > _brand.EVIDENCE_IMG_MAX_H:
                    s = _brand.EVIDENCE_IMG_MAX_H / h; w *= s; h *= s
                img.drawWidth = w; img.drawHeight = h
                cells.append([Paragraph(f"<b>{lbl}</b>", label_style),
                              Spacer(1, 0.05 * inch), img])
            except Exception:
                continue
        if not cells:
            continue

        if len(cells) == 1:
            grid = Table([cells], colWidths=[9.6 * inch])
        else:
            grid_rows = []
            paired = len(cells) - (len(cells) % 2)
            for i in range(0, paired, 2):
                grid_rows.append([cells[i], cells[i + 1]])
            if len(cells) % 2 == 1:
                grid_rows.append([cells[-1]])
            grid = Table(grid_rows, colWidths=[4.8 * inch, 4.8 * inch])
            if len(cells) % 2 == 1:
                last = len(grid_rows) - 1
                grid.setStyle(TableStyle([("SPAN", (0, last), (1, last))]))
        grid.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        elements.append(KeepTogether([
            info_t, Spacer(1, 0.15 * inch), grid, Spacer(1, 0.3 * inch),
        ]))
    return elements


# -----------------------------------------------------------------------------
# Per-wagon detail pages for one camera
# -----------------------------------------------------------------------------

def _camera_wagon_pages(
    *, camera_id, items, cache_root, per_camera_meta, styles,
):
    """Strict global order GW_1..GW_n -- never skip or reorder.  Each wagon:
    overview quartile grid (this camera) + one detail page per camera-
    authoritative detection that has data.  Wagons not visible to this camera
    still render (the overview page shows "NOT VISIBLE TO THIS CAMERA")."""
    meta = per_camera_meta.get(camera_id, {})
    # `items` is already built in global train order by _build_camera_items
    # (enumerate over state.wagons).  Keep that order verbatim so every camera
    # report enumerates GW_1 .. GW_n; anomalies are surfaced by the per-severity
    # Anomaly Summary section, not by reordering the per-wagon pages.
    ordered = items

    out: List[Any] = []
    for it in ordered:
        gw_id = it["gw_id"]
        cls = it["classification"]

        # Non-wagon classes get a single simple page (top cameras especially)
        if camera_id in C.TOP_CAMERAS and cls in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
            kind = "engine" if cls == C.CLASS_ENGINE else "brake_van"
            out.extend(_pages.simple_state_page(
                wagon_number=None, gw_id=gw_id, kind=kind,
                classification=cls, start_time=it["start_time"],
                end_time=it["end_time"], cache_root=cache_root,
                camera_id=camera_id, local_meta=meta, styles=styles,
            ))
            continue

        # Wagon overview quartile grid for this camera
        no_text = "" if it["visible"] else "NOT VISIBLE TO THIS CAMERA"
        out.extend(_pages.wagon_overview_page(
            wagon_number=it["sr"], gw_id=gw_id, classification=cls,
            cache_root=cache_root, camera_id=camera_id,
            start_time=it["start_time"], end_time=it["end_time"],
            local_meta=meta, no_detection_text=no_text, styles=styles,
        ))

        # Detail page per detection with a snapshot OR an anomaly
        anomalous_first = sorted(
            it["detections"],
            key=lambda d: (0 if (_brand.is_side_anomaly(d[1]) or d[1] == "DAMAGE") else 1),
        )
        for (lbl, st, conf, path) in anomalous_first:
            if not path and st in (None, "", "-", C.NO_DATA):
                continue
            ocr_line = (f" | OCR: {it['ocr']}"
                        if camera_id == C.CAMERA_RIGHT_UP and it["ocr"] != "-" else "")
            header = (f"<b>Wagon No: {it['sr']} | {gw_id} | {lbl}</b>{ocr_line}")
            out.extend(_pages.detail_page(
                header_html=header, state=st, confidence=conf,
                snapshot_path=path, styles=styles,
            ))
    return out


# -----------------------------------------------------------------------------
# Single camera report
# -----------------------------------------------------------------------------

def build_camera_report(
    *, camera_id: str, state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: Optional[str], wagon_states_root: Optional[str],
    cache_root: Optional[str], per_camera_tracking_path: Optional[str],
    output_pdf: str, batch_key: str, logo_path: Optional[str] = None,
    verbose: bool = True,
) -> Optional[str]:
    try:
        import reportlab  # noqa: F401
    except Exception as e:
        print(f"[CAMERA_REPORT/{camera_id}] reportlab unavailable: {e}")
        return None

    styles = _brand.build_styles()
    per_camera_meta = ev.load_per_camera_meta(per_camera_tracking_path)
    items = _build_camera_items(
        camera_id=camera_id, state=state, unified=unified,
        evidence_root=evidence_root, wagon_states_root=wagon_states_root,
        cache_root=cache_root,
    )

    doc = _pages.make_doc(
        output_pdf, f"WagonEye {camera_id} Report -- {batch_key}", logo_path)
    elements: List[Any] = []
    elements.extend(_camera_summary_page(
        camera_id=camera_id, items=items,
        total_wagons=state.total_wagons, styles=styles))
    elements.extend(_camera_wagon_pages(
        camera_id=camera_id, items=items, cache_root=cache_root,
        per_camera_meta=per_camera_meta, styles=styles))
    elements.extend(_camera_anomaly_summary(
        camera_id=camera_id, items=items, styles=styles))
    elements.extend(_camera_evidence_pages(
        camera_id=camera_id, items=items, styles=styles))

    try:
        doc.build(elements)
    except Exception as e:
        print(f"[CAMERA_REPORT/{camera_id}] doc.build FAILED: {e}")
        traceback.print_exc(limit=3)
        return None
    if verbose:
        print(f"[CAMERA_REPORT/{camera_id}] wrote {output_pdf}")
    return output_pdf


# -----------------------------------------------------------------------------
# Build all four camera reports
# -----------------------------------------------------------------------------

def build_all(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: str,
    output_dir: str,
    batch_key: str,
    wagon_states_root: Optional[str] = None,
    cache_root: Optional[str] = None,
    per_camera_tracking_path: Optional[str] = None,
    logo_path: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Build the camera-wise PDFs.  Returns {camera_id -> path|None}.

    ``cameras`` restricts regeneration to a subset, so a late camera rebuilds
    ONLY its own PDF; the other camera reports on disk are left untouched.
    Independent failures do not block the others."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    out: Dict[str, Optional[str]] = {}

    target = set(cameras) if cameras is not None else set(C.ALL_CAMERAS)
    for camera_id in C.ALL_CAMERAS:
        if camera_id not in target:
            continue
        path = os.path.join(output_dir, CAMERA_FILE[camera_id])
        try:
            out[camera_id] = build_camera_report(
                camera_id=camera_id, state=state, unified=unified,
                evidence_root=evidence_root,
                wagon_states_root=wagon_states_root,
                cache_root=cache_root,
                per_camera_tracking_path=per_camera_tracking_path,
                output_pdf=path, batch_key=batch_key,
                logo_path=logo_path, verbose=verbose,
            )
        except Exception as e:
            print(f"[CAMERA_REPORT/{camera_id}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            out[camera_id] = None

    if verbose:
        n_ok = sum(1 for v in out.values() if v)
        print(f"[CAMERA_REPORT] done {n_ok}/{len(out)} camera PDFs  "
              f"({time.time() - t0:.1f}s)")
    return out
