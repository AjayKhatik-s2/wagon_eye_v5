"""Stage 5 -- emit `combined_train_report.json` + `combined_train_report.pdf`.

Visual identity ported from the legacy WagonEye CombinedReportGenerator
(old_system/RIGHT_UP/combined_report_generator.py).  The PDF reproduces:

    1. Navy title banner ("COMBINED WAGON EYE REPORT" + IST date/time)
    2. VIDEO EVIDENCE table (5 cols: label + 4 cameras; RAW + PROCESSED)
    3. PARTIAL REPORT amber banner when any camera feed is missing
    4. DETAILED CAMERA REPORTS table (links to the 4 camera-wise PDFs)
    5. INSPECTION SUMMARY 10-column KPI table
    6. Wagon Inspection table (7 cols: SR.NO, WAGON#, LEFT DOORS,
       RIGHT DOORS, R-TOP, L-TOP, WAGON TYPE) with legacy issue-row
       highlighting rules
    7. Damaged Wagon Report -- per-anomaly-wagon evidence sections,
       grouped by wagon number, sorted by camera priority (Left, Right,
       Left-Side, Right-Side, Left-Top, Right-Top) with the legacy
       4.2 x 2.8 inch image grid

Data sources (read-only):
    * GlobalTrainState
    * UnifiedWagonState
    * wagon_states/<feature>/<gw>.json
    * evidence/<gw>/<feature>/{*.jpg, metadata.json}
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons

from . import _brand
from . import _adapter
from . import _evidence_lookup as ev


_REPORT_SCHEMA = "wagon_eye.combined_report.v4"


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def _now_ist():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _now_ist_iso() -> str:
    return _now_ist().isoformat(timespec="seconds")


def _date_str() -> str:
    return _now_ist().strftime("%d-%m-%Y")


def _time_str() -> str:
    return _now_ist().strftime("%H:%M IST")


def _datetime_str() -> str:
    return _now_ist().strftime("%d-%m-%Y - %H:%M:%S")


# -----------------------------------------------------------------------------
# JSON
# -----------------------------------------------------------------------------

def _evidence_pages(evidence_root: Optional[str], wagons) -> Dict[str, Dict[str, str]]:
    if not evidence_root or not os.path.isdir(evidence_root):
        return {}
    candidates = {
        "door":   ["left_best.jpg", "left_crop.jpg",
                    "right_best.jpg", "right_crop.jpg"],
        "ocr":    ["best_frame.jpg", "number_crop.jpg"],
        "damage": ["track_1.jpg", "track_2.jpg", "track_3.jpg"],
        "load":   ["best_frame.jpg"],
    }
    pages: Dict[str, Dict[str, str]] = {}
    for u in wagons:
        snaps: Dict[str, str] = {}
        for feat, files in candidates.items():
            for fn in files:
                p = os.path.join(evidence_root, u.global_id, feat, fn)
                if os.path.isfile(p):
                    key = f"{feat}_{os.path.splitext(fn)[0]}"
                    snaps[key] = os.path.relpath(p, start=evidence_root).replace(os.sep, "/")
        if snaps:
            pages[u.global_id] = snaps
    return pages


def _build_json(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    evidence_root: Optional[str] = None,
    legacy_view_model: Optional[_adapter.LegacyViewModel] = None,
    report_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wagons_in_order = [unified[w.global_id] for w in state.wagons
                       if w.global_id in unified]
    summary = summarize_wagons(wagons_in_order)
    doc: Dict[str, Any] = {
        "schema":      _REPORT_SCHEMA,
        "batch_key":   batch_key,
        "generated_at": _now_ist_iso(),
        "train_metadata": {
            "master_camera":       state.master_camera,
            "master_fps":          state.master_fps,
            "master_total_frames": state.master_total_frames,
            "source_video_urls":   dict(source_video_urls or {}),
            "processed_video_urls":dict(processed_video_urls or {}),
        },
        "summary": summary,
        "stage0_fallback_used":    state.fallback_used,
        "stage0_fallback_reason":  state.fallback_reason,
        "stage0_corrections_applied": list(state.corrections_applied),
        "per_camera_local_counts": dict(state.per_camera_local_counts),
        "wagons": [u.to_dict() for u in wagons_in_order],
        "evidence_pages": _evidence_pages(evidence_root, wagons_in_order),
    }
    # Incremental-lifecycle report metadata (report_revision / status /
    # camera availability / GST provenance).  Defaults describe a complete,
    # single-shot FINAL report so legacy callers are unaffected.
    doc["report_meta"] = report_meta or {
        "report_revision": 0,
        "report_status": "FINAL",
        "cameras_present": list(state.participating_cameras or C.ALL_CAMERAS),
        "cameras_pending": [],
        "cameras_missing_final": [],
        "generated_from_global_state_version": getattr(state, "sealed_at", "") or "",
        "generated_from_global_state_hash": "",
        "fusion_revision": 0,
        "partial_reason": "",
    }
    if legacy_view_model is not None:
        doc["legacy_view_model"] = {
            "summary_kpis": legacy_view_model.summary_kpis,
            "state_counts": legacy_view_model.state_counts,
            "merged_wagons": legacy_view_model.merged_wagons,
        }
    if extra_metadata:
        doc["train_metadata"].update(extra_metadata)
    return doc


# -----------------------------------------------------------------------------
# PDF -- legacy-identity reportlab body
# -----------------------------------------------------------------------------

def _build_pdf(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    vm: _adapter.LegacyViewModel,
    batch_key: str,
    output_pdf: str,
    evidence_root: Optional[str],
    source_video_urls: Dict[str, str],
    processed_video_urls: Dict[str, str],
    camera_pdf_urls: Dict[str, str],
    logo_path: Optional[str],
    missing_cameras: Sequence[str],
    cache_root: Optional[str] = None,
) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)

    page_w, page_h = landscape(A4)
    L = 0.5 * inch
    doc = BaseDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=L, rightMargin=L,
        topMargin=L,  bottomMargin=L,
        title=f"WagonEye Combined Report -- {batch_key}",
        author="WagonEye",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="content")
    on_page = _brand.make_logo_callback(logo_path)
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=on_page)
    ])

    styles = _brand.build_styles()
    elements: List[Any] = []

    # ----- 1. TITLE BANNER -----
    elements.append(Spacer(1, 0.25 * inch))
    banner_data = [[
        Paragraph("COMBINED WAGON EYE REPORT", styles["BannerTitle"])
    ], [
        Paragraph(f"{_date_str()}  |  {_time_str()}", styles["BannerDate"])
    ]]
    banner = Table(banner_data, colWidths=[_brand.PAGE_BODY_WIDTH])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _brand.NAVY_DARK),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (0, 0),   14),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING",    (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 12),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("BOX", (0, 0), (-1, -1), 1.5, _brand.NAVY_DARK),
    ]))
    elements.append(banner)
    elements.append(Spacer(1, 0.18 * inch))

    # ----- 2. VIDEO EVIDENCE -----
    cams = list(C.ALL_CAMERAS)
    label_w = 1.4 * inch
    cam_w   = 2.15 * inch

    def _cam_link(url: Optional[str], cam: str):
        return _brand.make_camera_link(
            url, "Click to View", cam, missing_cameras, styles,
        )

    raw_cells = [_cam_link(source_video_urls.get(cam), cam) for cam in cams]
    proc_cells = [_cam_link(processed_video_urls.get(cam), cam) for cam in cams]

    video_data = [
        [Paragraph("<b>VIDEO EVIDENCE</b>", styles["SectionTitleWhite"]),
         "", "", "", ""],
        [Paragraph("", styles["CameraLabel"])] + [
            Paragraph(f"<b>{cam}</b>", styles["CameraLabel"]) for cam in cams
        ],
        [Paragraph("<b>Raw Video</b>", styles["CameraLabel"])] + raw_cells,
        [Paragraph("<b>Processed Video</b>", styles["CameraLabel"])] + proc_cells,
    ]
    video_t = Table(video_data, colWidths=[label_w, cam_w, cam_w, cam_w, cam_w])
    video_style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_MID),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.SLATE_LIGHT),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
        ("BACKGROUND", (0, 2), (0, 2), _brand.SLATE_LIGHT),
        ("BACKGROUND", (1, 2), (-1, 2), _brand.WHITE),
        ("TOPPADDING", (0, 2), (-1, 2), 8),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 8),
        ("BACKGROUND", (0, 3), (0, 3), _brand.SLATE_LIGHT),
        ("BACKGROUND", (1, 3), (-1, 3), _brand.SLATE_BG),
        ("TOPPADDING", (0, 3), (-1, 3), 8),
        ("BOTTOMPADDING", (0, 3), (-1, 3), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, cam in enumerate(cams):
        if cam in missing_cameras:
            video_style.append(("BACKGROUND", (i + 1, 1), (i + 1, -1), _brand.NA_BG))
    video_t.setStyle(TableStyle(video_style))
    elements.append(video_t)
    elements.append(Spacer(1, 0.14 * inch))

    # ----- 3. PARTIAL REPORT WARNING -----
    warn = _brand.make_warning_banner(missing_cameras, styles)
    if warn is not None:
        elements.append(warn)
        elements.append(Spacer(1, 0.12 * inch))

    # ----- 4. DETAILED CAMERA REPORTS (legacy per-camera report links) -----
    camera_links_order = [
        (C.CAMERA_LEFT_UP,      "LEFT Detail Report"),
        (C.CAMERA_RIGHT_UP,     "RIGHT Detail Report"),
        (C.CAMERA_RIGHT_UP_TOP, "R-TOP Detail Report"),
        (C.CAMERA_LEFT_UP_TOP,  "L-TOP Detail Report"),
    ]
    feat_cells = []
    for cam, label in camera_links_order:
        url = (camera_pdf_urls or {}).get(cam)
        if cam in missing_cameras:
            feat_cells.append(Paragraph(
                '<font color="#C62828"><i>NO FEED</i></font>',
                styles["NoFeedCell"],
            ))
        elif url:
            feat_cells.append(Paragraph(
                f'<a href="{url}" color="#1565C0"><b><u>{label}</u></b></a>',
                styles["LinkCellPro"],
            ))
        else:
            feat_cells.append(Paragraph(
                f'<font color="#78909C">{label}</font>',
                styles["LinkCell"],
            ))

    report_data = [
        [Paragraph("<b>DETAILED CAMERA REPORTS</b>", styles["SectionTitleWhite"]),
         "", "", ""],
        [Paragraph(f"<b>{C.CAMERA_LEFT_UP}</b>",      styles["CameraLabel"]),
         Paragraph(f"<b>{C.CAMERA_RIGHT_UP}</b>",     styles["CameraLabel"]),
         Paragraph(f"<b>{C.CAMERA_RIGHT_UP_TOP}</b>", styles["CameraLabel"]),
         Paragraph(f"<b>{C.CAMERA_LEFT_UP_TOP}</b>",  styles["CameraLabel"])],
        feat_cells,
    ]
    report_t = Table(report_data, colWidths=[2.5 * inch] * 4)
    report_t.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.TEAL_ACCENT),
        ("ALIGN",  (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.SLATE_LIGHT),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
        ("BACKGROUND", (0, 2), (-1, 2), _brand.WHITE),
        ("TOPPADDING", (0, 2), (-1, 2), 9),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(report_t)
    elements.append(Spacer(1, 0.18 * inch))

    # ----- 5. INSPECTION SUMMARY -----
    kpi = vm.summary_kpis
    rake = kpi["rake_type"]
    status = kpi["status"]
    loco = " / ".join(kpi["loco_numbers"]) if kpi["loco_numbers"] else "Not Detected"

    def _na_if_missing(camera_id, val):
        return "N/A" if camera_id in missing_cameras else str(val)

    status_color = "#C62828" if status == "NOT OK" else "#2E7D32"
    rake_color = (
        "#1565C0" if rake == "LOADED RAKE"
        else ("#E65100" if rake == "EMPTY RAKE" else "#1A1A2E")
    )

    from reportlab.lib.styles import ParagraphStyle
    header_p = ParagraphStyle(
        "SummaryHeader", fontSize=8, alignment=1, textColor=_brand.WHITE,
        fontName="Helvetica-Bold", leading=11,
    )
    data_p = ParagraphStyle(
        "SummaryData", fontSize=9, alignment=1, textColor=_brand.TEXT_DARK,
        fontName="Helvetica", leading=12,
    )
    data_b = ParagraphStyle(
        "SummaryDataBold", fontSize=9, alignment=1, textColor=_brand.TEXT_DARK,
        fontName="Helvetica-Bold", leading=12,
    )

    title_row = [Paragraph("<b>INSPECTION SUMMARY</b>", styles["SectionTitleWhite"]),
                 "", "", "", "", "", "", "", "", ""]
    header_row = [
        Paragraph("DATE-TIME",       header_p),
        Paragraph("LOCO NUMBER",     header_p),
        Paragraph("TOTAL<br/>WAGONS", header_p),
        Paragraph("LEFT OPEN<br/>DOORS",  header_p),
        Paragraph("RIGHT OPEN<br/>DOORS", header_p),
        Paragraph("R-TOP<br/>DAMAGES",    header_p),
        Paragraph("L-TOP<br/>DAMAGES",    header_p),
        Paragraph("PARTIAL<br/>CLOSED",   header_p),
        Paragraph("RAKE<br/>TYPE",        header_p),
        Paragraph("STATUS",          header_p),
    ]
    partial_text = (
        f"L {_na_if_missing(C.CAMERA_LEFT_UP,  kpi['left_partial'])} / "
        f"R {_na_if_missing(C.CAMERA_RIGHT_UP, kpi['right_partial'])}"
    )
    data_row = [
        Paragraph(_datetime_str(), data_p),
        Paragraph(f"<b>{loco}</b>", data_b),
        Paragraph(f"<b>{kpi['total_wagons']}</b>", data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_LEFT_UP,  kpi['left_open'])}</b>",  data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_RIGHT_UP, kpi['right_open'])}</b>", data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_RIGHT_UP_TOP, kpi['top_damages'])}</b>", data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_LEFT_UP_TOP,  kpi['left_top_damages'])}</b>", data_b),
        Paragraph(partial_text, data_p),
        Paragraph(f'<b><font color="{rake_color}">{rake}</font></b>', data_b),
        Paragraph(f'<b><font color="{status_color}">{status}</font></b>', data_b),
    ]
    summary_data = [title_row, header_row, data_row]
    col_w = [1.2*inch, 1.1*inch, 0.7*inch, 0.8*inch, 0.8*inch,
             0.8*inch, 0.8*inch, 0.9*inch, 1.0*inch, 1.0*inch]
    summary_t = Table(summary_data, colWidths=col_w)

    status_bg = colors.HexColor("#FFEBEE") if status == "NOT OK" else colors.HexColor("#E8F5E9")
    rake_bg = (
        _brand.LOADED_BG if rake == "LOADED RAKE"
        else (_brand.EMPTY_BG if rake == "EMPTY RAKE" else _brand.WHITE)
    )
    summary_style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_MID),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.NAVY_DARK),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("BACKGROUND", (0, 2), (-1, 2), _brand.WHITE),
        ("TOPPADDING", (0, 2), (-1, 2), 10),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 10),
        ("BACKGROUND", (-1, 2), (-1, 2), status_bg),
        ("BACKGROUND", (8, 2), (8, 2), rake_bg),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 1, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    # Grey out missing-camera KPI cells
    cam_to_col = {
        C.CAMERA_LEFT_UP: 3, C.CAMERA_RIGHT_UP: 4,
        C.CAMERA_RIGHT_UP_TOP: 5, C.CAMERA_LEFT_UP_TOP: 6,
    }
    for cam, col in cam_to_col.items():
        if cam in missing_cameras:
            summary_style.append(("BACKGROUND", (col, 2), (col, 2), _brand.NA_BG))
    summary_t.setStyle(TableStyle(summary_style))
    elements.append(summary_t)
    elements.append(Spacer(1, 0.18 * inch))

    # ----- 6. WAGON INSPECTION TABLE -----
    _wagon_table = _build_wagon_table(vm, styles, missing_cameras)
    elements.append(_wagon_table)

    # ----- 6b. MULTI-ANGLE WAGON EVIDENCE (wagon-centric, all 4 cameras) -----
    multi_angle = _build_multi_angle_section(
        state=state, unified=unified,
        evidence_root=evidence_root, cache_root=cache_root,
        styles=styles, missing_cameras=missing_cameras,
    )
    if multi_angle:
        elements.extend(multi_angle)

    # ----- 7. DAMAGED WAGON REPORT (evidence pages) -----
    evidence_blocks = _build_evidence_section(
        vm=vm, styles=styles, evidence_root=evidence_root,
    )
    if evidence_blocks:
        elements.append(PageBreak())
        elements.extend(evidence_blocks)

    doc.build(elements)
    return output_pdf


# -----------------------------------------------------------------------------
# Wagon table -- legacy 7-column with issue-row highlighting (legacy 797-1015)
# -----------------------------------------------------------------------------

def _build_wagon_table(vm: _adapter.LegacyViewModel, styles, missing_cameras):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Table, TableStyle

    NO_FEED_TEXT = "⚠ NO FEED"
    # Column layout: SR(0) WAGON#(1) CLASS(2) LEFT(3) RIGHT(4) R-TOP(5) L-TOP(6) TYPE(7)
    cam_col = {
        C.CAMERA_LEFT_UP: 3, C.CAMERA_RIGHT_UP: 4,
        C.CAMERA_RIGHT_UP_TOP: 5, C.CAMERA_LEFT_UP_TOP: 6,
    }
    missing_cols = {cam_col[c] for c in missing_cameras if c in cam_col}

    col_header_p = ParagraphStyle(
        "WagonColHeader", fontSize=8, alignment=1, textColor=_brand.WHITE,
        fontName="Helvetica-Bold", leading=11,
    )
    cell = ParagraphStyle(
        "WagonCell", fontSize=8, alignment=1, textColor=_brand.TEXT_BODY,
        fontName="Helvetica", leading=11,
    )
    cell_b = ParagraphStyle(
        "WagonCellBold", fontSize=8, alignment=1, textColor=_brand.TEXT_DARK,
        fontName="Helvetica-Bold", leading=11,
    )
    issue = ParagraphStyle(
        "WagonIssue", fontSize=8, alignment=1, textColor=_brand.COLOR_NOT_OK,
        fontName="Helvetica-Bold", leading=11,
    )
    nofeed = ParagraphStyle(
        "WagonNoFeed", fontSize=7, alignment=1, textColor=_brand.TEXT_LIGHT,
        fontName="Helvetica-Oblique", leading=10,
    )

    title_row = [Paragraph("<b>WAGON INSPECTION DETAILS</b>",
                            styles["SectionTitleWhite"]),
                 "", "", "", "", "", "", ""]
    header_row = [
        Paragraph("SR.NO",               col_header_p),
        Paragraph("WAGON NUMBER",        col_header_p),
        Paragraph("CLASS",               col_header_p),
        Paragraph("LEFT CAMERA<br/>DOORS",  col_header_p),
        Paragraph("RIGHT CAMERA<br/>DOORS", col_header_p),
        Paragraph("R-TOP<br/>DAMAGES",   col_header_p),
        Paragraph("L-TOP<br/>DAMAGES",   col_header_p),
        Paragraph("WAGON<br/>TYPE",      col_header_p),
    ]
    rows = [title_row, header_row]
    highlight_info = []

    for wagon in vm.merged_wagons:
        row_idx = len(rows)
        sr = str(wagon["wagon_sr_no"])
        wn = wagon.get("ocr_wagon_number") or "-"
        wn_disp = wn if wn != "-" else "-"

        has_l = wagon.get("has_open_left")     and 2 not in missing_cols
        has_r = wagon.get("has_open_right")    and 3 not in missing_cols
        has_t = wagon.get("has_open_top")      and 4 not in missing_cols
        has_lt = wagon.get("has_open_left_top") and 5 not in missing_cols

        l_text = NO_FEED_TEXT if 2 in missing_cols else wagon.get("left_doors_text",  "NO DATA")
        r_text = NO_FEED_TEXT if 3 in missing_cols else wagon.get("right_doors_text", "NO DATA")
        t_text = NO_FEED_TEXT if 4 in missing_cols else wagon.get("top_doors_text",   "NO DATA")
        lt_text = NO_FEED_TEXT if 5 in missing_cols else wagon.get("left_top_doors_text", "NO DATA")

        wt_text = wagon.get("wagon_type", "-")
        if wt_text == "LOADED":
            wt_style = ParagraphStyle("WagonLoaded", parent=cell,
                                       textColor=_brand.COLOR_LOADED,
                                       fontName="Helvetica-Bold")
        elif wt_text == "EMPTY":
            wt_style = ParagraphStyle("WagonEmpty", parent=cell,
                                       textColor=_brand.COLOR_EMPTY,
                                       fontName="Helvetica-Bold")
        else:
            wt_style = cell

        # Global wagon class (authoritative from GlobalTrainState).  Never
        # defaults to "WAGON": an unclassified wagon shows UNKNOWN.
        cls_raw = str(wagon.get("classification") or C.CLASS_UNKNOWN)
        cls_disp = cls_raw.replace("_", " ")
        if cls_raw == C.CLASS_ENGINE:
            cls_style = ParagraphStyle("WagonClsEng", parent=cell_b,
                                       textColor=_brand.COLOR_EMPTY)
        elif cls_raw == C.CLASS_BRAKE_VAN:
            cls_style = ParagraphStyle("WagonClsBv", parent=cell_b,
                                       textColor=_brand.COLOR_LOADED)
        else:
            cls_style = cell_b

        l_s = issue if has_l else (nofeed if 3 in missing_cols else cell)
        r_s = issue if has_r else (nofeed if 4 in missing_cols else cell)
        t_s = issue if has_t else (nofeed if 5 in missing_cols else cell)
        lt_s = issue if has_lt else (nofeed if 6 in missing_cols else cell)

        rows.append([
            Paragraph(f"<b>{sr}</b>", cell_b),
            Paragraph(f"<b>{wn_disp}</b>", cell_b) if wn_disp != "-" else Paragraph(wn_disp, cell),
            Paragraph(cls_disp, cls_style),
            Paragraph(l_text, l_s),
            Paragraph(r_text, r_s),
            Paragraph(t_text, t_s),
            Paragraph(lt_text, lt_s),
            Paragraph(wt_text, wt_style),
        ])

        issue_cols = []
        if has_l:  issue_cols.append(3)
        if has_r:  issue_cols.append(4)
        if has_t:  issue_cols.append(5)
        if has_lt: issue_cols.append(6)
        if issue_cols:
            highlight_info.append((row_idx, issue_cols))

    t = Table(
        rows,
        colWidths=[0.5*inch, 1.3*inch, 1.0*inch, 1.9*inch, 1.9*inch,
                   0.9*inch, 0.9*inch, 0.9*inch],
        repeatRows=2,
    )
    style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_MID),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.NAVY_DARK),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 1, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 2), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 2), (-1, -1), 8),
    ]
    # Alternating row backgrounds
    n_data = len(vm.merged_wagons)
    for i in range(n_data):
        row_idx = i + 2
        bg = _brand.WHITE if i % 2 == 0 else _brand.SLATE_BG
        style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))
    # Grey columns for missing cameras
    for col in missing_cols:
        if n_data > 0:
            style.append(("BACKGROUND", (col, 2), (col, n_data + 1), _brand.NA_BG))
    # Issue-row highlighting (>=2 cameras whole row; 1 camera SR+WN+col)
    for row_idx, issue_cols in highlight_info:
        if len(issue_cols) >= 2:
            style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), _brand.ISSUE_BG))
        else:
            style.append(("BACKGROUND", (0, row_idx), (2, row_idx), _brand.ISSUE_BG))
            for col in issue_cols:
                style.append(("BACKGROUND", (col, row_idx), (col, row_idx), _brand.ISSUE_BG))
    t.setStyle(TableStyle(style))
    return t


# -----------------------------------------------------------------------------
# Multi-Angle Wagon Evidence  --  one page per anomalous wagon showing all four
# camera perspectives of the SAME global wagon (presentation enhancement).
# -----------------------------------------------------------------------------

# Grid order: top row RIGHT_UP | LEFT_UP ; bottom row RIGHT_UP_TOP | LEFT_UP_TOP
_MULTI_ANGLE_GRID = (
    (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP),
    (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP),
)


def _cache_mid_frame(cache_root: Optional[str], gw_id: str, camera_id: str) -> Optional[str]:
    """A representative wagon_cache frame (the temporal middle) for one camera,
    so we can still show the wagon from that angle even with no detection."""
    if not cache_root:
        return None
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    d = os.path.join(cache_root, gw_id, folder)
    if not os.path.isdir(d):
        return None
    try:
        jpgs = sorted(fn for fn in os.listdir(d) if fn.endswith(".jpg"))
    except OSError:
        return None
    if not jpgs:
        return None
    return os.path.join(d, jpgs[len(jpgs) // 2])


def _damage_track_cams(evidence_root: Optional[str], gw_id: str) -> set:
    """Set of top-camera ids that actually produced a damage track for a wagon."""
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    cams = set()
    for tr in (md.get("tracks") or []):
        if isinstance(tr, dict) and tr.get("camera_id"):
            cams.add(tr["camera_id"])
    return cams


def _top_damage_snapshot(evidence_root: Optional[str], gw_id: str, camera_id: str) -> Optional[str]:
    """Best damage-track snapshot for ONE top camera (highest best_confidence)."""
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    best = None
    best_conf = -1.0
    for tr in (md.get("tracks") or []):
        if not isinstance(tr, dict) or tr.get("camera_id") != camera_id:
            continue
        idx = tr.get("track_idx")
        if idx is None:
            continue
        conf = float(tr.get("best_confidence") or 0.0)
        if conf > best_conf:
            snap = ev.evidence_snapshot(evidence_root, gw_id, "damage", f"track_{int(idx)}")
            if snap:
                best, best_conf = snap, conf
    return best


def _wagon_is_anomalous(u: UnifiedWagonState) -> bool:
    """True when a wagon has any reportable anomaly: open/damaged door, top or
    side damage, OCR missing on a WAGON, or a load NO_DATA on a WAGON."""
    if u is None:
        return False
    if _brand.is_side_anomaly(u.left_door) or _brand.is_side_anomaly(u.right_door):
        return True
    if u.top_damage == C.DAMAGE_PRESENT or u.side_damage == C.DAMAGE_PRESENT:
        return True
    if u.classification == C.CLASS_WAGON:
        if u.wagon_identifier in (None, "", C.NO_DATA):
            return True
        if u.load_status == C.NO_DATA:
            return True
    return False


def _authoritative_cams(u: UnifiedWagonState, damage_cams: set) -> set:
    """Cameras whose authority actually detected one of the wagon's anomalies."""
    cams = set()
    if _brand.is_side_anomaly(u.right_door):
        cams.add(C.CAMERA_RIGHT_UP)
    if _brand.is_side_anomaly(u.left_door):
        cams.add(C.CAMERA_LEFT_UP)
    if u.classification == C.CLASS_WAGON and u.wagon_identifier in (None, "", C.NO_DATA):
        cams.add(C.CAMERA_RIGHT_UP)   # OCR authority
    if u.top_damage == C.DAMAGE_PRESENT:
        cams |= (damage_cams or {C.CAMERA_RIGHT_UP_TOP})
    if u.classification == C.CLASS_WAGON and u.load_status == C.NO_DATA:
        cams.add(C.CAMERA_RIGHT_UP_TOP)  # load authority
    return cams


def _panel_state_text(u: UnifiedWagonState, camera_id: str, damage_cams: set,
                      has_frames: bool) -> str:
    """The per-camera 'detected state' label shown under each panel."""
    if camera_id == C.CAMERA_RIGHT_UP:
        door = (_brand.format_door_status(u.right_door)
                if u.right_door not in (None, "", C.NO_DATA) else "NO DATA")
        s = f"Right Door: {door}"
        if u.classification == C.CLASS_WAGON and u.wagon_identifier in (None, "", C.NO_DATA):
            s += "  |  OCR: MISSING"
        elif u.wagon_identifier not in (None, "", C.NO_DATA):
            s += f"  |  OCR: {u.wagon_identifier}"
        return s
    if camera_id == C.CAMERA_LEFT_UP:
        door = (_brand.format_door_status(u.left_door)
                if u.left_door not in (None, "", C.NO_DATA) else "NO DATA")
        return f"Left Door: {door}"
    if camera_id == C.CAMERA_RIGHT_UP_TOP:
        dmg = "DAMAGE" if C.CAMERA_RIGHT_UP_TOP in damage_cams else ("OK" if has_frames else "NO DATA")
        load = u.load_status if u.load_status not in (None, "") else C.NO_DATA
        return f"Top Damage: {dmg}  |  Load: {load}"
    # LEFT_UP_TOP
    dmg = "DAMAGE" if C.CAMERA_LEFT_UP_TOP in damage_cams else ("OK" if has_frames else "NO DATA")
    return f"Top Damage (support): {dmg}"


def _best_damage_snapshot_any(evidence_root: Optional[str], gw_id: str) -> Optional[str]:
    """Highest-confidence damage-track snapshot across BOTH top cameras."""
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    best = None
    best_conf = -1.0
    for tr in (md.get("tracks") or []):
        if not isinstance(tr, dict):
            continue
        idx = tr.get("track_idx")
        if idx is None:
            continue
        conf = float(tr.get("best_confidence") or 0.0)
        if conf > best_conf:
            snap = ev.evidence_snapshot(evidence_root, gw_id, "damage", f"track_{int(idx)}")
            if snap:
                best, best_conf = snap, conf
    return best


def _panel_snapshot(u: UnifiedWagonState, camera_id: str,
                    evidence_root: Optional[str], cache_root: Optional[str],
                    gw_id: str) -> Optional[str]:
    """Resolve a snapshot for one camera: feature evidence first, else a
    representative wagon_cache frame (so all four angles show the wagon).

    ITEM 7: for TOP cameras, when the wagon is damaged we ALWAYS prefer a boxed
    damage snapshot (own-camera first, then the best across BOTH top cameras) so
    a detected anomaly is never shown as a clean frame in the multi-angle grid.
    """
    snap = None
    if camera_id == C.CAMERA_RIGHT_UP:
        snap = ev.evidence_snapshot(evidence_root, gw_id, "door", "right_best")
    elif camera_id == C.CAMERA_LEFT_UP:
        snap = ev.evidence_snapshot(evidence_root, gw_id, "door", "left_best")
    elif camera_id == C.CAMERA_RIGHT_UP_TOP:
        snap = _top_damage_snapshot(evidence_root, gw_id, C.CAMERA_RIGHT_UP_TOP)
        if snap is None and u.top_damage == C.DAMAGE_PRESENT:
            snap = _best_damage_snapshot_any(evidence_root, gw_id)
        if snap is None:
            snap = ev.evidence_snapshot(evidence_root, gw_id, "load", "best_frame")
    elif camera_id == C.CAMERA_LEFT_UP_TOP:
        snap = _top_damage_snapshot(evidence_root, gw_id, C.CAMERA_LEFT_UP_TOP)
        if snap is None and u.top_damage == C.DAMAGE_PRESENT:
            snap = _best_damage_snapshot_any(evidence_root, gw_id)
    if snap and os.path.isfile(snap):
        return snap
    return _cache_mid_frame(cache_root, gw_id, camera_id)


def _make_camera_panel(*, camera_id, state_text, snap_path, authoritative,
                       missing, has_frames):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Table, TableStyle, Image

    border = colors.HexColor("#C62828") if authoritative else colors.HexColor("#9E9E9E")
    hdr_bg = colors.HexColor("#C62828") if authoritative else _brand.NAVY_MID
    cam_p = ParagraphStyle("PanelCam", fontSize=9, alignment=1, textColor=colors.white,
                           fontName="Helvetica-Bold", leading=12)
    st_p = ParagraphStyle("PanelState", fontSize=8, alignment=1,
                          textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold", leading=11)
    tag = "  ● DETECTED HERE" if authoritative else ""
    header = Paragraph(f"{camera_id}{tag}", cam_p)

    def _placeholder(text, color):
        p = ParagraphStyle("PH", fontSize=13, alignment=1, textColor=color,
                           fontName="Helvetica-Bold")
        t = Table([[Paragraph(text, p)]], colWidths=[4.3 * inch], rowHeights=[2.4 * inch])
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
            ("BOX", (0, 0), (-1, -1), 1, colors.gray),
        ]))
        return t

    body = None
    if missing:
        body = _placeholder("NO FEED", colors.HexColor("#C62828"))
    elif snap_path and os.path.isfile(snap_path):
        try:
            img = Image(snap_path, width=4.3 * inch, height=2.4 * inch)
            img.hAlign = "CENTER"
            body = img
        except Exception:
            body = None
    if body is None and not missing:
        body = _placeholder("NOT VISIBLE", colors.gray)

    inner = Table([[header], [Paragraph(state_text, st_p)], [body]],
                  colWidths=[4.6 * inch])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), hdr_bg),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 2, border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (0, 0), 5),
        ("BOTTOMPADDING", (0, 0), (0, 0), 5),
    ]))
    return inner


def _wagon_anomaly_list(u: UnifiedWagonState) -> List[str]:
    """Concise list of the ACTUAL anomalies on a wagon (no normal states).

    Mirrors the anomaly definition in _wagon_is_anomalous + the per-camera
    severity rules in camera_reports so all surfaces agree.  Returns [] for a
    clean wagon.
    """
    out: List[str] = []
    if _brand.is_side_anomaly(u.right_door):
        out.append(f"RIGHT DOOR {_brand.format_door_status(u.right_door)}")
    elif u.right_door == C.DOOR_PARTIAL:
        out.append("RIGHT DOOR PARTIAL CLOSED")
    if _brand.is_side_anomaly(u.left_door):
        out.append(f"LEFT DOOR {_brand.format_door_status(u.left_door)}")
    elif u.left_door == C.DOOR_PARTIAL:
        out.append("LEFT DOOR PARTIAL CLOSED")
    if u.top_damage == C.DAMAGE_PRESENT:
        out.append("TOP DAMAGE")
    if u.side_damage == C.DAMAGE_PRESENT:
        out.append("SIDE DAMAGE")
    if u.classification == C.CLASS_WAGON:
        if u.wagon_identifier in (None, "", C.NO_DATA):
            out.append("OCR MISSING")
        if u.load_status == C.NO_DATA:
            out.append("LOAD NO DATA")
    return out


def _issue_summary_table(u: UnifiedWagonState, styles):
    """Concise issue strip: a single 'NO ISSUES' chip for a clean wagon, or one
    red chip listing only the ACTUAL anomalies.  Normal CLOSED/OK/EMPTY states
    are never printed."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    hdr_p = ParagraphStyle("IssHdr", fontSize=9, alignment=1, textColor=_brand.WHITE,
                           fontName="Helvetica-Bold", leading=12)
    ok_p = ParagraphStyle("IssOk", fontSize=10, alignment=1,
                          textColor=_brand.COLOR_OK_GREEN,
                          fontName="Helvetica-Bold", leading=13)
    bad_p = ParagraphStyle("IssBad", fontSize=10, alignment=1,
                           textColor=_brand.COLOR_NOT_OK,
                           fontName="Helvetica-Bold", leading=13)

    issues = _wagon_anomaly_list(u)
    if not issues:
        value_row = [Paragraph("NO ISSUES", ok_p)]
        anom = False
    else:
        value_row = [Paragraph("  |  ".join(issues), bad_p)]
        anom = True

    t = Table([[Paragraph("DETECTED ISSUES", hdr_p)], value_row],
              colWidths=[_brand.PAGE_BODY_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_DARK),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 1), (-1, 1),
         _brand.ISSUE_BG if anom else _brand.OK_BG),
    ]))
    return t


def _build_multi_angle_section(*, state, unified, evidence_root, cache_root,
                               styles, missing_cameras):
    """One dedicated page per anomalous wagon: header + issue summary + a 2x2
    grid of all four camera perspectives of that SAME global wagon.  Always
    shows all four views (representative wagon_cache frame when a camera has no
    detection); panels whose camera authoritatively detected the anomaly are
    emphasised with a red header + border."""
    from reportlab.lib.units import inch
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak,
    )

    anomalous = []
    for idx, gw in enumerate(state.wagons, start=1):
        u = unified.get(gw.global_id)
        if _wagon_is_anomalous(u):
            anomalous.append((idx, gw, u))
    if not anomalous:
        return []

    missing_set = set(missing_cameras or [])
    elements = [PageBreak()]
    elements.append(Paragraph(
        "<b>Multi-Angle Wagon Evidence</b>", styles["ReportTitle"]))
    elements.append(Spacer(1, 0.05 * inch))
    elements.append(Paragraph(
        f"<b>Anomalous Wagons: {len(anomalous)} &mdash; all four camera "
        f"perspectives per wagon</b>", styles["ReportSubtitle"]))
    elements.append(Spacer(1, 0.15 * inch))

    hdr_style = ParagraphStyle("MAHeader", fontSize=15, alignment=TA_CENTER,
                               textColor=_brand.WHITE, fontName="Helvetica-Bold",
                               leading=19)

    for sr, gw, u in anomalous:
        damage_cams = _damage_track_cams(evidence_root, gw.global_id)
        auth = _authoritative_cams(u, damage_cams)

        # 1. Wagon header banner
        banner = Table([[Paragraph(
            f"Wagon No: {sr}  |  {gw.global_id}  |  {u.classification}",
            hdr_style)]], colWidths=[_brand.PAGE_BODY_WIDTH])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _brand.NAVY_DARK),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 1.5, _brand.NAVY_DARK),
        ]))

        # 2. Issue summary
        issue_t = _issue_summary_table(u, styles)

        # 3. 2x2 multi-angle grid
        grid_rows = []
        for cam_row in _MULTI_ANGLE_GRID:
            cells = []
            for cam in cam_row:
                has_frames = _cache_mid_frame(cache_root, gw.global_id, cam) is not None
                cells.append(_make_camera_panel(
                    camera_id=cam,
                    state_text=_panel_state_text(u, cam, damage_cams, has_frames),
                    snap_path=_panel_snapshot(u, cam, evidence_root, cache_root, gw.global_id),
                    authoritative=(cam in auth),
                    missing=(cam in missing_set),
                    has_frames=has_frames,
                ))
            grid_rows.append(cells)
        grid = Table(grid_rows, colWidths=[4.9 * inch, 4.9 * inch])
        grid.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))

        elements.append(KeepTogether([
            banner, Spacer(1, 0.08 * inch),
            issue_t, Spacer(1, 0.12 * inch),
            grid, PageBreak(),
        ]))
    return elements


# -----------------------------------------------------------------------------
# Damaged Wagon Report -- per-anomaly evidence pages (legacy 1017-1365)
# -----------------------------------------------------------------------------

def _build_evidence_section(*, vm, styles, evidence_root):
    """Return a list of reportlab elements implementing the per-anomaly
    "Damaged Wagon Report" section.  Skips entirely when no snapshots
    are available."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
    )

    # Gather all anomalous images
    images: List[Dict[str, Any]] = []
    for d in vm.left_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Left",
                "issue_type": "Damage" if "damage" in str(d.get("state","")).lower() else "Open Door",
                "label": d.get("state", ""),
            })
    for d in vm.right_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Right",
                "issue_type": "Damage" if "damage" in str(d.get("state","")).lower() else "Open Door",
                "label": d.get("state", ""),
            })
    for d in vm.top_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Right-Top",
                "issue_type": "Damage",
                "label": d.get("state", ""),
            })
    for d in vm.left_top_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Left-Top",
                "issue_type": "Damage",
                "label": d.get("state", ""),
            })

    if not images:
        return []

    images.sort(key=lambda x: (
        int(x.get("wagon_number") or 999999),
        _brand.CAMERA_PRIORITY.get(x.get("camera", ""), 99),
    ))

    # Group by wagon_number preserving order
    by_wagon: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()
    for img in images:
        by_wagon.setdefault(int(img["wagon_number"]), []).append(img)

    elements: List[Any] = []
    elements.append(Paragraph(
        "<b>Damaged Wagon Report</b>",
        styles["ReportTitle"],
    ))
    elements.append(Spacer(1, 0.05 * inch))

    total_damaged = len(by_wagon)
    elements.append(Paragraph(
        f"<b>Total Damaged Wagons: {total_damaged}</b>",
        styles["ReportSubtitle"],
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Wagon# -> ocr lookup for the info table
    wagon_lookup = {w["wagon_sr_no"]: w for w in vm.merged_wagons}

    label_style = ParagraphStyle(
        "SnapLabel", fontSize=8, leading=10, alignment=1,
        textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold",
    )

    timestamp = _now_ist().strftime("%d-%m-%Y %H:%M:%S IST")
    sn = 0
    for wagon_num, imgs in by_wagon.items():
        sn += 1
        wagon_info = wagon_lookup.get(wagon_num, {})
        ocr = wagon_info.get("ocr_wagon_number", "-") or "-"
        camera_angles = ", ".join(sorted(set(i["camera"] for i in imgs)))

        # Info table
        header_row = [
            Paragraph("<b>SN</b>",           styles["TableHeader"]),
            Paragraph("<b>Wagon ID</b>",     styles["TableHeader"]),
            Paragraph("<b>Wagon No.</b>",    styles["TableHeader"]),
            Paragraph("<b>Camera Angles</b>", styles["TableHeader"]),
            Paragraph("<b>Issues</b>",       styles["TableHeader"]),
            Paragraph("<b>Date &amp; Time</b>", styles["TableHeader"]),
        ]
        data_row = [
            Paragraph(f"{sn}.",                  styles["TableCell"]),
            Paragraph(str(wagon_num),            styles["TableCell"]),
            Paragraph(str(ocr),                  styles["TableCell"]),
            Paragraph(f"<b>{camera_angles}</b>", styles["TableCell"]),
            Paragraph(f"<b>{len(imgs)}</b>",     styles["TableCell"]),
            Paragraph(timestamp,                 styles["TableCell"]),
        ]
        info_t = Table([header_row, data_row],
                       colWidths=[0.5*inch, 0.8*inch, 2.2*inch,
                                  1.8*inch, 0.7*inch, 2.6*inch])
        info_t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), _brand.HEADER_GRAY),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))

        # Image cells
        cells: List[List[Any]] = []
        for img_info in imgs:
            try:
                display_label = _brand.CAMERA_LABELS.get(
                    img_info["camera"],
                    f"{img_info['camera']} Camera – {img_info['issue_type']}",
                )
                img = Image(img_info["path"])
                w, h = img.drawWidth, img.drawHeight
                if w > _brand.EVIDENCE_IMG_MAX_W:
                    s = _brand.EVIDENCE_IMG_MAX_W / w
                    w *= s; h *= s
                if h > _brand.EVIDENCE_IMG_MAX_H:
                    s = _brand.EVIDENCE_IMG_MAX_H / h
                    w *= s; h *= s
                img.drawWidth = w
                img.drawHeight = h
                cells.append([
                    Paragraph(f"<b>{display_label}</b>", label_style),
                    Spacer(1, 0.05 * inch),
                    img,
                ])
            except Exception as e:
                print(f"  [combined report] image load failed {img_info.get('path')}: {e}")

        if not cells:
            continue

        full_grid_w = 9.6 * inch
        col_w = 4.8 * inch
        if len(cells) == 1:
            grid = Table([cells], colWidths=[full_grid_w])
        else:
            grid_rows: List[List[Any]] = []
            paired = len(cells) - (len(cells) % 2)
            for i in range(0, paired, 2):
                grid_rows.append([cells[i], cells[i + 1]])
            if len(cells) % 2 == 1:
                grid_rows.append([cells[-1]])
            grid = Table(grid_rows, colWidths=[col_w, col_w])
            if len(cells) % 2 == 1:
                last = len(grid_rows) - 1
                grid.setStyle(TableStyle([("SPAN", (0, last), (1, last))]))

        grid.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
        ]))

        block = KeepTogether([
            info_t,
            Spacer(1, 0.15 * inch),
            grid,
            Spacer(1, 0.3 * inch),
        ])
        elements.append(block)

    return elements


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    evidence_root: Optional[str] = None,
    wagon_states_root: Optional[str] = None,
    cache_root: Optional[str] = None,
    missing_cameras: Optional[Sequence[str]] = None,
    camera_pdf_urls: Optional[Dict[str, str]] = None,
    logo_path: Optional[str] = None,
    report_meta: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Stage 5 public entry.  Writes JSON always; PDF if reportlab is OK.

    `report_meta` carries the incremental-lifecycle fields (report_revision,
    report_status INTERIM/FINAL/FINAL_PARTIAL, cameras_present/pending/
    missing_final, GST version+hash, fusion_revision, partial_reason) into the
    JSON.  The PDF's existing partial-report banner still renders from
    `missing_cameras` -- branding + grids are unchanged.

    Returns:
        {"json_path": "...", "pdf_path": "..." | None}
    """
    os.makedirs(output_dir, exist_ok=True)
    missing_cameras = list(missing_cameras or [])

    # Always build the view-model.  Even if the PDF fails, the JSON has it.
    vm = _adapter.build_legacy_view_model(
        state=state, unified=unified,
        wagon_states_root=wagon_states_root,
        evidence_root=evidence_root,
        missing_cameras=missing_cameras,
    )

    t0 = time.time()
    json_doc = _build_json(
        state=state, unified=unified, batch_key=batch_key,
        source_video_urls=source_video_urls,
        processed_video_urls=processed_video_urls,
        extra_metadata=extra_metadata,
        evidence_root=evidence_root,
        legacy_view_model=vm,
        report_meta=report_meta,
    )
    json_path = os.path.join(output_dir, "combined_train_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_doc, f, indent=2, default=str)
    if verbose:
        print(f"[STAGE5] wrote {json_path}")

    pdf_path: Optional[str] = os.path.join(output_dir, "combined_train_report.pdf")
    try:
        _build_pdf(
            state=state, unified=unified, vm=vm,
            batch_key=batch_key, output_pdf=pdf_path,
            evidence_root=evidence_root,
            source_video_urls=dict(source_video_urls or {}),
            processed_video_urls=dict(processed_video_urls or {}),
            camera_pdf_urls=dict(camera_pdf_urls or {}),
            logo_path=logo_path,
            missing_cameras=missing_cameras,
            cache_root=cache_root,
        )
        if verbose:
            print(f"[STAGE5] wrote {pdf_path}")
    except Exception as e:
        print(f"[STAGE5] PDF generation FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        pdf_path = None

    if verbose:
        print(f"[STAGE5] done in {time.time() - t0:.1f}s")
    return {"json_path": json_path, "pdf_path": pdf_path}
