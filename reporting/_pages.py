"""Shared reportlab page widgets for the camera + combined reports.

These are presentation primitives ported from the legacy per-camera
generators (old_system/{CAMERA}/report_generator.py).  They are pure
layout helpers -- no model loads, no video decoding; every image comes
from a path already on disk (evidence/ or wagon_cache/).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import _brand
from . import _evidence_lookup as ev


# -----------------------------------------------------------------------------
# Document
# -----------------------------------------------------------------------------

def make_doc(output_pdf: str, title: str, logo_path: Optional[str]):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import inch
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)
    doc = BaseDocTemplate(
        output_pdf, pagesize=landscape(A4),
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        title=title, author="WagonEye",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="content")
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame],
                     onPage=_brand.make_logo_callback(logo_path))
    ])
    return doc


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

def bordered_image(path: str, w_inch: float, h_inch: float, border_color):
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Table, TableStyle
    img = Image(path, width=w_inch * inch, height=h_inch * inch)
    img.hAlign = "CENTER"
    t = Table([[img]], colWidths=[(w_inch + 0.2) * inch])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 2, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def centered(elem, width_inch: float = 10.0):
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle
    t = Table([[elem]], colWidths=[width_inch * inch])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return t


def no_snapshot_placeholder():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Table, TableStyle
    p = ParagraphStyle("PHCenter", fontSize=16, alignment=1,
                       textColor=colors.Color(0.4, 0.4, 0.4))
    t = Table([[Paragraph("Snapshot Not Available", p)]],
              colWidths=[9 * inch], rowHeights=[4 * inch])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
        ("BOX", (0, 0), (-1, -1), 2, colors.gray),
    ]))
    return t


# -----------------------------------------------------------------------------
# Detection summary table (cover-page KPI grid) -- legacy report_generator 518-543
# -----------------------------------------------------------------------------

def detection_summary_table(rows: List[List[str]]):
    """rows[0] is the header; remaining rows are [metric, count]."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]
    for i, row in enumerate(rows[1:], start=1):
        metric = (row[0] or "").lower()
        if "open" in metric or "missing" in metric or "damaged" in metric:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.Color(1, 0.8, 0.8)))
        elif "closed" in metric or "captured" in metric or metric.strip() == "ok":
            style.append(("BACKGROUND", (0, i), (-1, i), colors.Color(0.8, 1, 0.8)))
        elif "damage" in metric:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.Color(1, 0.95, 0.95)))
        elif "loaded" in metric:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#E3F2FD")))
        elif "empty" in metric:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFF3E0")))
    t = Table(rows, colWidths=[3 * inch, 1.5 * inch])
    t.setStyle(TableStyle(style))
    return t


# -----------------------------------------------------------------------------
# Wagon overview (2x2 quartile grid for ONE camera) -- legacy 854-934
# -----------------------------------------------------------------------------

def wagon_overview_page(
    *, wagon_number: int, gw_id: str, classification: str,
    cache_root: Optional[str], camera_id: str,
    start_time: float, end_time: float, local_meta: Dict[str, Any],
    no_detection_text: str, styles,
):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, Image,
    )

    paths = ev.quartile_cache_paths(
        cache_root=cache_root, gw_id=gw_id, camera_id=camera_id,
        wagon_start_time=start_time, wagon_end_time=end_time,
        local_fps=float(local_meta.get("fps", 0.0)),
        local_total_frames=int(local_meta.get("total_frames", 0)),
    )

    header_p = ParagraphStyle("OvHeader", fontSize=24, alignment=TA_CENTER,
                              textColor=colors.black, fontName="Helvetica-Bold")
    sub_p = ParagraphStyle("OvSub", fontSize=12, alignment=TA_CENTER,
                           textColor=_brand.TEXT_MUTED, fontName="Helvetica")
    label_p = ParagraphStyle("OvLabel", fontSize=10, alignment=TA_CENTER,
                             textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold")

    page: List[Any] = [
        Paragraph(f"<b>Wagon No: {wagon_number}</b>", header_p),
        Spacer(1, 0.05 * inch),
    ]
    if no_detection_text:
        page.append(Paragraph(no_detection_text, sub_p))
    page.append(Paragraph(
        f"{gw_id} | {classification} | Camera: {camera_id} | "
        f"Time: {start_time:.1f}s &ndash; {end_time:.1f}s", sub_p,
    ))
    page.append(Spacer(1, 0.15 * inch))

    titles = ["Start (12.5%)", "Middle-1 (37.5%)", "Middle-2 (62.5%)", "End (87.5%)"]
    cells: List[List[Any]] = []
    for label, p in zip(titles, paths):
        if p and os.path.isfile(p):
            try:
                img = Image(p, width=4.3 * inch, height=2.4 * inch)
                img.hAlign = "CENTER"
                cells.append([Paragraph(f"<b>{label}</b>", label_p),
                              Spacer(1, 0.04 * inch), img])
                continue
            except Exception:
                pass
        cells.append([Paragraph(f"<b>{label}</b>", label_p),
                      Spacer(1, 0.04 * inch),
                      Paragraph("[no snapshot]", sub_p)])

    grid = Table([[cells[0], cells[1]], [cells[2], cells[3]]],
                 colWidths=[4.8 * inch, 4.8 * inch])
    grid.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    page.append(grid)
    return [KeepTogether(page), PageBreak()]


# -----------------------------------------------------------------------------
# Generic state-tinted detail page (door / damage / ocr / load) -- legacy 721-845
# -----------------------------------------------------------------------------

def detail_page(
    *, header_html: str, state: Optional[str], confidence: float,
    snapshot_path: Optional[str], styles,
    info_rows: Optional[List[List[str]]] = None,
    img_w: float = 9.0, img_h: float = 4.5,
):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak,
    )

    text_c, bg_c, border_c = _brand.state_colors(state)
    state_disp = (state or "UNKNOWN").upper().replace("_", " ")

    head_p = ParagraphStyle("DetHead", fontSize=20, alignment=TA_CENTER,
                            textColor=colors.black, fontName="Helvetica-Bold")
    head_t = Table([[Paragraph(header_html, head_p)]], colWidths=[10 * inch])
    head_t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))

    state_p = ParagraphStyle("DetStateVal", fontSize=12,
                             textColor=text_c, fontName="Helvetica-Bold")
    rows = info_rows or [
        ["Final State", f"{state_disp}"],
        ["Confidence",  f"{float(confidence or 0.0):.1%}"],
    ]
    info_data = []
    for k, v in rows:
        val_style = state_p if k.lower() in ("final state", "state") else \
            ParagraphStyle("DetVal", fontSize=12, fontName="Helvetica-Bold")
        info_data.append([
            Paragraph(f"<b>{k}</b>", styles["TableCell"]),
            Paragraph(f"<b>{v}</b>", val_style),
        ])
    info_t = Table(info_data, colWidths=[1.7 * inch, 2.6 * inch])
    info_t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), bg_c),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 2, border_c),
        ("LINEBELOW", (0, 0), (-1, 0), 1, border_c),
    ]))

    page: List[Any] = [head_t, Spacer(1, 8), centered(info_t), Spacer(1, 10)]
    if snapshot_path and os.path.isfile(snapshot_path):
        try:
            page.append(centered(bordered_image(snapshot_path, img_w, img_h, colors.gray)))
        except Exception:
            page.append(no_snapshot_placeholder())
    else:
        page.append(no_snapshot_placeholder())

    return [KeepTogether(page), PageBreak()]


# -----------------------------------------------------------------------------
# Simple state page (loaded / no-damage / non-wagon) -- legacy 713-1131
# -----------------------------------------------------------------------------

def simple_state_page(
    *, wagon_number: Optional[int], gw_id: str, kind: str,
    classification: str, start_time: float, end_time: float,
    cache_root: Optional[str], camera_id: str, local_meta: Dict[str, Any],
    styles,
):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak,
    )

    presets = {
        "loaded":    (colors.Color(0.2, 0.4, 0.7),  colors.Color(0.2, 0.4, 0.7),
                      "LOADED – FLOOR NOT VISIBLE", f"<b>Wagon No: {wagon_number}</b>"),
        "no_damage": (colors.Color(0.4, 0.4, 0.4),  colors.gray,
                      "NO DAMAGE DETECTED", f"<b>Wagon No: {wagon_number}</b>"),
        "empty":     (colors.Color(0.9, 0.4, 0.0),  colors.Color(0.9, 0.4, 0.0),
                      "EMPTY", f"<b>Wagon No: {wagon_number}</b>"),
        "engine":    (colors.Color(0.4, 0.2, 0.6),  colors.Color(0.5, 0.3, 0.7),
                      "NOT COUNTED AS WAGON", "<b>ENGINE</b>"),
        "brake_van": (colors.Color(0.2, 0.45, 0.45), colors.Color(0.3, 0.55, 0.55),
                      "NOT COUNTED AS WAGON", "<b>BREAKVAN</b>"),
        "no_data":   (colors.gray, colors.gray,
                      "NO DATA", f"<b>Wagon No: {wagon_number}</b>"),
    }
    text_color, border, subtitle, header = presets.get(
        kind, (colors.black, colors.gray, "", f"<b>Wagon No: {wagon_number}</b>"))

    header_p = ParagraphStyle("SimpH", fontSize=24, alignment=TA_CENTER,
                              textColor=text_color, fontName="Helvetica-Bold")
    sub_p = ParagraphStyle("SimpSub", fontSize=18, alignment=TA_CENTER,
                           textColor=text_color, fontName="Helvetica-Bold")
    frame_p = ParagraphStyle("SimpFrame", fontSize=10, alignment=TA_CENTER,
                             textColor=_brand.TEXT_MUTED, fontName="Helvetica")

    page: List[Any] = []
    head_t = Table([[Paragraph(header, header_p)]], colWidths=[10 * inch])
    head_t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    page.append(head_t)
    page.append(Spacer(1, 8))
    if subtitle:
        page.append(Paragraph(subtitle, sub_p))
    page.append(Paragraph(
        f"{gw_id} | {classification} | Camera: {camera_id} | "
        f"Time: {start_time:.1f}s &ndash; {end_time:.1f}s", frame_p,
    ))
    page.append(Spacer(1, 10))

    mid = ev.midpoint_cache_path(
        cache_root=cache_root, gw_id=gw_id, camera_id=camera_id,
        wagon_start_time=start_time, wagon_end_time=end_time,
        local_fps=float(local_meta.get("fps", 0.0)),
        local_total_frames=int(local_meta.get("total_frames", 0)),
    )
    if mid and os.path.isfile(mid):
        page.append(centered(bordered_image(mid, 9.0, 4.5, border)))
    else:
        page.append(no_snapshot_placeholder())

    return [KeepTogether(page), PageBreak()]
