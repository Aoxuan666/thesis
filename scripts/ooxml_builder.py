# -*- coding: utf-8 -*-
"""
Pure OOXML thesis builder compliant with Tianjin Agricultural University (TJAU)
2019 Academic Master's Thesis Formatting Specification.

- A4 (210 mm x 297 mm), Margins: Left 30 mm, Right 20 mm, Top 25 mm, Bottom 25 mm
- Header distance: 14 mm, Footer distance: 18 mm, centered page numbers
- Odd/Even different headers in body section:
    Odd  -> 天津农学院硕士学位论文 (5号宋体居中)
    Even -> STYLEREF 1 (current chapter H1 title, 5号宋体居中)
- Heading 1 (三号黑体居中, 1.5 lines before, 1 line after, pageBreakBefore)
- Heading 2 (四号黑体居左, 0.5 lines before/after)
- Heading 3 (小四号黑体居左, 0.2 lines before/after)
- Heading 4 (小四号宋体加粗居左)
- Body text (小四号宋体 + Times New Roman, 2-char first-line indent, 21 pt fixed line spacing)
- Three-line tables (top 1.5 pt, header bottom 0.75 pt, bottom 1.5 pt, no vertical borders)
- Bilingual captions (中文五号黑体 + 英文五号 Times New Roman)
- Native Word OMML (m:oMath) equations centered with right-aligned (x-y) numbers
- Live PAGEREF Chinese & English TOCs with cached real page numbers from 2-pass LibreOffice build
"""
import os
import re
import zipfile
from xml.sax.saxutils import escape as xml_escape
from PIL import Image

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def mm_to_twips(mm):
    return int(round(mm * 1440.0 / 25.4))


def mm_to_emu(mm):
    return int(round(mm * 36000.0))


def pt_to_halfpt(pt):
    return int(round(pt * 2))


def pt_to_twips(pt):
    return int(round(pt * 20))


def xesc(text):
    return xml_escape(str(text), {'"': "&quot;", "'": "&apos;"})


# =====================================================================
# OMML (Word native math) helper builders
# =====================================================================
def mr(text, italic=None):
    """OMML math run <m:r>. Numbers/operators upright by default; Latin letters italic unless italic=False."""
    t = xesc(text)
    rpr = ""
    if italic is False:
        rpr = '<m:rPr><m:nor/></m:rPr>'
    elif italic is True:
        rpr = '<m:rPr><m:sty m:val="i"/></m:rPr>'
    wrpr = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Cambria Math"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
    return f'<m:r>{rpr}{wrpr}<m:t xml:space="preserve">{t}</m:t></m:r>'


def mn(text):
    """Upright math text (numbers, operators, function names like tanh, Conv1D, Low)."""
    return mr(text, italic=False)


def mv(text):
    """Italic math variable."""
    return mr(text, italic=True)


def msub(base, sub):
    return f'<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>'


def msup(base, sup):
    return f'<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>'


def msubsup(base, sub, sup):
    return f'<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>'


def mfrac(num, den):
    return f'<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>'


def msqrt(expr):
    return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{expr}</m:e></m:rad>'


def mdelim(expr, beg="(", end=")"):
    return f'<m:d><m:dPr><m:begChr m:val="{xesc(beg)}"/><m:endChr m:val="{xesc(end)}"/></m:dPr><m:e>{expr}</m:e></m:d>'


def macc(expr, chr_val="\u0302"):
    """Accent: hat=\u0302, tilde=\u0303, bar=\u0304."""
    return f'<m:acc><m:accPr><m:chr m:val="{chr_val}"/></m:accPr><m:e>{expr}</m:e></m:acc>'


def msum(sub_expr, sup_expr, body_expr, op="\u2211"):
    return (
        f'<m:nary><m:naryPr><m:chr m:val="{op}"/><m:limLoc m:val="undOvr"/></m:naryPr>'
        f'<m:sub>{sub_expr}</m:sub><m:sup>{sup_expr}</m:sup><m:e>{body_expr}</m:e></m:nary>'
    )


def mmatrix(rows):
    """rows is list of list of OMML strings."""
    mrs = "".join("<m:mr>" + "".join(f"<m:e>{cell}</m:e>" for cell in r) + "</m:mr>" for r in rows)
    return f"<m:m>{mrs}</m:m>"


def mcases(rows):
    """Piecewise curly-brace cases: rows is list of (expr_omml, cond_omml)."""
    mrs = "".join(
        f'<m:mr><m:e>{expr}{mn(",  ")}</m:e><m:e>{cond}</m:e></m:mr>'
        for expr, cond in rows
    )
    return mdelim(f"<m:m>{mrs}</m:m>", beg="{", end="")


# =====================================================================
# Rich text run helper (supports *italic*, ^sup^, ~sub~ markers in prose)
# =====================================================================
def make_run_xml(text, east_asia="宋体", ascii_font="Times New Roman",
                 sz_pt=12.0, bold=False, italic=False, color=None, vert_align=None, underline=False):
    sz = pt_to_halfpt(sz_pt)
    parts = [
        f'<w:rFonts w:ascii="{xesc(ascii_font)}" w:hAnsi="{xesc(ascii_font)}" '
        f'w:eastAsia="{xesc(east_asia)}" w:cs="{xesc(ascii_font)}"/>'
    ]
    if bold:
        parts.append("<w:b/><w:bCs/>")
    if italic:
        parts.append("<w:i/><w:iCs/>")
    if underline:
        parts.append('<w:u w:val="single"/>')
    if color:
        parts.append(f'<w:color w:val="{color}"/>')
    parts.append(f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>')
    if vert_align:
        parts.append(f'<w:vertAlign w:val="{vert_align}"/>')
    return f'<w:r><w:rPr>{"".join(parts)}</w:rPr><w:t xml:space="preserve">{xesc(text)}</w:t></w:r>'


def render_inline_markup(text, east_asia="宋体", ascii_font="Times New Roman",
                         sz_pt=12.0, bold=False, italic=False):
    """
    Parse lightweight inline markers:
      - *italic text* (only when not mathematical multiplication like 1.8*T)
      - R² / CO₂ / NH₃ / m³ / m⁻³ / m⁻¹ / A₃ / D₁ / D₂ / D₃ superscripts & subscripts rendered cleanly
    """
    # Convert unicode superscripts/subscripts selectively into clean Word vertAlign runs for publication look
    # while also supporting explicit <<I>>...<</I>> italic tags used by bibliography/scientific terms.
    tokens = re.split(r"(<<I>>.*?<</I>>)", text)
    out = []
    for tok in tokens:
        if not tok:
            continue
        is_it = italic
        seg = tok
        if tok.startswith("<<I>>") and tok.endswith("<</I>>"):
            is_it = True
            seg = tok[5:-6]
        # Handle chemical/unit superscripts & subscripts nicely
        sub_sup_pat = re.compile(r"(CO₂|NH₃|R²|m³|m⁻³|m⁻¹|s⁻¹|μg·m⁻³|mg·m⁻³|m·s⁻¹|O₂)")
        pieces = sub_sup_pat.split(seg)
        for p in pieces:
            if not p:
                continue
            if p == "CO₂":
                out.append(make_run_xml("CO", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("2", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="subscript"))
            elif p == "NH₃":
                out.append(make_run_xml("NH", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("3", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="subscript"))
            elif p == "R²":
                out.append(make_run_xml("R", east_asia, ascii_font, sz_pt, bold, True))
                out.append(make_run_xml("2", east_asia, ascii_font, sz_pt, bold, False, vert_align="superscript"))
            elif p == "m³":
                out.append(make_run_xml("m", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("3", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "m⁻³":
                out.append(make_run_xml("m", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("-3", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "m⁻¹":
                out.append(make_run_xml("m", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("-1", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "s⁻¹":
                out.append(make_run_xml("s", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("-1", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "μg·m⁻³":
                out.append(make_run_xml("μg·m", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("-3", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "mg·m⁻³":
                out.append(make_run_xml("mg·m", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("-3", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "m·s⁻¹":
                out.append(make_run_xml("m·s", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("-1", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="superscript"))
            elif p == "O₂":
                out.append(make_run_xml("O", east_asia, ascii_font, sz_pt, bold, is_it))
                out.append(make_run_xml("2", east_asia, ascii_font, sz_pt, bold, is_it, vert_align="subscript"))
            else:
                out.append(make_run_xml(p, east_asia, ascii_font, sz_pt, bold, is_it))
    return "".join(out)


# =====================================================================
# Main TJAU Academic Master's Thesis OOXML Builder
# =====================================================================
class TJAUThesisBuilder:
    # A4 content width = 210 - 30 - 20 = 160 mm
    CONTENT_WIDTH_MM = 160.0
    CONTENT_WIDTH_TWIPS = mm_to_twips(160.0)  # 9071 twips

    def __init__(self, toc_page_map=None):
        """
        toc_page_map: optional dict {bookmark_name: display_page_str} populated on Pass 2
        from LibreOffice rendered PDF.
        """
        self.body_parts = []
        self.images = []       # list of (rid, media_name, bytes, ext)
        self.next_rid = 10     # rId1..rId9 reserved for styles/settings/headers/footers
        self.next_bm_id = 1
        self.next_docpr_id = 1
        self.toc_entries = []  # list of (level, zh_title, en_title, bm_name, is_front)
        self.toc_page_map = toc_page_map or {}
        self.zh_toc_placeholder_idx = None
        self.en_toc_placeholder_idx = None

    def _alloc_bm(self, name):
        bid = self.next_bm_id
        self.next_bm_id += 1
        return bid, name

    def make_sect_pr(self, hdr_odd_rid=None, hdr_even_rid=None,
                     ftr_odd_rid=None, ftr_even_rid=None,
                     pg_num_fmt=None, pg_num_start=None):
        parts = []
        if hdr_odd_rid:
            parts.append(f'<w:headerReference w:type="default" r:id="{hdr_odd_rid}"/>')
        if hdr_even_rid:
            parts.append(f'<w:headerReference w:type="even" r:id="{hdr_even_rid}"/>')
        if ftr_odd_rid:
            parts.append(f'<w:footerReference w:type="default" r:id="{ftr_odd_rid}"/>')
        if ftr_even_rid:
            parts.append(f'<w:footerReference w:type="even" r:id="{ftr_even_rid}"/>')
        parts.append(f'<w:pgSz w:w="{mm_to_twips(210.0)}" w:h="{mm_to_twips(297.0)}"/>')
        parts.append(
            f'<w:pgMar w:top="{mm_to_twips(25.0)}" w:right="{mm_to_twips(20.0)}" '
            f'w:bottom="{mm_to_twips(25.0)}" w:left="{mm_to_twips(30.0)}" '
            f'w:header="{mm_to_twips(14.0)}" w:footer="{mm_to_twips(18.0)}" w:gutter="0"/>'
        )
        if pg_num_fmt:
            start_attr = f' w:start="{pg_num_start}"' if pg_num_start is not None else ""
            parts.append(f'<w:pgNumType w:fmt="{pg_num_fmt}"{start_attr}/>')
        parts.append('<w:cols w:space="425"/>')
        parts.append('<w:docGrid w:type="lines" w:linePitch="312"/>')
        return f'<w:sectPr>{"".join(parts)}</w:sectPr>'

    def add_section_break(self, sect_pr_xml):
        self.body_parts.append(f'<w:p><w:pPr>{sect_pr_xml}</w:pPr></w:p>')

    def add_page_break(self):
        self.body_parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_blank_lines(self, n=1, sz_pt=12.0):
        for _ in range(n):
            self.body_parts.append(
                f'<w:p><w:pPr><w:spacing w:line="{pt_to_twips(21)}" w:lineRule="exact"/></w:pPr>'
                f'{make_run_xml("", sz_pt=sz_pt)}</w:p>'
            )

    def add_custom_p(self, text, align="left", east_asia="宋体", ascii_font="Times New Roman",
                     sz_pt=12.0, bold=False, italic=False, line_pt=21.0,
                     before_pt=0.0, after_pt=0.0, first_indent_chars=0,
                     hanging_twips=0, left_twips=0, page_break_before=False,
                     keep_next=False, underline=False):
        ppr = []
        if page_break_before:
            ppr.append("<w:pageBreakBefore/>")
        if keep_next:
            ppr.append("<w:keepNext/>")
        ppr.append(
            f'<w:spacing w:before="{pt_to_twips(before_pt)}" w:after="{pt_to_twips(after_pt)}" '
            f'w:line="{pt_to_twips(line_pt)}" w:lineRule="exact"/>'
        )
        ind_attrs = []
        if first_indent_chars:
            ind_attrs.append(f'w:firstLineChars="{int(first_indent_chars * 100)}"')
            ind_attrs.append(f'w:firstLine="{int(round(first_indent_chars * sz_pt * 20))}"')
        if hanging_twips:
            ind_attrs.append(f'w:hanging="{hanging_twips}"')
        if left_twips:
            ind_attrs.append(f'w:left="{left_twips}"')
        if ind_attrs:
            ppr.append(f'<w:ind {" ".join(ind_attrs)}/>')
        ppr.append(f'<w:jc w:val="{align}"/>')
        if underline:
            runs = make_run_xml(text, east_asia, ascii_font, sz_pt, bold, italic, underline=True)
        else:
            runs = render_inline_markup(text, east_asia, ascii_font, sz_pt, bold, italic)
        self.body_parts.append(f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{runs}</w:p>')

    def add_body(self, text, first_indent_chars=2):
        """Standard body paragraph: 小四号(12pt) 宋体 + Times New Roman, 21pt exact line spacing, justified."""
        self.add_custom_p(
            text, align="both", east_asia="宋体", ascii_font="Times New Roman",
            sz_pt=12.0, line_pt=21.0, first_indent_chars=first_indent_chars
        )

    def add_en_body(self, text, first_indent_chars=2):
        """English Abstract body: Times New Roman 小四号(12pt), 21pt exact line spacing, justified."""
        self.add_custom_p(
            text, align="both", east_asia="宋体", ascii_font="Times New Roman",
            sz_pt=12.0, line_pt=21.0, first_indent_chars=first_indent_chars
        )

    def add_h1(self, zh_title, en_title=None, bm_name=None, in_toc=True,
               page_break=True, is_front=False, style_id="Heading1", sz_pt=16.0,
               before_pt=24.0, after_pt=16.0, east_asia="黑体", bold=False, is_en_only=False):
        """
        Chapter / Level-1 heading:
        TJAU spec: 三号(16pt)黑体居中，上空1.5行(24pt)下空1行(16pt).
        """
        bm_start, bm_end = "", ""
        if bm_name:
            bid, bname = self._alloc_bm(bm_name)
            bm_start = f'<w:bookmarkStart w:id="{bid}" w:name="{xesc(bname)}"/>'
            bm_end = f'<w:bookmarkEnd w:id="{bid}"/>'
            if in_toc:
                self.toc_entries.append((1, zh_title, en_title or zh_title, bname, is_front))

        ppr = []
        if style_id:
            ppr.append(f'<w:pStyle w:val="{style_id}"/>')
        if page_break:
            ppr.append("<w:pageBreakBefore/>")
        ppr.append("<w:keepNext/>")
        ppr.append(
            f'<w:spacing w:before="{pt_to_twips(before_pt)}" w:after="{pt_to_twips(after_pt)}" '
            f'w:line="{pt_to_twips(24)}" w:lineRule="exact"/>'
        )
        ppr.append('<w:jc w:val="center"/>')
        runs = make_run_xml(
            zh_title,
            east_asia=east_asia,
            ascii_font="Times New Roman",
            sz_pt=sz_pt,
            bold=bold or is_en_only,
        )
        self.body_parts.append(f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{bm_start}{runs}{bm_end}</w:p>')

    def add_h2(self, zh_title, en_title=None, bm_name=None, in_toc=True):
        """Level-2 heading (1.1): 四号(14pt)黑体居左，段前、段后0.5行(7pt)."""
        bm_start, bm_end = "", ""
        if bm_name:
            bid, bname = self._alloc_bm(bm_name)
            bm_start = f'<w:bookmarkStart w:id="{bid}" w:name="{xesc(bname)}"/>'
            bm_end = f'<w:bookmarkEnd w:id="{bid}"/>'
            if in_toc:
                self.toc_entries.append((2, zh_title, en_title or zh_title, bname, False))

        ppr = (
            '<w:pStyle w:val="Heading2"/><w:keepNext/>'
            f'<w:spacing w:before="{pt_to_twips(7)}" w:after="{pt_to_twips(7)}" '
            f'w:line="{pt_to_twips(21)}" w:lineRule="exact"/>'
            '<w:jc w:val="left"/>'
        )
        runs = make_run_xml(zh_title, east_asia="黑体", ascii_font="Times New Roman", sz_pt=14.0, bold=False)
        self.body_parts.append(f'<w:p><w:pPr>{ppr}</w:pPr>{bm_start}{runs}{bm_end}</w:p>')

    def add_h3(self, zh_title):
        """Level-3 heading (1.1.1): 小四号(12pt)黑体/小标体居左，段前、段后0.2行(3pt)."""
        ppr = (
            '<w:pStyle w:val="Heading3"/><w:keepNext/>'
            f'<w:spacing w:before="{pt_to_twips(3)}" w:after="{pt_to_twips(3)}" '
            f'w:line="{pt_to_twips(21)}" w:lineRule="exact"/>'
            '<w:jc w:val="left"/>'
        )
        runs = make_run_xml(zh_title, east_asia="黑体", ascii_font="Times New Roman", sz_pt=12.0, bold=False)
        self.body_parts.append(f'<w:p><w:pPr>{ppr}</w:pPr>{runs}</w:p>')

    def add_h4(self, zh_title):
        """Level-4 heading (1.1.1.1): 小四号(12pt)宋体加粗居左."""
        ppr = (
            '<w:keepNext/>'
            f'<w:spacing w:before="{pt_to_twips(2)}" w:after="{pt_to_twips(2)}" '
            f'w:line="{pt_to_twips(21)}" w:lineRule="exact"/>'
            '<w:jc w:val="left"/>'
        )
        runs = make_run_xml(zh_title, east_asia="宋体", ascii_font="Times New Roman", sz_pt=12.0, bold=True)
        self.body_parts.append(f'<w:p><w:pPr>{ppr}</w:pPr>{runs}</w:p>')

    # -----------------------------------------------------------------
    # TOC Placeholders (filled right before save so all bookmarks exist)
    # -----------------------------------------------------------------
    def reserve_zh_toc(self):
        self.zh_toc_placeholder_idx = len(self.body_parts)
        self.body_parts.append("<!--ZH_TOC-->")

    def reserve_en_toc(self):
        self.en_toc_placeholder_idx = len(self.body_parts)
        self.body_parts.append("<!--EN_TOC-->")

    def _make_toc_line(self, label, bm_name, indent_twips=0, is_en=False):
        cached = self.toc_page_map.get(bm_name, "1")
        tab_pos = self.CONTENT_WIDTH_TWIPS
        ppr = (
            f'<w:spacing w:before="0" w:after="0" w:line="{pt_to_twips(22)}" w:lineRule="exact"/>'
            f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{tab_pos}"/></w:tabs>'
            f'<w:ind w:left="{indent_twips}"/>'
            '<w:jc w:val="left"/>'
        )
        ea = "Times New Roman" if is_en else "宋体"
        r_label = make_run_xml(label, east_asia=ea, ascii_font="Times New Roman", sz_pt=12.0)
        r_tab = '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr><w:tab/></w:r>'
        fld = (
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve"> PAGEREF {xesc(bm_name)} \\h </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f'{make_run_xml(str(cached), east_asia="宋体", ascii_font="Times New Roman", sz_pt=12.0)}'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        )
        return f'<w:p><w:pPr>{ppr}</w:pPr>{r_label}{r_tab}{fld}</w:p>'

    def _render_tocs(self):
        if self.zh_toc_placeholder_idx is not None:
            parts = []
            # 二号(22pt)黑体居中，段前段后1行(22pt)
            parts.append(
                '<w:p><w:pPr><w:spacing w:before="440" w:after="440" w:line="480" w:lineRule="exact"/>'
                '<w:jc w:val="center"/></w:pPr>'
                f'{make_run_xml("目  录", east_asia="黑体", ascii_font="Times New Roman", sz_pt=22.0)}</w:p>'
            )
            for lvl, zh, _en, bm, _front in self.toc_entries:
                ind = 0 if lvl == 1 else 420
                parts.append(self._make_toc_line(zh, bm, indent_twips=ind, is_en=False))
            self.body_parts[self.zh_toc_placeholder_idx] = "".join(parts)

        if self.en_toc_placeholder_idx is not None:
            parts = []
            # Times New Roman 二号(22pt)加粗居中，段前段后1行(22pt)
            parts.append(
                '<w:p><w:pPr><w:pageBreakBefore/><w:spacing w:before="440" w:after="440" w:line="480" w:lineRule="exact"/>'
                '<w:jc w:val="center"/></w:pPr>'
                f'{make_run_xml("Contents", east_asia="Times New Roman", ascii_font="Times New Roman", sz_pt=22.0, bold=True)}</w:p>'
            )
            for lvl, _zh, en, bm, _front in self.toc_entries:
                ind = 0 if lvl == 1 else 420
                parts.append(self._make_toc_line(en, bm, indent_twips=ind, is_en=True))
            self.body_parts[self.en_toc_placeholder_idx] = "".join(parts)

    # -----------------------------------------------------------------
    # Native Word OMML Equations (centered math + right-aligned (x-y))
    # -----------------------------------------------------------------
    def add_equation(self, omml_inner, eq_num):
        """
        Borderless 3-column layout table (12% | 76% | 12%) guarantees exact centered OMML
        and exact right-aligned equation number (e.g. (3-1)) across both Word & LibreOffice,
        with zero borders/lines between equation and number (per TJAU spec P25).
        """
        w_side = int(self.CONTENT_WIDTH_TWIPS * 0.12)
        w_mid = self.CONTENT_WIDTH_TWIPS - 2 * w_side
        no_borders = (
            '<w:tblBorders>'
            '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '</w:tblBorders>'
        )
        cell_mar = (
            '<w:tblCellMar>'
            '<w:top w:w="40" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
            '<w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/>'
            '</w:tblCellMar>'
        )
        math_xml = f'<m:oMath>{omml_inner}</m:oMath>'
        num_xml = make_run_xml(f"({eq_num})", east_asia="宋体", ascii_font="Times New Roman", sz_pt=12.0)
        tbl = (
            f'<w:tbl><w:tblPr><w:tblW w:w="{self.CONTENT_WIDTH_TWIPS}" w:type="dxa"/>'
            f'<w:jc w:val="center"/>{no_borders}{cell_mar}<w:tblLayout w:type="fixed"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{w_side}"/><w:gridCol w:w="{w_mid}"/><w:gridCol w:w="{w_side}"/></w:tblGrid>'
            '<w:tr><w:trPr><w:cantSplit/></w:trPr>'
            f'<w:tc><w:tcPr><w:tcW w:w="{w_side}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p/></w:tc>'
            f'<w:tc><w:tcPr><w:tcW w:w="{w_mid}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:before="60" w:after="60" w:line="280" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{math_xml}</w:p></w:tc>'
            f'<w:tc><w:tcPr><w:tcW w:w="{w_side}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:before="60" w:after="60"/><w:jc w:val="right"/></w:pPr>{num_xml}</w:p></w:tc>'
            '</w:tr></w:tbl>'
        )
        self.body_parts.append(tbl)

    # -----------------------------------------------------------------
    # Bilingual Three-Line Tables (TJAU Spec P21)
    # -----------------------------------------------------------------
    def add_three_line_table(self, zh_caption, en_caption, headers, rows,
                             col_weights=None, note=None, align_cols=None):
        """
        TJAU Spec:
        - Caption above table centered: 中文五号(10.5pt)黑体, 英文五号(10.5pt) Times New Roman
        - Three-line borders: top 1.5pt (sz=12), header bottom 0.75pt (sz=6), bottom 1.5pt (sz=12), no vertical lines
        - Cell font: 五号(10.5pt) 宋体 + Times New Roman
        - Table note: 小五号(9pt) 宋体
        """
        if zh_caption:
            self.add_custom_p(
                zh_caption, align="center", east_asia="黑体", ascii_font="Times New Roman",
                sz_pt=10.5, bold=False, line_pt=16.0, before_pt=8.0, after_pt=2.0, keep_next=True
            )
        if en_caption:
            self.add_custom_p(
                en_caption, align="center", east_asia="Times New Roman", ascii_font="Times New Roman",
                sz_pt=10.5, bold=False, line_pt=15.0, before_pt=0.0, after_pt=4.0, keep_next=True
            )

        ncols = len(headers)
        if col_weights and len(col_weights) == ncols:
            tot = float(sum(col_weights))
            widths = [int(round(self.CONTENT_WIDTH_TWIPS * w / tot)) for w in col_weights]
            widths[-1] = self.CONTENT_WIDTH_TWIPS - sum(widths[:-1])
        else:
            w0 = self.CONTENT_WIDTH_TWIPS // ncols
            widths = [w0] * ncols
            widths[-1] = self.CONTENT_WIDTH_TWIPS - w0 * (ncols - 1)

        tbl_borders = (
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '</w:tblBorders>'
        )
        cell_mar = (
            '<w:tblCellMar>'
            '<w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/>'
            '<w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>'
            '</w:tblCellMar>'
        )
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

        # Header row (with 0.75pt bottom border + tblHeader for repeat across page breaks)
        hdr_cells = []
        for ci, h in enumerate(headers):
            h_lines = str(h).split("\n")
            p_xmls = []
            for li, hl in enumerate(h_lines):
                p_xmls.append(
                    '<w:p><w:pPr><w:spacing w:before="20" w:after="20" w:line="280" w:lineRule="exact"/>'
                    '<w:jc w:val="center"/></w:pPr>'
                    f'{render_inline_markup(hl, east_asia="宋体", ascii_font="Times New Roman", sz_pt=10.5 if li == 0 else 9.0, bold=(li == 0))}'
                    '</w:p>'
                )
            hdr_cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>'
                '<w:vAlign w:val="center"/></w:tcPr>'
                f'{"".join(p_xmls)}</w:tc>'
            )
        tr_hdr = f'<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>{"".join(hdr_cells)}</w:tr>'

        # Body rows
        body_trs = []
        for ri, row in enumerate(rows):
            is_last = (ri == len(rows) - 1)
            row_cells = []
            for ci in range(ncols):
                val = str(row[ci]) if ci < len(row) else ""
                al = "center"
                if align_cols and ci < len(align_cols):
                    al = align_cols[ci]
                elif len(val) > 22:
                    al = "left"
                bot_bdr = '<w:tcBorders><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/></w:tcBorders>' if is_last else ""
                cell_p = (
                    '<w:p><w:pPr><w:spacing w:before="20" w:after="20" w:line="280" w:lineRule="exact"/>'
                    f'<w:jc w:val="{al}"/></w:pPr>'
                    f'{render_inline_markup(val, east_asia="宋体", ascii_font="Times New Roman", sz_pt=10.5)}'
                    '</w:p>'
                )
                row_cells.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{bot_bdr}'
                    '<w:vAlign w:val="center"/></w:tcPr>'
                    f'{cell_p}</w:tc>'
                )
            body_trs.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(row_cells)}</w:tr>')

        tbl = (
            f'<w:tbl><w:tblPr><w:tblW w:w="{self.CONTENT_WIDTH_TWIPS}" w:type="dxa"/>'
            f'<w:jc w:val="center"/>{tbl_borders}{cell_mar}<w:tblLayout w:type="fixed"/></w:tblPr>'
            f'<w:tblGrid>{grid}</w:tblGrid>{tr_hdr}{"".join(body_trs)}</w:tbl>'
        )
        self.body_parts.append(tbl)

        if note:
            self.add_custom_p(
                note, align="left", east_asia="宋体", ascii_font="Times New Roman",
                sz_pt=9.0, line_pt=14.0, before_pt=3.0, after_pt=6.0
            )
        else:
            # Small spacing paragraph after table
            self.body_parts.append('<w:p><w:pPr><w:spacing w:before="0" w:after="80" w:line="100" w:lineRule="exact"/></w:pPr></w:p>')

    # -----------------------------------------------------------------
    # Embedded Figures with Bilingual Captions (TJAU Spec P23)
    # -----------------------------------------------------------------
    def add_figure(self, img_path, zh_caption, en_caption, max_width_mm=142.0, max_height_mm=175.0):
        """
        Embeds image centered with aspect ratio preserved and bounded within (max_width_mm, max_height_mm),
        followed by bilingual centered captions (中文五号黑体 + 英文五号 Times New Roman).
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing figure file: {img_path}")
        with Image.open(img_path) as im:
            px_w, px_h = im.size
        aspect = px_h / float(px_w)
        w_mm = min(max_width_mm, self.CONTENT_WIDTH_MM - 4.0)
        h_mm = w_mm * aspect
        if h_mm > max_height_mm:
            h_mm = max_height_mm
            w_mm = h_mm / aspect

        cx = mm_to_emu(w_mm)
        cy = mm_to_emu(h_mm)

        ext = os.path.splitext(img_path)[1].lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        rid = f"rId{self.next_rid}"
        self.next_rid += 1
        media_name = f"image{len(self.images) + 1}.{ext}"
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        self.images.append((rid, media_name, img_bytes, ext))

        docpr_id = self.next_docpr_id
        self.next_docpr_id += 1
        drawing = (
            f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{docpr_id}" name="{xesc(media_name)}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{A_NS}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="{A_NS}"><a:graphicData uri="{PIC_NS}">'
            f'<pic:pic xmlns:pic="{PIC_NS}">'
            f'<pic:nvPicPr><pic:cNvPr id="{docpr_id}" name="{xesc(media_name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>'
        )
        self.body_parts.append(
            '<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r>{drawing}</w:r></w:p>'
        )
        self.add_custom_p(
            zh_caption, align="center", east_asia="黑体", ascii_font="Times New Roman",
            sz_pt=10.5, bold=False, line_pt=16.0, before_pt=2.0, after_pt=2.0, keep_next=True
        )
        self.add_custom_p(
            en_caption, align="center", east_asia="Times New Roman", ascii_font="Times New Roman",
            sz_pt=10.5, bold=False, line_pt=15.0, before_pt=0.0, after_pt=8.0
        )

    # -----------------------------------------------------------------
    # Bibliography paragraph (TJAU Spec P56: 五号宋体, 17pt fixed line, hanging indent)
    # -----------------------------------------------------------------
    def add_reference_item(self, text):
        self.add_custom_p(
            text, align="both", east_asia="宋体", ascii_font="Times New Roman",
            sz_pt=10.5, line_pt=17.0, before_pt=0.0, after_pt=1.5,
            left_twips=420, hanging_twips=420
        )

    # -----------------------------------------------------------------
    # Cover Metadata Table Helper (clean aligned key-value rows with bottom underlines)
    # -----------------------------------------------------------------
    def add_cover_fields(self, rows, is_en=False):
        w_key = mm_to_twips(42.0 if not is_en else 46.0)
        w_val = mm_to_twips(72.0 if not is_en else 74.0)
        no_bdr = (
            '<w:tblBorders>'
            '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '</w:tblBorders>'
        )
        trs = []
        for k, v in rows:
            k_run = make_run_xml(k, east_asia="宋体", ascii_font="Times New Roman", sz_pt=14.0, bold=is_en)
            v_run = make_run_xml(v, east_asia="宋体", ascii_font="Times New Roman", sz_pt=14.0, bold=False)
            trs.append(
                '<w:tr><w:trPr><w:trHeight w:val="520" w:hRule="atLeast"/></w:trPr>'
                f'<w:tc><w:tcPr><w:tcW w:w="{w_key}" w:type="dxa"/><w:vAlign w:val="bottom"/></w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="60" w:after="40"/><w:jc w:val="right"/></w:pPr>{k_run}</w:p></w:tc>'
                f'<w:tc><w:tcPr><w:tcW w:w="{w_val}" w:type="dxa"/>'
                '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>'
                '<w:vAlign w:val="bottom"/></w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="60" w:after="40"/><w:jc w:val="center"/></w:pPr>{v_run}</w:p></w:tc>'
                '</w:tr>'
            )
        tbl = (
            f'<w:tbl><w:tblPr><w:tblW w:w="{w_key + w_val}" w:type="dxa"/><w:jc w:val="center"/>'
            f'{no_bdr}<w:tblLayout w:type="fixed"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{w_key}"/><w:gridCol w:w="{w_val}"/></w:tblGrid>'
            f'{"".join(trs)}</w:tbl>'
        )
        self.body_parts.append(tbl)

    # -----------------------------------------------------------------
    # Package Assembly & Save
    # -----------------------------------------------------------------
    def save(self, out_path, final_sect_pr_xml):
        self._render_tocs()

        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{W_NS}" xmlns:m="{M_NS}" xmlns:r="{R_NS}" '
            f'xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">'
            '<w:body>'
            f'{"".join(self.body_parts)}'
            f'{final_sect_pr_xml}'
            '</w:body></w:document>'
        )

        # Styles XML
        styles_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault><w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>
      <w:sz w:val="24"/><w:szCs w:val="24"/>
      <w:lang w:val="en-US" w:eastAsia="zh-CN"/>
    </w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr>
      <w:spacing w:line="420" w:lineRule="exact"/>
    </w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:outlineLvl w:val="0"/><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:outlineLvl w:val="2"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
</w:styles>'''

        # Settings XML: enable evenAndOddHeaders + updateFields on open
        settings_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}">
  <w:evenAndOddHeaders/>
  <w:updateFields w:val="true"/>
  <w:defaultTabStop w:val="420"/>
  <w:characterSpacingControl w:val="compressPunctuation"/>
</w:settings>'''

        # Headers & Footers:
        # rId3 = empty header (front matter)
        # rId4 = empty footer (covers / declaration)
        # rId5 = odd body header ("天津农学院硕士学位论文", 五号宋体居中 + 底边框线)
        # rId6 = even body header (STYLEREF 1, 五号宋体居中 + 底边框线)
        # rId7 = centered PAGE footer (五号宋体居中)
        hdr_empty = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr xmlns:w="{W_NS}"><w:p/></w:hdr>'
        ftr_empty = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="{W_NS}"><w:p/></w:ftr>'

        hdr_odd = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{W_NS}">
  <w:p>
    <w:pPr>
      <w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/></w:pBdr>
      <w:jc w:val="center"/>
    </w:pPr>
    {make_run_xml("天津农学院硕士学位论文", east_asia="宋体", ascii_font="Times New Roman", sz_pt=10.5)}
  </w:p>
</w:hdr>'''

        hdr_even = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{W_NS}">
  <w:p>
    <w:pPr>
      <w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/></w:pBdr>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:r><w:fldChar w:fldCharType="begin"/></w:r>
    <w:r><w:instrText xml:space="preserve"> STYLEREF "Heading 1" </w:instrText></w:r>
    <w:r><w:fldChar w:fldCharType="separate"/></w:r>
    {make_run_xml("第一章 绪论", east_asia="宋体", ascii_font="Times New Roman", sz_pt=10.5)}
    <w:r><w:fldChar w:fldCharType="end"/></w:r>
  </w:p>
</w:hdr>'''

        ftr_page = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W_NS}">
  <w:p>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:r><w:fldChar w:fldCharType="begin"/></w:r>
    <w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>
    <w:r><w:fldChar w:fldCharType="separate"/></w:r>
    {make_run_xml("1", east_asia="宋体", ascii_font="Times New Roman", sz_pt=10.5)}
    <w:r><w:fldChar w:fldCharType="end"/></w:r>
  </w:p>
</w:ftr>'''

        # Relationships
        rels = [
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header_empty.xml"/>',
            '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer_empty.xml"/>',
            '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header_odd.xml"/>',
            '<Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header_even.xml"/>',
            '<Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer_page.xml"/>',
        ]
        exts = set()
        for rid, media_name, _bytes, ext in self.images:
            exts.add(ext)
            rels.append(
                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
            )

        doc_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(rels)}</Relationships>'
        )

        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>'
        )

        ct_defaults = [
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
        ]
        mime_map = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "gif": "image/gif"}
        for ext in sorted(exts):
            ct_defaults.append(f'<Default Extension="{ext}" ContentType="{mime_map.get(ext, "image/png")}"/>')

        content_types_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  {"".join(ct_defaults)}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/header_empty.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer_empty.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
  <Override PartName="/word/header_odd.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/header_even.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer_page.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>'''

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types_xml.encode("utf-8"))
            z.writestr("_rels/.rels", root_rels_xml.encode("utf-8"))
            z.writestr("word/document.xml", document_xml.encode("utf-8"))
            z.writestr("word/_rels/document.xml.rels", doc_rels_xml.encode("utf-8"))
            z.writestr("word/styles.xml", styles_xml.encode("utf-8"))
            z.writestr("word/settings.xml", settings_xml.encode("utf-8"))
            z.writestr("word/header_empty.xml", hdr_empty.encode("utf-8"))
            z.writestr("word/footer_empty.xml", ftr_empty.encode("utf-8"))
            z.writestr("word/header_odd.xml", hdr_odd.encode("utf-8"))
            z.writestr("word/header_even.xml", hdr_even.encode("utf-8"))
            z.writestr("word/footer_page.xml", ftr_page.encode("utf-8"))
            for _rid, media_name, img_bytes, _ext in self.images:
                z.writestr(f"word/media/{media_name}", img_bytes)
