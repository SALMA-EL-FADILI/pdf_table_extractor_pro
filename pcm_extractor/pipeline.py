import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd

from .config import (
    PCM_SECTIONS, MULTI_PAGE_SECTIONS, SECTION_DISPLAY_NAMES, RESET_SECTION_RE,
)
from .models import ParsedSection, normalize_text
from .extraction import (
    extract_lines, calibrate_zones, page_has_data, extract_table_hybrid,
    detect_section, _page_belongs_to_known_section,
)
from .parsers import PARSER_MAP

log = logging.getLogger(__name__)


def extract_all_sections(pdf_path: str) -> List[ParsedSection]:
    import pdfplumber

    sections: List[ParsedSection] = []
    current_section: Optional[str]         = None
    current_lines:   List                  = []
    current_zones:   Optional[List[float]] = None
    current_lmax:    float                 = 200.0
    current_pw:      float                 = 595.0
    current_page:    int                   = 0
    current_page_obj: Optional[Any]        = None

    def flush_section():
        nonlocal current_section, current_lines, current_zones, current_lmax, current_page_obj
        if current_section and current_lines:
            cls    = PARSER_MAP.get(current_section, PARSER_MAP.get("bilan_actif"))
            parser = cls(current_lines, current_section, current_lmax, current_pw, current_zones)
            df     = parser.parse()

            if (current_page_obj is not None
                    and current_section not in MULTI_PAGE_SECTIONS
                    and df is not None and not df.empty):
                df_hybrid, mode = extract_table_hybrid(
                    current_page_obj, current_section, current_pw, current_lines
                )
                if df_hybrid is not None and not df_hybrid.empty:
                    def _count_num(d: pd.DataFrame) -> int:
                        return sum(
                            1 for c in d.columns if c != "Libellé"
                            for v in d[c]
                            if v is not None and not (isinstance(v, float) and pd.isna(v))
                            and isinstance(v, (int, float))
                        )
                    if _count_num(df_hybrid) > _count_num(df):
                        df = df_hybrid

            if df is not None and not df.empty:
                display = SECTION_DISPLAY_NAMES.get(current_section, current_section)
                sections.append(ParsedSection(
                    section_key=current_section,
                    display_name=display,
                    page=current_page,
                    dataframe=df,
                ))
                log.info(f"   {display} P{current_page} → {len(df)} lignes")
        current_section  = None
        current_lines    = []
        current_zones    = None
        current_page_obj = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pnum, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if len(text.strip()) < 20:
                    continue

                pw    = float(page.width)
                lines = extract_lines(page, pnum)

                first_lines_text = " ".join(ln.full_text() for ln in lines[:6]).upper()
                first_lines_norm = normalize_text(first_lines_text)
                is_known_reset   = RESET_SECTION_RE.search(first_lines_norm)
                is_any_pcm       = any(
                    pat.search(first_lines_norm) or pat.search(first_lines_text)
                    for _, pat in PCM_SECTIONS
                )

                if (current_section in MULTI_PAGE_SECTIONS
                        and not is_any_pcm and not is_known_reset
                        and len(first_lines_text.strip()) > 20):
                    log.info(f"    P{pnum} hors-périmètre PCM → reset [{current_section}]")
                    flush_section(); continue

                if current_section in MULTI_PAGE_SECTIONS and is_known_reset:
                    flush_section(); continue

                header_text = " ".join(ln.full_text() for ln in lines[:8])
                new_sec = detect_section(header_text)

                if new_sec and new_sec != current_section:
                    flush_section()
                    current_section  = new_sec
                    current_page     = pnum
                    current_pw       = pw
                    current_page_obj = page
                    zones, lmax, mode = calibrate_zones(page, new_sec, pw)
                    current_zones    = zones
                    current_lmax     = lmax
                    log.info(f"   P{pnum} → [{new_sec}] zones={mode}")

                if current_section:
                    if not page_has_data(lines) and current_section not in MULTI_PAGE_SECTIONS:
                        continue
                    if (current_section not in MULTI_PAGE_SECTIONS
                            and not new_sec
                            and not _page_belongs_to_known_section(header_text)
                            and pnum > current_page):
                        log.info(f"    P{pnum} : page annexe, arrêt [{current_section}]")
                        flush_section(); continue
                    current_lines.extend(lines)
                elif current_section in MULTI_PAGE_SECTIONS:
                    current_lines.extend(lines)

            flush_section()
    except Exception as e:
        log.error(f" Erreur extraction : {e}", exc_info=True)

    return sections


def _repair_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """[P21] Fusionne les libellés fragmentés par une coupure de page."""
    if df.empty or len(df) < 2:
        return df
    lbl_col  = "Libellé" if "Libellé" in df.columns else df.columns[0]
    val_cols = [c for c in df.columns if c != lbl_col]
    if not val_cols: return df

    rows = df.to_dict("records"); result = []; i = 0
    while i < len(rows):
        row = rows[i]
        lbl = row.get(lbl_col)
        has_label = lbl and str(lbl).strip() not in ("", "nan", "None")
        has_vals  = any(
            row.get(c) is not None
            and not (isinstance(row.get(c), float) and pd.isna(row.get(c)))
            for c in val_cols
        )
        if has_label and not has_vals and i + 1 < len(rows):
            nxt = rows[i + 1]
            nxt_lbl = nxt.get(lbl_col)
            nxt_has_label = nxt_lbl and str(nxt_lbl).strip() not in ("", "nan", "None")
            nxt_has_vals  = any(
                nxt.get(c) is not None
                and not (isinstance(nxt.get(c), float) and pd.isna(nxt.get(c)))
                for c in val_cols
            )
            if nxt_has_vals and not nxt_has_label:
                merged_row = {lbl_col: lbl}
                for c in val_cols: merged_row[c] = nxt.get(c)
                result.append(merged_row); i += 2; continue
        result.append(row); i += 1
    return pd.DataFrame(result).reset_index(drop=True)


def merge_sections(sections: List[ParsedSection]) -> List[ParsedSection]:
    groups: Dict[str, List[ParsedSection]] = defaultdict(list)
    for s in sections:
        groups[s.section_key].append(s)
    merged: List[ParsedSection] = []
    for key in [k for k, _ in PCM_SECTIONS]:
        if key not in groups: continue
        parts = groups[key]
        if len(parts) == 1:
            merged.append(parts[0])
        else:
            dfs = [p.dataframe for p in parts if p.dataframe is not None and not p.dataframe.empty]
            if not dfs: continue
            combined = pd.concat(dfs, ignore_index=True).dropna(how="all")
            combined = _repair_split_labels(combined)
            merged.append(ParsedSection(
                section_key=parts[0].section_key,
                display_name=parts[0].display_name,
                page=parts[0].page,
                dataframe=combined,
            ))
            log.info(f"   Merge {key} : {len(parts)} pages → {len(combined)} lignes")
    return merged


def post_clean(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    df = df.copy()
    for col in df.columns:
        if col != "Libellé":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(how="all")
    if "Libellé" in df.columns:
        df["Libellé"] = df["Libellé"].fillna("").astype(str).str.strip()
    return df.reset_index(drop=True) if not df.empty else None
