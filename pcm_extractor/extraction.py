import re
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .config import (
    PCM_SECTIONS, MULTI_PAGE_SECTIONS, SECTION_DISPLAY_NAMES,
    FALLBACK_ZONES_RATIOS, FALLBACK_LMAX_RATIO, MIN_N_COLS,
    HEADER_KEYWORDS, RESET_SECTION_RE, MIN_CHARS_PER_PAGE,
    MIN_NUMERIC_RATIO, LEFT_DECO_RATIO, _NOISE_RE,
)
from .models import _ALPHA_RE
from .models import Token, PageLine, ParsedSection, parse_amount, normalize_text, _NUM_FRAG

log = logging.getLogger(__name__)


def _page_belongs_to_known_section(header_text: str) -> bool:
    norm = normalize_text(header_text)
    for _, pattern in PCM_SECTIONS:
        if pattern.search(norm) or pattern.search(header_text):
            return True
    return bool(RESET_SECTION_RE.search(norm))


def apply_fallback_zones(section_key: str, page_width: float) -> Tuple[List[float], float]:
    ratios = FALLBACK_ZONES_RATIOS.get(section_key, [0.60, 0.80])
    zones  = [round(r * page_width, 1) for r in ratios]
    lmax   = round(FALLBACK_LMAX_RATIO.get(section_key, 0.35) * page_width, 1)
    return zones, lmax


def _merge_num_fragments(raw: List[dict]) -> List[dict]:
    result, i = [], 0
    words = sorted(raw, key=lambda w: (round(float(w["top"])), float(w["x0"])))
    while i < len(words):
        w = dict(words[i])
        txt_clean = w["text"].strip().replace("†", "")
        if _NUM_FRAG.match(txt_clean) or _NUM_FRAG.match(w["text"].strip()):
            j = i + 1
            while j < len(words):
                nw = words[j]
                gap = float(nw["x0"]) - float(w["x1"])
                cur_has_sep  = any(c in w["text"] for c in (".", ","))
                nw_clean     = nw["text"].strip().replace("†", "")
                candidate    = (w["text"] + nw["text"]).replace("†", "").replace(" ", "")
                multi_sep    = candidate.count(".") + candidate.count(",") > 1
                parseable    = (
                    not cur_has_sep and not multi_sep
                    and (_NUM_FRAG.match(nw["text"].strip()) or _NUM_FRAG.match(nw_clean))
                    and parse_amount(candidate) is not None
                )
                if (abs(float(nw["top"]) - float(w["top"])) < 3.0
                        and 0 <= gap <= 7.0 and parseable):
                    w = {"text": w["text"] + nw["text"],
                         "x0": w["x0"], "x1": float(nw["x1"]),
                         "top": w["top"], "bottom": w.get("bottom", w["top"])}
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1
        result.append(w)
    return result


def _estimate_label_zone(header_words: List[dict], section_key: str, page_width: float) -> float:
    keywords = HEADER_KEYWORDS.get(section_key, [])
    if not keywords:
        return page_width * 0.35

    lines_by_y: Dict[int, List[dict]] = defaultdict(list)
    for w in header_words:
        bucket = int(round(float(w["top"]) / 4.0) * 4)
        lines_by_y[bucket].append(w)

    matches_x: List[float] = []
    for bucket, wlist in lines_by_y.items():
        wlist_sorted = sorted(wlist, key=lambda w: float(w["x0"]))
        line_text    = " ".join(w["text"] for w in wlist_sorted)
        line_text_n  = normalize_text(line_text)
        for kw_pat, _ in keywords:
            pat = re.compile(kw_pat, re.I)
            if pat.search(line_text_n) or pat.search(line_text):
                num_tokens = [w for w in wlist_sorted
                              if any(c.isdigit() for c in w["text"])
                              and float(w["x0"]) > page_width * 0.2]
                if num_tokens:
                    matches_x.append(float(min(num_tokens, key=lambda w: float(w["x0"]))["x0"]))
                else:
                    right_tokens = [w for w in wlist_sorted if float(w["x0"]) > page_width * 0.3]
                    if right_tokens:
                        w0 = right_tokens[0]
                        matches_x.append((float(w0["x0"]) + float(w0["x1"])) / 2)
                break

    if not matches_x:
        return page_width * 0.35
    return max(30.0, min(matches_x) - 10.0)


def calibrate_zones(page, section_key: str, page_width: float) -> Tuple[List[float], float, str]:
    n_min = MIN_N_COLS.get(section_key, 2)
    try:
        raw_words = page.extract_words(x_tolerance=3, y_tolerance=3,
                                       keep_blank_chars=False, use_text_flow=False)
    except Exception:
        z, lm = apply_fallback_zones(section_key, page_width)
        return z, lm, "fallback"

    if not raw_words:
        z, lm = apply_fallback_zones(section_key, page_width)
        return z, lm, "fallback"

    header_words   = [w for w in raw_words if float(w["top"]) < 200]
    label_zone_max = _estimate_label_zone(header_words, section_key, page_width)
    merged         = _merge_num_fragments(raw_words)

    numeric_x1s: List[float] = []
    for w in merged:
        txt = w["text"].strip()
        v   = parse_amount(txt)
        if v is None or not re.search(r"\d", txt):
            continue
        x0, x1 = float(w["x0"]), float(w["x1"])
        if x0 >= label_zone_max and x1 < page_width - 5:
            numeric_x1s.append(x1)

    if len(numeric_x1s) < n_min:
        z, lm = apply_fallback_zones(section_key, page_width)
        return z, lm, "fallback"

    DENSE_SECTIONS = {"tableau_provisions", "tableau_amortissements", "titres_participation"}
    if section_key in DENSE_SECTIONS:
        CLUSTER_TOL = 15.0
    elif page_width < 500:
        CLUSTER_TOL = 17.0
    elif page_width > 700:
        CLUSTER_TOL = 26.0
    else:
        CLUSTER_TOL = 22.0

    sorted_x1s = sorted(numeric_x1s)
    clusters: List[List[float]] = []
    for x in sorted_x1s:
        placed = False
        for c in clusters:
            if abs(x - float(np.mean(c))) < CLUSTER_TOL:
                c.append(x); placed = True; break
        if not placed:
            clusters.append([x])

    zones_all      = sorted([float(np.median(c)) for c in clusters])
    min_obs        = max(2, len(raw_words) // 60)
    zones_filtered = sorted([float(np.median(c)) for c in clusters if len(c) >= min_obs])

    if len(zones_filtered) >= n_min:
        zones = [round(z, 1) for z in zones_filtered]
    elif len(zones_all) >= n_min:
        zones = [round(z, 1) for z in zones_all]
    else:
        z, lm = apply_fallback_zones(section_key, page_width)
        return z, lm, "fallback"

    gap  = (zones[-1] - zones[0]) / max(len(zones) - 1, 1) if len(zones) > 1 else 90
    lmax = max(30.0, min(zones[0] - gap * 0.45, label_zone_max + 5))

    log.info(f"     Calibré ({section_key}) : {len(zones)} zones={[round(z) for z in zones]}, lmax={lmax:.0f}")
    return zones, lmax, "calibrated"


def extract_lines(page, page_num: int) -> List[PageLine]:
    raw = page.extract_words(x_tolerance=3, y_tolerance=2,
                              keep_blank_chars=False, use_text_flow=False)
    if not raw:
        return []
    pw = float(page.width)

    long_text_x0s = [
        float(w["x0"]) for w in raw
        if len(w["text"].strip()) >= 3 and not re.search(r"^\d", w["text"].strip())
    ]
    if long_text_x0s:
        min_text_x0 = min(long_text_x0s)
        adaptive_left_limit = max(pw * 0.02, min(pw * 0.08, min_text_x0 - 5.0))
    else:
        adaptive_left_limit = pw * LEFT_DECO_RATIO

    raw = [w for w in raw
           if not (len(w["text"].strip()) == 1
                   and w["text"].strip().isalpha()
                   and float(w["x0"]) < adaptive_left_limit)]
    merged = _merge_num_fragments(raw)
    y_tol  = 4.0
    ld: Dict[int, PageLine] = {}
    for w in merged:
        bucket = int(round(float(w["top"]) / y_tol) * y_tol)
        if bucket not in ld:
            ld[bucket] = PageLine(y=float(w["top"]), page=page_num)
        text = w["text"].strip()
        v    = parse_amount(text)
        is_n = v is not None and bool(re.search(r"\d", text))
        ld[bucket].tokens.append(Token(
            text=text, x0=float(w["x0"]), x1=float(w["x1"]),
            y=float(w["top"]), is_numeric=is_n, value=v))

    lines = sorted(ld.values(), key=lambda l: l.y)
    for ln in lines:
        ln.tokens.sort(key=lambda t: t.x0)

    result: List[PageLine] = []
    for ln in lines:
        ha = any(_ALPHA_RE.search(t.text) for t in ln.tokens)
        hn = any(t.is_numeric for t in ln.tokens)
        if not ha and hn and result and (ln.y - result[-1].y) < 15.0:
            result[-1].tokens.extend(ln.tokens)
            result[-1].tokens.sort(key=lambda t: t.x0)
            continue
        result.append(ln)

    final: List[PageLine] = []
    i = 0
    while i < len(result):
        ln = result[i]
        ha = any(_ALPHA_RE.search(t.text) for t in ln.tokens)
        hn = any(t.is_numeric for t in ln.tokens)
        if not ha and hn and i + 1 < len(result):
            nxt = result[i + 1]
            if nxt.y - ln.y < 15.0:
                nxt.tokens.extend(ln.tokens)
                nxt.tokens.sort(key=lambda t: t.x0)
                i += 1
                continue
        final.append(ln)
        i += 1
    return final


def page_has_data(lines: List[PageLine]) -> bool:
    total = sum(len(ln.tokens) for ln in lines)
    nums  = sum(1 for ln in lines for t in ln.tokens if t.is_numeric)
    return total > 0 and (nums / total) >= MIN_NUMERIC_RATIO


def extract_table_hybrid(page, section_key: str, page_width: float,
                         lines_fallback: List[PageLine]) -> Tuple[Optional[pd.DataFrame], str]:
    """[V35-A] Extraction hybride via pdfplumber.extract_table() si les bordures sont présentes."""
    try:
        table_settings = {
            "vertical_strategy": "lines", "horizontal_strategy": "lines",
            "snap_tolerance": 3, "join_tolerance": 3,
            "edge_min_length": 10, "min_words_vertical": 1, "min_words_horizontal": 1,
        }
        tables = page.extract_tables(table_settings)
        if not tables:
            return None, "fallback"

        best_table = max(tables, key=lambda t: sum(1 for row in t for c in row if c))
        if not best_table or len(best_table) < 2:
            return None, "fallback"

        rows_data = []
        for raw_row in best_table:
            if not any(c and str(c).strip() for c in raw_row):
                continue
            rows_data.append([str(cell or "").strip() or None for cell in raw_row])

        if not rows_data:
            return None, "fallback"

        n_numeric_table = sum(1 for row in rows_data for cell in row
                              if cell and parse_amount(cell) is not None)
        n_numeric_words = sum(1 for ln in lines_fallback for t in ln.tokens if t.is_numeric)

        if n_numeric_table > n_numeric_words * 1.15 and n_numeric_table >= 3:
            n_cols = max(len(r) for r in rows_data)
            col_text_len = [0] * n_cols
            for row in rows_data:
                for ci, cell in enumerate(row):
                    if cell and parse_amount(cell) is None:
                        col_text_len[ci] += len(cell)
            label_col_idx = col_text_len.index(max(col_text_len)) if col_text_len else 0

            col_names_raw = [f"Col_{i}" for i in range(n_cols)]
            col_names_raw[label_col_idx] = "Libellé"

            df_rows = []
            for raw_row in rows_data:
                row_dict: Dict[str, Any] = {c: None for c in col_names_raw}
                for ci, cell in enumerate(raw_row):
                    if ci < n_cols:
                        col_name = col_names_raw[ci]
                        if ci == label_col_idx:
                            row_dict[col_name] = cell
                        else:
                            v = parse_amount(cell) if cell else None
                            row_dict[col_name] = v if v is not None else cell
                df_rows.append(row_dict)

            df = pd.DataFrame(df_rows).dropna(how="all").reset_index(drop=True)
            log.info(f"    [V35-A] Table structurée ({section_key}) : "
                     f"{n_numeric_table} valeurs vs {n_numeric_words} mots")
            return df, "table"

        return None, "fallback"
    except Exception as e:
        log.debug(f"    [V35-A] extract_table_hybrid échec ({section_key}) : {e}")
        return None, "fallback"


def detect_pdf_type(pdf_path: str) -> str:
    import pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:
                text = page.extract_text() or ""
                if len(text.strip()) >= MIN_CHARS_PER_PAGE:
                    return "text"
        return "scanned"
    except Exception:
        return "unknown"


def detect_exercise_year(pdf_path: str) -> str:
    import pdfplumber
    patterns = [
        (re.compile(r"du\s*:?\s*(\d{2}/\d{2}/\d{4})\s+au\s+(\d{2}/\d{2}/\d{4})", re.I),
         lambda m: f"{m.group(1)} au {m.group(2)}"),
        (re.compile(r"clos\s+le\s*:?\s*(\d{2}/\d{2}/\d{4})", re.I),
         lambda m: m.group(1).split("/")[-1]),
        (re.compile(r"(?:31/12|30/06|31/03|30/09)/(\d{4})"),
         lambda m: m.group(1)),
        (re.compile(r"(\d{4})\s*/\s*(\d{4})"),
         lambda m: f"{m.group(1)}/{m.group(2)}"),
    ]
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:5]:
                text = page.extract_text() or ""
                for pat, extractor in patterns:
                    m = pat.search(text)
                    if m:
                        result = extractor(m)
                        log.info(f"   Exercice : {result}")
                        return result
    except Exception:
        pass
    return "N/A"


def detect_section(text: str) -> Optional[str]:
    norm = normalize_text(text)
    for key, pattern in PCM_SECTIONS:
        if pattern.search(norm) or pattern.search(text):
            return key
    return None


def _assign(x1: float, zones: List[float]) -> int:
    if not zones:
        return 0
    dists = [abs(x1 - z) for z in zones]
    best  = int(np.argmin(dists))
    if len(zones) > 1:
        sorted_idx = sorted(range(len(dists)), key=lambda i: dists[i])
        if len(sorted_idx) >= 2 and abs(dists[sorted_idx[0]] - dists[sorted_idx[1]]) < 8.0:
            best = sorted_idx[0]
    return best
