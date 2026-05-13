import re
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    MULTI_PAGE_SECTIONS, RESET_SECTION_RE, _NOISE_RE,
)
from .models import _ALPHA_RE
from .models import PageLine, ParsedSection, parse_amount, normalize_text
from .extraction import _assign, apply_fallback_zones

log = logging.getLogger(__name__)

_CGNC_PASSIF_LABELS: List[str] = [
    "CAPITAUX PROPRES", "Capital social ou personnel",
    "Actionnaires capital souscrit non appelé",
    "Primes d'émission de fusion d'apport", "Ecarts de réévaluation",
    "Réserve légale", "Autres réserves", "Report à nouveau",
    "Résultats nets en instance d'affectation", "RÉSULTAT NET DE L'EXERCICE",
    "Total des capitaux propres", "CAPITAUX PROPRES ASSIMILES",
    "Subventions d'investissement", "Provisions réglementées",
    "DETTES DE FINANCEMENT", "Emprunts obligataires",
    "Autres dettes de financement",
    "PROVISIONS DURABLES POUR RISQUES ET CHARGES",
    "Provisions pour risques", "Provisions pour charges",
    "ECARTS DE CONVERSION PASSIF", "Augmentation des créances immobilisées",
    "Diminution des dettes de financement", "FINANCEMENT PERMANENT",
    "DETTES DU PASSIF CIRCULANT", "Fournisseurs et comptes rattachés",
    "Clients créditeurs avances et acomptes", "Personnel", "Organismes sociaux",
    "Etat", "Comptes d'associés", "Autres créanciers",
    "Comptes de régularisation Passif",
    "AUTRES PROVISIONS POUR RISQUES ET CHARGES",
    "ECARTS DE CONVERSION PASSIF Elements circulants",
    "PASSIF CIRCULANT", "TRESORERIE PASSIF", "Crédits d'escompte",
    "Crédits de trésorerie", "Banques soldes créditeurs", "TOTAL GENERAL",
]


def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _repair_truncated_label(label: str, vocabulary: List[str],
                             max_prefix_missing: int = 3,
                             max_lev_ratio: float = 0.35) -> str:
    if not label or len(label) < 3:
        return label
    label_norm = normalize_text(label).upper().strip()
    first_word = label_norm.split()[0] if label_norm.split() else ""
    is_likely_truncated = (
        len(first_word) <= 5
        and not any(normalize_text(v).upper().startswith(first_word) for v in vocabulary)
    )
    if not is_likely_truncated:
        return label

    best_match: Optional[str] = None
    best_score = float("inf")
    for ref in vocabulary:
        ref_norm = normalize_text(ref).upper()
        for cut in range(1, min(max_prefix_missing + 1, len(ref_norm))):
            ref_suffix = ref_norm[cut:]
            if len(ref_suffix) < 4:
                continue
            dist = _levenshtein(label_norm, ref_suffix)
            ratio = dist / max(len(label_norm), len(ref_suffix))
            if ratio < best_score and ratio <= max_lev_ratio:
                best_score = ratio
                best_match = ref
    if best_match:
        log.info(f"    [P23a] Label réparé: {repr(label)} → {repr(best_match)} (score={best_score:.2f})")
        return best_match
    return label


def _deduplicate_consecutive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """[V36-8] Déduplique les libellés consécutifs identiques."""
    if df is None or df.empty:
        return df
    lbl_col = "Libellé" if "Libellé" in df.columns else df.columns[0]
    val_cols = [c for c in df.columns if c != lbl_col]
    if not val_cols:
        return df

    def _has_vals(row):
        return any(
            row.get(c) is not None
            and not (isinstance(row.get(c), float) and pd.isna(row.get(c)))
            for c in val_cols
        )

    rows = df.to_dict("records")
    result: List[dict] = []
    i = 0; removed = 0
    while i < len(rows):
        row = rows[i]
        lbl = normalize_text(str(row.get(lbl_col) or "")).upper().strip()
        if i + 1 < len(rows):
            nxt = rows[i + 1]
            nxt_lbl = normalize_text(str(nxt.get(lbl_col) or "")).upper().strip()
            if lbl and lbl == nxt_lbl and lbl not in ("", "NONE", "NAN"):
                row_has = _has_vals(row); nxt_has = _has_vals(nxt)
                if row_has and not nxt_has:
                    result.append(row); i += 2; removed += 1; continue
                elif nxt_has and not row_has:
                    result.append(nxt); i += 2; removed += 1; continue
                else:
                    result.append(row); i += 2; removed += 1; continue
        result.append(row)
        i += 1
    if removed > 0:
        log.info(f"    [V36-8] {removed} doublon(s) consécutif(s) supprimé(s)")
    return pd.DataFrame(result).reset_index(drop=True)


class PCMParser:
    COL_NAMES: List[str] = []

    def __init__(self, lines: List[PageLine], sk: str, lmax: float,
                 pw: float, zones: Optional[List[float]] = None):
        self.lines  = lines
        self.sk     = sk
        self.lmax   = lmax
        self.pw     = pw
        if zones is not None:
            self.zones = zones
        else:
            self.zones, _ = apply_fallback_zones(sk, pw)
        self.n_cols = len(self.zones) if self.zones else max(len(self.COL_NAMES), 2)
        self._merge_flags: List[bool] = self._build_merge_flags()

    def _build_merge_flags(self) -> List[bool]:
        n = len(self.lines)
        flags = [False] * n
        for i, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 775:
                continue
            for j in range(i + 1, n):
                nxt = self.lines[j]
                gap = nxt.y - ln.y
                if gap > 10.0: break
                if gap < 0: continue
                has_val = any(t.is_numeric and t.x0 >= self.lmax for t in nxt.tokens)
                has_lbl = any(_ALPHA_RE.search(t.text) and t.x0 < self.lmax for t in nxt.tokens)
                if has_val and not has_lbl:
                    flags[i] = True; break
                if has_lbl: break
        return flags

    def _is_section_header(self, label: str) -> bool:
        if not label: return False
        words = label.replace("•", "").strip().split()
        upper = [w for w in words if w and w[0].isupper() and len(w) > 2]
        return len(upper) >= 2 and len(label) > 8

    def _noise(self, label: str) -> bool:
        return bool(_NOISE_RE.match(label)) if label else False

    _HARD_NOISE_RE = re.compile(r'^[A-Z](\s[A-Z]){1,6}$', re.I)

    def _hard_noise(self, label: str) -> bool:
        return bool(self._HARD_NOISE_RE.match(label)) if label else False

    def _truncate_at_reset(self, lines: List[PageLine]) -> List[PageLine]:
        """[P29-3] Coupe les lignes dès qu'un titre de tableau annexe est détecté."""
        for i, ln in enumerate(lines):
            line_text = " ".join(t.text for t in ln.tokens)
            line_norm = normalize_text(line_text)
            if RESET_SECTION_RE.search(line_norm) or RESET_SECTION_RE.search(line_text):
                if i > 0:
                    return lines[:i]
        return lines

    def _col_names(self) -> List[str]:
        cn = list(self.COL_NAMES)
        while len(cn) < self.n_cols:
            cn.append(f"Col_{len(cn)+1}")
        return cn[:self.n_cols]

    def _emit_label_row(self, label: str, cn: List[str]) -> dict:
        row = {"Libellé": label}
        for c in cn: row[c] = None
        return row

    def parse(self) -> pd.DataFrame:
        if self.sk not in MULTI_PAGE_SECTIONS:
            self.lines = self._truncate_at_reset(self.lines)

        cn = self._col_names(); rows: List[dict] = []; acc = ""
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 775: continue
            label = ln.label_text(self.lmax)
            vtoks = ln.value_tokens(self.lmax)
            if label and not vtoks:
                if self._noise(label): continue
                should_merge = self._merge_flags[idx]
                if should_merge:
                    if acc and self._is_section_header(label):
                        if not self._noise(acc):
                            rows.append(self._emit_label_row(acc, cn))
                        acc = label
                    else:
                        acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            values: Dict[int, float] = {}
            for t in vtoks:
                idx_col = _assign(t.x1, self.zones) if self.zones else 0
                if idx_col < self.n_cols and idx_col not in values:
                    values[idx_col] = t.value
            if not label and not values: continue
            if label and self._noise(label) and not values: continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc):
            rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)


class BilanActifParser(PCMParser):
    COL_NAMES = ["Brut", "Amort. & Prov.", "Net Exercice N", "Net Exercice N-1"]
    LMAX_TOL_FACTOR = 0.5; LMAX_TOL_MIN = 3.0; LMAX_TOL_MAX = 12.0

    def _compute_lmax_tolerance(self) -> float:
        x0_vals = [
            t.x0 for ln in self.lines for t in ln.tokens
            if t.is_numeric and self.lmax - 20 <= t.x0 <= self.lmax + 5
        ]
        if len(x0_vals) < 3: return 5.0
        tol = self.LMAX_TOL_FACTOR * float(np.std(x0_vals))
        return max(self.LMAX_TOL_MIN, min(self.LMAX_TOL_MAX, tol))

    def parse(self) -> pd.DataFrame:
        zones = self.zones; lmax = self.lmax; cn = self._col_names()
        lmax_relaxed = lmax - self._compute_lmax_tolerance()
        val_lines: List[Tuple] = []; lbl_only: Dict[float, str] = {}
        for ln in self.lines:
            if ln.y < 90 or ln.y > 790: continue
            label = ln.label_text(lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= lmax_relaxed]
            if vtoks:
                values: Dict[int, float] = {}
                for t in vtoks:
                    idx = _assign(t.x1, zones)
                    if idx < self.n_cols and idx not in values: values[idx] = t.value
                lbl = label if (label and not self._noise(label)) else ""
                val_lines.append((ln.y, lbl, values))
            elif label and not self._noise(label):
                lbl_only[ln.y] = label

        used: set = set(); result: List[Tuple] = []
        for vy, vlbl, values in sorted(val_lines, key=lambda x: x[0]):
            final_lbl = vlbl
            if not final_lbl:
                best_y, best_d = None, 999.0
                for ly, lbl in lbl_only.items():
                    if ly in used: continue
                    d = abs(ly - vy)
                    if d <= 10 and d < best_d: best_y, best_d = ly, d
                if best_y is not None:
                    final_lbl = lbl_only[best_y]; used.add(best_y)
            result.append((vy, final_lbl or None, values))

        orphan_labels = {ly: lbl for ly, lbl in lbl_only.items() if ly not in used}
        rows = []
        res_iter  = iter(sorted(result, key=lambda x: x[0]))
        orph_iter = iter(sorted(orphan_labels.items()))
        try:
            vy, vlbl, values = next(res_iter)
            for oly, olbl in orph_iter:
                if oly < vy:
                    rows.append({"Libellé": olbl, **{c: None for c in cn}})
                else:
                    row = {"Libellé": vlbl}
                    for i, c in enumerate(cn): row[c] = values.get(i)
                    rows.append(row)
                    try: vy, vlbl, values = next(res_iter)
                    except StopIteration: break
            else:
                row_else = {"Libellé": vlbl}
                for i_e, c_e in enumerate(cn): row_else[c_e] = values.get(i_e)
                rows.append(row_else)
                for vy2, vlbl2, values2 in res_iter:
                    row = {"Libellé": vlbl2}
                    for i, c in enumerate(cn): row[c] = values2.get(i)
                    rows.append(row)
        except StopIteration:
            pass
        if not rows:
            for _, vlbl, values in sorted(result, key=lambda x: x[0]):
                row = {"Libellé": vlbl}
                for i, c in enumerate(cn): row[c] = values.get(i)
                rows.append(row)
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        for c in cn:
            if c not in df.columns: df[c] = None
        return df[["Libellé"] + cn].dropna(how="all").reset_index(drop=True)


class BilanPassifParser(PCMParser):
    COL_NAMES = ["Exercice N", "Exercice N-1"]

    _SECTION_TOTAL_MAP = {
        normalize_text("CAPITAUX PROPRES").upper():           "Total des capitaux propres",
        normalize_text("FINANCEMENT PERMANENT").upper():      "FINANCEMENT PERMANENT",
        normalize_text("DETTES DU PASSIF CIRCULANT").upper(): "PASSIF CIRCULANT",
        normalize_text("PASSIF CIRCULANT").upper():           "PASSIF CIRCULANT",
        normalize_text("TRESORERIE PASSIF").upper():          "TRESORERIE PASSIF",
    }

    def parse(self) -> pd.DataFrame:
        df = super().parse()
        if df.empty: return df
        lbl_col  = "Libellé"
        num_cols = [c for c in df.columns if c != lbl_col]

        # [V36-5] Supprime les entiers négatifs ]-20,0[ sur lignes avec "(1)"/"(2)" (références de notes)
        for idx, row in df.iterrows():
            lbl = str(row.get(lbl_col) or "")
            if re.search(r'\([12]\)', lbl):
                for nc in num_cols:
                    v = row.get(nc)
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        try:
                            fv = float(v)
                            if -20 <= fv < 0 and fv == int(fv):
                                df.at[idx, nc] = None
                        except (ValueError, TypeError):
                            pass

        # [V36-9] Labellisation des sous-totaux orphelins (libellé vide + valeurs)
        if lbl_col in df.columns:
            df[lbl_col] = df[lbl_col].apply(
                lambda x: _repair_truncated_label(str(x), _CGNC_PASSIF_LABELS)
                if x and str(x) not in ("None", "nan") else x
            )
            last_section_total_key = None
            for idx, row in df.iterrows():
                lbl = row.get(lbl_col); lbl_str = str(lbl).strip() if lbl else ""
                lbl_norm = normalize_text(lbl_str).upper()
                has_vals = any(
                    row.get(nc) is not None
                    and not (isinstance(row.get(nc), float) and pd.isna(row.get(nc)))
                    for nc in num_cols
                )
                for key in self._SECTION_TOTAL_MAP:
                    if key in lbl_norm:
                        last_section_total_key = key; break
                if (not lbl_str or lbl_str in ("None", "nan")) and has_vals:
                    if last_section_total_key and last_section_total_key in self._SECTION_TOTAL_MAP:
                        candidate = self._SECTION_TOTAL_MAP[last_section_total_key]
                        df.at[idx, lbl_col] = candidate
                        last_section_total_key = None
        return df


class CPCParser(PCMParser):
    COL_NAMES = ["Propres exercice", "Exercices préc.", "Total N", "Total N-1"]
    N_FIXED_COLS = 4

    def __init__(self, lines, sk, lmax, pw, zones=None):
        super().__init__(lines, sk, lmax, pw, zones)
        self.n_cols = self.N_FIXED_COLS

    def _build_zone_col_map(self) -> Dict[int, int]:
        n = len(self.zones)
        if n >= 4: return {i: i for i in range(4)}
        if n <= 1: return {0: 2}

        zone_vals: Dict[int, List[float]] = defaultdict(list)
        for ln in self.lines:
            if ln.y < 90 or ln.y > 790: continue
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax
                     and t.value is not None and abs(t.value) > 10]
            if len(vtoks) < 2: continue
            for t in vtoks:
                zi = _assign(t.x1, self.zones)
                if zi < n: zone_vals[zi].append(t.value)

        if n == 2: return {0: 0, 1: 3}

        v0 = zone_vals.get(0, []); v1 = zone_vals.get(1, [])
        set0 = set(round(v, 0) for v in v0); set1 = set(round(v, 0) for v in v1)
        overlap_ratio = len(set0 & set1) / max(len(set0), len(set1), 1)

        if overlap_ratio >= 0.5:
            return {0: 0, 1: 2, 2: 3}
        else:
            return {0: 0, 1: 1, 2: 3}

    _CPC_NOISE = re.compile(
        r"^\s*(\(\s*[IVX]+\s*[+\-]\s*[IVX]+\s*\)|"
        r"total\s+des\s+(produits|charges)\s*[-–]\s*total|"
        r"\(total\s+des|Variation\s+de\s+stocks.*augmentation|"
        r"Achats\s+revendus.*consomm)", re.I)
    _CPC_RESULT_RE = re.compile(r"R.SULTAT\s+NET", re.I)

    def parse(self) -> pd.DataFrame:
        cn = self._col_names(); rows = []; acc = ""
        zone_col = self._build_zone_col_map()
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 775: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
            if vtoks and not label:
                if all(t.value is not None and 1 <= t.value <= 9 and
                       t.value == int(t.value) for t in vtoks): continue
            if label and not vtoks:
                if self._noise(label) or self._CPC_NOISE.match(label): continue
                should_merge = self._merge_flags[idx]
                if should_merge:
                    if acc and self._is_section_header(label):
                        if not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
                        acc = label
                    else:
                        acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            if label and self._CPC_NOISE.match(label) and not vtoks: continue
            values: Dict[int, float] = {}
            for t in vtoks:
                zone_idx = _assign(t.x1, self.zones)
                col_idx  = zone_col.get(zone_idx, zone_idx)
                if col_idx < self.N_FIXED_COLS and col_idx not in values:
                    values[col_idx] = t.value
            if not label and not values: continue
            if label and self._noise(label) and not values: continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        return self._fix_resultat_net_sign(
            pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)
        )

    def _fix_resultat_net_sign(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        num_cols = [c for c in df.columns if c != "Libellé"]
        if not num_cols: return df
        tot_col = next((c for c in num_cols if "total n" in c.lower() and "n-1" not in c.lower()),
                       num_cols[-1] if num_cols else None)
        if not tot_col: return df
        tp = tc = None
        for _, row in df.iterrows():
            lbl = normalize_text(str(row.get("Libellé", "") or "")).upper()
            v = pd.to_numeric(row.get(tot_col), errors="coerce")
            if pd.isna(v): continue
            if "TOTAL" in lbl and "PRODUIT" in lbl: tp = v
            elif "TOTAL" in lbl and "CHARGE" in lbl: tc = v
        if tp is None or tc is None: return df
        expected_sign = 1 if tp >= tc else -1
        for idx, row in df.iterrows():
            lbl = normalize_text(str(row.get("Libellé", "") or "")).upper()
            if self._CPC_RESULT_RE.search(lbl):
                v = pd.to_numeric(row.get(tot_col), errors="coerce")
                if pd.notna(v) and abs(v) > 0 and (1 if v > 0 else -1) != expected_sign:
                    df.at[idx, tot_col] = -v
        return df


class ESGParser(PCMParser):
    COL_NAMES = ["Exercice N", "Exercice N-1"]
    _EBE_RE = re.compile(r"\bEBE\b|\bEXCEDENT\s+BRUT\b", re.I)
    _IBE_RE = re.compile(r"\bIBE\b|\bINSUFFISANCE\s+BRUTE\b|\bINSUFFISANCE\b", re.I)

    def _resolve_ebe_ibe(self, df: pd.DataFrame) -> pd.DataFrame:
        """[P29-4] Normalise les libellés EBE/IBE mixtes."""
        if df.empty or "Libellé" not in df.columns: return df
        for idx, row in df.iterrows():
            lbl = str(row.get("Libellé", "") or "")
            lbl_norm = normalize_text(lbl).upper()
            if self._EBE_RE.search(lbl_norm) and self._IBE_RE.search(lbl_norm):
                cleaned = self._EBE_RE.sub("", lbl).strip(" /=()- ")
                if cleaned: df.at[idx, "Libellé"] = cleaned
        return df

    def parse(self) -> pd.DataFrame:
        cn = self._col_names(); rows = []; acc = ""
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 775: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax
                     and not (t.value is not None and abs(t.value) <= 9
                              and t.value == int(t.value) if t.value is not None else False)]
            if label and not vtoks:
                if self._noise(label): continue
                should_merge = self._merge_flags[idx]
                if should_merge:
                    if acc and self._is_section_header(label):
                        if not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
                        acc = label
                    else:
                        acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            values: Dict[int, float] = {}
            for t in vtoks:
                idx_col = _assign(t.x1, self.zones) if self.zones else 0
                if idx_col < self.n_cols and idx_col not in values: values[idx_col] = t.value
            if not label and not values: continue
            if label and self._noise(label) and not values: continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        return self._resolve_ebe_ibe(pd.DataFrame(rows).dropna(how="all").reset_index(drop=True))


class CAFParser(PCMParser):
    """[V36-1] Parseur CAF / Tableau Financement Partie II — détecte layout 2 ou 4 colonnes."""
    COL_NAMES_2 = ["Exercice N", "Exercice N-1"]
    COL_NAMES_4 = ["Emplois N", "Ressources N", "Emplois N-1", "Ressources N-1"]

    def _detect_4col_layout(self) -> bool:
        x1_vals: List[float] = []
        for ln in self.lines:
            if ln.y < 90 or ln.y > 790: continue
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax
                     and t.value is not None and abs(t.value) > 10]
            if len(vtoks) >= 2:
                for t in vtoks: x1_vals.append(t.x1)
        if len(x1_vals) < 6: return False
        CTOL = 18.0; clusters: List[List[float]] = []
        for x in sorted(x1_vals):
            placed = False
            for c in clusters:
                if abs(x - float(np.mean(c))) < CTOL: c.append(x); placed = True; break
            if not placed: clusters.append([x])
        n_clusters = sum(1 for c in clusters if len(c) >= 3)
        return n_clusters >= 3

    def _build_4col_split(self) -> Tuple[float, float]:
        x1_all: List[float] = []
        for ln in self.lines:
            if ln.y < 90 or ln.y > 790: continue
            for t in ln.tokens:
                if t.is_numeric and t.x0 >= self.lmax and t.value is not None and abs(t.value) > 10:
                    x1_all.append(t.x1)
        if len(x1_all) < 4:
            pw_half = self.pw * 0.65
            return pw_half, pw_half - 40.0
        CTOL = 18.0; clusters: List[List[float]] = []
        for x in sorted(x1_all):
            placed = False
            for c in clusters:
                if abs(x - float(np.mean(c))) < CTOL: c.append(x); placed = True; break
            if not placed: clusters.append([x])
        centroids = sorted([float(np.median(c)) for c in clusters if len(c) >= 2])
        if len(centroids) < 2:
            pw_half = self.pw * 0.65
            return pw_half, pw_half - 40.0
        gaps = [(centroids[i+1] - centroids[i], i) for i in range(len(centroids)-1)]
        max_gap_val, max_gap_idx = max(gaps, key=lambda g: g[0])
        split_N_N1 = (centroids[max_gap_idx] + centroids[max_gap_idx + 1]) / 2.0
        left_c  = [c for c in centroids if c < split_N_N1]
        right_c = [c for c in centroids if c >= split_N_N1]
        split_E_R_left = (
            (left_c[0] + left_c[1]) / 2.0 if len(left_c) >= 2
            else split_N_N1 - (max_gap_val / 2)
        )
        return split_N_N1, split_E_R_left

    def parse(self) -> pd.DataFrame:
        is_4col = self._detect_4col_layout()
        if is_4col:
            cn = list(self.COL_NAMES_4)
            split_N_N1, split_E_R = self._build_4col_split()
            split_E_R_right = split_N_N1 + (split_N_N1 - split_E_R)
            rows: List[dict] = []; acc = ""
            for idx, ln in enumerate(self.lines):
                if ln.y < 90 or ln.y > 790: continue
                label = ln.label_text(self.lmax)
                vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax
                         and t.value is not None and abs(t.value) > 0.001]
                if label and not vtoks:
                    if self._noise(label): continue
                    if self._merge_flags[idx]:
                        acc = (acc + " " + label).strip() if acc else label
                    else:
                        if acc and not self._noise(acc):
                            rows.append(self._emit_label_row(acc, cn)); acc = ""
                        rows.append(self._emit_label_row(label, cn))
                    continue
                if acc:
                    label = (acc + " " + label).strip() if label else acc; acc = ""
                if not label and not vtoks: continue
                if label and self._noise(label) and not vtoks: continue
                if label and self._hard_noise(label): continue
                values: Dict[int, float] = {}
                for t in vtoks:
                    x = t.x1
                    if x < split_N_N1:
                        col_idx = 0 if x <= split_E_R else 1
                    else:
                        col_idx = 2 if x <= split_E_R_right else 3
                    if col_idx not in values: values[col_idx] = t.value
                row = {"Libellé": label or None}
                for i, c in enumerate(cn): row[c] = values.get(i)
                rows.append(row)
            if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
            if not rows: return pd.DataFrame()
            return pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)
        else:
            cn = self.COL_NAMES_2; rows2: List[dict] = []; acc2 = ""
            for idx, ln in enumerate(self.lines):
                if ln.y < 90 or ln.y > 790: continue
                label = ln.label_text(self.lmax)
                vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
                if label and not vtoks:
                    if self._noise(label): continue
                    if self._merge_flags[idx]:
                        acc2 = (acc2 + " " + label).strip() if acc2 else label
                    else:
                        if acc2 and not self._noise(acc2):
                            rows2.append(self._emit_label_row(acc2, cn)); acc2 = ""
                        rows2.append(self._emit_label_row(label, cn))
                    continue
                if acc2:
                    label = (acc2 + " " + label).strip() if label else acc2; acc2 = ""
                if not label and not vtoks: continue
                if label and self._noise(label) and not vtoks: continue
                if label and self._hard_noise(label): continue
                values3: Dict[int, float] = {}
                for t in vtoks:
                    col_idx = _assign(t.x1, self.zones) if self.zones else 0
                    if col_idx < 2 and col_idx not in values3: values3[col_idx] = t.value
                row = {"Libellé": label or None}
                for i, c in enumerate(cn): row[c] = values3.get(i)
                rows2.append(row)
            if acc2 and not self._noise(acc2): rows2.append(self._emit_label_row(acc2, cn))
            if not rows2: return pd.DataFrame()
            return pd.DataFrame(rows2).dropna(how="all").reset_index(drop=True)


class FinancementParser(PCMParser):
    COL_NAMES = ["Exercice N (a)", "Exercice N-1 (b)", "Emplois (c)", "Ressources (d)"]

    def parse(self) -> pd.DataFrame:
        cn = self._col_names(); rows = []; acc = ""
        all_text = " ".join(t.text for ln in self.lines for t in ln.tokens).upper()
        if "SYNTHESE" in all_text or "MASSES" in all_text:
            cn = ["Exercice N (a)", "Exercice N-1 (b)", "Emplois (c)", "Ressources (d)"]
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 790: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
            if label and not vtoks:
                if self._noise(label): continue
                if self._merge_flags[idx]:
                    acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            if not label and not vtoks: continue
            if label and self._noise(label) and not vtoks: continue
            values: Dict[int, float] = {}
            for t in vtoks:
                idx_col = _assign(t.x1, self.zones)
                if idx_col < self.n_cols and idx_col not in values: values[idx_col] = t.value
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows); df.columns = ["Libellé"] + cn
        return df.dropna(how="all").reset_index(drop=True)


class ImmobilisationsParser(PCMParser):
    COL_NAMES    = ["Montant Début", "Acquisitions", "Production", "Cessions/Retraits", "Montant Fin"]
    N_FIXED_COLS = 5

    def _build_zone_col_map(self) -> Dict[int, int]:
        n = len(self.zones)
        if n >= 5: return {i: i for i in range(5)}
        if n <= 1: return {0: 4}

        zone_sums: Dict[int, float]  = defaultdict(float)
        zone_counts: Dict[int, int]  = defaultdict(int)
        for ln in self.lines:
            if ln.y < 90 or ln.y > 790: continue
            for t in ln.tokens:
                if t.is_numeric and t.x0 >= self.lmax and t.value is not None:
                    zi = _assign(t.x1, self.zones)
                    if zi < n: zone_sums[zi] += abs(t.value); zone_counts[zi] += 1

        if n == 2:
            z0, z1 = self.zones[0], self.zones[1]
            span_ratio = (z1 - z0) / max(self.pw, 1)
            if span_ratio >= 0.45:
                mid_values_x: List[float] = []
                for ln in self.lines:
                    if ln.y < 90 or ln.y > 790: continue
                    for t in ln.tokens:
                        if t.is_numeric and t.x0 >= self.lmax and t.value is not None:
                            if z0 + 5.0 < t.x1 < z1 - 15.0: mid_values_x.append(t.x1)
                if mid_values_x:
                    zone_mid = float(np.median(mid_values_x))
                elif any(ln.y <= 120 for ln in self.lines):
                    acq_header_x: List[float] = []
                    _ACQ_RE = re.compile(r"acquisit", re.I)
                    for ln in self.lines:
                        if ln.y > 120: continue
                        for t in ln.tokens:
                            if _ACQ_RE.search(t.text): acq_header_x.append((t.x0 + t.x1) / 2.0)
                    zone_mid = float(np.median(acq_header_x)) if acq_header_x else round(z0 + (z1 - z0) * 0.25, 1)
                else:
                    zone_mid = round(z0 + (z1 - z0) * 0.25, 1)
                self.zones = sorted([z0, round(zone_mid, 1), z1])
                return {0: 0, 1: 1, 2: 4}
            return {0: 0, 1: 4}
        elif n == 3:
            s0 = zone_sums.get(0, 0.0); s1 = zone_sums.get(1, 0.0); s2 = zone_sums.get(2, 0.0)
            if s0 > 0 and s2 > 0:
                ratio_ends = min(s0, s2) / max(s0, s2)
                if ratio_ends >= 0.3 and s1 <= max(s0, s2): return {0: 0, 1: 1, 2: 4}
            if zone_counts.get(1, 0) == 0: return {0: 0, 1: 1, 2: 4}
            return {0: 0, 1: 1, 2: 4}
        elif n == 4:
            return {0: 0, 1: 1, 2: 3, 3: 4}
        return {i: i for i in range(n)}

    def parse(self) -> pd.DataFrame:
        cn = self._col_names(); rows = []; acc = ""
        zone_col = self._build_zone_col_map()
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 790: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
            if label and not vtoks:
                if self._noise(label): continue
                if self._merge_flags[idx]:
                    acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            values: Dict[int, float] = {}
            for t in vtoks:
                zone_idx = _assign(t.x1, self.zones)
                col_idx  = zone_col.get(zone_idx, zone_idx)
                if col_idx < self.N_FIXED_COLS and col_idx not in values: values[col_idx] = t.value
            if not label and not values: continue
            if label and self._noise(label) and not values: continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)


class AmortissementsParser(PCMParser):
    COL_NAMES = ["Cumul début exercice", "Dotation exercice",
                 "Amort. sur sorties", "Cumul fin exercice"]


class ProvisionsParser(PCMParser):
    COL_NAMES = [
        "Début exercice",
        "Dot. Exploit.", "Dot. Fin.", "Dot. Non-Cour.",
        "Rep. Exploit.", "Rep. Fin.", "Rep. Non-Cour.",
        "Fin exercice",
    ]
    N_FIXED_COLS = 8

    def __init__(self, lines, sk, lmax, pw, zones=None):
        super().__init__(lines, sk, lmax, pw, zones)
        self.n_cols = self.N_FIXED_COLS

    def _build_zone_col_map(self) -> Dict[int, int]:
        n = len(self.zones)
        if n >= 7: return {i: i for i in range(7)}
        elif n == 6: return {0:0, 1:1, 2:2, 3:4, 4:5, 5:6, 6:7}
        elif n == 5: return {0:0, 1:1, 2:4, 3:5, 4:6, 5:7}
        elif n == 4: return {0:0, 1:1, 2:4, 3:7}
        return {0:0, 1:1, 2:7}

    def parse(self) -> pd.DataFrame:
        cn = self.COL_NAMES[:self.N_FIXED_COLS]; rows = []; acc = ""
        zone_col = self._build_zone_col_map()
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 790: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
            if label and not vtoks:
                if self._noise(label): continue
                if self._merge_flags[idx]:
                    acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            values: Dict[int, float] = {}
            for t in vtoks:
                zone_idx = _assign(t.x1, self.zones)
                col_idx  = zone_col.get(zone_idx, zone_idx)
                if col_idx < self.N_FIXED_COLS and col_idx not in values: values[col_idx] = t.value
            if not label and not values: continue
            if label and self._noise(label) and not values: continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)

        # [V36-2] Si "Fin exercice" vide mais dotations présentes → décalage +1
        col_fin = "Fin exercice"; col_dot = "Dot. Exploit."
        if col_fin in df.columns and col_dot in df.columns:
            sum_fin = sum(abs(float(v)) for v in df[col_fin]
                         if v is not None and not (isinstance(v, float) and pd.isna(v)))
            sum_dot = sum(abs(float(v)) for v in df[col_dot]
                         if v is not None and not (isinstance(v, float) and pd.isna(v)))
            if sum_fin < 1.0 and sum_dot > 0:
                new_zone_col = {zi: min(ci + 1, self.N_FIXED_COLS - 1)
                                for zi, ci in zone_col.items()}
                rows2: List[dict] = []; acc2 = ""
                for idx2, ln in enumerate(self.lines):
                    if ln.y < 90 or ln.y > 790: continue
                    label = ln.label_text(self.lmax)
                    vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
                    if label and not vtoks:
                        if self._noise(label): continue
                        if self._merge_flags[idx2]:
                            acc2 = (acc2 + " " + label).strip() if acc2 else label
                        else:
                            if acc2 and not self._noise(acc2):
                                rows2.append(self._emit_label_row(acc2, cn)); acc2 = ""
                            rows2.append(self._emit_label_row(label, cn))
                        continue
                    if acc2:
                        label = (acc2 + " " + label).strip() if label else acc2; acc2 = ""
                    values2: Dict[int, float] = {}
                    for t in vtoks:
                        col_idx = new_zone_col.get(_assign(t.x1, self.zones), 0)
                        if col_idx < self.N_FIXED_COLS and col_idx not in values2:
                            values2[col_idx] = t.value
                    if not label and not values2: continue
                    if label and self._noise(label) and not values2: continue
                    if label and self._hard_noise(label): continue
                    row2 = {"Libellé": label or None}
                    for i, c in enumerate(cn): row2[c] = values2.get(i)
                    rows2.append(row2)
                if acc2 and not self._noise(acc2): rows2.append(self._emit_label_row(acc2, cn))
                if rows2:
                    df = pd.DataFrame(rows2).dropna(how="all").reset_index(drop=True)
        return df


class CreancesParser(PCMParser):
    COL_NAMES = ["Total", "Plus d'1 an", "Moins d'1 an", "Échues"]
    def parse(self) -> pd.DataFrame:
        return _deduplicate_consecutive_labels(super().parse())


class DettesParser(PCMParser):
    COL_NAMES = ["Total", "Plus d'1 an", "Moins d'1 an", "Échues"]
    def parse(self) -> pd.DataFrame:
        return _deduplicate_consecutive_labels(super().parse())


class TitresParticipationParser(PCMParser):
    COL_NAMES = ["Début exercice", "Acquisitions", "Cessions/Retraits",
                 "Cumul amort.", "Net N", "Net N-1"]

    def parse(self) -> pd.DataFrame:
        df = super().parse()
        if df is None or df.empty: return df
        num_cols = [c for c in df.columns if c != "Libellé"]

        def _is_col_index_row(row: pd.Series) -> bool:
            """[V36-3] Détecte les lignes contenant des numéros de colonnes [1,15]."""
            vals = []
            for nc in num_cols:
                v = row.get(nc)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    try:
                        fv = float(v)
                        if 1.0 <= fv <= 15.0 and fv == int(fv): vals.append(int(fv))
                        else: return False
                    except (ValueError, TypeError): return False
            if len(vals) < 2: return False
            vals_sorted = sorted(vals)
            return all(vals_sorted[i+1] - vals_sorted[i] <= 2 for i in range(len(vals_sorted)-1))

        mask = df.apply(_is_col_index_row, axis=1)
        if mask.sum() > 0:
            df = df[~mask].reset_index(drop=True)
        return df


class DetailCPCParser(PCMParser):
    COL_NAMES = ["Exercice N", "Exercice N-1"]

    def parse(self) -> pd.DataFrame:
        df = super().parse()
        if df is None or df.empty: return df
        col_n = "Exercice N"; col_n1 = "Exercice N-1"; col_lbl = "Libellé"
        if col_n not in df.columns: return df

        _TOTAL_MARKER_RE = re.compile(r"\bTOTAL\b|\bSOUS.TOTAL\b|\bSOMMAIRE\b", re.I)
        total_indices = [i for i, row in df.iterrows()
                         if _TOTAL_MARKER_RE.search(str(row.get(col_lbl) or ""))]
        reconstructed = 0; prev_total_idx = -1

        for tidx in total_indices:
            val_n  = df.at[tidx, col_n]
            val_n1 = df.at[tidx, col_n1] if col_n1 in df.columns else None
            n_is_none  = (val_n is None or (isinstance(val_n, float) and pd.isna(val_n)))
            n1_present = (val_n1 is not None and not (isinstance(val_n1, float) and pd.isna(val_n1)))

            if n_is_none and n1_present:
                detail_n_vals: List[float] = []; detail_n1_vals: List[float] = []
                for didx in range(prev_total_idx + 1, tidx):
                    if didx not in df.index: continue
                    vn = df.at[didx, col_n]
                    if vn is not None and not (isinstance(vn, float) and pd.isna(vn)):
                        try: detail_n_vals.append(abs(float(vn)))
                        except (ValueError, TypeError): pass
                    if col_n1 in df.columns:
                        vn1 = df.at[didx, col_n1]
                        if vn1 is not None and not (isinstance(vn1, float) and pd.isna(vn1)):
                            try: detail_n1_vals.append(abs(float(vn1)))
                            except (ValueError, TypeError): pass

                if detail_n_vals:
                    sum_n  = sum(detail_n_vals)
                    sum_n1 = sum(detail_n1_vals) if detail_n1_vals else 0.0
                    count_n  = len(detail_n_vals); count_n1 = len(detail_n1_vals) if detail_n1_vals else 0
                    # [V36-6] Deux critères d'incomplétude
                    incomplete = (sum_n1 > 0 and sum_n < sum_n1 * 0.30) or \
                                 (count_n1 > 1 and count_n < count_n1 * 0.50)
                    if not incomplete:
                        df.at[tidx, col_n] = sum(
                            float(df.at[d, col_n]) for d in range(prev_total_idx + 1, tidx)
                            if d in df.index and df.at[d, col_n] is not None
                            and not (isinstance(df.at[d, col_n], float) and pd.isna(df.at[d, col_n]))
                        )
                        reconstructed += 1
            prev_total_idx = tidx

        if reconstructed > 0:
            log.info(f"    [P31-3] DetailCPC : {reconstructed} total(aux) Exercice N reconstruit(s)")
        return df


class ResultatFiscalParser(PCMParser):
    COL_NAMES    = ["Bénéfice net", "Déficit net"]
    _BENEFICE_RE = re.compile(r"BENEFICE|BÉNÉFICE|PROFIT|EXCEDENT|TOTAL\s+\d", re.I)
    _DEFICIT_RE  = re.compile(r"DEFICIT|DÉFICIT|PERTE", re.I)

    def _label_col_hint(self, label: str) -> Optional[int]:
        lbl_n = normalize_text(label).upper()
        if self._BENEFICE_RE.search(lbl_n): return 0
        if self._DEFICIT_RE.search(lbl_n):  return 1
        return None

    def parse(self) -> pd.DataFrame:
        cn = self._col_names(); rows = []; acc = ""
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 790: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric and t.x0 >= self.lmax]
            if label and not vtoks:
                if self._noise(label): continue
                if self._merge_flags[idx]:
                    acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            hint = self._label_col_hint(label) if label else None
            values: Dict[int, float] = {}
            for t in vtoks:
                if hint is not None and len(vtoks) == 1: idx_col = hint
                elif len(self.zones) < 2:
                    idx_col = 0 if (t.value is not None and t.value >= 0) else 1
                else: idx_col = _assign(t.x1, self.zones)
                if idx_col < len(cn) and idx_col not in values:
                    values[idx_col] = abs(t.value) if t.value is not None else t.value
            if not label and not values: continue
            if label and self._noise(label) and not values: continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)

        _REINTEGRATION_RE = re.compile(
            r"r[ée]int[ée]gra|reintegr|r\.i\.\s|r\.i\b|"
            r"d[ée]ductions?\s+fiscales?|reports?\s+d[ée]ficitaires?", re.I)
        _TOTAL_FISCAL_RE = re.compile(r"[.\s]*\bTOTAL\b", re.I)
        col_ben = cn[0] if len(cn) > 0 else "Bénéfice net"
        col_def = cn[1] if len(cn) > 1 else "Déficit net"

        if col_def in df.columns and col_ben in df.columns:
            # [P31-2a] Déplacer les réintégrations fiscales de Déficit → Bénéfice
            for idx, row in df.iterrows():
                lbl = str(row.get("Libellé") or "")
                val_def = row.get(col_def); val_ben = row.get(col_ben)
                if (_REINTEGRATION_RE.search(lbl)
                        and val_def is not None and not (isinstance(val_def, float) and pd.isna(val_def))
                        and float(val_def) > 0
                        and (val_ben is None or (isinstance(val_ben, float) and pd.isna(val_ben)))):
                    df.at[idx, col_ben] = float(val_def); df.at[idx, col_def] = None

            # [P31-2b] Cohérence de colonne pour les TOTAL : suit ses détails
            total_positions = [i for i, row in df.iterrows()
                               if _TOTAL_FISCAL_RE.search(str(row.get("Libellé") or ""))]
            prev_t = -1
            for tidx in total_positions:
                val_def = df.at[tidx, col_def]; val_ben = df.at[tidx, col_ben]
                total_in_deficit = (
                    val_def is not None and not (isinstance(val_def, float) and pd.isna(val_def))
                    and (val_ben is None or (isinstance(val_ben, float) and pd.isna(val_ben)))
                )
                if total_in_deficit:
                    detail_in_ben = sum(
                        1 for didx in range(prev_t + 1, tidx)
                        if didx in df.index
                        and df.at[didx, col_ben] is not None
                        and not (isinstance(df.at[didx, col_ben], float) and pd.isna(df.at[didx, col_ben]))
                    )
                    detail_in_def = sum(
                        1 for didx in range(prev_t + 1, tidx)
                        if didx in df.index
                        and df.at[didx, col_def] is not None
                        and not (isinstance(df.at[didx, col_def], float) and pd.isna(df.at[didx, col_def]))
                    )
                    if detail_in_ben > 0 and detail_in_def == 0:
                        df.at[tidx, col_ben] = float(val_def); df.at[tidx, col_def] = None
                prev_t = tidx
        return df


class DeterminationResultatParser(PCMParser):
    COL_NAMES = ["Montant"]


class AffectationResultatsParser(PCMParser):
    COL_NAMES = ["Montant A (origine)", "Montant B (affectation)"]

    def __init__(self, lines, sk, lmax, pw, zones=None):
        super().__init__(lines, sk, lmax, pw, zones)
        self.n_cols = len(self.COL_NAMES)

    def parse(self) -> pd.DataFrame:
        cn = self._col_names(); rows = []; acc = ""
        all_x1 = sorted([t.x1 for ln in self.lines for t in ln.tokens
                          if t.is_numeric and 10 < abs(t.value or 0) < 1_000_000])
        if len(all_x1) >= 4:
            gaps = [(all_x1[i+1] - all_x1[i], i) for i in range(len(all_x1)-1)]
            max_gap, max_idx = max(gaps, key=lambda g: g[0])
            split_x = (all_x1[max_idx] + all_x1[max_idx+1]) / 2 if max_gap > 30 else self.pw * 0.55
        else:
            split_x = self.pw * 0.55
        for idx, ln in enumerate(self.lines):
            if ln.y < 90 or ln.y > 790: continue
            label = ln.label_text(self.lmax)
            vtoks = [t for t in ln.tokens if t.is_numeric
                     and t.value is not None and 0.01 < abs(t.value) < 1_000_000]
            if label and not vtoks:
                if self._noise(label): continue
                if self._merge_flags[idx]:
                    acc = (acc + " " + label).strip() if acc else label
                else:
                    if acc and not self._noise(acc):
                        rows.append(self._emit_label_row(acc, cn)); acc = ""
                    rows.append(self._emit_label_row(label, cn))
                continue
            if acc:
                label = (acc + " " + label).strip() if label else acc; acc = ""
            values: Dict[int, float] = {}
            for t in vtoks:
                col = 0 if t.x1 <= split_x else 1
                if col not in values: values[col] = t.value
            if not label and not values: continue
            if label and self._noise(label): continue
            if label and self._hard_noise(label): continue
            row = {"Libellé": label or None}
            for i, c in enumerate(cn): row[c] = values.get(i)
            rows.append(row)
        if acc and not self._noise(acc): rows.append(self._emit_label_row(acc, cn))
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)


class Resultats3AnsParser(PCMParser):
    COL_NAMES = ["Année N-2", "Année N-1", "Année N"]


PARSER_MAP: Dict[str, type] = {
    "bilan_actif":             BilanActifParser,
    "bilan_passif":            BilanPassifParser,
    "cpc":                     CPCParser,
    "esg":                     ESGParser,
    "caf":                     CAFParser,
    "tableau_financement":     FinancementParser,
    "tableau_immobilisations": ImmobilisationsParser,
    "tableau_amortissements":  AmortissementsParser,
    "tableau_provisions":      ProvisionsParser,
    "tableau_creances":        CreancesParser,
    "tableau_dettes":          DettesParser,
    "titres_participation":    TitresParticipationParser,
    "detail_cpc":              DetailCPCParser,
    "resultat_fiscal":         ResultatFiscalParser,
    "determination_resultat":  DeterminationResultatParser,
    "affectation_resultats":   AffectationResultatsParser,
    "resultats_3ans":          Resultats3AnsParser,
}
