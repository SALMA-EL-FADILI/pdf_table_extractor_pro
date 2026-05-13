import re
import logging
from typing import List, Dict, Optional

import pandas as pd

from .config import CONFIDENCE_THRESHOLD
from .models import ParsedSection, ValidationAnomaly, SectionScore, normalize_text

log = logging.getLogger(__name__)


class StructuralValidator:
    """Valide chaque section contre les règles CGNC (R01–R12) et retourne un score 0–100."""

    TOLERANCE = 0.02

    def validate(self, section: ParsedSection) -> SectionScore:
        anomalies: List[ValidationAnomaly] = []
        df = section.dataframe; sk = section.section_key

        if df is None or df.empty:
            return SectionScore(sk, 0.0, [
                ValidationAnomaly(sk, "EMPTY", "critical", "Section vide")
            ], True)

        lbl_col = [c for c in df.columns if "libellé" in c.lower() or "lib" in c.lower()]

        # R07 — libellés vides
        if lbl_col:
            lbl_series = df[lbl_col[0]].astype(str).str.strip()
            empty_pct  = (lbl_series.isin(["", "nan", "None"])).mean()
            if empty_pct > 0.20:
                anomalies.append(ValidationAnomaly(
                    sk, "R07", "warning",
                    f"{empty_pct*100:.0f}% de libellés vides (seuil 20%)",
                    expected=0.20, observed=empty_pct))

        # R11 — doublons
        if lbl_col:
            lbls = df[lbl_col[0]].dropna().astype(str).str.strip()
            lbls = lbls[lbls.str.len() > 3]
            dups = lbls[lbls.duplicated()].tolist()
            if dups:
                anomalies.append(ValidationAnomaly(
                    sk, "R11", "warning",
                    f"Libellés en double : {dups[:3]}",
                    observed=dups))

        num_cols = [c for c in df.columns if c != "Libellé" and c not in lbl_col]
        if not num_cols:
            score = max(0, 100 - len(anomalies) * 15)
            return SectionScore(sk, score, anomalies, score < CONFIDENCE_THRESHOLD)

        def _col_vals(col: str) -> pd.Series:
            return pd.to_numeric(df[col], errors="coerce").dropna()

        # R06 — valeurs aberrantes
        for col in num_cols:
            vals = _col_vals(col)
            if len(vals) < 3: continue
            med = vals.abs().median()
            if med > 0:
                outliers = vals[vals.abs() > med * 200]
                if not outliers.empty:
                    anomalies.append(ValidationAnomaly(
                        sk, "R06", "critical",
                        f"Valeur aberrante colonne '{col}': {outliers.iloc[0]:,.0f} (médiane={med:,.0f})",
                        row_indices=outliers.index.tolist(), observed=outliers.iloc[0]))

        # R08 — colonnes identiques
        if len(num_cols) >= 2:
            v1 = _col_vals(num_cols[0]); v2 = _col_vals(num_cols[1])
            common = v1.index.intersection(v2.index)
            if len(common) > 3 and (v1[common] == v2[common]).all():
                anomalies.append(ValidationAnomaly(
                    sk, "R08", "warning",
                    f"Colonnes '{num_cols[0]}' et '{num_cols[1]}' identiques — possible erreur de colonnage",
                    observed="identical"))

        # R12 — cohérence N vs N-1
        n_cols  = [c for c in num_cols if "N-1" not in c and "préc" not in c.lower()]
        n1_cols = [c for c in num_cols if "N-1" in c or "préc" in c.lower()]
        if n_cols and n1_cols:
            vn  = _col_vals(n_cols[0]).abs(); vn1 = _col_vals(n1_cols[0]).abs()
            common = vn.index.intersection(vn1.index)
            if len(common) > 2:
                ratios = (vn[common] / (vn1[common] + 1)).dropna()
                extreme = ratios[ratios > 10]
                if len(extreme) > len(common) * 0.3:
                    anomalies.append(ValidationAnomaly(
                        sk, "R12", "warning",
                        f"{len(extreme)} lignes avec ratio N/N-1 > 10× — possible confusion de colonnes",
                        row_indices=extreme.index.tolist()))

        if sk == "bilan_actif":
            anomalies.extend(self._validate_bilan_actif(df, sk))
        elif sk == "bilan_passif":
            anomalies.extend(self._validate_bilan_passif(df, sk))
        elif sk == "cpc":
            anomalies.extend(self._validate_cpc(df, sk))

        penalty = sum(
            20 if a.severity == "critical" else 10 if a.severity == "warning" else 3
            for a in anomalies
        )
        score = max(0.0, 100.0 - penalty)
        return SectionScore(sk, score, anomalies, score < CONFIDENCE_THRESHOLD)

    def _find_total_row(self, df: pd.DataFrame, col: str) -> Optional[float]:
        for _, row in df.iterrows():
            lbl = str(row.get("Libellé", "") or "").upper()
            if re.search(r"\bTOTAL\b", lbl):
                v = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(v): return float(v)
        return None

    def _validate_bilan_actif(self, df: pd.DataFrame, sk: str) -> List[ValidationAnomaly]:
        anomalies = []
        brut_col  = next((c for c in df.columns if "brut" in c.lower()), None)
        amort_col = next((c for c in df.columns if "amort" in c.lower()), None)
        net_col   = next((c for c in df.columns if "net" in c.lower() and "n-1" not in c.lower()), None)

        if brut_col and amort_col and net_col:
            for idx, row in df.iterrows():
                brut  = pd.to_numeric(row.get(brut_col),  errors="coerce")
                amort = pd.to_numeric(row.get(amort_col), errors="coerce")
                net   = pd.to_numeric(row.get(net_col),   errors="coerce")
                if pd.notna(brut) and pd.notna(amort) and pd.notna(net):
                    expected_net = brut - amort
                    if abs(expected_net) > 100 and abs(net - expected_net) / abs(expected_net) > self.TOLERANCE:
                        anomalies.append(ValidationAnomaly(
                            sk, "R04", "critical",
                            f"Ligne {idx}: Net({net:,.0f}) ≠ Brut({brut:,.0f}) - Amort({amort:,.0f}) = {expected_net:,.0f}",
                            row_indices=[idx], expected=expected_net, observed=net))

        if amort_col:
            amort_vals = pd.to_numeric(df[amort_col], errors="coerce").dropna()
            neg_amort  = amort_vals[amort_vals < -1]
            if not neg_amort.empty:
                anomalies.append(ValidationAnomaly(
                    sk, "R09", "warning",
                    f"{len(neg_amort)} amortissements négatifs",
                    row_indices=neg_amort.index.tolist()))
        return anomalies

    def _validate_bilan_passif(self, df: pd.DataFrame, sk: str) -> List[ValidationAnomaly]:
        anomalies = []
        n_col = next((c for c in df.columns if "exercice n" in c.lower() and "n-1" not in c.lower()), None)
        if n_col:
            total = self._find_total_row(df, n_col)
            if total:
                non_total = [
                    pd.to_numeric(row.get(n_col), errors="coerce")
                    for _, row in df.iterrows()
                    if not re.search(r"\bTOTAL\b", str(row.get("Libellé", "") or "").upper())
                    and pd.notna(pd.to_numeric(row.get(n_col), errors="coerce"))
                ]
                if non_total:
                    soma = sum(v for v in non_total if pd.notna(v))
                    if abs(total) > 1000 and abs(soma - total) / abs(total) > self.TOLERANCE:
                        anomalies.append(ValidationAnomaly(
                            sk, "R02", "warning",
                            f"Total Passif ({total:,.0f}) ≠ Somme des postes ({soma:,.0f})",
                            expected=soma, observed=total))
        return anomalies

    def _validate_cpc(self, df: pd.DataFrame, sk: str) -> List[ValidationAnomaly]:
        anomalies = []
        tot_col = next((c for c in df.columns if "total n" in c.lower() and "n-1" not in c.lower()), None)
        if tot_col:
            total = self._find_total_row(df, tot_col)
            if total and abs(total) > 1000:
                tp = tc = None
                for _, row in df.iterrows():
                    lbl = str(row.get("Libellé", "") or "").upper()
                    v = pd.to_numeric(row.get(tot_col), errors="coerce")
                    if pd.isna(v): continue
                    if "TOTAL PRODUITS" in lbl: tp = v
                    elif "TOTAL CHARGES" in lbl: tc = v
                if pd.notna(tp) and pd.notna(tc):
                    expected_res = tp - tc
                    res_rows = df[df["Libellé"].astype(str).str.upper().str.contains(
                        r"R.SULTAT\s+NET", regex=True, na=False)]
                    if not res_rows.empty:
                        observed_res = pd.to_numeric(res_rows.iloc[0][tot_col], errors="coerce")
                        if pd.notna(observed_res) and abs(expected_res) > 100:
                            if abs(observed_res - expected_res) / abs(expected_res) > self.TOLERANCE:
                                anomalies.append(ValidationAnomaly(
                                    sk, "R05", "critical",
                                    f"Résultat Net ({observed_res:,.0f}) ≠ Produits - Charges ({expected_res:,.0f})",
                                    expected=expected_res, observed=observed_res))
        return anomalies
