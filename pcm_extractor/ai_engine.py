import re
import os
import json
import time
import hashlib
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd

from .config import (
    AI_BACKEND, AI_MODEL, AI_CACHE_FILE, AI_SKIP_THRESHOLD,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    CGNC_SECTION_RULES, HAS_ANTHROPIC,
)
from .models import ParsedSection, ValidationAnomaly, normalize_text

log = logging.getLogger(__name__)

_CGNC_LABEL_VOCAB_EXTENDED: Dict[str, List[str]] = {
    "bilan_actif": [
        "ACTIF IMMOBILISÉ", "IMMOBILISATIONS EN NON-VALEURS", "FRAIS PRÉLIMINAIRES",
        "CHARGES À RÉPARTIR", "PRIMES DE REMBOURSEMENT", "IMMOBILISATIONS INCORPORELLES",
        "IMMOBILISATIONS CORPORELLES", "TERRAINS", "CONSTRUCTIONS",
        "INSTALLATIONS TECHNIQUES MATÉRIEL ET OUTILLAGE", "MATÉRIEL DE TRANSPORT",
        "IMMOBILISATIONS FINANCIÈRES", "PRÊTS IMMOBILISÉS", "TITRES DE PARTICIPATION",
        "ACTIF CIRCULANT", "STOCKS", "MARCHANDISES", "PRODUITS EN COURS", "PRODUITS FINIS",
        "CRÉANCES DE L'ACTIF CIRCULANT", "CLIENTS ET COMPTES RATTACHÉS",
        "TRÉSORERIE ACTIF", "BANQUES TRÉSORERIE GÉNÉRALE ET CHÈQUES POSTAUX", "TOTAL GÉNÉRAL",
    ],
    "bilan_passif": [
        "FINANCEMENT PERMANENT", "CAPITAUX PROPRES", "CAPITAL SOCIAL OU PERSONNEL",
        "RÉSERVE LÉGALE", "AUTRES RÉSERVES", "REPORT À NOUVEAU", "RÉSULTAT NET DE L'EXERCICE",
        "CAPITAUX PROPRES ASSIMILÉS", "DETTES DE FINANCEMENT", "EMPRUNTS OBLIGATAIRES",
        "PROVISIONS DURABLES POUR RISQUES ET CHARGES", "ECARTS DE CONVERSION PASSIF",
        "PASSIF CIRCULANT", "DETTES DU PASSIF CIRCULANT",
        "FOURNISSEURS ET COMPTES RATTACHÉS", "ORGANISMES SOCIAUX", "ETAT CRÉDITEUR",
        "AUTRES CRÉANCIERS", "TRÉSORERIE PASSIF", "CRÉDITS DE TRÉSORERIE",
        "BANQUES SOLDES CRÉDITEURS", "TOTAL GÉNÉRAL",
    ],
    "cpc": [
        "PRODUITS D'EXPLOITATION", "VENTES DE MARCHANDISES", "CHIFFRE D'AFFAIRES",
        "CHARGES D'EXPLOITATION", "ACHATS REVENDUS DE MARCHANDISES",
        "CHARGES DE PERSONNEL", "DOTATIONS D'EXPLOITATION", "RÉSULTAT D'EXPLOITATION",
        "PRODUITS FINANCIERS", "CHARGES FINANCIÈRES", "RÉSULTAT FINANCIER",
        "RÉSULTAT COURANT", "PRODUITS NON COURANTS", "CHARGES NON COURANTES",
        "RÉSULTAT NON COURANT", "RÉSULTAT AVANT IMPÔTS",
        "IMPÔTS SUR LES RÉSULTATS", "RÉSULTAT NET DE L'EXERCICE",
        "TOTAL DES PRODUITS", "TOTAL DES CHARGES",
    ],
    "esg": [
        "MARGE BRUTE SUR VENTES EN L'ÉTAT", "PRODUCTION DE L'EXERCICE",
        "CONSOMMATION DE L'EXERCICE", "VALEUR AJOUTÉE",
        "EXCÉDENT BRUT D'EXPLOITATION", "INSUFFISANCE BRUTE D'EXPLOITATION",
        "RÉSULTAT D'EXPLOITATION", "RÉSULTAT FINANCIER",
        "RÉSULTAT COURANT", "RÉSULTAT NON COURANT",
        "RÉSULTAT NET DE L'EXERCICE", "CAPACITÉ D'AUTOFINANCEMENT", "AUTOFINANCEMENT",
    ],
    "tableau_financement": [
        "FINANCEMENT PERMANENT", "ACTIF IMMOBILISÉ",
        "FONDS DE ROULEMENT FONCTIONNEL", "BESOIN DE FINANCEMENT GLOBAL",
        "TRÉSORERIE NETTE", "AUTOFINANCEMENT",
    ],
    "resultat_fiscal": [
        "RÉSULTAT NET COMPTABLE", "RÉINTÉGRATIONS FISCALES",
        "DÉDUCTIONS FISCALES", "RÉSULTAT BRUT FISCAL",
        "REPORTS DÉFICITAIRES IMPUTÉS", "BÉNÉFICE NET FISCAL", "DÉFICIT NET FISCAL",
    ],
}

_DECO_LETTER_PREFIX_RE = re.compile(r"^[A-Z]{1,3}\s+(?=[A-ZÀÂÄÉÈÊËÎÏÔÙÛÜ])", re.UNICODE)
_DECO_LETTER_SUFFIX_RE = re.compile(r"\s+[A-Z]{1,2}$", re.UNICODE)
_KNOWN_DUPLICATES: Dict[str, List[str]] = {
    "bilan_passif": ["AUTRES DETTES DE FINANCEMENT", "AUTRES CRÉANCIERS"],
}


class AICorrectionsCache:
    """Cache JSON pour éviter les appels API redondants."""

    def __init__(self, cache_file: str = AI_CACHE_FILE):
        self.cache_file = cache_file
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {}

    def _save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _hash(self, df: pd.DataFrame, anomalies: List[ValidationAnomaly]) -> str:
        payload = df.to_csv(index=False) + str([a.rule + a.description for a in anomalies])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, df: pd.DataFrame, anomalies: List[ValidationAnomaly]) -> Optional[Dict]:
        return self._data.get(self._hash(df, anomalies))

    def set(self, df: pd.DataFrame, anomalies: List[ValidationAnomaly], result: Dict):
        self._data[self._hash(df, anomalies)] = result
        self._save()


class ContextBuilder:
    """[V28-B] Construit les prompts enrichis pour l'IA."""
    MAX_ROWS_IN_PROMPT = 150

    def build_prompt(self, section: ParsedSection,
                     anomalies: List[ValidationAnomaly],
                     cross_context: Optional[str] = None) -> str:
        df = section.dataframe; sk = section.section_key; name = section.display_name
        anomaly_text = "\n".join(
            f"  - [{a.severity.upper()}] Règle {a.rule}: {a.description}"
            + (f" | Attendu: {a.expected}" if a.expected is not None else "")
            + (f" | Observé: {a.observed}" if a.observed is not None else "")
            for a in anomalies
        ) if anomalies else "  Aucune anomalie — vérification systématique demandée."

        csv_repr = df.head(self.MAX_ROWS_IN_PROMPT).to_csv(index=True, float_format="{:.2f}".format)
        rules    = CGNC_SECTION_RULES.get(sk, "")
        cross    = f"\n CONTEXTE INTER-SECTIONS \n{cross_context}\n" if cross_context else ""

        return f"""Tu es un expert-comptable marocain CGNC.
Tu analyses la section "{name}" (clé: {sk}) extraite d'un bilan PCM marocain.

{cross}
 RÈGLES CGNC
{rules}

 ANOMALIES
{anomaly_text}

 DONNÉES (CSV, index=numéro de ligne)
{csv_repr}

 MISSION
1. Vérifie TOUTES les règles CGNC pour cette section
2. Corrige les valeurs erronées (signe, colonne, None calculable)
3. Répare les libellés tronqués ou mal formatés
4. N'invente JAMAIS une valeur absente du document source

Réponds UNIQUEMENT en JSON valide strict :
{{
  "corrections": [{{"row_index": <int>, "column": "<col>", "old_value": <val|null>,
    "new_value": <val>, "reason": "<règle CGNC>", "confidence": <0.0-1.0>}}],
  "labels_corrections": [{{"row_index": <int>, "old_label": "<old>",
    "new_label": "<new>", "reason": "<justification>"}}],
  "computed_values": [{{"row_index": <int>, "column": "<col>",
    "formula": "<formule>", "value": <val>, "reason": "<règle>"}}],
  "global_assessment": "<résumé>",
  "confidence": <0-100>,
  "unfixable": false
}}
"""

    def build_cross_check_prompt(self, sections: Dict[str, pd.DataFrame]) -> str:
        summaries = []
        for sk, df in sections.items():
            if df is None or df.empty: continue
            from .config import SECTION_DISPLAY_NAMES
            name = SECTION_DISPLAY_NAMES.get(sk, sk); totals = {}
            for _, row in df.iterrows():
                lbl = normalize_text(str(row.get("Libellé", "") or "")).upper()
                if "TOTAL GENERAL" in lbl or "TOTAL I" in lbl:
                    for col in df.columns:
                        if col != "Libellé":
                            v = pd.to_numeric(row.get(col), errors="coerce")
                            if pd.notna(v): totals[col] = v; break
            summaries.append(f"  {name}: totaux={totals}")

        return f"""Tu es expert-comptable CGNC. Vérifie la cohérence inter-sections :

TOTAUX EXTRAITS :
{chr(10).join(summaries)}

RÈGLES :
1. Total Général Bilan Actif = Total Général Bilan Passif
2. Résultat Net CPC = Résultat Net Bilan Passif = Résultat Net ESG
3. CAF ESG ≈ Autofinancement Tableau Financement

Réponds en JSON :
{{"coherence_issues": [{{"rule": "<règle>", "sections_involved": ["<sk1>", "<sk2>"],
  "description": "<écart>", "likely_error_in": "<section>", "suggested_value": <val|null>}}],
  "overall_coherent": <true/false>}}
"""

    def build_reconstruction_prompt(self, section: ParsedSection, missing_analysis: str) -> str:
        df = section.dataframe; sk = section.section_key
        rules = CGNC_SECTION_RULES.get(sk, "")
        csv_repr = df.to_csv(index=True, float_format="{:.2f}".format)
        return f"""Tu es expert-comptable CGNC. Reconstruis les valeurs manquantes (None/NaN)
de la section "{section.display_name}" par déduction depuis les autres valeurs présentes.

RÈGLES CGNC :
{rules}

ANALYSE DES MANQUES :
{missing_analysis}

DONNÉES ACTUELLES :
{csv_repr}

MISSION : Ne propose QUE des valeurs déductibles mathématiquement.

Réponds en JSON :
{{"reconstructed": [{{"row_index": <int>, "column": "<col>",
  "value": <val>, "formula": "<formule>", "confidence": <0.0-1.0>}}]}}
"""

    def build_explain_prompt(self, section: ParsedSection,
                             anomalies: List[ValidationAnomaly]) -> str:
        anomaly_text = "\n".join(f"  - [{a.severity.upper()}] {a.rule}: {a.description}"
                                  for a in anomalies)
        return f"""Tu es un expert-comptable marocain CGNC. Explique en français simple les anomalies
suivantes dans la section "{section.display_name}" :

{anomaly_text}

Pour chaque anomalie : cause probable + comment corriger manuellement. Sois concis (1-3 phrases).
"""


def _analyze_missing_values(df: pd.DataFrame, sk: str) -> str:
    if df is None or df.empty: return "Section vide"
    missing = []
    for idx, row in df.iterrows():
        lbl = str(row.get("Libellé", "") or "")
        for col in df.columns:
            if col == "Libellé": continue
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                others = {c: row[c] for c in df.columns
                          if c != "Libellé" and c != col
                          and row.get(c) is not None
                          and not (isinstance(row.get(c), float) and pd.isna(row.get(c)))}
                if others:
                    missing.append(f"  L{idx} '{lbl[:30]}' col '{col}' manquante")
    return "\n".join(missing[:20]) if missing else "Aucune valeur manquante calculable"


class OllamaClient:
    """[V30-B] Client HTTP léger vers Ollama /api/chat (stdlib seulement)."""

    SYSTEM_PROMPT = (
        "Tu es un expert-comptable marocain certifié CGNC. "
        "Tu réponds TOUJOURS en JSON valide strict, sans markdown ni backticks. "
        "Tu appliques strictement les règles du CGNC. "
        "Tu n'inventes jamais de valeur absente du document source."
    )

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL,
                 timeout: int = OLLAMA_TIMEOUT):
        self.url = url.rstrip("/"); self.model = model; self.timeout = timeout

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False

    def chat(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        payload = json.dumps({
            "model": self.model, "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1, "top_p": 0.9},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                content = data.get("message", {}).get("content", "")
                return content if content else None
        except Exception as e:
            log.warning(f"     Ollama erreur: {e}")
            return None


class LabelRefinementEngine:
    """[V30-C] Correction déterministe des libellés — zéro IA, zéro coût."""

    LEV_THRESHOLD = 0.65

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """[V31-A] Distance Levenshtein normalisée avec normalisation asymétrique pour troncatures."""
        s1 = s1.upper().strip(); s2 = s2.upper().strip()
        if s1 == s2: return 1.0
        if not s1 or not s2: return 0.0
        len1, len2 = len(s1), len(s2)
        is_truncation = (len1 < len2 * 0.6 and len1 >= 4)
        if not is_truncation and abs(len1 - len2) / max(len1, len2) > 0.5: return 0.0
        if is_truncation and s1 in s2: return 1.0 - (len2 - len1) / len2 * 0.1
        dp = list(range(len2 + 1))
        for i, c1 in enumerate(s1, 1):
            prev = dp[:]
            dp[0] = i
            for j, c2 in enumerate(s2, 1):
                dp[j] = min(prev[j] + 1, dp[j-1] + 1, prev[j-1] + (0 if c1 == c2 else 1))
        dist = dp[len2]
        return (1.0 - dist / max(len1, 1)) if is_truncation else (1.0 - dist / max(len1, len2))

    @classmethod
    def _repair_label(cls, label: str, section_key: str) -> str:
        if not label or label.strip() in ("", "NaN", "nan"): return label
        lbl = _DECO_LETTER_PREFIX_RE.sub("", label.strip()).strip()
        lbl = _DECO_LETTER_SUFFIX_RE.sub("", lbl).strip()
        lbl = re.sub(r"\s{2,}", " ", lbl)

        vocab = (_CGNC_LABEL_VOCAB_EXTENDED.get(section_key, []) +
                 _CGNC_LABEL_VOCAB_EXTENDED.get("bilan_actif", []))
        lbl_norm = normalize_text(lbl).upper()
        best_ratio = 0.0; best_match = lbl

        for candidate in vocab:
            cand_norm = normalize_text(candidate).upper()
            ratio = cls._levenshtein_ratio(lbl_norm, cand_norm)
            if ratio > best_ratio: best_ratio = ratio; best_match = candidate

        if best_ratio >= cls.LEV_THRESHOLD:
            if len(lbl) / max(len(best_match), 1) < 0.80 or best_ratio >= 0.92:
                return best_match
        return lbl

    @classmethod
    def process_section(cls, section: ParsedSection) -> ParsedSection:
        df = section.dataframe; sk = section.section_key
        if df is None or df.empty: return section
        lbl_col = next((c for c in df.columns
                        if "libellé" in c.lower() or c.lower() == "lib"), None)
        if lbl_col is None: return section

        df = df.copy(); corrections_count = 0

        for idx in df.index:
            original = str(df.at[idx, lbl_col] or "")
            repaired = cls._repair_label(original, sk)
            if repaired != original:
                df.at[idx, lbl_col] = repaired; corrections_count += 1

        for idx in df.index:
            lbl = str(df.at[idx, lbl_col] or "")
            if lbl.strip() in ("NaN", "nan", "None", ""):
                inferred = cls._infer_nan_label(df, idx, sk, lbl_col)
                if inferred: df.at[idx, lbl_col] = inferred; corrections_count += 1

        # Supprimer doublons connus
        seen_labels: Dict[str, int] = {}; rows_to_drop = []
        for idx in df.index:
            lbl_norm = normalize_text(str(df.at[idx, lbl_col] or "").strip()).upper()
            if lbl_norm in seen_labels:
                first_idx = seen_labels[lbl_norm]
                if any(normalize_text(d.upper()) == lbl_norm
                       for d in _KNOWN_DUPLICATES.get(sk, [])):
                    num_cols = [c for c in df.columns if c != lbl_col]
                    if [df.at[first_idx, c] for c in num_cols] == [df.at[idx, c] for c in num_cols]:
                        rows_to_drop.append(idx); corrections_count += 1
            else:
                seen_labels[lbl_norm] = idx

        if rows_to_drop:
            df = df.drop(index=rows_to_drop).reset_index(drop=True)

        if corrections_count > 0:
            log.info(f"      [LRE] {sk}: {corrections_count} libellé(s) corrigé(s)")
            return ParsedSection(section.section_key, section.display_name,
                                 section.page, df)
        return section

    @classmethod
    def _infer_nan_label(cls, df: pd.DataFrame, idx: int, sk: str, lbl_col: str) -> Optional[str]:
        vocab = _CGNC_LABEL_VOCAB_EXTENDED.get(sk, [])
        if not vocab: return None
        labels_before = [str(df.at[i, lbl_col] or "") for i in df.index
                         if i < idx and str(df.at[i, lbl_col] or "").strip() not in ("NaN", "nan", "")]
        prev_label = labels_before[-1].upper() if labels_before else ""
        if sk == "esg":
            if "VALEUR" in prev_label and "AJOUT" in prev_label: return "EXCÉDENT BRUT D'EXPLOITATION"
            if "EXCEDENT" in prev_label or "INSUFFISANCE" in prev_label: return "RÉSULTAT D'EXPLOITATION"
        if sk == "tableau_financement":
            if "FINANCEMENT PERMANENT" in prev_label or "ACTIF IMMOBILIS" in prev_label:
                return "FONDS DE ROULEMENT FONCTIONNEL"
            if "ACTIF CIRCULANT" in prev_label or "PASSIF CIRCULANT" in prev_label:
                return "BESOIN DE FINANCEMENT GLOBAL"
        return None


class RefinedMerger:
    """[V28-D] Applique les corrections JSON de l'IA sur un DataFrame pandas."""

    @staticmethod
    def apply(df: pd.DataFrame, corrections: Dict) -> pd.DataFrame:
        for corr in corrections.get("corrections", []):
            try:
                idx = int(corr["row_index"]); col = str(corr["column"])
                new_val = corr.get("new_value"); conf = float(corr.get("confidence", 1.0))
                if idx < len(df) and col in df.columns and conf >= 0.7:
                    if new_val is None: df.at[idx, col] = None
                    else: df.at[idx, col] = float(new_val) if isinstance(new_val, (int, float)) else new_val
            except (KeyError, ValueError, TypeError): pass

        for item in corrections.get("computed_values", []):
            try:
                idx = int(item["row_index"]); col = str(item["column"])
                val = item.get("value"); conf = float(item.get("confidence", 1.0))
                if idx < len(df) and col in df.columns and conf >= 0.8:
                    current = df.at[idx, col]
                    if current is None or (isinstance(current, float) and pd.isna(current)):
                        df.at[idx, col] = float(val) if val is not None else None
            except (KeyError, ValueError, TypeError): pass

        for corr in corrections.get("labels_corrections", []):
            try:
                idx = int(corr["row_index"]); new_lbl = str(corr["new_label"])
                lbl_col = next((c for c in df.columns
                                if "libellé" in c.lower() or "lib" in c.lower()), "Libellé")
                if idx < len(df) and lbl_col in df.columns:
                    df.at[idx, lbl_col] = new_lbl
            except (KeyError, ValueError, TypeError): pass

        return df


class AIRefinementEngine:
    """[V30] Orchestrateur IA hybride : LLM local (Ollama) + fallback Claude API."""

    MISSING_THRESHOLD = 0.15

    def __init__(self):
        self._ollama: Optional[OllamaClient] = None
        self._anthropic: Optional[Any]        = None
        self._backend: str                    = AI_BACKEND
        self._active_backend: str             = "none"
        self._init_backends()
        self.cache   = AICorrectionsCache()
        self.builder = ContextBuilder()

    def _init_backends(self):
        if self._backend in ("ollama", "auto"):
            client = OllamaClient(url=OLLAMA_URL, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT)
            if client.is_available():
                self._ollama = client
                log.info(f"   Ollama disponible: {OLLAMA_URL} | modèle: {OLLAMA_MODEL}")
            elif self._backend == "ollama":
                raise RuntimeError(f"Ollama non disponible ({OLLAMA_URL}). "
                                   f"Lancez: ollama pull {OLLAMA_MODEL} && ollama serve")
            else:
                log.info("  ℹ  Ollama absent — basculement sur Anthropic")

        if self._backend in ("anthropic", "auto"):
            if not HAS_ANTHROPIC:
                if self._backend == "anthropic":
                    raise ImportError("anthropic non installé — pip install anthropic")
            else:
                import anthropic as _anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if api_key:
                    self._anthropic = _anthropic.Anthropic(api_key=api_key)
                elif self._backend == "anthropic":
                    raise RuntimeError("ANTHROPIC_API_KEY non définie")

        if self._ollama and self._backend != "anthropic":
            self._active_backend = "ollama"
        elif self._anthropic:
            self._active_backend = "anthropic"
        else:
            raise RuntimeError("Aucun backend IA disponible.")
        log.info(f"   Backend IA actif: {self._active_backend.upper()}")

    def _call_api(self, prompt: str, max_tokens: int = 2000,
                  retries: int = 3) -> Optional[str]:
        if self._active_backend == "ollama":
            result = self._call_ollama(prompt, max_tokens, retries)
            if result is None and self._anthropic and self._backend == "auto":
                result = self._call_anthropic(prompt, max_tokens, retries)
            return result
        return self._call_anthropic(prompt, max_tokens, retries)

    def _call_ollama(self, prompt: str, max_tokens: int, retries: int) -> Optional[str]:
        if not self._ollama: return None
        for attempt in range(retries):
            try:
                result = self._ollama.chat(prompt, max_tokens=max_tokens)
                if result: return result
            except Exception as e:
                log.warning(f"     Ollama erreur (tentative {attempt+1}/{retries}): {e}")
            if attempt < retries - 1: time.sleep(2 ** attempt)
        return None

    def _call_anthropic(self, prompt: str, max_tokens: int, retries: int) -> Optional[str]:
        if not self._anthropic: return None
        SYSTEM = (
            "Tu es un expert-comptable marocain certifié CGNC. "
            "Tu réponds TOUJOURS en JSON valide strict, sans markdown ni backticks. "
            "Tu n'inventes jamais de valeur absente du document source."
        )
        for attempt in range(retries):
            try:
                response = self._anthropic.messages.create(
                    model=AI_MODEL, max_tokens=max_tokens,
                    system=SYSTEM, messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                log.warning(f"     Claude API erreur (tentative {attempt+1}/{retries}): {e}")
                if attempt < retries - 1: time.sleep(2 ** attempt)
        return None

    def _parse_json(self, raw: Optional[str], section_key: str) -> Optional[Dict]:
        if not raw: return None
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except json.JSONDecodeError as e:
            log.error(f"     JSON invalide [{section_key}]: {e} — raw: {raw[:200]}")
            return None

    def refine(self, section: ParsedSection,
               anomalies: List[ValidationAnomaly],
               cross_context: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        cached = self.cache.get(section.dataframe, anomalies)
        if cached:
            result = cached
        else:
            prompt = self.builder.build_prompt(section, anomalies, cross_context)
            raw    = self._call_api(prompt)
            result = self._parse_json(raw, section.section_key)
            if result: self.cache.set(section.dataframe, anomalies, result)

        if not result: return section.dataframe, {"error": "API non disponible"}
        if result.get("unfixable"):
            return section.dataframe, result

        refined_df = RefinedMerger.apply(section.dataframe.copy(), result)
        meta = {
            "mode": "targeted",
            "global_assessment":        result.get("global_assessment", ""),
            "ai_confidence":            result.get("confidence", 0),
            "corrections_count":        len(result.get("corrections", [])),
            "labels_corrections_count": len(result.get("labels_corrections", [])),
            "computed_count":           len(result.get("computed_values", [])),
        }
        return refined_df, meta

    def reconstruct_missing(self, section: ParsedSection) -> Tuple[pd.DataFrame, Dict]:
        df = section.dataframe; sk = section.section_key
        if sk not in CGNC_SECTION_RULES: return df, {}
        num_cols = [c for c in df.columns if c != "Libellé"]
        if not num_cols: return df, {}
        total   = df[num_cols].size
        missing = df[num_cols].isna().sum().sum()
        if missing == 0 or missing / total < self.MISSING_THRESHOLD: return df, {}

        missing_analysis = _analyze_missing_values(df, sk)
        if "Aucune valeur" in missing_analysis: return df, {}

        prompt = self.builder.build_reconstruction_prompt(section, missing_analysis)
        raw    = self._call_api(prompt, max_tokens=1500)
        result = self._parse_json(raw, sk)
        if not result or not result.get("reconstructed"): return df, {}

        df = df.copy(); count = 0
        for item in result["reconstructed"]:
            try:
                idx = int(item["row_index"]); col = str(item["column"])
                val = item["value"]; conf = float(item.get("confidence", 0))
                if idx < len(df) and col in df.columns and conf >= 0.8:
                    current = df.at[idx, col]
                    if current is None or (isinstance(current, float) and pd.isna(current)):
                        df.at[idx, col] = float(val); count += 1
            except (KeyError, ValueError, TypeError): pass
        return df, {"mode": "reconstruction", "reconstructed_count": count}

    def cross_check(self, sections: List[ParsedSection]) -> Dict[str, Any]:
        sec_map = {s.section_key: s.dataframe for s in sections
                   if s.dataframe is not None and not s.dataframe.empty}
        if not {"bilan_actif", "bilan_passif", "cpc"}.intersection(sec_map.keys()):
            return {}
        prompt = self.builder.build_cross_check_prompt(sec_map)
        raw    = self._call_api(prompt, max_tokens=1000)
        result = self._parse_json(raw, "cross_check")
        if not result: return {}
        issues = result.get("coherence_issues", [])
        if issues:
            log.info(f"     {len(issues)} incohérence(s) inter-sections détectée(s)")
        return result

    def explain(self, section: ParsedSection, anomalies: List[ValidationAnomaly]) -> str:
        prompt = self.builder.build_explain_prompt(section, anomalies)
        resp   = self._call_api(prompt, max_tokens=1500)
        return resp or "Explication non disponible"
