import os
import re
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from .config import (
    AI_SKIP_THRESHOLD, CONFIDENCE_THRESHOLD, AI_BACKEND,
    AI_REFINE_ENABLED, OLLAMA_MODEL, OLLAMA_URL,
    SECTION_DISPLAY_NAMES,
)
from .models import ParsedSection, SectionScore, ValidationAnomaly
from .extraction import detect_pdf_type, detect_exercise_year
from .pipeline import extract_all_sections, merge_sections
from .validator import StructuralValidator
from .ai_engine import AIRefinementEngine, LabelRefinementEngine
from .export import export_to_excel

log = logging.getLogger(__name__)


def extract_pdf(pdf_path: str,
                output_path: Optional[str] = None,
                ai_refine: bool = False,
                validate_only: bool = False,
                explain_mode: bool = False) -> Dict[str, Any]:

    from . import config as cfg

    start    = datetime.now()
    pdf_path = str(Path(pdf_path).resolve())
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")
    if output_path is None:
        stem = Path(pdf_path).stem
        output_path = str(Path(pdf_path).parent / f"{stem}_extrait_v36.xlsx")

    log.info("=" * 70)
    log.info(f"   PCM MAROC EXTRACTOR v36 — Architecture Hybride Heuristique + IA")
    log.info(f"   {Path(pdf_path).name}")
    log.info("=" * 70)

    if detect_pdf_type(pdf_path) == "scanned":
        msg = " PDF scanné — traite uniquement les PDFs texte natifs."
        log.error(msg)
        return {"success": False, "error": msg, "output_path": None}

    exercise_label = detect_exercise_year(pdf_path)
    sections       = extract_all_sections(pdf_path)

    if not sections:
        return {"success": False, "error": "Aucune section PCM détectée", "output_path": None}

    merged = merge_sections(sections)

    validator         = StructuralValidator()
    validation_scores: Dict[str, SectionScore] = {}
    ai_metadata:       Dict[str, Dict]          = {}
    explanations:      Dict[str, str]            = {}

    log.info("\n   COUCHE 2 — Validation structurelle CGNC")
    for sec in merged:
        score_obj = validator.validate(sec)
        validation_scores[sec.section_key] = score_obj
        emoji = "" if score_obj.score >= 90 else "" if score_obj.score >= 70 else ""
        log.info(f"    {emoji} {sec.display_name} : {score_obj.score:.0f}/100 "
                 f"({len(score_obj.anomalies)} anomalie(s))")
        for a in score_obj.anomalies:
            log.info(f"       [{'!' if a.severity == 'critical' else 'w'}] [{a.rule}] {a.description}")

    # R03 — Vérification équilibre Actif/Passif
    actif_sec  = next((s for s in merged if s.section_key == "bilan_actif"),  None)
    passif_sec = next((s for s in merged if s.section_key == "bilan_passif"), None)
    if actif_sec is not None and passif_sec is not None:
        def _total_net(sec, col_hint):
            df = sec.dataframe
            col = next((c for c in df.columns if col_hint in c.lower()), None)
            if col is None: return None
            for _, row in df.iterrows():
                if re.search(r"\bTOTAL\b", str(row.get("Libellé", "")).upper()):
                    import pandas as pd
                    v = pd.to_numeric(row.get(col), errors="coerce")
                    if pd.notna(v): return float(v)
            return None

        tot_actif  = _total_net(actif_sec,  "net")
        tot_passif = _total_net(passif_sec, "exercice n")
        if tot_actif is not None and tot_passif is not None and abs(tot_actif) > 1000:
            diff_pct = abs(tot_actif - tot_passif) / abs(tot_actif)
            if diff_pct > validator.TOLERANCE:
                r03 = ValidationAnomaly(
                    "bilan_actif", "R03", "critical",
                    f"Déséquilibre Bilan : Actif Net ({tot_actif:,.0f}) ≠ Passif ({tot_passif:,.0f}) — {diff_pct*100:.1f}%",
                    expected=tot_passif, observed=tot_actif)
                for sk_target in ("bilan_actif", "bilan_passif"):
                    if sk_target in validation_scores:
                        validation_scores[sk_target].anomalies.append(r03)
                        validation_scores[sk_target].score = max(0.0, validation_scores[sk_target].score - 20)
                        validation_scores[sk_target].needs_ai = True
                log.warning(f"     [R03] Déséquilibre Actif/Passif : {diff_pct*100:.1f}%")

    if validate_only:
        log.info("   Mode validation seule — pas d'export")
        return {
            "success": True, "validate_only": True,
            "validation_scores": {k: {"score": v.score, "anomalies": [
                {"rule": a.rule, "severity": a.severity, "description": a.description}
                for a in v.anomalies
            ]} for k, v in validation_scores.items()},
            "sections": [{"key": s.section_key, "name": s.display_name} for s in merged],
        }

    log.info("\n    PASSE 2.5 — Raffinement déterministe libellés (LRE)")
    for i, sec in enumerate(merged):
        merged[i] = LabelRefinementEngine.process_section(sec)
    for sec in merged:
        score_obj = validator.validate(sec)
        old_score = validation_scores.get(sec.section_key)
        if old_score and score_obj.score > old_score.score:
            validation_scores[sec.section_key] = score_obj

    if ai_refine or AI_REFINE_ENABLED:
        try:
            engine = AIRefinementEngine()
        except (RuntimeError, ImportError) as e:
            log.warning(f"   IA non disponible: {e} — skip COUCHE 3")
            engine = None

        if engine is not None:
            log.info(f"\n   COUCHE 3 — IA (backend: {engine._active_backend.upper()}, 3 passes)")

            log.info("   Passe 3A — Reconstruction valeurs manquantes")
            for sec in merged:
                refined_df, meta = engine.reconstruct_missing(sec)
                if meta.get("reconstructed_count", 0) > 0:
                    sec.dataframe = refined_df
                    ai_metadata.setdefault(sec.section_key, {}).update(meta)

            log.info("   Passe 3B — Cohérence inter-sections")
            cross_result   = engine.cross_check(merged)
            cross_contexts: Dict[str, str] = {}
            for issue in cross_result.get("coherence_issues", []):
                desc = issue.get("description", ""); rule = issue.get("rule", "")
                likely_err = issue.get("likely_error_in", "")
                ctx_line = f"INCOHÉRENCE [{rule}]: {desc}"
                if issue.get("suggested_value") is not None:
                    ctx_line += f" → valeur suggérée: {issue['suggested_value']}"
                if likely_err:
                    cross_contexts.setdefault(likely_err, []).append(ctx_line)
            for sk in cross_contexts:
                cross_contexts[sk] = "\n".join(cross_contexts[sk])

            log.info("    Passe 3C — Raffinement ciblé")
            for sec in merged:
                score_obj = validation_scores.get(sec.section_key)
                if not score_obj: continue
                score_pre = score_obj.score

                if score_pre >= AI_SKIP_THRESHOLD:
                    log.info(f"      [{sec.display_name}] score={score_pre:.0f} ≥ {AI_SKIP_THRESHOLD} → skip IA")
                    continue

                df_pre_ai = sec.dataframe.copy()
                refined_df, meta = engine.refine(sec, score_obj.anomalies, cross_contexts.get(sec.section_key))
                sec.dataframe = refined_df

                score_post_obj = validator.validate(sec)
                score_post     = score_post_obj.score

                if score_post < score_pre:
                    log.warning(f"      [V35-B] Rollback [{sec.display_name}] : "
                                f"{score_pre:.0f} → {score_post:.0f} → restauration")
                    sec.dataframe = df_pre_ai
                    meta["rolled_back"] = True
                else:
                    if score_post > score_pre:
                        validation_scores[sec.section_key] = score_post_obj
                meta["score_pre"] = score_pre; meta["score_post"] = score_post
                existing = ai_metadata.get(sec.section_key, {})
                existing.update(meta); ai_metadata[sec.section_key] = existing

                if explain_mode and score_obj.anomalies:
                    explanations[sec.section_key] = engine.explain(sec, score_obj.anomalies)

    log.info("\n   COUCHE 4 — Export Excel")
    result_path = export_to_excel(
        merged, output_path, exercise_label,
        validation_scores=validation_scores,
        ai_metadata=ai_metadata,
    )

    dur       = (datetime.now() - start).total_seconds()
    avg_score = sum(s.score for s in validation_scores.values()) / max(len(validation_scores), 1)
    log.info(f"\n TERMINÉ | {len(merged)} onglets | {dur:.1f}s | score moyen: {avg_score:.0f}/100")

    return {
        "success":           True,
        "output_path":       result_path,
        "exercise":          exercise_label,
        "sections":          [{"key": s.section_key, "name": s.display_name,
                               "page": s.page, "rows": len(s.dataframe)} for s in merged],
        "duration":          dur,
        "avg_quality_score": avg_score,
        "validation_scores": {k: {"score": v.score, "anomalies_count": len(v.anomalies)}
                              for k, v in validation_scores.items()},
        "ai_metadata":       ai_metadata,
        "explanations":      explanations,
    }


def process_directory(input_dir, output_dir=None, recursive=False,
                       ai_refine=False, validate_only=False):
    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs    = sorted(input_dir.glob("**/*.pdf" if recursive else "*.pdf"))
    results = []
    for i, pdf in enumerate(pdfs, 1):
        log.info(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            r = extract_pdf(str(pdf),
                            str(output_dir / f"{pdf.stem}_extrait_v36.xlsx"),
                            ai_refine=ai_refine, validate_only=validate_only)
            r["source"] = pdf.name; results.append(r)
        except Exception as e:
            log.error(f" {pdf.name} : {e}")
            results.append({"success": False, "source": pdf.name, "error": str(e)})
    ok = sum(1 for r in results if r.get("success"))
    log.info(f"Batch terminé : {ok}/{len(pdfs)} OK")
    return results


def main():
    from . import config as cfg
    import argparse

    p = argparse.ArgumentParser(
        description="Extracteur PCM Marocain v36 — Architecture Hybride Heuristique + IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m pcm_extractor bilan.pdf
  python -m pcm_extractor bilan.pdf --ai-refine
  python -m pcm_extractor bilan.pdf --ai-refine --ai-backend ollama
  python -m pcm_extractor bilan.pdf --ai-refine --ai-backend anthropic
  python -m pcm_extractor bilan.pdf --validate-only
  python -m pcm_extractor dossier/ -r --ai-refine
        """)
    p.add_argument("path")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--ai-refine", action="store_true")
    p.add_argument("--ai-refine-all", action="store_true")
    p.add_argument("--ai-backend", default=cfg.AI_BACKEND,
                   choices=["ollama", "anthropic", "auto"])
    p.add_argument("--ollama-model", default=cfg.OLLAMA_MODEL)
    p.add_argument("--ollama-url", default=cfg.OLLAMA_URL)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--confidence-threshold", type=int, default=cfg.CONFIDENCE_THRESHOLD)
    p.add_argument("--ai-skip-threshold", type=int, default=cfg.AI_SKIP_THRESHOLD)
    a = p.parse_args()

    cfg.CONFIDENCE_THRESHOLD = a.confidence_threshold
    cfg.AI_SKIP_THRESHOLD    = a.ai_skip_threshold
    cfg.AI_BACKEND           = a.ai_backend
    cfg.OLLAMA_MODEL         = a.ollama_model
    cfg.OLLAMA_URL           = a.ollama_url

    if a.ai_refine or getattr(a, "ai_refine_all", False):
        cfg.AI_REFINE_ENABLED = True

    target = Path(a.path)
    try:
        if target.is_dir():
            process_directory(str(target), a.output, a.recursive,
                              ai_refine=a.ai_refine, validate_only=a.validate_only)
        elif target.is_file():
            r = extract_pdf(str(target), a.output,
                            ai_refine=a.ai_refine or getattr(a, "ai_refine_all", False),
                            validate_only=a.validate_only, explain_mode=a.explain)
            if r.get("validate_only"):
                print("\n RAPPORT DE VALIDATION :")
                for sk, data in r.get("validation_scores", {}).items():
                    emoji = "" if data["score"] >= 90 else "" if data["score"] >= 70 else ""
                    print(f"  {emoji} {SECTION_DISPLAY_NAMES.get(sk, sk)}: {data['score']:.0f}/100 "
                          f"({data['anomalies_count']} anomalie(s))")
            elif r["success"]:
                print(f"\n {r['output_path']}")
                print(f"   Exercice      : {r.get('exercise', '?')}")
                print(f"   Score qualité : {r.get('avg_quality_score', 0):.0f}/100")
                print(f"   Durée         : {r.get('duration', 0):.1f}s")
            else:
                print(f" {r.get('error')}")
                sys.exit(1)
        else:
            print(f" Chemin invalide : {a.path}")
            sys.exit(1)
    except Exception as e:
        log.exception(e); sys.exit(1)


if __name__ == "__main__":
    main()
