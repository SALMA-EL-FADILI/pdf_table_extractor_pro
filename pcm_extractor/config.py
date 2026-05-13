import re
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pcm_v23.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    log.error("pdfplumber manquant — pip install pdfplumber")

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    log.error("openpyxl manquant — pip install openpyxl")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

AI_REFINE_ENABLED    = False
CONFIDENCE_THRESHOLD = 80
AI_SKIP_THRESHOLD    = 97
AI_MODEL             = "claude-sonnet-4-20250514"
AI_MAX_TOKENS        = 2000
AI_CACHE_FILE        = "logs/ai_corrections_cache.json"
AI_BACKEND           = "auto"
OLLAMA_URL           = "http://localhost:11434"
OLLAMA_MODEL         = "mistral:7b-instruct"
OLLAMA_TIMEOUT       = 120

MIN_CHARS_PER_PAGE = 80
MIN_NUMERIC_RATIO  = 0.04
MAX_MERGE_GAP      = 15.0
LEFT_DECO_RATIO    = 0.06

RESET_SECTION_RE = re.compile(
    r"BIENS\s+EN\s+CR.DIT.BAIL|PLUS\s+OU\s+MOINS\s+VALUES|SURETES\s+R.ELLES|"
    r"OPERATIONS\s+EN\s+DEVISES|DOTATIONS\s+AUX\s+AMORTISSEMENTS\s+RELATIFS|"
    r"ETAT\s+DES\s+PLUS.VALUES|INT.R.TS\s+DES\s+EMPRUNTS|LOCATIONS\s+ET\s+BAUX|"
    r"D.TAIL\s+DES\s+STOCKS|R.PARTITION\s+DU\s+CAPITAL|CALCUL\s+DE\s+L.IMP.T|"
    r"DATATION\s+ET\s+.V.NEMENTS|ETAT\s+DES\s+DEROGATIONS|"
    r"CHANGEMENTS\s+DE\s+M.THODES|DETAIL\s+DES\s+NON.VALEURS",
    re.I
)

PCM_SECTIONS: List[Tuple[str, re.Pattern]] = [
    ("bilan_actif",             re.compile(r"BILAN\s*[\(]\s*actif",                                re.I)),
    ("bilan_passif",            re.compile(r"BILAN\s*[\(]\s*passif",                               re.I)),
    ("cpc",                     re.compile(r"COMPTE\s+DE\s+PRODUITS\s+ET\s+CHARGES?",              re.I)),
    ("esg",                     re.compile(r"ETAT\s+DES\s+SOLDES\s+DE\s+GESTION|"
                                           r"TABLEAU\s+DE\s+FORMATION\s+DES\s+RESULTATS",          re.I)),
    ("caf",                     re.compile(r"CAPACITE\s+D.AUTOFINANCEMENT(?!\s*\(C\.A\.F\)\s*-\s*AUTO)",
                                           re.I)),
    ("tableau_financement",     re.compile(r"TABLEAU\s+DE\s+FINANCEMENT",                          re.I)),
    ("tableau_immobilisations", re.compile(r"TABLEAU\s+DES\s+IMMOBILISATIONS",                     re.I)),
    ("tableau_amortissements",  re.compile(r"TABLEAU\s+DES\s+AMORTISSEMENTS",                      re.I)),
    ("tableau_provisions",      re.compile(r"TABLEAU\s+DES\s+PROVISIONS",                          re.I)),
    ("tableau_creances",        re.compile(r"TABLEAU\s+DES\s+CREANCES",                            re.I)),
    ("tableau_dettes",          re.compile(r"TABLEAU\s+DES\s+DETTES",                              re.I)),
    ("titres_participation",    re.compile(r"TABLEAU\s+DES\s+TITRES\s+DE\s+PARTICIPATION",         re.I)),
    ("detail_cpc",              re.compile(r"DETAIL\s+DES\s+POSTES\s+DU\s+C\.?P\.?C",             re.I)),
    ("resultat_fiscal",         re.compile(r"PASSAGE\s+DU\s+RESULTAT\s+NET\s+COMPTABLE",           re.I)),
    ("determination_resultat",  re.compile(r"DETERMINATION\s+DU\s+RESULTAT\s+COURANT",             re.I)),
    ("affectation_resultats",   re.compile(r"AFFECTATION\s+DES\s+RESULTATS",                       re.I)),
    ("resultats_3ans",          re.compile(r"ELEMENTS\s+CARACTERISTIQUES|"
                                           r"TROIS\s+DERNIERS\s+EXERCICES",                        re.I)),
]

KNOWN_PCM_KEYS: set = {k for k, _ in PCM_SECTIONS}

MULTI_PAGE_SECTIONS: set = {
    "cpc", "esg", "caf", "tableau_financement",
    "tableau_immobilisations", "detail_cpc",
}

SECTION_DISPLAY_NAMES: Dict[str, str] = {
    "bilan_actif":             "Bilan Actif",
    "bilan_passif":            "Bilan Passif",
    "cpc":                     "CPC",
    "esg":                     "ESG — TFR",
    "caf":                     "CAF",
    "tableau_financement":     "Tableau Financement",
    "tableau_immobilisations": "Tableau Immobilisations",
    "tableau_amortissements":  "Tableau Amortissements",
    "tableau_provisions":      "Tableau Provisions",
    "tableau_creances":        "Tableau Créances",
    "tableau_dettes":          "Tableau Dettes",
    "titres_participation":    "Titres Participation",
    "detail_cpc":              "Détail Postes CPC",
    "resultat_fiscal":         "Résultat Fiscal",
    "determination_resultat":  "Résultat Courant",
    "affectation_resultats":   "Affectation Résultats",
    "resultats_3ans":          "Résultats 3 Exercices",
}

HEADER_KEYWORDS: Dict[str, List[Tuple[str, str]]] = {
    "bilan_actif": [
        (r"Brut", "Brut"), (r"Amortissements", "Amort. & Prov."),
        (r"^Net$", "Net N"), (r"PRECEDENT", "Net N-1"),
    ],
    "bilan_passif": [
        (r"EXERCICE\s*$", "Exercice N"), (r"PRECEDENT", "Exercice N-1"),
    ],
    "cpc": [
        (r"Propres\s+[àa]", "Propres exercice"), (r"Concernant", "Exercices préc."),
        (r"TOTAUX\s+DE\s+L", "Total N"), (r"PRECEDENT", "Total N-1"),
    ],
    "esg": [
        (r"EXERCICE\b", "Exercice N"), (r"PRECEDENT", "Exercice N-1"),
    ],
    "caf": [
        (r"EXERCICE\b", "Exercice N"), (r"PRECEDENT", "Exercice N-1"),
    ],
    "tableau_financement": [
        (r"^Emplois$", "Emplois"), (r"^Ressources$", "Ressources"),
        (r"EXERCICE\b", "Exercice N"), (r"PRECEDENT", "Exercice N-1"),
    ],
    "tableau_immobilisations": [
        (r"DEBUT|D.BUT\s+EX", "Début"), (r"Acquisition", "Acquisitions"),
        (r"FIN\s+EXERCICE|BRUT\s+FIN", "Fin"),
    ],
    "tableau_amortissements": [
        (r"Cumul\s+d.but|début\s+exercice", "Cumul début"),
        (r"Dotation", "Dotation"), (r"sorties", "Sorties"),
        (r"Cumul.*fin|fin\s+exerc", "Cumul fin"),
    ],
    "tableau_provisions": [
        (r"d.but|début\s+exercice", "Début"), (r"exploitation", "Dot. Exploit."),
        (r"financ", "Dot. Fin."), (r"non\s+cour|courantes", "Dot. Non-Cour."),
        (r"fin\s+d.exercice|fin\s+exerc", "Fin"),
    ],
    "tableau_creances": [
        (r"TOTAL", "Total"), (r"Plus", "Plus d'1 an"),
        (r"Moins", "Moins d'1 an"), (r"Echues", "Échues"),
    ],
    "tableau_dettes": [
        (r"TOTAL", "Total"), (r"Plus", "Plus d'1 an"),
        (r"Moins", "Moins d'1 an"), (r"Echues", "Échues"),
    ],
    "detail_cpc": [
        (r"EXERCICE\b", "Exercice N"), (r"PRECEDENT", "Exercice N-1"),
    ],
    "resultat_fiscal": [
        (r"B.n.fice\s+net|Bénéfice\s+net", "Bénéfice net"),
        (r"D.ficit\s+net|Déficit\s+net", "Déficit net"),
    ],
    "affectation_resultats": [(r"MONTANT", "Montant")],
    "resultats_3ans":        [(r"20[012]\d|19[89]\d", "Année")],
}

FALLBACK_ZONES_RATIOS: Dict[str, List[float]] = {
    "bilan_actif":             [0.58, 0.69, 0.80, 0.91],
    "bilan_passif":            [0.73, 0.88],
    "cpc":                     [0.47, 0.58, 0.80, 0.91],
    "esg":                     [0.73, 0.88],
    "caf":                     [0.73, 0.88],
    "tableau_financement":     [0.47, 0.60, 0.73, 0.88],
    "tableau_immobilisations": [0.30, 0.42, 0.85],
    "tableau_amortissements":  [0.45, 0.57, 0.75, 0.88],
    "tableau_provisions":      [0.23, 0.32, 0.45, 0.54, 0.63, 0.72, 0.87],
    "tableau_creances":        [0.28, 0.42, 0.54, 0.70],
    "tableau_dettes":          [0.28, 0.42, 0.54, 0.70],
    "titres_participation":    [0.25, 0.32, 0.45, 0.54, 0.72, 0.87],
    "detail_cpc":              [0.73, 0.88],
    "resultat_fiscal":         [0.58, 0.73],
    "determination_resultat":  [0.80],
    "affectation_resultats":   [0.47, 0.85],
    "resultats_3ans":          [0.67, 0.79, 0.91],
}

FALLBACK_LMAX_RATIO: Dict[str, float] = {
    "bilan_actif":             0.35,
    "bilan_passif":            0.60,
    "cpc":                     0.35,
    "esg":                     0.60,
    "caf":                     0.60,
    "tableau_financement":     0.30,
    "tableau_immobilisations": 0.22,
    "tableau_amortissements":  0.35,
    "tableau_provisions":      0.20,
    "tableau_creances":        0.20,
    "tableau_dettes":          0.20,
    "titres_participation":    0.20,
    "detail_cpc":              0.20,
    "resultat_fiscal":         0.48,
    "determination_resultat":  0.52,
    "affectation_resultats":   0.36,
    "resultats_3ans":          0.55,
}

MIN_N_COLS: Dict[str, int] = {
    "bilan_actif":             4,
    "bilan_passif":            2,
    "cpc":                     2,
    "esg":                     2,
    "caf":                     2,
    "tableau_financement":     3,
    "tableau_immobilisations": 2,
    "tableau_amortissements":  3,
    "tableau_provisions":      5,
    "tableau_creances":        2,
    "tableau_dettes":          2,
    "titres_participation":    2,
    "detail_cpc":              2,
    "resultat_fiscal":         1,
    "determination_resultat":  1,
    "affectation_resultats":   1,
    "resultats_3ans":          2,
}

_NOISE_RE = re.compile(
    r"^(Tableau\s+n[°o]?\s*\d*|Tableau\s+\d+|Exercice\s+(clos|du)|"
    r"mod.le\s+normal|mod.le\s+simplifi.|Etats?\s+financiers|Page\s+\d+|"
    r"EXERCICE\s+SOCIAL|REF\.\s*[A-Z]|R.F\.\s*[A-Z]|N°\s*CIN|"
    r"[A-Za-z\s]{2,30}\.(COM|MA|net|org)\s*$|"
    r"BILAN\s*[\(]|COMPTE\s+DE\s+PRODUITS|ETAT\s+DES\s+SOLDES|"
    r"TABLEAU\s+DE\s+F|TABLEAU\s+DES\s+[A-Z]|CAPACITE\s+D|"
    r"NATURE\s*$|RUBRIQUES|SYNTHESE\s+DES|DU\s+BILAN|"
    r"EMPLOIS\s+ET|Emplois\s*$|Ressources\s*$|OPERATIONS\s*$|"
    r"Propres\s+[àa]|Concernant\s+les|TOTAUX\s+DE|PRECEDENT\s*$|"
    r"l['\']exercice\s*$|exercices?\s+pr.c.dents\s*$|"
    r"ACTIF\s*$|Brut\s*$|Net\s*$|Amortissements?\s*$|"
    r"EXERCICE\s*$|\d+\s*=\s*\d+|\d+\s*\+\s*\d+|"
    r"\([12]\)\s*(Capital|B.n.ficiaire)|Capital\s+personnel|"
    r"Analyse\s+par|par\s+.ch.ance|Autres\s+analyses|"
    r"Montants\s*$|TOTAL\s*$|Plus\s*$|Moins\s*$|Echues?\s*$|"
    r"d['\']un\s*$|an\s*$|non\s+recouvrées|devises\s*$|"
    r"organis\.\s*publics|entrepris\.\s+liées|représentés\s+par|effets\s*$|"
    r"POSTE\s*$|Etat\s+[A-Z]\d+\s*$|I\.F\s*:|ASSOCIATION\s+|"
    r"DETAIL\s+DES\s+POSTES|CHARGES\s+D.EXPLOITATION\s*$|"
    r"PRODUITS\s+D.EXPLOITATION\s*$|CHARGES\s+FINANCIERES\s*$|"
    r"PRODUITS\s+FINANCIERS\s*$|CHARGES\s+NON\s+COURANTES\s*$|"
    r"PRODUITS\s+NON\s+COURANTS\s*$|"
    r"[AÀ]\s+l['\']exclusion\s+des\s+dot|[AÀ]\s+l['\']exclusion\s+des\s+rep|"
    r"Y\s+compris\s+reprises\s+sur|NOTA\s*:\s*Le\s+calcul|"
    r"Dans\s+la\s+limite\s+du\s+montant|Achats\s+revendus\s+ou\s+consomm|"
    r"Variation\s+de\s+stocks\s*:\s*stock\s+final|\(1\)\s*Capital\s+personnel|"
    r"\(2\)\s*B.n.ficiaire|\(1\)\s*Quand\s+le\s+nombre|"
    r"^[A-Z]{1,2}\s*$|^[A-Z](\s+[A-Z]){1,6}$|^\W{1,3}$)",
    re.I,
)

_TOTAL_RE = re.compile(
    r"\bTOTAL\b|\bMARGE\b|\bVALEUR\s+AJOUTEE\b|\bEXCEDENT\b|"
    r"\bCAF\b|\bAUTOFINANCEMENT\b|\bCAPACITE\b|\bRESULTAT\b|"
    r"\bINSUFFISANCE\b|\bFINANCEMENT\s+PERMANENT\b|"
    r"\bBENEFICE\s+NET\b|\bDEFICIT\s+NET\b|\bRESULTAT\s+BRUT\b|"
    r"\bSOLDE\s+DE\s+TRESORERIE\b|\bVARIATION\s+DU\s+BFR\b|"
    r"\bIBE\b|\bINSUFFISANCE\s+BRUTE\b|\bEBE\b",
    re.I)

COLORS = {
    "hbg": "1F3A5F",
    "hfg": "FFFFFF",
    "sbg": "2E5F8A",
    "tbg": "D6E4F0",
    "alt": "F0F7FF",
}

CGNC_SECTION_RULES: Dict[str, str] = {
    "bilan_actif": """Structure CGNC Bilan Actif :
- Colonnes : Brut | Amort. & Prov. | Net Exercice N | Net Exercice N-1
- Règle fondamentale : Net = Brut - Amortissements (toujours vrai par ligne)
- ACTIF IMMOBILISÉ : Non-valeurs + Incorporelles + Corporelles + Financières + Écarts
- ACTIF CIRCULANT : Stocks + Créances + TVP + Écarts
- TRÉSORERIE ACTIF : Chèques + Banques/TG/CP + Caisses
- TOTAL GÉNÉRAL = ACTIF IMMOBILISÉ + ACTIF CIRCULANT + TRÉSORERIE
- Les lignes avec Brut=None mais Net≠None : le Brut = Net (trésorerie, non amortissable)
- Valeurs toujours positives (sauf cas exceptionnels)""",

    "bilan_passif": """Structure CGNC Bilan Passif :
- Colonnes : Exercice N | Exercice N-1
- FINANCEMENT PERMANENT = Capitaux Propres + Cap.Propres Assimilés + Dettes Financement + Provisions durables + Écarts
- PASSIF CIRCULANT = Dettes fournisseurs + Personnel + Organismes sociaux + État + Autres + Provisions courantes + Écarts
- TRÉSORERIE PASSIF = Crédits escompte + Crédits trésorerie + Banques soldes créditeurs
- TOTAL GÉNÉRAL = FINANCEMENT PERMANENT + PASSIF CIRCULANT + TRÉSORERIE PASSIF
- TOTAL GÉNÉRAL doit égaler TOTAL GÉNÉRAL du Bilan Actif (équilibre comptable)
- RÉSULTAT NET DE L'EXERCICE : positif si bénéfice, négatif si perte""",

    "cpc": """Structure CGNC CPC (Compte de Produits et Charges) :
- Colonnes : Propres exercice | Exercices préc. | Total N | Total N-1
- Total N = Propres exercice + Exercices préc.
- RÉSULTAT D'EXPLOITATION = PRODUITS I - CHARGES II
- RÉSULTAT FINANCIER = PRODUITS IV - CHARGES V
- RÉSULTAT COURANT = RÉSULTAT D'EXPLOITATION + RÉSULTAT FINANCIER
- RÉSULTAT NET = Total Produits - Total Charges""",

    "esg": """Structure CGNC ESG :
- VALEUR AJOUTÉE = Marge Brute + Production - Consommation
- EBE/IBE = Valeur Ajoutée + Subventions - Impôts - Charges personnel
- RÉSULTAT D'EXPLOITATION = EBE + Autres produits - Autres charges + Reprises - Dotations
- RÉSULTAT COURANT = Résultat exploitation + Résultat financier
- CAF = Résultat Net + Dotations - Reprises - Produits cessions + VNA cessions""",

    "tableau_financement": """Structure CGNC Tableau de Financement :
- PARTIE I : Financement Permanent, Actif Immobilisé, FRF, BFG, Trésorerie Nette
- PARTIE II : Ressources stables vs Emplois stables
- FRF = Financement Permanent - Actif Immobilisé
- Trésorerie Nette = FRF - BFG""",

    "tableau_immobilisations": """Structure CGNC Tableau des Immobilisations :
- Colonnes : Montant Début | Acquisitions | Production | Cessions/Retraits | Montant Fin
- Règle : Montant Fin = Montant Début + Acquisitions + Production - Cessions/Retraits""",

    "tableau_amortissements": """Structure CGNC Tableau des Amortissements :
- Colonnes : Cumul début exercice | Dotation exercice | Amort. sur sorties | Cumul fin exercice
- Règle : Cumul fin = Cumul début + Dotation - Amort. sur sorties""",

    "resultat_fiscal": """Structure CGNC Résultat Fiscal :
- Colonnes : Bénéfice net fiscal | Déficit net fiscal
- Les réintégrations fiscales augmentent le bénéfice imposable""",

    "affectation_resultats": """Structure CGNC Affectation des Résultats :
- Colonne A : Report à nouveau N-1 + Résultat net + Prélèvements réserves = TOTAL A
- Colonne B : Réserve légale + Autres réserves + Dividendes + Report à nouveau N = TOTAL B
- TOTAL A = TOTAL B (équilibre obligatoire)""",
}
