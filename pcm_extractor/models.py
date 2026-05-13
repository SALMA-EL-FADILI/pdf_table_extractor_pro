import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import pandas as pd


def normalize_text(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


_CLEAN_NUM_RE = re.compile(r"[\s\u202f\xa0†]")
_NUM_FRAG     = re.compile(r"^-?\(?\d[\d,.'†\s]*\)?$")
_ALPHA_RE     = re.compile(r"[a-zA-ZÀ-ÿ•]")


def parse_amount(s: str) -> Optional[float]:
    if not s or not s.strip():
        return None
    s   = s.strip()
    s   = s.replace("–", "-").replace("—", "-").replace("\u2009", "").replace("\u202f", "").replace("\xa0", "")
    neg = s.startswith("(") or s.startswith("-")
    s   = s.strip("()- \u202f\xa0†")
    s   = _CLEAN_NUM_RE.sub("", s).replace(",", ".")
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    if not s or not re.search(r"\d", s):
        return None
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


@dataclass
class Token:
    text: str; x0: float; x1: float; y: float
    is_numeric: bool = False; value: Optional[float] = None


@dataclass
class PageLine:
    y: float; tokens: List[Token] = field(default_factory=list); page: int = 0

    def label_text(self, max_x: float) -> str:
        return " ".join(
            t.text for t in self.tokens if t.x0 < max_x and not t.is_numeric
        ).strip()

    def value_tokens(self, min_x: float) -> List[Token]:
        return [t for t in self.tokens if t.is_numeric and t.x0 >= min_x]

    def full_text(self) -> str:
        return " ".join(t.text for t in self.tokens)


@dataclass
class ParsedSection:
    section_key: str; display_name: str; page: int; dataframe: pd.DataFrame


@dataclass
class ValidationAnomaly:
    section_key: str
    rule: str
    severity: str
    description: str
    row_indices: List[int] = field(default_factory=list)
    expected: Optional[Any] = None
    observed: Optional[Any] = None


@dataclass
class SectionScore:
    section_key: str
    score: float
    anomalies: List[ValidationAnomaly] = field(default_factory=list)
    needs_ai: bool = False
