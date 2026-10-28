"""Reaction Lab database used to restrict the Inverse Experiment engine.

Every reaction listed here can be reproduced in the frontend Reaction Lab
(the 14 reagents exposed in the Solution A / Solution B dropdowns). The AI is
only allowed to *pick* one of these IDs; all chemistry shown to the student
(reagents, equation, outcome) is read from this table, never from AI text.

Keep this file in sync with ``js/lab-reactions.js`` (same IDs and pairs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reagents available in the Reaction Lab dropdowns
# ---------------------------------------------------------------------------

LAB_REAGENTS: Dict[str, Dict[str, object]] = {
    "AgNO3": {"name": "Silver Nitrate", "aliases": ["agno3", "silver nitrate", "perak nitrat"]},
    "Pb(NO3)2": {
        "name": "Lead(II) Nitrate",
        "aliases": ["pb(no3)2", "pbno32", "lead nitrate", "lead(ii) nitrate", "timbal nitrat", "timbal(ii) nitrat"],
    },
    "BaCl2": {"name": "Barium Chloride", "aliases": ["bacl2", "barium chloride", "barium klorida"]},
    "CaCl2": {"name": "Calcium Chloride", "aliases": ["cacl2", "calcium chloride", "kalsium klorida"]},
    "NaCl": {
        "name": "Sodium Chloride",
        "aliases": ["nacl", "sodium chloride", "natrium klorida", "table salt", "garam dapur"],
    },
    "KI": {"name": "Potassium Iodide", "aliases": ["ki", "potassium iodide", "kalium iodida"]},
    "Na2SO4": {
        "name": "Sodium Sulfate",
        "aliases": ["na2so4", "sodium sulfate", "sodium sulphate", "natrium sulfat"],
    },
    "Na2CO3": {
        "name": "Sodium Carbonate",
        "aliases": ["na2co3", "sodium carbonate", "washing soda", "natrium karbonat", "soda abu"],
    },
    "HCl": {"name": "Hydrochloric Acid", "aliases": ["hcl", "hydrochloric acid", "asam klorida"]},
    "CH3COOH": {
        "name": "Acetic Acid",
        "aliases": ["ch3cooh", "acetic acid", "vinegar", "asam asetat", "asam cuka", "cuka"],
    },
    "NaOH": {
        "name": "Sodium Hydroxide",
        "aliases": ["naoh", "sodium hydroxide", "natrium hidroksida", "soda api", "lye"],
    },
    "NaHCO3": {
        "name": "Sodium Bicarbonate",
        "aliases": [
            "nahco3", "baking soda", "sodium bicarbonate", "sodium hydrogen carbonate",
            "soda kue", "natrium bikarbonat",
        ],
    },
    "Ca(OH)2": {
        "name": "Calcium Hydroxide",
        "aliases": ["ca(oh)2", "caoh2", "limewater", "lime water", "calcium hydroxide", "air kapur", "kalsium hidroksida"],
    },
    "NH3": {
        "name": "Aqueous Ammonia",
        "aliases": ["nh3", "ammonia", "amonia", "ammonium hydroxide", "nh4oh"],
    },
}

# ---------------------------------------------------------------------------
# Observable products the lab can show
# ---------------------------------------------------------------------------

PRODUCTS: Dict[str, Dict[str, object]] = {
    "AgCl": {"name": "Silver Chloride", "color": "white",
             "keywords": ["agcl", "silver chloride", "perak klorida"]},
    "AgI": {"name": "Silver Iodide", "color": "yellow",
            "keywords": ["agi", "silver iodide", "perak iodida"]},
    "PbI2": {"name": "Lead(II) Iodide", "color": "yellow",
             "keywords": ["pbi2", "lead iodide", "lead(ii) iodide", "timbal iodida", "timbal(ii) iodida",
                          "golden rain", "hujan emas"]},
    "PbCl2": {"name": "Lead(II) Chloride", "color": "white",
              "keywords": ["pbcl2", "lead chloride", "lead(ii) chloride", "timbal klorida"]},
    "PbCO3": {"name": "Lead(II) Carbonate", "color": "white",
              "keywords": ["pbco3", "lead carbonate", "lead(ii) carbonate", "timbal karbonat"]},
    "BaSO4": {"name": "Barium Sulfate", "color": "white",
              "keywords": ["baso4", "barium sulfate", "barium sulphate", "barium sulfat"]},
    "BaCO3": {"name": "Barium Carbonate", "color": "white",
              "keywords": ["baco3", "barium carbonate", "barium karbonat"]},
    "CaCO3": {"name": "Calcium Carbonate", "color": "white",
              "keywords": ["caco3", "calcium carbonate", "kalsium karbonat", "chalk", "kapur tulis",
                           "limestone", "batu kapur"]},
    "CO2": {"name": "Carbon Dioxide", "color": "colorless",
            "keywords": ["co2", "carbon dioxide", "karbon dioksida", "karbondioksida"]},
    "NH3": {"name": "Ammonia Gas", "color": "colorless",
            "keywords": ["ammonia gas", "gas amonia", "gas ammonia", "nh3 gas", "gas nh3", "bau amonia",
                         "ammonia smell"]},
    "H2O": {"name": "Water (Neutralization)", "color": "colorless",
            "keywords": ["h2o", "salt and water", "garam dan air"]},
}


@dataclass(frozen=True)
class LabReaction:
    id: str
    type: str  # "precipitate" | "gas" | "neutralization"
    reactant_a: str
    reactant_b: str
    product: str
    equation: str
    conc_a: float = 1.0
    conc_b: float = 1.0
    priority: int = 0

    @property
    def product_info(self) -> Dict[str, object]:
        return PRODUCTS[self.product]

    @property
    def color(self) -> str:
        return str(self.product_info["color"])

    @property
    def gas(self) -> Optional[str]:
        return self.product if self.type == "gas" else None

    @property
    def precipitate(self) -> Optional[str]:
        return self.product if self.type == "precipitate" else None

    @property
    def ph_final(self) -> str:
        if self.type == "neutralization":
            return "7.0"
        if self.type == "gas" and self.product == "CO2":
            return "~6.5"
        return "~7.0"

    @property
    def name(self) -> str:
        info = self.product_info
        if self.type == "precipitate":
            return f"{info['name']} Precipitate ({self.product})"
        if self.type == "gas":
            return f"{info['name']} Gas Evolution ({self.product}↑)" if self.product == "CO2" \
                else f"{info['name']} Evolution (NH₃↑)"
        return f"Acid-Base Neutralization ({LAB_REAGENTS[self.reactant_a]['name']} + " \
               f"{LAB_REAGENTS[self.reactant_b]['name']})"

    @property
    def explanation(self) -> str:
        name_a = LAB_REAGENTS[self.reactant_a]["name"]
        name_b = LAB_REAGENTS[self.reactant_b]["name"]
        info = self.product_info
        if self.type == "precipitate":
            return (
                f"Mixing {name_a} ({self.reactant_a}) with {name_b} ({self.reactant_b}) brings together ions "
                f"whose product is insoluble in water, so a {info['color']} solid of {info['name']} "
                f"({self.product}) precipitates out. The remaining ions stay dissolved as spectator ions."
            )
        if self.type == "gas" and self.product == "CO2":
            return (
                f"The acid ({self.reactant_a if self.reactant_a in ('HCl', 'CH3COOH') else self.reactant_b}) "
                f"donates H⁺ to the carbonate/bicarbonate ions, forming unstable carbonic acid (H₂CO₃) that "
                f"immediately decomposes into water and carbon dioxide gas, seen as fizzing bubbles."
            )
        if self.type == "gas":
            return (
                f"Hydroxide ions (OH⁻) from {name_b if self.reactant_a == 'NH3' else name_a} push the "
                f"ammonium/ammonia equilibrium toward free ammonia (NH₄⁺ + OH⁻ → NH₃ + H₂O), releasing "
                f"pungent ammonia gas."
            )
        return (
            f"H⁺ ions from the acid react with OH⁻ ions from the base (H⁺ + OH⁻ → H₂O). Using matching "
            f"amounts of {name_a} and {name_b} brings the solution close to neutral pH and releases heat "
            f"(exothermic)."
        )


def _r(rid: str, rtype: str, a: str, b: str, product: str, equation: str,
       conc_a: float = 1.0, conc_b: float = 1.0, priority: int = 0) -> LabReaction:
    return LabReaction(rid, rtype, a, b, product, equation, conc_a, conc_b, priority)


# Every pair below triggers a rule in the Reaction Lab engine (js/chemistry-engine.js).
SUPPORTED_REACTIONS: Tuple[LabReaction, ...] = (
    # --- Precipitation -----------------------------------------------------
    _r("agcl_nacl", "precipitate", "AgNO3", "NaCl", "AgCl",
       "AgNO₃(aq) + NaCl(aq) → AgCl(s)↓ + NaNO₃(aq)", priority=10),
    _r("agcl_bacl2", "precipitate", "AgNO3", "BaCl2", "AgCl",
       "2AgNO₃(aq) + BaCl₂(aq) → 2AgCl(s)↓ + Ba(NO₃)₂(aq)", conc_a=2.0, conc_b=1.0, priority=3),
    _r("agcl_cacl2", "precipitate", "AgNO3", "CaCl2", "AgCl",
       "2AgNO₃(aq) + CaCl₂(aq) → 2AgCl(s)↓ + Ca(NO₃)₂(aq)", conc_a=2.0, conc_b=1.0, priority=3),
    _r("agcl_hcl", "precipitate", "AgNO3", "HCl", "AgCl",
       "AgNO₃(aq) + HCl(aq) → AgCl(s)↓ + HNO₃(aq)", priority=4),
    _r("agi_ki", "precipitate", "AgNO3", "KI", "AgI",
       "AgNO₃(aq) + KI(aq) → AgI(s)↓ + KNO₃(aq)", priority=6),
    _r("pbi2_ki", "precipitate", "Pb(NO3)2", "KI", "PbI2",
       "Pb(NO₃)₂(aq) + 2KI(aq) → PbI₂(s)↓ + 2KNO₃(aq)", conc_a=1.0, conc_b=2.0, priority=9),
    _r("pbcl2_nacl", "precipitate", "Pb(NO3)2", "NaCl", "PbCl2",
       "Pb(NO₃)₂(aq) + 2NaCl(aq) → PbCl₂(s)↓ + 2NaNO₃(aq)", conc_a=1.0, conc_b=2.0, priority=5),
    _r("pbcl2_bacl2", "precipitate", "Pb(NO3)2", "BaCl2", "PbCl2",
       "Pb(NO₃)₂(aq) + BaCl₂(aq) → PbCl₂(s)↓ + Ba(NO₃)₂(aq)", priority=2),
    _r("pbcl2_cacl2", "precipitate", "Pb(NO3)2", "CaCl2", "PbCl2",
       "Pb(NO₃)₂(aq) + CaCl₂(aq) → PbCl₂(s)↓ + Ca(NO₃)₂(aq)", priority=2),
    _r("pbcl2_hcl", "precipitate", "Pb(NO3)2", "HCl", "PbCl2",
       "Pb(NO₃)₂(aq) + 2HCl(aq) → PbCl₂(s)↓ + 2HNO₃(aq)", conc_a=1.0, conc_b=2.0, priority=2),
    _r("pbco3_na2co3", "precipitate", "Pb(NO3)2", "Na2CO3", "PbCO3",
       "Pb(NO₃)₂(aq) + Na₂CO₃(aq) → PbCO₃(s)↓ + 2NaNO₃(aq)", priority=4),
    _r("baso4_na2so4", "precipitate", "BaCl2", "Na2SO4", "BaSO4",
       "BaCl₂(aq) + Na₂SO₄(aq) → BaSO₄(s)↓ + 2NaCl(aq)", priority=7),
    _r("baco3_na2co3", "precipitate", "BaCl2", "Na2CO3", "BaCO3",
       "BaCl₂(aq) + Na₂CO₃(aq) → BaCO₃(s)↓ + 2NaCl(aq)", priority=4),
    _r("caco3_cacl2", "precipitate", "CaCl2", "Na2CO3", "CaCO3",
       "CaCl₂(aq) + Na₂CO₃(aq) → CaCO₃(s)↓ + 2NaCl(aq)", priority=8),
    _r("caco3_caoh2", "precipitate", "Ca(OH)2", "Na2CO3", "CaCO3",
       "Ca(OH)₂(aq) + Na₂CO₃(aq) → CaCO₃(s)↓ + 2NaOH(aq)", priority=6),
    # --- Gas evolution -----------------------------------------------------
    _r("co2_hcl_nahco3", "gas", "NaHCO3", "HCl", "CO2",
       "NaHCO₃(aq) + HCl(aq) → NaCl(aq) + H₂O(l) + CO₂(g)↑", priority=10),
    _r("co2_ch3cooh_nahco3", "gas", "CH3COOH", "NaHCO3", "CO2",
       "CH₃COOH(aq) + NaHCO₃(aq) → CH₃COONa(aq) + H₂O(l) + CO₂(g)↑", priority=9),
    _r("co2_hcl_na2co3", "gas", "Na2CO3", "HCl", "CO2",
       "Na₂CO₃(aq) + 2HCl(aq) → 2NaCl(aq) + H₂O(l) + CO₂(g)↑", conc_a=1.0, conc_b=2.0, priority=6),
    _r("co2_ch3cooh_na2co3", "gas", "CH3COOH", "Na2CO3", "CO2",
       "2CH₃COOH(aq) + Na₂CO₃(aq) → 2CH₃COONa(aq) + H₂O(l) + CO₂(g)↑", conc_a=2.0, conc_b=1.0, priority=5),
    _r("nh3_naoh", "gas", "NH3", "NaOH", "NH3",
       "NH₄⁺(aq) + OH⁻(aq) → NH₃(g)↑ + H₂O(l)", priority=8),
    _r("nh3_caoh2", "gas", "NH3", "Ca(OH)2", "NH3",
       "2NH₄⁺(aq) + Ca(OH)₂(aq) → 2NH₃(g)↑ + 2H₂O(l) + Ca²⁺(aq)", priority=5),
    # --- Neutralization ----------------------------------------------------
    _r("neut_hcl_naoh", "neutralization", "HCl", "NaOH", "H2O",
       "HCl(aq) + NaOH(aq) → NaCl(aq) + H₂O(l)", priority=10),
    _r("neut_hcl_caoh2", "neutralization", "HCl", "Ca(OH)2", "H2O",
       "2HCl(aq) + Ca(OH)₂(aq) → CaCl₂(aq) + 2H₂O(l)", conc_a=2.0, conc_b=1.0, priority=5),
    _r("neut_hcl_nh3", "neutralization", "HCl", "NH3", "H2O",
       "HCl(aq) + NH₃(aq) → NH₄Cl(aq)", priority=5),
    _r("neut_ch3cooh_naoh", "neutralization", "CH3COOH", "NaOH", "H2O",
       "CH₃COOH(aq) + NaOH(aq) → CH₃COONa(aq) + H₂O(l)", priority=7),
    _r("neut_ch3cooh_caoh2", "neutralization", "CH3COOH", "Ca(OH)2", "H2O",
       "2CH₃COOH(aq) + Ca(OH)₂(aq) → Ca(CH₃COO)₂(aq) + 2H₂O(l)", conc_a=2.0, conc_b=1.0, priority=4),
    _r("neut_ch3cooh_nh3", "neutralization", "CH3COOH", "NH3", "H2O",
       "CH₃COOH(aq) + NH₃(aq) → CH₃COONH₄(aq)", priority=4),
)

REACTIONS_BY_ID: Dict[str, LabReaction] = {r.id: r for r in SUPPORTED_REACTIONS}

SUGGESTIONS: List[Dict[str, str]] = [
    {"label": "🟡 Yellow Precipitate", "prompt": "Form a yellow precipitate (PbI2)"},
    {"label": "⚪ White Precipitate", "prompt": "Form a white precipitate of Silver Chloride (AgCl)"},
    {"label": "💨 CO₂ Gas Evolution", "prompt": "Produce effervescent carbon dioxide (CO2) gas"},
    {"label": "⚖️ Neutral pH (7.0)", "prompt": "Neutralize acid and base to reach neutral pH 7.0"},
    {"label": "🧱 Chalk (CaCO₃)", "prompt": "Form white chalk precipitate (CaCO3)"},
    {"label": "🧪 Ammonia Gas", "prompt": "Release ammonia gas (NH3)"},
]

# Lab slider limits (index.html): concentration 0.1–2.0 M, volume 10–100 mL, temperature 0–100 °C.
CONC_RANGE = (0.1, 2.0)
VOL_RANGE = (10.0, 100.0)
TEMP_RANGE = (0.0, 100.0)

# ---------------------------------------------------------------------------
# Goal matching
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "gas": ["gas", "gases", "bubble", "bubbles", "bubbling", "fizz", "fizzy", "fizzing", "effervescent",
            "effervescence", "foam", "foaming", "gelembung", "buih", "berbuih", "busa", "berbusa"],
    "precipitate": ["precipitate", "precipitates", "precipitation", "solid", "endapan", "mengendap",
                    "padatan", "ppt", "insoluble", "tidak larut"],
    "neutralization": ["neutral", "neutralize", "neutralise", "neutralization", "neutralisation",
                       "netral", "penetralan", "netralisasi", "ph 7", "ph7", "ph 7.0", "acid base",
                       "acid-base", "asam basa", "asam-basa"],
}

COLOR_KEYWORDS: Dict[str, List[str]] = {
    "white": ["white", "putih", "milky"],
    "yellow": ["yellow", "kuning", "golden", "gold", "emas"],
}

UNSUPPORTED_COLOR_KEYWORDS: List[str] = [
    "blue", "biru", "red", "merah", "green", "hijau", "brown", "coklat", "cokelat", "black", "hitam",
    "orange", "jingga", "oranye", "purple", "ungu", "pink", "violet",
]

UNSUPPORTED_TERMS: List[str] = [
    "hydrogen", "hidrogen", "oxygen", "oksigen", "chlorine", "klorin", "gas klor", "copper", "tembaga",
    "iron", "besi", "zinc", "seng", "magnesium", "aluminium", "aluminum", "sulfur dioxide", "belerang",
    "hydrogen sulfide", "nitrogen", "bromide", "bromida", "bromine", "permanganate", "permanganat",
    "silver mirror", "cermin perak", "rust", "karat", "combustion", "pembakaran", "explosion", "ledakan",
    "electrolysis", "elektrolisis", "redox", "displacement",
]

# Phrases that contain an "unsupported" word but refer to a supported reagent.
SAFE_PHRASES: List[str] = ["hydrogen carbonate", "hidrogen karbonat"]

ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "U", "W", "Ir", "Os", "Re", "Ta", "Hf",
}

# Formulas (lowercase, no parentheses) that belong to the lab's chemistry.
KNOWN_FORMULAS = {
    # reagents
    "agno3", "pbno32", "bacl2", "cacl2", "nacl", "ki", "na2so4", "na2co3", "hcl", "ch3cooh", "naoh",
    "nahco3", "caoh2", "nh3",
    # products
    "agcl", "agi", "pbi2", "pbcl2", "pbco3", "baso4", "baco3", "caco3", "co2", "h2o",
    # by-products, ions and intermediates
    "nano3", "kno3", "bano32", "cano32", "hno3", "nh4cl", "ch3coona", "cach3coo2", "ch3coonh4", "h2co3",
    "nh4oh", "nh4", "co3", "hco3", "so4", "no3", "oh", "ch3coo",
}

# Tokens that look like formulas but are ordinary words/abbreviations.
FORMULA_IGNORE = {"ph", "ph7", "ok", "no", "hi"}

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_CHARS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]")
_CHARGE = re.compile(r"(?<=[A-Za-z0-9)])\^?\d?[+\-](?![A-Za-z0-9])")
_UPPER_FORMULA = re.compile(r"(?<![A-Za-z0-9])((?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*)+)(?![A-Za-z0-9])")
_LOWER_FORMULA = re.compile(r"(?<![a-z0-9])([a-z][a-z0-9()]*\d[a-z0-9()]*)(?![a-z0-9])")
_ELEMENT_SPLIT = re.compile(r"[A-Z][a-z]?")


def normalize_text(text: str) -> str:
    """Normalize subscripts, strip charges/superscripts, keep original casing."""
    cleaned = str(text or "").translate(_SUBSCRIPTS)
    cleaned = _SUPERSCRIPT_CHARS.sub("", cleaned)
    cleaned = _CHARGE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains(text_lower: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text_lower) is not None


def _formula_key(token: str) -> str:
    return token.lower().replace("(", "").replace(")", "")


def find_unsupported_terms(goal_text: str) -> List[str]:
    """Return substances/colors in the goal that the Reaction Lab cannot simulate."""
    text = normalize_text(goal_text)
    lower = text.lower()
    for phrase in SAFE_PHRASES:
        lower = lower.replace(phrase, " ")

    found: List[str] = []
    for term in UNSUPPORTED_TERMS:
        if _contains(lower, term):
            found.append(term)
    for color in UNSUPPORTED_COLOR_KEYWORDS:
        if _contains(lower, color):
            found.append(color)

    # Capitalized formulas such as "BaBr2", "CuSO4", "CO", "H2".
    for match in _UPPER_FORMULA.finditer(text):
        token = match.group(1)
        symbols = _ELEMENT_SPLIT.findall(token)
        if not symbols or any(sym not in ELEMENT_SYMBOLS for sym in symbols):
            continue
        has_digit = any(ch.isdigit() for ch in token)
        if not has_digit and len(symbols) < 2:
            continue
        key = _formula_key(token)
        if key in FORMULA_IGNORE or key in KNOWN_FORMULAS:
            continue
        found.append(token)

    # Lowercase formulas that contain a digit, such as "babr2" or "h2".
    for match in _LOWER_FORMULA.finditer(lower):
        key = _formula_key(match.group(1))
        if key in FORMULA_IGNORE or key in KNOWN_FORMULAS:
            continue
        if not any(f.lower().replace("(", "").replace(")", "") == key for f in found):
            found.append(match.group(1))

    # De-duplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for item in found:
        k = item.lower()
        if k not in seen:
            seen.add(k)
            unique.append(item)
    return unique


@dataclass
class MatchResult:
    reaction: Optional[LabReaction]
    score: float = 0.0
    strong: bool = False
    unsupported_terms: List[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.reaction is not None


def detect_intent(lower: str) -> Optional[str]:
    for intent in ("gas", "neutralization", "precipitate"):
        if any(_contains(lower, kw) for kw in INTENT_KEYWORDS[intent]):
            return intent
    return None


def detect_color(lower: str) -> Optional[str]:
    for color, words in COLOR_KEYWORDS.items():
        if any(_contains(lower, w) for w in words):
            return color
    return None


def mentioned_reagents(lower: str) -> List[str]:
    hits = []
    for reagent_id, info in LAB_REAGENTS.items():
        if any(_contains(lower, alias) for alias in info["aliases"]):  # type: ignore[union-attr]
            hits.append(reagent_id)
    return hits


def score_reaction(reaction: LabReaction, lower: str, intent: Optional[str], color: Optional[str],
                   reagents: List[str]) -> Tuple[float, bool]:
    score = 0.0
    strong = False

    if any(_contains(lower, kw) for kw in reaction.product_info["keywords"]):  # type: ignore[union-attr]
        score += 6
        strong = True

    pair = {reaction.reactant_a, reaction.reactant_b}
    matched_reagents = [r for r in reagents if r in pair]
    score += 4 * len(matched_reagents)
    score -= 2 * len([r for r in reagents if r not in pair])
    if len(matched_reagents) == 2:
        strong = True

    if intent:
        score += 3 if intent == reaction.type else -4

    if color and reaction.type == "precipitate":
        score += 3 if color == reaction.color else -5
    elif color and reaction.type != "precipitate":
        score -= 2

    score += reaction.priority * 0.01
    return score, strong


def match_goal(goal_text: str) -> MatchResult:
    """Deterministically map a free-text goal to a supported lab reaction."""
    unsupported = find_unsupported_terms(goal_text)
    if unsupported:
        return MatchResult(reaction=None, unsupported_terms=unsupported)

    lower = normalize_text(goal_text).lower()
    for phrase in SAFE_PHRASES:
        lower = lower.replace(phrase, "sodium bicarbonate")
    intent = detect_intent(lower)
    color = detect_color(lower)
    reagents = mentioned_reagents(lower)

    best: Optional[LabReaction] = None
    best_score = float("-inf")
    best_strong = False
    for reaction in SUPPORTED_REACTIONS:
        score, strong = score_reaction(reaction, lower, intent, color, reagents)
        if score > best_score:
            best, best_score, best_strong = reaction, score, strong

    if best is None or best_score < 3:
        return MatchResult(reaction=None, score=max(best_score, 0.0))
    return MatchResult(reaction=best, score=best_score, strong=best_strong)


def confidence_from_score(score: float) -> int:
    return int(max(60, min(99, round(70 + score * 3))))


def clamp(value: float, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, float(value)))


def describe_for_prompt() -> str:
    """Compact listing of the database for the AI prompt."""
    lines = []
    for r in SUPPORTED_REACTIONS:
        observable = (
            f"{r.color} precipitate {r.product}" if r.type == "precipitate"
            else f"{r.product} gas bubbles" if r.type == "gas"
            else "neutral pH ~7, warm (exothermic)"
        )
        lines.append(f"- {r.id}: {r.reactant_a} + {r.reactant_b} -> {observable}")
    return "\n".join(lines)
