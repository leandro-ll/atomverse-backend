"""Catalog-derived AI tutor fallback service.

Replaces hardcoded reaction explanations with intent-dispatched composition
over canonical catalog facts and the current experiment result.

Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 16.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from app.domain.chemistry_evaluator import evaluate_experiment
from app.generated.chemistry_catalog import CATALOG, CATALOG_VERSION

LIMITATION_ID_TEXT = (
    "Pertanyaan ini berada di luar cakupan simulasi kimia yang dimodelkan saat ini. "
    "Kamu dapat bertanya tentang reaksi pengendapan, netralisasi asam-basa, "
    "persamaan reaksi, pereaksi pembatas, atau kesetimbangan kelarutan (Ksp)."
)

LIMITATION_EN_TEXT = (
    "This question is outside the scope of the current chemical simulation. "
    "You can ask about precipitation reactions, acid-base neutralization, "
    "reaction equations, limiting reagents, or solubility product equilibrium (Ksp)."
)

SUPPORTED_SUGGESTIONS = (
    "Apa reaksi yang terjadi antara kedua larutan?",
    "Berapa pH campuran akhir?",
    "Apakah terbentuk endapan?",
    "Tunjukkan persamaan reaksi ion bersih",
)

SUPPORTED_SUGGESTIONS_EN = (
    "What reaction occurs between the two solutions?",
    "What is the final pH of the mixture?",
    "Does a precipitate form?",
    "Show the net ionic equation",
)


def is_english_text(message: str) -> bool:
    """Heuristic check to determine if student asked in English."""
    msg_lower = (message or "").lower()
    en_words = {
        "what", "how", "why", "is", "are", "does", "will", "can", "tell", "explain",
        "the", "this", "that", "reaction", "precipitate", "solution", "color", "mixing",
        "between", "formed", "produce", "happen", "happens", "acid", "base", "limiting",
        "temperature", "ph", "particles", "atoms", "ions", "equation", "equations",
        "formula", "yield", "excess", "solubility", "neutral", "neutralization", "solid"
    }
    id_words = {
        "apakah", "bagaimana", "mengapa", "apa", "terbentuk", "larutan", "endapan",
        "suhu", "reaksi", "pereaksi", "persamaan", "kisi", "kristal", "pembatas", "jenuh", "kuning"
    }
    tokens = set(re.findall(r"\b[a-z]+\b", msg_lower))
    en_matches = len(tokens.intersection(en_words))
    id_matches = len(tokens.intersection(id_words))

    if en_matches > id_matches:
        return True
    return False


class FallbackIntent(StrEnum):
    """Closed enumeration of recognized fallback tutor intents."""

    CURRENT_OUTCOME = "current_outcome"
    EQUATIONS = "equations"
    LIMITING_REAGENT = "limiting_reagent"
    SATURATION = "saturation"
    PH_BOUNDARY = "ph_boundary"
    PARTICLE_VIEW = "particle_view"
    MODEL_BOUNDARY = "model_boundary"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class FallbackReply:
    """Structured response from the local tutor fallback."""

    text: str
    source: str = "fallback"
    scope: str = "answered"  # "answered" or "unsupported"
    catalog_version: str = CATALOG_VERSION
    degraded: bool = True
    reason_code: str = "provider_unavailable"
    suggestions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.text,
            "text": self.text,
            "source": self.source,
            "scope": self.scope,
            "catalog_version": self.catalog_version,
            "degraded": self.degraded,
            "reason_code": self.reason_code,
            "suggestions": list(self.suggestions),
        }


def fmt_sci(val: float) -> str:
    """Format floating point number into scientific notation for display only."""
    if val == 0.0 or not math.isfinite(val):
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    coeff = val / (10**exp)
    return f"{coeff:.2f} x 10^{exp}"


def _get_substance(sub_id: str, catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    substances = catalog.get("substances", ())
    if isinstance(substances, (list, tuple)):
        for s in substances:
            if s.get("id") == sub_id or s.get("formulaToken") == sub_id:
                return s
    return {"id": sub_id, "indonesianName": sub_id, "notation": {"unicode": sub_id}}


def classify_intent(message: str, result: Optional[Mapping[str, Any]] = None) -> FallbackIntent:
    """Keyword-plus-result intent classification (Bilingual: ID & EN)."""
    msg_lower = (message or "").lower()

    # Specific educational intents
    if any(k in msg_lower for k in [
        "persamaan", "reaksi ion", "net ionic", "molekul", "rumus reaksi",
        "equation", "equations", "net-ionic", "reaction formula", "chemical formula"
    ]):
        return FallbackIntent.EQUATIONS

    if any(k in msg_lower for k in [
        "pembatas", "limiting", "pereaksi sisa", "habis bereaksi", "stoikiometri",
        "excess", "stoichiometry", "limiting reagent", "limiting reactant"
    ]):
        return FallbackIntent.LIMITING_REAGENT

    if any(k in msg_lower for k in [
        "ksp", "kelarutan", "jenuh", "mengendap", "hasil kali ion", "nilai q",
        "solubility", "saturated", "saturation", "precipitate", "q-value", "q value"
    ]):
        return FallbackIntent.SATURATION

    if any(k in msg_lower for k in [
        "ph", "asam", "basa", "netral", "derajat keasaman",
        "acid", "base", "neutral", "neutralize", "neutralization", "acidity"
    ]):
        return FallbackIntent.PH_BOUNDARY

    if any(k in msg_lower for k in [
        "partikel", "submikro", "mikroskopik", "kisi", "kristal", "gerak ion",
        "particle", "particles", "submicroscopic", "molecular", "lattice", "crystal"
    ]):
        return FallbackIntent.PARTICLE_VIEW

    if any(k in msg_lower for k in [
        "asumsi", "model", "tekanan", "batas", "lingkup simulasi",
        "assumption", "assumptions", "boundary", "boundaries", "scope", "pressure", "simulation model"
    ]):
        return FallbackIntent.MODEL_BOUNDARY

    # Out of scope keywords (topics not in current catalog phase)
    out_of_scope_patterns = [
        r"\b(redoks|redox)\b",
        r"\b(gas|karbon dioksida|co2|oksigen|o2|oxygen)\b",
        r"\b(organik|organic|hidrokarbon|hydrocarbon|alkana|alkena)\b",
        r"\b(baterai|battery|batteries|sel galvani|galvanic|elektrolisis|electrolysis|katoda|cathode|anoda|anode)\b",
        r"\b(harga|beli|price|cost|buy|shop|store|toko|pasar)\b",
        r"\b(penemu|inventor|sejarah|history|biografi|biography)\b",
        r"\b(cuaca|weather|suhu bumi|climate|iklim global)\b",
    ]
    for pattern in out_of_scope_patterns:
        if re.search(pattern, msg_lower):
            return FallbackIntent.OUT_OF_SCOPE

    return FallbackIntent.CURRENT_OUTCOME


# ---------------------------------------------------------
# INDONESIAN COMPOSERS (PRESERVED VERBATIM FOR COMPATIBILITY)
# ---------------------------------------------------------

def compose_below_saturation(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    """Indonesian explanation for below_saturation state citing authored Ksp and Q."""
    cand_id = result.get("candidateSolidId") or result.get("productId") or ""
    solid = _get_substance(cand_id, catalog)
    solid_name = solid.get("indonesianName", cand_id)
    solid_unicode = solid.get("notation", {}).get("unicode", cand_id)

    q_val = result.get("QApprox", 0.0)
    q_str = fmt_sci(q_val)

    ksp_literal = result.get("kspLiteral")
    if not ksp_literal and result.get("kspRef"):
        ksp_literal = result["kspRef"].get("value", "-")
    ksp_literal = ksp_literal or "-"

    ksp_source = "Katalog Atomverse"
    if result.get("kspRef") and result["kspRef"].get("sourceId"):
        ksp_source = result["kspRef"]["sourceId"]

    return (
        f"Pada jumlah yang kamu masukkan, {solid_name} ({solid_unicode}) belum mengendap. "
        f"Hasil kali ion Q = {q_str} masih lebih kecil daripada tetapan hasil kali kelarutan "
        f"Ksp = {ksp_literal} (sumber: {ksp_source}). "
        f"Endapan baru muncul ketika Q melampaui Ksp, misalnya dengan menaikkan konsentrasi "
        f"atau memperbesar jumlah zat terlarut."
    )


def compose_current_outcome(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    """Composes current reaction outcome summary from catalog facts and ledger."""
    outcome = result.get("outcome")

    sub_a = _get_substance(result.get("reagentAId", ""), catalog)
    sub_b = _get_substance(result.get("reagentBId", ""), catalog)
    name_a = sub_a.get("indonesianName", result.get("reagentAId", ""))
    name_b = sub_b.get("indonesianName", result.get("reagentBId", ""))
    uni_a = sub_a.get("notation", {}).get("unicode", name_a)
    uni_b = sub_b.get("notation", {}).get("unicode", name_b)

    if outcome == "precipitate_forms":
        cand_id = result.get("candidateSolidId") or result.get("productId") or ""
        prod = _get_substance(cand_id, catalog)
        prod_name = prod.get("indonesianName", cand_id)
        prod_uni = prod.get("notation", {}).get("unicode", cand_id)
        net_ionic = result.get("netIonicEquation", "")

        limiting_id = (result.get("ledger") or {}).get("limitingId") or result.get("limitingReagentId")
        lim_sub = _get_substance(limiting_id, catalog) if limiting_id else None
        lim_text = f" Pereaksi pembatas: {lim_sub['indonesianName']}." if lim_sub else ""

        return (
            f"{uni_a} ({name_a}) dan {uni_b} ({name_b}) bereaksi membentuk endapan {prod_name} ({prod_uni}). "
            f"Persamaan ion bersih: {net_ionic}.{lim_text}"
        )

    if outcome == "below_saturation":
        return compose_below_saturation(result, catalog)

    if outcome == "neutralization":
        net_ionic = result.get("netIonicEquation", "")
        ph_info = result.get("pH", {})
        ph_val = ph_info.get("value")
        ph_str = f" dengan pH sekitar {ph_val:.2f}" if ph_val is not None else ""
        return (
            f"{uni_a} dan {uni_b} mengalami reaksi netralisasi asam-basa eksoterm{ph_str}. "
            f"Persamaan ion bersih: {net_ionic}."
        )

    if outcome == "no_modeled_reaction":
        expl = catalog.get("presentation", {}).get("explanations", {}).get("no_modeled_reaction")
        return (
            expl
            or f"Campuran {uni_a} dan {uni_b} tidak menunjukkan reaksi kimia yang dapat diamati; "
            "semua ion tetap terlarut bebas di dalam air."
        )

    if outcome == "unsupported":
        expl_id = result.get("explanationId")
        expl = catalog.get("presentation", {}).get("explanations", {}).get(expl_id) if expl_id else None
        return expl or "Kombinasi pereaksi ini belum dimodelkan pada versi katalog saat ini."

    return f"Campuran antara {uni_a} dan {uni_b} berada dalam kesetimbangan larutan."


def compose_equations(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    mol_eq = result.get("molecularEquation")
    net_eq = result.get("netIonicEquation")

    if mol_eq and net_eq:
        return f"Persamaan molekul: {mol_eq}\nPersamaan ion bersih: {net_eq}"
    if mol_eq:
        return f"Persamaan reaksi: {mol_eq}"
    return "Tidak terjadi persamaan reaksi kimia pada pasangan larutan ini karena semua produk tetap terlarut."


def compose_limiting_reagent(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    ledger = result.get("ledger") or {}
    limiting_id = ledger.get("limitingId") or result.get("limitingReagentId")

    if not limiting_id:
        return (
            "Pada percobaan ini tidak ada pereaksi pembatas karena kedua larutan tidak bereaksi "
            "atau berada dalam jumlah stoikiometri yang tepat."
        )

    lim_sub = _get_substance(limiting_id, catalog)
    lim_name = lim_sub.get("indonesianName", limiting_id)
    lim_uni = lim_sub.get("notation", {}).get("unicode", limiting_id)

    excess_id = ledger.get("excessId")
    excess_text = ""
    if excess_id:
        exc_sub = _get_substance(excess_id, catalog)
        excess_text = f" Sementara {exc_sub.get('indonesianName', excess_id)} merupakan pereaksi berlebih."

    return (
        f"Pereaksi pembatas pada reaksi ini adalah {lim_name} ({lim_uni}) karena habis bereaksi terlebih dahulu "
        f"dan menentukan jumlah maksimum produk yang dapat terbentuk.{excess_text}"
    )


def compose_ph(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    ph_data = result.get("pH") or {}
    if ph_data.get("supported") and ph_data.get("value") is not None:
        ph_val = float(ph_data["value"])
        desc = "netral" if abs(ph_val - 7.0) < 0.1 else "asam" if ph_val < 7.0 else "basa"
        return f"Campuran memiliki pH {ph_val:.2f} ({desc})."

    msg = ph_data.get("message")
    if msg:
        return f"Perhitungan pH: {msg}"
    return "Nilai pH untuk campuran ini tidak dimodelkan pada fase simulasi saat ini."


def compose_particle_view(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    outcome = result.get("outcome")
    if outcome == "precipitate_forms":
        return (
            "Pada tingkat submikroskopik, ion-ion yang saling tarik-menarik kuat keluar dari larutan "
            "dan berkumpul membentuk kisi kristal padat yang tampak sebagai endapan."
        )
    return (
        "Pada tingkat submikroskopik, ion-ion dari kedua larutan tetap bergerak bebas di antara "
        "molekul-molekul air sebagai ion terhidrasi tanpa membentuk kisi padat."
    )


def compose_model_boundary(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    return (
        "Model kimia Atomverse saat ini mencakup reaksi metatesis larutan berair, "
        "kesetimbangan kelarutan (Ksp) pada 25°C, dan reaksi asam-basa kuat. "
        "Reaksi redoks, pelepasan gas, dan kesetimbangan asam lemah berada di luar lingkup fase ini."
    )


COMPOSERS: dict[FallbackIntent, Callable[[Mapping[str, Any], Mapping[str, Any]], str]] = {
    FallbackIntent.CURRENT_OUTCOME: compose_current_outcome,
    FallbackIntent.EQUATIONS: compose_equations,
    FallbackIntent.LIMITING_REAGENT: compose_limiting_reagent,
    FallbackIntent.SATURATION: compose_below_saturation,
    FallbackIntent.PH_BOUNDARY: compose_ph,
    FallbackIntent.PARTICLE_VIEW: compose_particle_view,
    FallbackIntent.MODEL_BOUNDARY: compose_model_boundary,
}


# ---------------------------------------------------------
# ENGLISH COMPOSERS
# ---------------------------------------------------------

def compose_below_saturation_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    cand_id = result.get("candidateSolidId") or result.get("productId") or ""
    solid = _get_substance(cand_id, catalog)
    solid_unicode = solid.get("notation", {}).get("unicode", cand_id)

    q_val = result.get("QApprox", 0.0)
    q_str = fmt_sci(q_val)

    ksp_literal = result.get("kspLiteral")
    if not ksp_literal and result.get("kspRef"):
        ksp_literal = result["kspRef"].get("value", "-")
    ksp_literal = ksp_literal or "-"

    return (
        f"At the specified quantities, {solid_unicode} does not precipitate. "
        f"The ion product quotient Q = {q_str} is still less than the solubility product constant "
        f"Ksp = {ksp_literal}. Precipitation will only occur when Q exceeds Ksp."
    )


def compose_current_outcome_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    outcome = result.get("outcome")

    sub_a = _get_substance(result.get("reagentAId", ""), catalog)
    sub_b = _get_substance(result.get("reagentBId", ""), catalog)
    uni_a = sub_a.get("notation", {}).get("unicode") or sub_a.get("formulaToken") or str(result.get("reagentAId", ""))
    uni_b = sub_b.get("notation", {}).get("unicode") or sub_b.get("formulaToken") or str(result.get("reagentBId", ""))

    if outcome == "precipitate_forms":
        cand_id = result.get("candidateSolidId") or result.get("productId") or ""
        prod = _get_substance(cand_id, catalog)
        prod_uni = prod.get("notation", {}).get("unicode") or prod.get("formulaToken") or cand_id
        net_ionic = result.get("netIonicEquation", "")

        limiting_id = (result.get("ledger") or {}).get("limitingId") or result.get("limitingReagentId")
        lim_sub = _get_substance(limiting_id, catalog) if limiting_id else None
        lim_uni = (lim_sub.get("notation", {}).get("unicode") or lim_sub.get("formulaToken") or limiting_id) if lim_sub else ""
        lim_text = f" Limiting reagent: {lim_uni}." if lim_uni else ""

        return (
            f"{uni_a} and {uni_b} react to form an insoluble precipitate of {prod_uni}. "
            f"Net ionic equation: {net_ionic}.{lim_text}"
        )

    if outcome == "below_saturation":
        return compose_below_saturation_en(result, catalog)

    if outcome == "neutralization":
        net_ionic = result.get("netIonicEquation", "")
        ph_info = result.get("pH", {})
        ph_val = ph_info.get("value")
        ph_str = f" with a final pH of ~{ph_val:.2f}" if ph_val is not None else ""
        return (
            f"{uni_a} and {uni_b} undergo an exothermic acid-base neutralization reaction{ph_str}. "
            f"Net ionic equation: {net_ionic}."
        )

    if outcome == "no_modeled_reaction":
        return (
            f"The mixture of {uni_a} and {uni_b} exhibits no observable reaction; "
            "all ions remain freely dissolved in aqueous solution as spectator ions."
        )

    if outcome == "unsupported":
        return "This combination of reagents is not yet modeled in the current catalog version."

    return f"The mixture of {uni_a} and {uni_b} is in solution equilibrium."


def compose_equations_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    mol_eq = result.get("molecularEquation")
    net_eq = result.get("netIonicEquation")

    if mol_eq and net_eq:
        return f"Molecular equation: {mol_eq}\nNet ionic equation: {net_eq}"
    if mol_eq:
        return f"Reaction equation: {mol_eq}"
    return "No chemical reaction equation applies for this pair because all products remain dissolved."


def compose_limiting_reagent_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    ledger = result.get("ledger") or {}
    limiting_id = ledger.get("limitingId") or result.get("limitingReagentId")

    if not limiting_id:
        return "There is no limiting reagent in this trial because the solutions do not react or are present in exact stoichiometric proportions."

    lim_sub = _get_substance(limiting_id, catalog)
    lim_uni = lim_sub.get("notation", {}).get("unicode", limiting_id)

    excess_id = ledger.get("excessId")
    excess_text = ""
    if excess_id:
        exc_sub = _get_substance(excess_id, catalog)
        exc_uni = exc_sub.get("notation", {}).get("unicode", excess_id)
        excess_text = f" Meanwhile, {exc_uni} is the excess reagent."

    return (
        f"The limiting reagent in this reaction is {lim_uni} as it is consumed completely first, "
        f"governing the maximum amount of precipitate or product formed.{excess_text}"
    )


def compose_ph_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    ph_data = result.get("pH") or {}
    if ph_data.get("supported") and ph_data.get("value") is not None:
        ph_val = float(ph_data["value"])
        desc = "neutral" if abs(ph_val - 7.0) < 0.1 else "acidic" if ph_val < 7.0 else "basic"
        return f"The mixture has a calculated pH of {ph_val:.2f} ({desc})."

    msg = ph_data.get("message")
    if msg:
        return f"pH Calculation: {msg}"
    return "The pH for this solution mixture is not modeled in the current simulation phase."


def compose_particle_view_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    outcome = result.get("outcome")
    if outcome == "precipitate_forms":
        return (
            "At the submicroscopic level, strongly attracting counter-ions collide, leave solution, "
            "and assemble into an orderly crystalline lattice that aggregates as solid precipitate."
        )
    return (
        "At the submicroscopic level, ions from both solutions remain hydrated, moving freely "
        "among water molecules without nucleating a crystal lattice."
    )


def compose_model_boundary_en(result: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    return (
        "Atomverse's model covers aqueous metathesis reactions, solubility equilibria (Ksp) at 25°C, "
        "and strong acid-base stoichiometry. Complex redox cycles and weak acid buffering "
        "fall outside this simulation phase."
    )


COMPOSERS_EN: dict[FallbackIntent, Callable[[Mapping[str, Any], Mapping[str, Any]], str]] = {
    FallbackIntent.CURRENT_OUTCOME: compose_current_outcome_en,
    FallbackIntent.EQUATIONS: compose_equations_en,
    FallbackIntent.LIMITING_REAGENT: compose_limiting_reagent_en,
    FallbackIntent.SATURATION: compose_below_saturation_en,
    FallbackIntent.PH_BOUNDARY: compose_ph_en,
    FallbackIntent.PARTICLE_VIEW: compose_particle_view_en,
    FallbackIntent.MODEL_BOUNDARY: compose_model_boundary_en,
}


def build_chat_fallback(
    message: str,
    lab_state: Optional[Mapping[str, Any]] = None,
    result: Optional[Mapping[str, Any]] = None,
    catalog: Mapping[str, Any] = CATALOG,
) -> FallbackReply:
    """Builds an educational tutor reply derived from catalog facts and experiment state."""
    lab_state = lab_state or {}

    # If result not provided, evaluate on the fly from lab_state
    if result is None:
        reagent_a = lab_state.get("reagentA") or lab_state.get("reagentAId") or "sub.agno3"
        reagent_b = lab_state.get("reagentB") or lab_state.get("reagentBId") or "sub.nacl"
        conc_a = float(lab_state.get("mA") or lab_state.get("concentrationA_M") or 1.0)
        vol_a = float(lab_state.get("vA") or lab_state.get("volumeA_mL") or 50.0)
        conc_b = float(lab_state.get("mB") or lab_state.get("concentrationB_M") or 1.0)
        vol_b = float(lab_state.get("vB") or lab_state.get("volumeB_mL") or 50.0)
        temp_c = float(lab_state.get("temp") or lab_state.get("temperatureC") or 25.0)

        exp_input = {
            "reagentAId": reagent_a,
            "reagentBId": reagent_b,
            "concentrationA_M": conc_a,
            "volumeA_mL": vol_a,
            "concentrationB_M": conc_b,
            "volumeB_mL": vol_b,
            "temperatureC": temp_c,
        }
        try:
            res_dict = dict(evaluate_experiment(catalog, exp_input))
            res_dict["reagentAId"] = reagent_a
            res_dict["reagentBId"] = reagent_b
            result = res_dict
        except Exception:
            result = {
                "outcome": "no_modeled_reaction",
                "reagentAId": reagent_a,
                "reagentBId": reagent_b,
            }

    intent = classify_intent(message, result)
    is_en = is_english_text(message)

    if intent == FallbackIntent.OUT_OF_SCOPE:
        return FallbackReply(
            text=LIMITATION_EN_TEXT if is_en else LIMITATION_ID_TEXT,
            scope="unsupported",
            catalog_version=catalog.get("catalogVersion", CATALOG_VERSION),
            degraded=True,
            reason_code="provider_unavailable",
            suggestions=SUPPORTED_SUGGESTIONS_EN if is_en else SUPPORTED_SUGGESTIONS,
        )

    composers = COMPOSERS_EN if is_en else COMPOSERS
    composer = composers.get(intent, compose_current_outcome_en if is_en else compose_current_outcome)
    reply_text = composer(result, catalog)

    return FallbackReply(
        text=reply_text,
        source="fallback",
        scope="answered",
        catalog_version=catalog.get("catalogVersion", CATALOG_VERSION),
        degraded=True,
        reason_code="provider_unavailable",
        suggestions=SUPPORTED_SUGGESTIONS_EN if is_en else SUPPORTED_SUGGESTIONS,
    )

