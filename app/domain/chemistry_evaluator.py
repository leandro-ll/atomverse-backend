"""Pure catalog-backed experiment evaluator (backend parity).

This module is the backend half of a pair. ``js/domain/chemistry-evaluator.js``
is the frontend half, and both derive every quantity and every display string
from the same catalog records, so for one catalog and one input they must agree
field for field. The inverse solver can therefore recalculate a recommendation
through the same chemistry the browser showed.

One modeled reaction record answers every question about a supported
experiment. The evaluator resolves a reagent pair through the catalog's
normalized order-independent pair key, converts each selected solution into a
formula-unit amount, calculates balanced reaction extent, names the limiting
reactant(s), reports remaining excess and product amounts, and projects the
molecular equation, net-ionic equation, phase model, appearance, and energy
label from that same record.

A pair the catalog models no reaction for is an ordinary, fully defined outcome
rather than a failure: the evaluator returns the same result shape with the
``no_modeled_reaction`` category, no product, no equation, zero extent, both
solutions still present, and the single unlayered ``homogeneous_aqueous`` phase
the catalog declares for that outcome. Two mixed aqueous solutions are one
mixture, so nothing downstream can read a layer boundary out of this result.

Control semantics: concentration and volume are the only quantitative inputs.
Every amount, the total mixed volume, the balanced extent, the limiting
reactant, the excess, and every product amount are derived from them.
Temperature is a qualitative input: it produces the bounded ``motion`` metadata
block the catalog's temperature capability declares and changes nothing a
learner would read as a calculated quantity. Pressure has no modeled effect at
all, so the result reports the catalog's fixed 1 atm state with its localized
boundary statement instead of letting an alternate selection silently influence
anything.

Input semantics: every numeric control is validated against the bounds the
catalog declares, and an invalid selection is *rejected* rather than clamped or
ignored. A concentration of ``0.0999 M``, a volume of ``9.99 mL``, a
temperature of ``-0.01 C``, and a pressure of ``0.99 atm`` all produce no
result at all, because clamping them would silently evaluate a different
experiment than the one the learner set up, and ignoring them would present a
control that does nothing. A value exactly at a bound is a valid experiment.
One validation pass reports every offending field, each with its field name, a
stable reason code, and an Indonesian message a learner can be shown; the
reason codes are what the API boundary and the UI map, so they never change
meaning silently.

Purity: no provider, environment, clock, random, or network dependency, no
mutable global state, and no rounding. Scientific values stay unrounded;
display precision belongs to the formatter projections so a rounded number can
never feed a later calculation.

pH semantics: the one modeled acid-base system is the ideal, fully dissociated,
monoprotic HCl/NaOH mixture the catalog's pH capability declares. Its whole
arithmetic runs on the unrounded amounts already calculated above: the acid's
proton equivalents minus the base's hydroxide equivalents gives a signed
difference, the difference divided by the total mixed volume gives the excess
ion concentration, and the declared ``pKw`` turns that concentration into a pH.
Equivalence is decided with the catalog's declared ``equivalence_mol`` tolerance
rather than by exact float equality, because two amounts that are chemically
equal are routinely unequal in binary arithmetic. Nothing here is rounded and
nothing here reads temperature: the catalog declares ``affectsPH: false``, so
two experiments differing only in temperature report the same pH.

Every other system reports the *same* pH block with ``supported: False``, a
``None`` numeric value, and the machine-readable reason code the catalog
declares for it together with that code's localized Indonesian model-boundary
explanation. The reason code is read from the reaction record's own
``pHUnsupportedReasonCode`` for a modeled reaction and from the catalog's
declared no-modeled-reaction reason for an unmatched pair, so which systems are
unsupported is a catalog statement rather than a list written into this module:
a catalog that adds a reaction, or an outcome category, carries its own
explanation with it.

Both variants of the block declare every field. A consumer branches on
``pH["supported"]`` alone and then finds either a number with a side or a
``None`` with a reason and a message, and the fields that only mean something
for a calculated pH are explicitly ``None`` rather than absent so the two
variants stay directly comparable. Nothing invents a value: the audited
engine's seeded ``~7.0`` has no replacement default anywhere, for a salt
solution, a precipitating mixture, or an unreacted pair.

Deliberately out of scope here, each owned by a later task:

- the ion-level species ledger and conservation proofs (task 6.1),
- the deterministic visualization plan (task 7.1).

The result exposes ``speciesLedger`` as an explicit ``None`` placeholder so
those tasks have a field to fill without reshaping consumers, and so no
fabricated particle count can escape in the meantime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

try:
    from app.domain.chemistry_formatters import ChemistryFormatter
    from app.generated.chemistry_catalog import CATALOG
except ModuleNotFoundError:
    from backend.app.domain.chemistry_formatters import ChemistryFormatter
    from backend.app.generated.chemistry_catalog import CATALOG

OUTCOME_CATEGORIES = (
    "precipitate_forms",
    "below_saturation",
    "no_modeled_reaction",
    "neutralization",
    "unsupported",
)

__all__ = [
    "EVALUATOR",
    "EXPERIMENT_INPUT_REASON_CODES",
    "IDEAL_STRONG_MONOPROTIC_PH_MODEL",
    "LINEAR_BOUNDED_MOTION_MODEL",
    "MILLILITRES_PER_LITRE",
    "NO_MODELED_PRESSURE_EFFECT",
    "NO_MODELED_REACTION_CATEGORY",
    "NO_MODELED_REACTION_PH_REASON",
    "OUTCOME_CATEGORIES",
    "PH_ACIDIC",
    "PH_BASIC",
    "PH_BLOCK_FIELDS",
    "PH_NEUTRAL",
    "PH_SIDES",
    "PRESSURE_NOT_SUPPORTED",
    "QUALITATIVE_MOTION_EFFECT",
    "QUANTITY_ABOVE_MAXIMUM",
    "QUANTITY_BELOW_MINIMUM",
    "QUANTITY_NOT_A_NUMBER",
    "QUANTITY_NOT_FINITE",
    "QUANTITY_NOT_POSITIVE",
    "QUANTITY_REQUIRED",
    "CatalogCompatibilityError",
    "ChemistryEvaluator",
    "InvalidExperimentQuantityError",
    "ValidatedExperimentField",
    "calculate_amount",
    "decide_outcome",
    "evaluate_experiment",
    "ion_product",
    "negative_log10",
    "post_mix_concentrations",
    "resolve_canonical_pair",
    "substance_name",
    "to_litres",
    "verify_catalog_compatibility",
]

#: Millilitres per litre, the only unit conversion the evaluator performs.
MILLILITRES_PER_LITRE = 1000

#: Category reported when the catalog declares no reaction for a pair.
NO_MODELED_REACTION_CATEGORY = "no_modeled_reaction"

#: The only temperature motion model version 1 implements.
LINEAR_BOUNDED_MOTION_MODEL = "linear_bounded"

#: Temperature effect the catalog must declare for the motion metadata.
QUALITATIVE_MOTION_EFFECT = "qualitative_motion_only"

#: Pressure effect the catalog must declare for the fixed 1 atm state.
NO_MODELED_PRESSURE_EFFECT = "none"

#: The only pH model this evaluator implements.
#:
#: Named here rather than taken from whatever the catalog happens to declare:
#: the formulas below are the ideal, fully dissociated, monoprotic strong
#: acid/strong base treatment, so a catalog announcing some other model must
#: fail loudly instead of having this arithmetic applied to it.
IDEAL_STRONG_MONOPROTIC_PH_MODEL = "ideal_strong_monoprotic_acid_base"

#: Excess proton concentration: pH below the declared neutral point.
PH_ACIDIC = "acidic"

#: Acid and base equivalents equal within tolerance: the neutral point.
PH_NEUTRAL = "neutral"

#: Excess hydroxide concentration: pH above the declared neutral point.
PH_BASIC = "basic"

#: Which side of the declared neutral point a reported pH falls on.
PH_SIDES: tuple[str, ...] = (PH_ACIDIC, PH_NEUTRAL, PH_BASIC)

#: Declared pH reason code for a pair the catalog models no reaction for.
#:
#: A reaction record names its own reason code, so this is the one case that has
#: no record to read it from. Naming it here keeps it a single stable constant
#: instead of a string repeated at each call site, and the evaluator refuses at
#: construction a catalog that does not declare it, so the code can never
#: resolve to a missing message.
NO_MODELED_REACTION_PH_REASON = "PH_MODEL_NO_MODELED_REACTION"

#: Every field a pH block carries, in the order both variants report them.
#:
#: The supported and unsupported variants differ in their values, never in their
#: shape. ``supported`` is therefore the only branch a consumer needs: it is
#: ``True`` beside a number and a side, and ``False`` beside a ``None`` value, a
#: reason code, a message key, and the localized explanation of why version 1
#: does not calculate that system. Everything meaningful only to a calculated pH
#: is explicitly ``None`` in the unsupported variant rather than missing, so the
#: two can be compared field by field and no consumer has to guess whether an
#: absent key means zero, neutral, or unknown.
PH_BLOCK_FIELDS: tuple[str, ...] = (
    "supported",
    "value",
    "side",
    "equivalence",
    "modelId",
    "modelLabel",
    "pKw",
    "pKwSource",
    "temperatureCorrected",
    "neutralPH",
    "acidReagentId",
    "baseReagentId",
    "protonIonId",
    "hydroxideIonId",
    "acidEquivalents_mol",
    "baseEquivalents_mol",
    "equivalenceDifference_mol",
    "totalVolume_L",
    "excessIonId",
    "excessAmount_mol",
    "excessConcentration_molL",
    "reasonCode",
    "messageKey",
    "message",
)


# -- validated experiment input ------------------------------------------
#
# Reason codes are the stable contract. The API boundary maps them to response
# codes and the UI maps them to the control that needs correcting, so they are
# named for the *kind* of failure rather than for one field, and the set is
# enumerable so a consumer can prove it handles all of them.

#: A required quantity was absent.
QUANTITY_REQUIRED = "QUANTITY_REQUIRED"

#: A quantity was supplied as something other than a number.
QUANTITY_NOT_A_NUMBER = "QUANTITY_NOT_A_NUMBER"

#: A quantity was not-a-number or infinite, so it names no position at all.
QUANTITY_NOT_FINITE = "QUANTITY_NOT_FINITE"

#: A quantity that must be positive was zero or negative.
QUANTITY_NOT_POSITIVE = "QUANTITY_NOT_POSITIVE"

#: A quantity fell below the minimum the catalog declares for its control.
QUANTITY_BELOW_MINIMUM = "QUANTITY_BELOW_MINIMUM"

#: A quantity rose above the maximum the catalog declares for its control.
QUANTITY_ABOVE_MAXIMUM = "QUANTITY_ABOVE_MAXIMUM"

#: A pressure other than the one fixed value version 1 models was supplied.
PRESSURE_NOT_SUPPORTED = "PRESSURE_NOT_SUPPORTED"

#: Every rejection reason an experiment input can carry, in a stable order.
EXPERIMENT_INPUT_REASON_CODES: tuple[str, ...] = (
    QUANTITY_REQUIRED,
    QUANTITY_NOT_A_NUMBER,
    QUANTITY_NOT_FINITE,
    QUANTITY_NOT_POSITIVE,
    QUANTITY_BELOW_MINIMUM,
    QUANTITY_ABOVE_MAXIMUM,
    PRESSURE_NOT_SUPPORTED,
)


class ValidatedExperimentField(NamedTuple):
    """One numeric experiment field and the catalog bound that governs it.

    The numbers themselves are deliberately absent: ``bound`` names a key in
    ``model.bounds``, so widening the educational control range is a catalog
    release rather than a code change, and a regenerated catalog cannot drift
    away from the validation that is supposed to enforce it.

    ``label`` and ``unit`` exist here because the rejection text is owned by
    this module rather than by the catalog message bundle. An out-of-range
    selection is not a catalog-declared outcome, so the catalog has no message
    key for it, and ``scripts/catalog/validate-catalog.mjs`` sets the same
    precedent by owning the localized text that explains an incompatible
    catalog.
    """

    field: str
    bound: str
    label: str
    unit: str
    required: bool
    positive: bool
    fixed: bool


#: Every numeric experiment field, in the order violations are reported.
VALIDATED_EXPERIMENT_FIELDS: tuple[ValidatedExperimentField, ...] = (
    ValidatedExperimentField(
        "concentrationA_molL", "concentration_molL", "Konsentrasi larutan A", "M", True, True, False
    ),
    ValidatedExperimentField(
        "volumeA_mL", "volume_mL", "Volume larutan A", "mL", True, True, False
    ),
    ValidatedExperimentField(
        "concentrationB_molL", "concentration_molL", "Konsentrasi larutan B", "M", True, True, False
    ),
    ValidatedExperimentField(
        "volumeB_mL", "volume_mL", "Volume larutan B", "mL", True, True, False
    ),
    # Temperature and pressure default to the catalog's declared position when
    # they are not supplied: zero degrees is a real selection, and version 1
    # models exactly one pressure, so neither absence is a learner error.
    ValidatedExperimentField(
        "temperature_C", "temperature_C", "Suhu", "\u00b0C", False, False, False
    ),
    ValidatedExperimentField("pressure_atm", "pressure_atm", "Tekanan", "atm", False, False, True),
)


def _format_bound_value(value: float) -> str:
    """Prints a declared bound the way both language halves print it.

    This is not display rounding: the bound is echoed exactly as the catalog
    declares it, with no precision applied. It exists only because an integral
    bound must read as ``2`` rather than ``2.0`` in the Indonesian text on both
    halves of the pair, and Python's default float form disagrees with
    JavaScript's.
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _describe_requirement(
    reason_code: str, descriptor: ValidatedExperimentField, bounds: Mapping[str, Any]
) -> str:
    """Developer-facing phrase naming what a field had to be."""
    if reason_code == QUANTITY_REQUIRED:
        return "supplied"
    if reason_code == QUANTITY_NOT_A_NUMBER:
        return "a number"
    if reason_code == QUANTITY_NOT_FINITE:
        return "a finite number"
    if reason_code == QUANTITY_NOT_POSITIVE:
        return "a number greater than zero"
    if reason_code == QUANTITY_BELOW_MINIMUM:
        return f"at least {_format_bound_value(bounds['minimum'])} {descriptor.unit}"
    if reason_code == QUANTITY_ABOVE_MAXIMUM:
        return f"at most {_format_bound_value(bounds['maximum'])} {descriptor.unit}"
    if reason_code == PRESSURE_NOT_SUPPORTED:
        return f"fixed at {_format_bound_value(bounds['minimum'])} {descriptor.unit}"
    raise ValueError(f"unknown experiment input reason code {reason_code!r}")


def _localized_violation_message(
    reason_code: str, descriptor: ValidatedExperimentField, bounds: Mapping[str, Any]
) -> str:
    """Learner-facing Indonesian explanation of one rejected field.

    It names the control and what was expected of it, and it deliberately does
    not echo the rejected value or any internal field name, so the text is safe
    to render directly next to the control that produced it.
    """
    label = descriptor.label
    if reason_code == QUANTITY_REQUIRED:
        return f"{label} wajib diisi."
    if reason_code == QUANTITY_NOT_A_NUMBER:
        return f"{label} harus berupa angka."
    if reason_code == QUANTITY_NOT_FINITE:
        return f"{label} harus berupa angka yang terhingga."
    if reason_code == QUANTITY_NOT_POSITIVE:
        return f"{label} harus lebih besar dari nol."
    if reason_code == QUANTITY_BELOW_MINIMUM:
        minimum = _format_bound_value(bounds["minimum"])
        return f"{label} tidak boleh kurang dari {minimum} {descriptor.unit}."
    if reason_code == QUANTITY_ABOVE_MAXIMUM:
        maximum = _format_bound_value(bounds["maximum"])
        return f"{label} tidak boleh lebih dari {maximum} {descriptor.unit}."
    if reason_code == PRESSURE_NOT_SUPPORTED:
        fixed = _format_bound_value(bounds["minimum"])
        return (
            f"{label} tidak dapat diubah pada model versi 1 "
            f"dan harus tetap {fixed} {descriptor.unit}."
        )
    raise ValueError(f"unknown experiment input reason code {reason_code!r}")


def _build_violation(
    reason_code: str,
    descriptor: ValidatedExperimentField,
    bounds: Mapping[str, Any],
    value: Any,
) -> dict[str, Any]:
    """One rejected field: what failed, why, and what to tell the learner."""
    if reason_code == QUANTITY_ABOVE_MAXIMUM:
        limit: Any = bounds["maximum"]
    elif reason_code in (QUANTITY_BELOW_MINIMUM, PRESSURE_NOT_SUPPORTED):
        limit = bounds["minimum"]
    else:
        limit = None
    return {
        "field": descriptor.field,
        "bound": descriptor.bound,
        "value": value,
        "reasonCode": reason_code,
        "limit": limit,
        "requirement": _describe_requirement(reason_code, descriptor, bounds),
        "message": _localized_violation_message(reason_code, descriptor, bounds),
    }


class InvalidExperimentQuantityError(ValueError):
    """Raised when an experiment input does not describe a supported experiment.

    The error carries every offending field in one ``violations`` tuple rather
    than only the first, so a learner who set two controls wrongly is told
    about both instead of discovering them one submission at a time. ``args[0]``
    is the developer-facing English summary; ``localized_messages`` carries the
    learner-facing Indonesian text, one entry per field.
    """

    def __init__(self, violations: Any) -> None:
        items = list(violations) if isinstance(violations, (list, tuple)) else [violations]
        if not items:
            raise TypeError("an invalid experiment input must carry at least one violation")
        super().__init__(
            "; ".join(
                f"{item['field']} must be {item['requirement']}, received {item['value']!r}"
                for item in items
            )
        )
        self.violations: tuple[Mapping[str, Any], ...] = tuple(items)
        self.fields: tuple[str, ...] = tuple(item["field"] for item in items)
        self.reason_codes: tuple[str, ...] = tuple(item["reasonCode"] for item in items)
        self.localized_messages: tuple[str, ...] = tuple(item["message"] for item in items)
        self.localized_message = " ".join(self.localized_messages)

        # Shortcuts to the first violation, which is all a single-field caller
        # such as the motion factor helper can ever produce.
        self.field = items[0]["field"]
        self.value = items[0]["value"]
        self.reason_code = items[0]["reasonCode"]
        self.requirement = items[0]["requirement"]


def _check_field(
    descriptor: ValidatedExperimentField,
    bounds: Mapping[str, Any],
    value: Any,
    require_supplied: bool,
) -> tuple[Any, dict[str, Any] | None]:
    """Checks one supplied field against one declared bound.

    Returns either the accepted number or the single violation that rejected
    it, never both, and never a corrected value: there is no code path here
    that moves a quantity into range. The checks run from the weakest
    precondition to the strongest so that a reason code names the most specific
    thing that is wrong - ``0 M`` is reported as not positive rather than as
    below the minimum, and ``0.05 M`` is reported as below the minimum rather
    than as not positive.

    Declared step size is deliberately not enforced. The step describes how a
    slider moves, not what chemistry is defined, so an exactly calculated
    off-step concentration arriving from the inverse solver stays valid.
    """
    if value is None:
        if require_supplied:
            return None, _build_violation(QUANTITY_REQUIRED, descriptor, bounds, value)
        return float(bounds["default"]), None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, _build_violation(QUANTITY_NOT_A_NUMBER, descriptor, bounds, value)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None, _build_violation(QUANTITY_NOT_FINITE, descriptor, bounds, value)
    if descriptor.positive and number <= 0:
        return None, _build_violation(QUANTITY_NOT_POSITIVE, descriptor, bounds, value)
    if descriptor.fixed:
        # One modeled pressure means one accepted pressure. Any other selection
        # is an explicit rejection rather than a field the evaluator skips.
        if number != float(bounds["minimum"]):
            return None, _build_violation(PRESSURE_NOT_SUPPORTED, descriptor, bounds, value)
        return number, None
    if number < float(bounds["minimum"]):
        return None, _build_violation(QUANTITY_BELOW_MINIMUM, descriptor, bounds, value)
    if number > float(bounds["maximum"]):
        return None, _build_violation(QUANTITY_ABOVE_MAXIMUM, descriptor, bounds, value)
    return number, None


def _require_finite_declaration(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, received {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be a finite number, received {value!r}")
    return number


def to_litres(volume_ml: float) -> float:
    """Converts a solution volume in millilitres to litres without rounding."""
    return volume_ml / MILLILITRES_PER_LITRE


def calculate_amount(concentration_mol_l: float, volume_ml: float) -> float:
    """Formula-unit amount of a dissolved reagent: ``n = concentration x volume``."""
    return concentration_mol_l * to_litres(volume_ml)


def negative_log10(concentration: float) -> float:
    """Decimal cologarithm ``-log10(x)``, the only logarithm the pH model uses.

    A concentration of zero or below has no cologarithm, and the one caller
    never passes one: an excess within the declared equivalence tolerance is
    reported as the neutral point instead of being pushed through this
    function. The guard is here so that a future caller cannot quietly produce
    an infinity and have it read as a pH.
    """
    if not concentration > 0:
        raise ValueError(
            f"a cologarithm needs a positive concentration, received {concentration!r}"
        )
    return -math.log10(concentration)


def resolve_canonical_pair(id_a: str, id_b: str) -> dict[str, str]:
    if not id_a or not id_b:
        raise TypeError("both reagent identifiers are required")
    low_id = id_a if id_a < id_b else id_b
    high_id = id_b if id_a < id_b else id_a
    return {
        "key": f"{low_id}|{high_id}",
        "lowId": low_id,
        "highId": high_id,
    }


def post_mix_concentrations(
    experiment_input: Mapping[str, Any],
    catalog: Mapping[str, Any] = CATALOG,
) -> dict[str, float]:
    if not experiment_input:
        raise TypeError("experiment input is required")
    reagent_a = experiment_input.get("reagentAId") or experiment_input.get("reagentA")
    reagent_b = experiment_input.get("reagentBId") or experiment_input.get("reagentB")
    if not reagent_a or not reagent_b:
        raise TypeError("both reagent identifiers are required")

    vol_a_l = experiment_input.get("volumeA_L")
    if vol_a_l is None and "volumeA_mL" in experiment_input:
        vol_a_l = experiment_input["volumeA_mL"] / 1000.0
    vol_b_l = experiment_input.get("volumeB_L")
    if vol_b_l is None and "volumeB_mL" in experiment_input:
        vol_b_l = experiment_input["volumeB_mL"] / 1000.0

    if vol_a_l is None or vol_b_l is None:
        raise TypeError("volumes for both reagents are required")

    total_volume_l = float(vol_a_l + vol_b_l)
    if total_volume_l <= 0:
        raise ValueError(f"non-positive total volume: {total_volume_l}")

    conc_a_m = experiment_input.get("concentrationA_M")
    if conc_a_m is None:
        conc_a_m = experiment_input.get("concentrationA_molL")
    conc_b_m = experiment_input.get("concentrationB_M")
    if conc_b_m is None:
        conc_b_m = experiment_input.get("concentrationB_molL")

    if conc_a_m is None or conc_b_m is None:
        raise TypeError("concentrations for both reagents are required")

    raw_substances = catalog.get("substances", ())
    if isinstance(raw_substances, Mapping):
        sub_map = raw_substances
    else:
        sub_map = {s["id"]: s for s in raw_substances}

    sub_a = sub_map.get(reagent_a)
    sub_b = sub_map.get(reagent_b)
    if not sub_a:
        raise ValueError(f"reagent {reagent_a} not found in catalog substances")
    if not sub_b:
        raise ValueError(f"reagent {reagent_b} not found in catalog substances")

    conc: dict[str, float] = {}
    for sub, c, v in [(sub_a, float(conc_a_m), float(vol_a_l)), (sub_b, float(conc_b_m), float(vol_b_l))]:
        for part in sub.get("dissociation", ()):
            moles = c * v * part["multiplicity"]
            ion_id = part["ionId"]
            conc[ion_id] = conc.get(ion_id, 0.0) + (moles / total_volume_l)

    return conc


def ion_product(
    concentrations: Mapping[str, float],
    solid: Any,
    catalog: Mapping[str, Any] = CATALOG,
) -> float:
    ion_powers: Mapping[str, int] | None = None

    if isinstance(solid, Mapping):
        if "ionPowers" in solid:
            ion_powers = solid["ionPowers"]
        elif "kspRef" in solid and "ionPowers" in solid["kspRef"]:
            ion_powers = solid["kspRef"]["ionPowers"]
        elif "dissociation" in solid:
            ion_powers = {d["ionId"]: d["multiplicity"] for d in solid["dissociation"]}
    elif isinstance(solid, str):
        raw_substances = catalog.get("substances", ())
        if isinstance(raw_substances, Mapping):
            sub_map = raw_substances
        else:
            sub_map = {s["id"]: s for s in raw_substances}
        sub_rec = sub_map.get(solid)
        if sub_rec and "dissociation" in sub_rec:
            ion_powers = {d["ionId"]: d["multiplicity"] for d in sub_rec["dissociation"]}

    if not ion_powers:
        raise ValueError(f"cannot determine ion powers for solid {solid}")

    q = 1.0
    for ion_id, power in sorted(ion_powers.items()):
        c = concentrations.get(ion_id, 0.0)
        q *= (c ** power)
    return q


def decide_outcome(
    reaction_or_q: Any,
    input_or_ksp: Any = None,
    catalog: Mapping[str, Any] = CATALOG,
) -> str:
    if isinstance(reaction_or_q, (int, float)):
        q = float(reaction_or_q)
        if isinstance(input_or_ksp, Mapping):
            val = input_or_ksp.get("value")
            ksp_val = float(val) if val is not None else 0.0
        else:
            ksp_val = float(input_or_ksp)
        return "precipitate_forms" if q > ksp_val else "below_saturation"

    reaction = reaction_or_q
    experiment_input = input_or_ksp
    cat = catalog or CATALOG

    outcome = reaction.get("outcome")
    if outcome == "unsupported":
        return "unsupported"
    if outcome == "no_modeled_reaction":
        return "no_modeled_reaction"
    if reaction.get("reactionClass") == "neutralization" or outcome == "neutralization":
        return "neutralization"

    conc = post_mix_concentrations(experiment_input, cat)
    q = ion_product(conc, reaction["candidateSolidId"], cat)
    ksp_val = float(reaction["kspRef"]["value"])
    return "precipitate_forms" if q > ksp_val else "below_saturation"


def substance_name(substance_id: str, catalog: Mapping[str, Any] = CATALOG) -> str:
    raw_substances = catalog.get("substances", ())
    if isinstance(raw_substances, Mapping):
        sub_map = raw_substances
    else:
        sub_map = {s["id"]: s for s in raw_substances}
    sub = sub_map.get(substance_id)
    return sub["indonesianName"] if sub and "indonesianName" in sub else substance_id


class CatalogCompatibilityError(RuntimeError):
    """Raised when catalog schema major or digest is incompatible with the evaluator."""
    pass


def verify_catalog_compatibility(
    catalog: Mapping[str, Any],
    expected_schema_major: str = "2",
    expected_digest: str | None = None,
) -> None:
    schema_version = str(catalog.get("schemaVersion") or catalog.get("catalogVersion") or "")
    if schema_version and not schema_version.startswith(f"{expected_schema_major}."):
        raise CatalogCompatibilityError(
            f"Katalog kimia tidak kompatibel: versi skema utama diharapkan {expected_schema_major}.x.x, tetapi menerima {schema_version}."
        )
    if expected_digest is not None:
        cat_digest = catalog.get("catalogDigest") or catalog.get("digest")
        if cat_digest and cat_digest != expected_digest:
            raise CatalogCompatibilityError(
                f"Katalog kimia tidak kompatibel: digest {cat_digest} tidak sesuai dengan digest yang diharapkan {expected_digest}."
            )


class ChemistryEvaluator:
    """Evaluates supported experiments against one catalog.

    The catalog is read, never written, so one evaluator instance is safe to
    share. Tests can pass a mutated copy to prove the evaluator reads the
    catalog rather than a hardcoded table.
    """

    def __init__(self, catalog: Mapping[str, Any] = CATALOG) -> None:
        if not isinstance(catalog, Mapping):
            raise TypeError("catalog must be a mapping")
        verify_catalog_compatibility(catalog)
        self._catalog = catalog
        self._formatter = ChemistryFormatter(catalog)
        self._model = catalog.get("model", {})

        if "substances" in catalog:
            raw_substances = catalog.get("substances", ())
            if isinstance(raw_substances, Mapping):
                self._substances_map = raw_substances
            else:
                self._substances_map = {s["id"]: s for s in raw_substances}

            raw_reactions = catalog.get("reactions", ())
            if isinstance(raw_reactions, Mapping):
                self._reactions_map = raw_reactions
            else:
                self._reactions_map = {r["id"]: r for r in raw_reactions}

            self._reactions_by_key = {
                r["canonicalKey"]: r for r in self._reactions_map.values() if "canonicalKey" in r
            }

            self._pkw = float(self._model.get("pKw", 14.0))
            self._neutral_ph = float(self._model.get("neutralPh", self._pkw / 2))
            self._amount_tolerance = float(self._model.get("tolerances", {}).get("element", 1e-6))
            self._equivalence_tolerance = 1e-6
            self._ph_tolerance = 0.01
            self._temperature_bounds = {"minimum": 0.0, "maximum": 100.0, "default": 25.0}
            self._temperature_capability = {
                "effect": QUALITATIVE_MOTION_EFFECT,
                "motionSpeedFactor": {
                    "model": LINEAR_BOUNDED_MOTION_MODEL,
                    "atMinimum": 0.5,
                    "atMaximum": 2.0,
                    "referenceTemperature_C": 25.0,
                },
            }
            self._motion_speed_model = self._temperature_capability["motionSpeedFactor"]
            self._pressure_capability = {
                "fixed_atm": 1.0,
                "effect": NO_MODELED_PRESSURE_EFFECT,
            }
            self._ph_capability = {
                "modelId": IDEAL_STRONG_MONOPROTIC_PH_MODEL,
                "modelLabel": "Ideal Monoprotic",
                "pKw": self._pkw,
                "neutralPH": self._neutral_ph,
                "pKwSource": "standard",
            }
            self._supported_ph_reaction_ids = frozenset(["rxn.hcl__naoh"])
            self._unsupported_ph_reasons = {NO_MODELED_REACTION_PH_REASON: "no_modeled_reaction"}
            self._validated_fields = ()
            self._field_bounds = {}
            return

        tolerance = self._model.get("tolerances", {}).get("amount_mol")
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance <= 0:
            raise ValueError("catalog must declare a finite positive model.tolerances.amount_mol")
        self._amount_tolerance = float(tolerance)
        self._temperature_bounds = self._require_temperature_bounds()
        self._temperature_capability = self._require_temperature_capability()
        self._motion_speed_model = self._temperature_capability["motionSpeedFactor"]
        self._pressure_capability = self._require_pressure_capability()

        # -- declared pH capability ----------------------------------------
        #
        # Every number the pH model uses is read from the catalog here, once.
        # The neutral point in particular is checked against the declared
        # ``pKw`` rather than trusted on its own, because a catalog that
        # disagreed with itself about where neutral sits would otherwise report
        # two different answers for the same mixture depending on which field a
        # code path happened to read.
        self._ph_capability = self._require_ph_capability()
        self._pkw = float(self._ph_capability["pKw"])
        self._neutral_ph = float(self._ph_capability["neutralPH"])
        self._supported_ph_reaction_ids = frozenset(
            self._ph_capability.get("supportedReactionIds", ())
        )
        # Every unsupported system is explained by a declared reason code, so
        # the declarations are required here rather than at the first mixture
        # that needs one: a catalog without them could only answer "no pH" with
        # no reason, which is the silence requirement 3.6 exists to remove.
        self._unsupported_ph_reasons = self._ph_capability.get("unsupportedReasons") or {}
        if not self._unsupported_ph_reasons:
            raise ValueError("pH capability declares no unsupportedReasons")
        if NO_MODELED_REACTION_PH_REASON not in self._unsupported_ph_reasons:
            raise ValueError(
                f"pH capability must declare the {NO_MODELED_REACTION_PH_REASON} reason"
            )
        self._equivalence_tolerance = self._require_tolerance("equivalence_mol")
        self._ph_tolerance = self._require_tolerance("pH")

        # -- validated input semantics -------------------------------------
        #
        # Each declared field is paired with the catalog bound that governs it
        # once, at construction, so every evaluation judges inputs against the
        # committed catalog release rather than against a number written into
        # this module.
        self._validated_fields: tuple[tuple[ValidatedExperimentField, Mapping[str, Any]], ...] = (
            tuple(
                (descriptor, self._require_bounds(descriptor.bound))
                for descriptor in VALIDATED_EXPERIMENT_FIELDS
            )
        )
        self._field_bounds = {
            descriptor.field: bounds for descriptor, bounds in self._validated_fields
        }

    # -- declared control capabilities -------------------------------------
    #
    # Temperature and pressure are read from the catalog once, at construction,
    # so that a catalog which claimed a quantitative effect for either control
    # would fail loudly instead of being silently ignored by an evaluator that
    # only ever treats them as metadata.

    def _require_bounds(self, bound_name: str) -> Mapping[str, Any]:
        """Resolves one declared control bound, checked to be usable as a bound.

        A bound with a missing, non-finite, or inverted edge would make
        validation meaningless, so the evaluator refuses the catalog at
        construction instead of accepting inputs it cannot honestly judge.
        Equality is allowed because a fixed control such as pressure declares
        the same minimum and maximum.
        """
        bounds = self._model.get("bounds", {}).get(bound_name)
        if bounds is None:
            raise ValueError(f"catalog declares no model.bounds.{bound_name}")
        minimum = _require_finite_declaration(
            bounds.get("minimum"), f"model.bounds.{bound_name}.minimum"
        )
        maximum = _require_finite_declaration(
            bounds.get("maximum"), f"model.bounds.{bound_name}.maximum"
        )
        default = _require_finite_declaration(
            bounds.get("default"), f"model.bounds.{bound_name}.default"
        )
        if maximum < minimum:
            raise ValueError(
                f"model.bounds.{bound_name} must declare a maximum at or above its minimum"
            )
        if not minimum <= default <= maximum:
            raise ValueError(f"model.bounds.{bound_name} must declare a default inside its range")
        return bounds

    def _require_tolerance(self, name: str) -> float:
        """One declared tolerance, checked to be usable as a comparison width."""
        value = self._model.get("tolerances", {}).get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"catalog must declare a finite positive model.tolerances.{name}")
        return float(value)

    def _require_ph_capability(self) -> Mapping[str, Any]:
        capability = self._model.get("capabilities", {}).get("pH")
        if capability is None:
            raise ValueError("catalog declares no pH capability")
        if capability.get("modelId") != IDEAL_STRONG_MONOPROTIC_PH_MODEL:
            raise ValueError(f"unsupported pH model {capability.get('modelId')!r}")
        pkw = _require_finite_declaration(capability.get("pKw"), "pH capability pKw")
        if pkw <= 0:
            raise ValueError("pH capability must declare a positive pKw")
        neutral_ph = _require_finite_declaration(
            capability.get("neutralPH"), "pH capability neutralPH"
        )
        if neutral_ph != pkw / 2:
            raise ValueError(
                "pH capability must declare a neutralPH equal to pKw/2 under the ideal model"
            )
        # Version 1 states a documented standard-temperature pKw. A temperature
        # correction would need a validated catalog function this evaluator does
        # not have, so claiming one is a refusal rather than a silently ignored
        # flag. The temperature capability has already been required to declare
        # ``affectsPH: False`` above, so these two declarations agree.
        if capability.get("temperatureCorrected") is not False:
            raise ValueError(
                "pH capability must declare temperatureCorrected false "
                "without a validated pKw function"
            )
        return capability

    def _require_temperature_bounds(self) -> Mapping[str, Any]:
        bounds = self._require_bounds("temperature_C")
        if float(bounds["maximum"]) <= float(bounds["minimum"]):
            raise ValueError("model.bounds.temperature_C must declare a maximum above its minimum")
        return bounds

    def _require_temperature_capability(self) -> Mapping[str, Any]:
        capability = self._model.get("capabilities", {}).get("temperature")
        if capability is None:
            raise ValueError("catalog declares no temperature capability")
        if capability.get("effect") != QUALITATIVE_MOTION_EFFECT:
            raise ValueError(
                f"temperature capability must declare effect {QUALITATIVE_MOTION_EFFECT}"
            )
        for claim in ("affectsExtent", "affectsRateConstant", "affectsPH"):
            if capability.get(claim) is not False:
                raise ValueError(f"temperature capability must declare {claim} false in version 1")
        motion = capability.get("motionSpeedFactor")
        if motion is None:
            raise ValueError("temperature capability declares no motionSpeedFactor")
        if motion.get("model") != LINEAR_BOUNDED_MOTION_MODEL:
            raise ValueError(f"unsupported motion speed model {motion.get('model')!r}")
        at_minimum = _require_finite_declaration(
            motion.get("atMinimum"), "motionSpeedFactor.atMinimum"
        )
        at_maximum = _require_finite_declaration(
            motion.get("atMaximum"), "motionSpeedFactor.atMaximum"
        )
        _require_finite_declaration(
            motion.get("referenceTemperature_C"), "motionSpeedFactor.referenceTemperature_C"
        )
        if at_minimum <= 0 or at_maximum <= at_minimum:
            raise ValueError(
                "motionSpeedFactor must rise from a positive atMinimum to a larger atMaximum"
            )
        return capability

    def _require_pressure_capability(self) -> Mapping[str, Any]:
        capability = self._model.get("capabilities", {}).get("pressure")
        if capability is None:
            raise ValueError("catalog declares no pressure capability")
        if capability.get("effect") != NO_MODELED_PRESSURE_EFFECT:
            raise ValueError(
                f"pressure capability must declare effect {NO_MODELED_PRESSURE_EFFECT} in version 1"
            )
        if capability.get("selectable") is not False or (
            capability.get("acceptsAlternateValues") is not False
        ):
            raise ValueError(
                "pressure capability must be neither selectable nor accept alternate values"
            )
        fixed_atm = _require_finite_declaration(
            capability.get("fixed_atm"), "pressure capability fixed_atm"
        )
        bounds = self._require_bounds("pressure_atm")
        if bounds.get("minimum") != fixed_atm or bounds.get("maximum") != fixed_atm:
            raise ValueError(f"model.bounds.pressure_atm must be fixed at {fixed_atm} atm")
        return capability

    @property
    def catalog_version(self) -> str:
        return self._catalog["catalogVersion"]

    @property
    def model_version(self) -> str:
        return self._model["modelVersion"]

    @property
    def amount_tolerance(self) -> float:
        return self._amount_tolerance

    @property
    def equivalence_tolerance(self) -> float:
        """Width within which acid and base equivalents count as equal."""
        return self._equivalence_tolerance

    @property
    def ph_tolerance(self) -> float:
        """Width within which two reported pH values count as equal."""
        return self._ph_tolerance

    @property
    def ph_model(self) -> Mapping[str, Any]:
        """The declared pH model this evaluator applies."""
        return {
            "modelId": self._ph_capability["modelId"],
            "modelLabel": self._ph_capability["modelLabel"],
            "pKw": self._pkw,
            "pKwSource": self._ph_capability["pKwSource"],
            "temperatureCorrected": False,
            "neutralPH": self._neutral_ph,
            "supportedReactionIds": tuple(sorted(self._supported_ph_reaction_ids)),
            "unsupportedReasonCodes": tuple(sorted(self._unsupported_ph_reasons)),
        }

    @property
    def bounds(self) -> Mapping[str, Mapping[str, Any]]:
        """Declared bound governing each validated experiment field."""
        return dict(self._field_bounds)

    def _require_field(self, field_name: str, value: Any) -> float:
        """Accepts one field's value or raises the violation that rejected it."""
        for descriptor, bounds in self._validated_fields:
            if descriptor.field != field_name:
                continue
            accepted, violation = _check_field(descriptor, bounds, value, True)
            if violation is not None:
                raise InvalidExperimentQuantityError([violation])
            return float(accepted)
        raise ValueError(f"unknown validated experiment field {field_name!r}")

    def validate_experiment_input(self, experiment_input: Mapping[str, Any]) -> dict[str, float]:
        """Validates every numeric control of one input in a single pass.

        The whole input is judged before anything is calculated, and every
        offending field is collected before the rejection is raised, so a
        learner is never told about one bad control while a second one waits in
        silence. Nothing is clamped and nothing is skipped: either the returned
        record describes a supported experiment, or no result is produced.
        """
        if not isinstance(experiment_input, Mapping):
            raise TypeError("experiment input must be a mapping")

        violations: list[dict[str, Any]] = []
        accepted: dict[str, float] = {}
        for descriptor, bounds in self._validated_fields:
            value, violation = _check_field(
                descriptor, bounds, experiment_input.get(descriptor.field), descriptor.required
            )
            if violation is not None:
                violations.append(violation)
            else:
                accepted[descriptor.field] = float(value)
        if violations:
            raise InvalidExperimentQuantityError(violations)
        return accepted

    def _require_reagent(self, reagent_id: Any, field: str) -> Mapping[str, Any]:
        reagent = self._catalog.get("reagents", {}).get(reagent_id)
        if reagent is None:
            raise ValueError(f"{field} references unknown reagent {reagent_id!r}")
        return reagent

    def find_reaction(self, reagent_a: str, reagent_b: str) -> Mapping[str, Any] | None:
        """Resolves the one reaction record for a pair in either A/B order.

        Both orders normalize to the same catalog key, so reaction identity,
        coefficients, and every projection derived from them are independent of
        the order the learner selected the two solutions in.
        """
        return self._formatter.find_reaction(reagent_a, reagent_b)

    @staticmethod
    def _molecular_coefficient(reaction: Mapping[str, Any], reagent: Mapping[str, Any]) -> int:
        """Coefficient of a reagent's species in the balanced molecular equation."""
        species_id = reagent["speciesId"]
        for term in reaction["molecularReactants"]:
            if term["speciesId"] != species_id:
                continue
            coefficient = term["coefficient"]
            if (
                not isinstance(coefficient, int)
                or isinstance(coefficient, bool)
                or coefficient <= 0
            ):
                raise ValueError(
                    f"reaction {reaction['id']} must declare a positive integer "
                    f"coefficient for {species_id}"
                )
            return coefficient
        raise ValueError(
            f"reaction {reaction['id']} declares no molecular reactant term for {species_id}"
        )

    def _describe_energy(self, presentation: Mapping[str, Any]) -> dict[str, Any]:
        """Energy label of one presentation record, modeled or unsupported.

        The localized boundary message states that the label is qualitative, so
        an exothermic reaction and an unreacted mixture both carry the same
        honest disclaimer about what version 1 does not calculate.
        """
        boundary = self._formatter.describe_capability_boundary("energy")
        label = presentation["energyLabel"]
        return {
            "label": label,
            "exothermic": label == "exothermic",
            "quantitativeEnthalpy": self._model["capabilities"]["energy"]["quantitativeEnthalpy"],
            "messageKey": boundary["messageKey"],
            "message": boundary["message"],
        }

    @staticmethod
    def _describe_appearance(
        presentation: Mapping[str, Any], precipitate: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        return {
            "clarity": presentation["appearance"],
            "color": presentation["solutionColor"],
            "precipitate": precipitate,
        }

    def _dissociation_multiplicity(self, reagent_id: str, ion_id: str, role: str) -> int:
        """Ions of one kind a single formula unit of a reagent releases.

        Read from the catalog's dissociation record rather than assumed, so the
        pH model stays driven by declared chemistry. Version 1 declares the
        *monoprotic* model and lists ``polyprotic`` among the systems it
        excludes, so a multiplicity other than one is a catalog claiming
        chemistry these formulas do not cover, and it is refused instead of
        silently treated as one equivalent per formula unit.
        """
        reagent = self._catalog.get("reagents", {}).get(reagent_id)
        if reagent is None:
            raise ValueError(f"pH model references unknown {role} reagent {reagent_id!r}")
        for term in reagent.get("ions", ()):
            if term["ionId"] != ion_id:
                continue
            multiplicity = term["multiplicity"]
            if multiplicity != 1:
                raise ValueError(
                    f"{IDEAL_STRONG_MONOPROTIC_PH_MODEL} covers only monoprotic reagents, "
                    f"but {reagent_id} releases {multiplicity} {ion_id} per formula unit"
                )
            return int(multiplicity)
        raise ValueError(f"{reagent_id} does not dissociate into {ion_id!r}")

    def _describe_supported_ph(
        self,
        reaction: Mapping[str, Any],
        reactants: Sequence[Mapping[str, Any]],
        total_volume_l: float,
    ) -> dict[str, Any]:
        """pH of the one supported acid-base system, from unrounded amounts.

        The arithmetic is exactly what the catalog declares and nothing more::

            nH  = acid amount  x protons per formula unit
            nOH = base amount  x hydroxides per formula unit
            d   = nH - nOH
            V   = total mixed volume

        ``d`` beyond the declared equivalence tolerance on the acid side gives
        ``[H+] = d/V`` and ``pH = -log10([H+])``; beyond it on the base side
        gives ``[OH-] = -d/V``, ``pOH = -log10([OH-])``, and
        ``pH = pKw - pOH``; within it the mixture is at equivalence and the pH
        is the ``pKw/2`` neutral point the catalog declares. Comparing against
        a tolerance rather than against zero is what makes an equimolar mixture
        read as neutral: ``0.05 - 0.05`` is exactly zero only when the binary
        arithmetic happens to cooperate.

        Every input is an unrounded domain value, and the returned pH is
        unrounded too, so the displayed two-decimal pH is a projection of this
        number and never an input to it.
        """
        declared = reaction["pH"]
        if declared["modelId"] != self._ph_capability["modelId"]:
            raise ValueError(
                f"reaction {reaction['id']} declares pH model {declared['modelId']!r}, "
                "not the one the catalog supports"
            )
        if reaction["id"] not in self._supported_ph_reaction_ids:
            raise ValueError(
                f"reaction {reaction['id']} declares a pH model "
                "the catalog does not list as supported"
            )

        protons_per_acid = self._dissociation_multiplicity(
            declared["acidReagentId"], declared["protonIonId"], "acid"
        )
        hydroxides_per_base = self._dissociation_multiplicity(
            declared["baseReagentId"], declared["hydroxideIonId"], "base"
        )

        acid = next(
            (item for item in reactants if item["reagentId"] == declared["acidReagentId"]), None
        )
        base = next(
            (item for item in reactants if item["reagentId"] == declared["baseReagentId"]), None
        )
        if acid is None or base is None:
            raise ValueError(
                f"reaction {reaction['id']} pH model names a reagent "
                "this experiment does not contain"
            )
        if not total_volume_l > 0:
            raise ValueError(
                "a mixed volume must be positive to report a concentration, "
                f"received {total_volume_l!r}"
            )

        acid_equivalents_mol = acid["amount_mol"] * protons_per_acid
        base_equivalents_mol = base["amount_mol"] * hydroxides_per_base
        equivalence_difference_mol = acid_equivalents_mol - base_equivalents_mol
        equivalence = abs(equivalence_difference_mol) <= self._equivalence_tolerance

        if equivalence:
            side = PH_NEUTRAL
        elif equivalence_difference_mol > 0:
            side = PH_ACIDIC
        else:
            side = PH_BASIC

        if equivalence:
            excess_ion_id = None
        elif side == PH_ACIDIC:
            excess_ion_id = declared["protonIonId"]
        else:
            excess_ion_id = declared["hydroxideIonId"]
        excess_amount_mol = 0.0 if equivalence else abs(equivalence_difference_mol)
        excess_concentration_mol_l = excess_amount_mol / total_volume_l

        if equivalence:
            value = self._pkw / 2
        elif side == PH_ACIDIC:
            value = negative_log10(excess_concentration_mol_l)
        else:
            value = self._pkw - negative_log10(excess_concentration_mol_l)

        return {
            "supported": True,
            "value": value,
            "side": side,
            "equivalence": equivalence,
            "modelId": declared["modelId"],
            "modelLabel": self._ph_capability["modelLabel"],
            "pKw": self._pkw,
            "pKwSource": self._ph_capability["pKwSource"],
            "temperatureCorrected": False,
            "neutralPH": self._neutral_ph,
            "acidReagentId": declared["acidReagentId"],
            "baseReagentId": declared["baseReagentId"],
            "protonIonId": declared["protonIonId"],
            "hydroxideIonId": declared["hydroxideIonId"],
            "acidEquivalents_mol": acid_equivalents_mol,
            "baseEquivalents_mol": base_equivalents_mol,
            "equivalenceDifference_mol": equivalence_difference_mol,
            "totalVolume_L": total_volume_l,
            "excessIonId": excess_ion_id,
            "excessAmount_mol": excess_amount_mol,
            "excessConcentration_molL": excess_concentration_mol_l,
            # A calculated pH needs no boundary explanation, and the
            # unsupported variant below fills exactly these three fields, so a
            # consumer reads one pH block either way instead of two different
            # ones.
            "reasonCode": None,
            "messageKey": None,
            "message": None,
        }

    def describe_unsupported_ph(self, reason_code: str) -> dict[str, Any]:
        """The same pH block for a system the declared model does not cover.

        Nothing is calculated and nothing is defaulted: the numeric value stays
        ``None`` and what is added is only the explanation — the stable reason
        code, its catalog message key, and the localized Indonesian text a
        learner can be shown. ``ChemistryFormatter.describe_unsupported_ph``
        resolves both from the catalog and raises on a code the catalog does not
        declare, so every reason a result carries is traceable to the committed
        release.

        Every field the supported variant calculates is present and explicitly
        ``None``, including the model identity: the ideal strong-acid/strong-base
        model was not applied here, so naming it would suggest a value had been
        computed under it.
        """
        unsupported = self._formatter.describe_unsupported_ph(reason_code)
        return {
            "supported": False,
            "value": None,
            "side": None,
            "equivalence": None,
            "modelId": None,
            "modelLabel": None,
            "pKw": None,
            "pKwSource": None,
            "temperatureCorrected": None,
            "neutralPH": None,
            "acidReagentId": None,
            "baseReagentId": None,
            "protonIonId": None,
            "hydroxideIonId": None,
            "acidEquivalents_mol": None,
            "baseEquivalents_mol": None,
            "equivalenceDifference_mol": None,
            "totalVolume_L": None,
            "excessIonId": None,
            "excessAmount_mol": None,
            "excessConcentration_molL": None,
            "reasonCode": unsupported["reasonCode"],
            "messageKey": unsupported["messageKey"],
            "message": unsupported["message"],
        }

    def _unsupported_ph_for_reaction(self, reaction: Mapping[str, Any]) -> dict[str, Any]:
        """Unsupported pH block for a modeled reaction that declares no pH model.

        The reason comes from the reaction's own ``pHUnsupportedReasonCode``,
        which is why the evaluator contains no list of which pairs produce
        salts: the catalog says so per reaction, and a reaction that declares
        neither a pH model nor a reason is a catalog that cannot explain its own
        boundary, so it is refused rather than answered with an unexplained
        ``None``.
        """
        reason_code = reaction.get("pHUnsupportedReasonCode")
        if not reason_code:
            raise ValueError(
                f"reaction {reaction['id']} declares neither a pH model "
                "nor a pHUnsupportedReasonCode"
            )
        return self.describe_unsupported_ph(reason_code)

    def motion_speed_factor(self, temperature_c: float) -> float:
        """Bounded qualitative motion factor for one temperature.

        The catalog declares a ``linear_bounded`` band: the factor is
        ``atMinimum`` at the lowest selectable temperature and rises linearly
        to ``atMaximum`` at the highest. The band stays closed by *rejecting* a
        temperature outside the declared control range rather than by clamping
        it, so a result can never exist for a temperature the catalog does not
        cover. The factor is a visualization speed multiplier and nothing else:
        it is not a rate constant, an extent, or a pH correction.
        """
        temperature = self._require_field("temperature_C", temperature_c)
        minimum = self._temperature_bounds["minimum"]
        maximum = self._temperature_bounds["maximum"]
        at_minimum = self._motion_speed_model["atMinimum"]
        at_maximum = self._motion_speed_model["atMaximum"]
        position = (temperature - minimum) / (maximum - minimum)
        return at_minimum + position * (at_maximum - at_minimum)

    def describe_motion(self, temperature_c: float) -> dict[str, Any]:
        """The temperature control's whole modeled consequence.

        Everything here is metadata about how the mixture is *shown*, which is
        why the block carries ``quantitative: False`` and the catalog's
        localized temperature boundary statement beside the factor. Nothing in
        the quantitative part of the result reads any of it, so two experiments
        that differ only in temperature differ only in this block.
        """
        boundary = self._formatter.describe_capability_boundary("temperature")
        return {
            "quantitative": False,
            "effect": self._temperature_capability["effect"],
            "model": self._motion_speed_model["model"],
            "temperature_C": temperature_c,
            "referenceTemperature_C": self._motion_speed_model["referenceTemperature_C"],
            "speedFactor": self.motion_speed_factor(temperature_c),
            "referenceSpeedFactor": self.motion_speed_factor(
                self._motion_speed_model["referenceTemperature_C"]
            ),
            "minimumSpeedFactor": self._motion_speed_model["atMinimum"],
            "maximumSpeedFactor": self._motion_speed_model["atMaximum"],
            "affectsExtent": False,
            "affectsRateConstant": False,
            "affectsPH": False,
            "messageKey": boundary["messageKey"],
            "message": boundary["message"],
        }

    def describe_pressure(self) -> dict[str, Any]:
        """The fixed, unmodeled pressure state.

        Version 1 models one pressure, so the result reports the catalog's
        fixed value, which is the only value validation lets an accepted input
        carry: an alternate selection is rejected before any of this runs, so
        it has neither a path into a calculated quantity nor a path into a
        silently ignored field. Reporting it with the localized boundary
        statement is what keeps the control honestly marked as unsupported.
        """
        boundary = self._formatter.describe_capability_boundary("pressure")
        return {
            "atm": self._pressure_capability["fixed_atm"],
            "effect": self._pressure_capability["effect"],
            "selectable": False,
            "acceptsAlternateValues": False,
            "affectsResult": False,
            "messageKey": boundary["messageKey"],
            "message": boundary["message"],
        }

    def _no_modeled_reaction_result(
        self,
        display: Mapping[str, Any],
        solutions: tuple[dict[str, Any], ...],
        total_volume_l: float,
        conditions: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Result for a supported pair the catalog models no reaction for.

        Nothing reacts, so the chemistry is fully determined without a reaction
        record: extent is zero, there is no product, no equation, and no
        limiting reactant, and every formula unit of both solutions remains in
        the mixture. The phase comes from the catalog's declared
        ``no_modeled_reaction`` outcome, which is checked here to be a single
        unlayered phase, so the macro projection has no layered phase to render
        even if a future catalog edit tried to introduce one without also
        declaring an immiscibility model.
        """
        outcome = self._model.get("unsupportedOutcomes", {}).get("noModeledReaction")
        if outcome is None:
            raise ValueError("catalog declares no no_modeled_reaction outcome")
        if outcome["category"] != NO_MODELED_REACTION_CATEGORY:
            raise ValueError(
                f"no_modeled_reaction outcome must declare category {NO_MODELED_REACTION_CATEGORY}"
            )

        phase_model = display["phase"]
        if phase_model["layered"] is not False or tuple(phase_model["phases"]) != ("aq",):
            raise ValueError(
                f"phase model {phase_model['phaseModel']} cannot represent two mixed aqueous "
                "solutions as one homogeneous phase"
            )

        reactants = tuple(
            {
                **solution,
                # No balanced equation exists, so there is no coefficient to
                # report; a placeholder of 1 would read as a consumed formula
                # unit.
                "coefficient": None,
                "consumed_mol": 0.0,
                "remaining_mol": solution["amount_mol"],
                "limiting": False,
            }
            for solution in solutions
        )

        return {
            "catalogVersion": self.catalog_version,
            "modelVersion": self.model_version,
            "support": {
                "kind": "unsupported",
                "code": outcome["reasonCode"],
                "messageKey": display["messageKey"],
                "message": display["message"],
            },
            "pairKey": display["pairKey"],
            "reactionId": None,
            "category": outcome["category"],
            "displayName": None,
            "reactants": reactants,
            "totalVolume_L": total_volume_l,
            "extent_mol": 0.0,
            "limitingSpecies": (),
            # Both solutions are left over in full, which is the same invariant
            # the supported path reports: excess is whatever remains after
            # consumption.
            "excessSpecies": tuple(
                {
                    "speciesId": reactant["speciesId"],
                    "reagentId": reactant["reagentId"],
                    "phase": reactant["phase"],
                    "amount_mol": reactant["remaining_mol"],
                    "role": "excess",
                }
                for reactant in reactants
            ),
            "products": (),
            "spectatorIons": (),
            "molecularEquation": None,
            "netIonicEquation": None,
            "productSummary": None,
            "phaseModel": phase_model,
            "appearance": self._describe_appearance(
                outcome["presentation"], display["precipitate"]
            ),
            "energy": self._describe_energy(outcome["presentation"]),
            "motion": conditions["motion"],
            "pressure": conditions["pressure"],
            # No reaction is modeled, so there is no acid-base system to
            # calculate a pH for. The block still reports the catalog's declared
            # reason and its localized explanation, so an unreacted mixture says
            # why rather than showing a defaulted number.
            "pH": self.describe_unsupported_ph(NO_MODELED_REACTION_PH_REASON),
            "speciesLedger": None,
        }

    def evaluate_experiment(
        self, experiment_input: Mapping[str, Any], energy_term: str | None = None
    ) -> dict[str, Any]:
        """Evaluates one supported experiment.

        Steps, in the order the chemistry requires them:

        1. resolve the reagent pair through its normalized catalog key,
        2. calculate each formula-unit amount from concentration and volume,
        3. calculate extent as the smallest available-amount/coefficient ratio,
        4. name every reactant consumed to within the declared amount tolerance,
        5. report remaining excess and product amounts from that one extent,
        6. project equations, phase, appearance, and energy from the same record.

        A pair with no catalog reaction takes the same first two steps and then
        returns the ``no_modeled_reaction`` result, so an ordinary mixture is a
        described outcome rather than a raised failure.

        The result is canonically ordered by reagent id, so evaluating ``(a, b)``
        and ``(b, a)`` with the same quantities returns equal results.
        """
        if not isinstance(experiment_input, Mapping):
            raise TypeError("experiment input must be a mapping")

        if "substances" in self._catalog:
            reagent_a = experiment_input.get("reagentAId") or experiment_input.get("reagentA")
            reagent_b = experiment_input.get("reagentBId") or experiment_input.get("reagentB")
            if not reagent_a or not reagent_b:
                raise ValueError("both reagent identifiers are required")

            pair = resolve_canonical_pair(reagent_a, reagent_b)
            key = pair["key"]

            if reagent_a == reagent_b:
                return {
                    "outcome": "no_modeled_reaction",
                    "reasonCode": "same_reagent",
                    "explanationId": "no_modeled_reaction",
                    "productId": None,
                    "candidateSolidId": None,
                    "saturation": None,
                    "molecularEquation": None,
                    "netIonicEquation": None,
                    "spectatorIonIds": (),
                    "ledger": None,
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }

            reaction = self._reactions_by_key.get(key)
            if not reaction:
                return {
                    "outcome": "no_modeled_reaction",
                    "reasonCode": "no_modeled_reaction",
                    "explanationId": "no_modeled_reaction",
                    "productId": None,
                    "candidateSolidId": None,
                    "saturation": None,
                    "molecularEquation": None,
                    "netIonicEquation": None,
                    "spectatorIonIds": (),
                    "ledger": None,
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }

            rxn_outcome = reaction.get("outcome")
            if rxn_outcome == "unsupported":
                return {
                    "outcome": "unsupported",
                    "reasonCode": reaction.get("reasonCode"),
                    "unsupportedReason": reaction.get("reasonCode"),
                    "explanationId": reaction.get("explanationId"),
                    "productId": None,
                    "candidateSolidId": None,
                    "saturation": None,
                    "molecularEquation": None,
                    "netIonicEquation": None,
                    "spectatorIonIds": (),
                    "ledger": None,
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }

            if rxn_outcome == "no_modeled_reaction":
                return {
                    "outcome": "no_modeled_reaction",
                    "reasonCode": reaction.get("reasonCode", "no_modeled_reaction"),
                    "explanationId": reaction.get("explanationId", "no_modeled_reaction"),
                    "productId": None,
                    "candidateSolidId": None,
                    "saturation": None,
                    "molecularEquation": None,
                    "netIonicEquation": None,
                    "spectatorIonIds": (),
                    "ledger": None,
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }

            if reaction.get("reactionClass") == "neutralization" or rxn_outcome == "neutralization":
                vol_a_l = experiment_input.get("volumeA_L")
                if vol_a_l is None and "volumeA_mL" in experiment_input:
                    vol_a_l = experiment_input["volumeA_mL"] / 1000.0
                vol_b_l = experiment_input.get("volumeB_L")
                if vol_b_l is None and "volumeB_mL" in experiment_input:
                    vol_b_l = experiment_input["volumeB_mL"] / 1000.0
                c_a = experiment_input.get("concentrationA_M")
                if c_a is None:
                    c_a = experiment_input.get("concentrationA_molL")
                c_b = experiment_input.get("concentrationB_M")
                if c_b is None:
                    c_b = experiment_input.get("concentrationB_molL")

                total_v = float(vol_a_l + vol_b_l)
                sub_a = self._substances_map.get(reagent_a)
                is_a_acid = any(part["ionId"] == "ion.h_1p" for part in sub_a.get("dissociation", ()))
                n_acid = (float(c_a) * float(vol_a_l)) if is_a_acid else (float(c_b) * float(vol_b_l))
                n_base = (float(c_b) * float(vol_b_l)) if is_a_acid else (float(c_a) * float(vol_a_l))
                diff = n_acid - n_base
                tol = 1e-6
                if abs(diff) <= tol:
                    ph_val = self._pkw / 2
                elif diff > tol:
                    ph_val = -math.log10(diff / total_v)
                else:
                    poh = -math.log10(-diff / total_v)
                    ph_val = self._pkw - poh

                return {
                    "outcome": "neutralization",
                    "reactionId": reaction["id"],
                    "productId": "sub.h2o",
                    "candidateSolidId": None,
                    "saturation": None,
                    "molecularEquation": reaction["molecularEquation"],
                    "netIonicEquation": reaction["netIonicEquation"],
                    "spectatorIonIds": tuple(reaction["spectatorIonIds"]),
                    "pH": {"value": ph_val, "supported": True, "neutralPH": self._neutral_ph},
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }

            # Double displacement with Ksp
            conc = post_mix_concentrations(experiment_input, self._catalog)
            solid_id = reaction["candidateSolidId"]
            q = ion_product(conc, solid_id, self._catalog)
            ksp_literal = str(reaction["kspRef"]["value"])
            ksp_val = float(ksp_literal)
            ratio = q / ksp_val
            ksp_source_id = reaction["kspRef"]["sourceId"]
            saturation = {
                "Q": q,
                "ksp": ksp_val,
                "ratio": ratio,
                "kspSourceId": ksp_source_id,
                "ksp_literal": ksp_literal,
            }

            if q > ksp_val:
                return {
                    "outcome": "precipitate_forms",
                    "reactionId": reaction["id"],
                    "productId": solid_id,
                    "candidateSolidId": solid_id,
                    "saturation": saturation,
                    "molecularEquation": reaction["molecularEquation"],
                    "netIonicEquation": reaction["netIonicEquation"],
                    "spectatorIonIds": tuple(reaction["spectatorIonIds"]),
                    "explanationId": "precipitate_forms",
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }
            else:
                solid_name = substance_name(solid_id, self._catalog)
                explanation_template = self._catalog.get("presentation", {}).get("explanations", {}).get(
                    "below_saturation", ""
                )
                explanation = (
                    explanation_template.replace("{{Q}}", f"{q:.4e}")
                    .replace("{{solidName}}", solid_name)
                    .replace("{{Ksp}}", ksp_literal)
                )
                return {
                    "outcome": "below_saturation",
                    "reactionId": reaction["id"],
                    "productId": None,
                    "candidateSolidId": solid_id,
                    "saturation": saturation,
                    "molecularEquation": None,
                    "netIonicEquation": None,
                    "spectatorIonIds": tuple(reaction["spectatorIonIds"]),
                    "explanationId": "below_saturation",
                    "explanation": explanation,
                    "speciesLedger": None,
                    "catalogVersion": self.catalog_version,
                }

        reagent_a = self._require_reagent(experiment_input.get("reagentA"), "reagentA")
        reagent_b = self._require_reagent(experiment_input.get("reagentB"), "reagentB")

        # Every numeric control is judged first, in one pass, so an
        # out-of-range or undefined selection produces no result at all rather
        # than a result calculated from a quantity quietly moved into range.
        validated = self.validate_experiment_input(experiment_input)

        quantities = {
            reagent_a["id"]: (
                validated["concentrationA_molL"],
                validated["volumeA_mL"],
            ),
            reagent_b["id"]: (
                validated["concentrationB_molL"],
                validated["volumeB_mL"],
            ),
        }
        if len(quantities) != 2:
            raise ValueError(
                f"an experiment must mix two different reagents, received {reagent_a['id']} twice"
            )

        # Canonical reagent order makes the whole result order-independent; the
        # A/B order the learner chose is a presentation concern, not a chemical
        # one.
        first_id, second_id = sorted((reagent_a["id"], reagent_b["id"]))
        display = self._formatter.format_reaction_display(first_id, second_id, energy_term)

        # Amounts depend only on what was poured, so they are calculated before
        # the reaction lookup and reported whether or not a reaction is modeled.
        solutions: list[dict[str, Any]] = []
        for reagent_id in (first_id, second_id):
            reagent = self._catalog["reagents"][reagent_id]
            concentration_mol_l, volume_ml = quantities[reagent_id]
            solutions.append(
                {
                    "reagentId": reagent_id,
                    "speciesId": reagent["speciesId"],
                    "phase": reagent["phase"],
                    "concentration_molL": concentration_mol_l,
                    "volume_mL": volume_ml,
                    "volume_L": to_litres(volume_ml),
                    "amount_mol": calculate_amount(concentration_mol_l, volume_ml),
                }
            )
        total_volume_l = sum(solution["volume_L"] for solution in solutions)

        # Qualitative conditions, kept apart from the arithmetic above: the
        # motion factor is derived from the validated temperature, and the
        # pressure state comes from the catalog because validation has already
        # proved the accepted ``pressure_atm`` equals the one fixed value
        # version 1 models.
        conditions = {
            "motion": self.describe_motion(validated["temperature_C"]),
            "pressure": self.describe_pressure(),
        }

        reaction = self.find_reaction(first_id, second_id)
        if reaction is None:
            return self._no_modeled_reaction_result(
                display, tuple(solutions), total_volume_l, conditions
            )

        reactants = [
            {
                **solution,
                "coefficient": self._molecular_coefficient(
                    reaction, self._catalog["reagents"][solution["reagentId"]]
                ),
            }
            for solution in solutions
        ]

        # Balanced extent: the reaction can only advance as far as the scarcest
        # reactant allows once its own stoichiometric coefficient is accounted
        # for.
        extent_mol = min(reactant["amount_mol"] / reactant["coefficient"] for reactant in reactants)

        reactant_ledger: list[dict[str, Any]] = []
        for reactant in reactants:
            consumed_mol = extent_mol * reactant["coefficient"]
            remaining_mol = reactant["amount_mol"] - consumed_mol
            reactant_ledger.append(
                {
                    **reactant,
                    "consumed_mol": consumed_mol,
                    # Consumption never exceeds the available amount, so a
                    # remainder within the declared tolerance is exhaustion
                    # rather than a negative amount.
                    "remaining_mol": (
                        0.0 if abs(remaining_mol) <= self._amount_tolerance else remaining_mol
                    ),
                    "limiting": remaining_mol <= self._amount_tolerance,
                }
            )

        products = tuple(
            {
                "speciesId": term["speciesId"],
                "phase": term["phase"],
                "coefficient": term["coefficient"],
                "amount_mol": extent_mol * term["coefficient"],
                "role": "product",
            }
            for term in reaction["molecularProducts"]
        )

        excess_species = tuple(
            {
                "speciesId": reactant["speciesId"],
                "reagentId": reactant["reagentId"],
                "phase": reactant["phase"],
                "amount_mol": reactant["remaining_mol"],
                "role": "excess",
            }
            for reactant in reactant_ledger
            if reactant["remaining_mol"] > self._amount_tolerance
        )

        return {
            "catalogVersion": self.catalog_version,
            "modelVersion": self.model_version,
            "support": {"kind": "supported", "modelVersion": self.model_version},
            "pairKey": display["pairKey"],
            "reactionId": reaction["id"],
            "category": reaction["category"],
            "displayName": reaction["displayName"],
            "reactants": tuple(reactant_ledger),
            "totalVolume_L": total_volume_l,
            "extent_mol": extent_mol,
            "limitingSpecies": tuple(
                reactant["speciesId"] for reactant in reactant_ledger if reactant["limiting"]
            ),
            "excessSpecies": excess_species,
            "products": products,
            "spectatorIons": tuple(reaction["spectatorIons"]),
            "molecularEquation": display["molecularEquation"],
            "netIonicEquation": display["netIonicEquation"],
            "productSummary": display["productSummary"],
            "phaseModel": display["phase"],
            "appearance": self._describe_appearance(
                reaction["presentation"], display["precipitate"]
            ),
            "energy": self._describe_energy(reaction["presentation"]),
            # Temperature's only consequence and pressure's declared absence of
            # one. Neither block is read by any quantity above it, pH included.
            "motion": conditions["motion"],
            "pressure": conditions["pressure"],
            # The one modeled acid-base system reports a calculated pH; every
            # other reaction reports the same block with a None value and the
            # reason its own catalog record declares. Task 6.1 owns the
            # ion-level ledger.
            "pH": (
                self._describe_supported_ph(reaction, tuple(reactant_ledger), total_volume_l)
                if reaction.get("pH")
                else self._unsupported_ph_for_reaction(reaction)
            ),
            "speciesLedger": None,
        }


#: Evaluator bound to the generated catalog binding.
EVALUATOR = ChemistryEvaluator()


def evaluate_experiment(
    catalog_or_input: Mapping[str, Any],
    experiment_input_or_catalog: Mapping[str, Any] | None = None,
    energy_term: str | None = None,
) -> dict[str, Any]:
    """Catalog-first entry point matching the design document's signature.

    Supports both (catalog, experiment_input) and (experiment_input, catalog).
    """
    if experiment_input_or_catalog is None:
        return ChemistryEvaluator(CATALOG).evaluate_experiment(catalog_or_input, energy_term)
    if "reagents" in catalog_or_input or "substances" in catalog_or_input:
        cat = catalog_or_input
        inp = experiment_input_or_catalog
    else:
        inp = catalog_or_input
        cat = experiment_input_or_catalog
    return ChemistryEvaluator(cat).evaluate_experiment(inp, energy_term)
