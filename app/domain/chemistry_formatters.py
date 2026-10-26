"""Display formatters for the canonical chemistry catalog (backend parity).

This module is the backend half of a pair. ``js/domain/chemistry-formatters.js``
is the frontend half, and both derive every formula, equation, and localized
model-boundary string from the same structural catalog terms. For one catalog
they must produce identical output; ``tests/fixtures/chemistry/display-formatting.json``
is the shared golden snapshot both suites assert against.

The catalog stores no Unicode formula and no equation string, so these
formatters are the only place a chemistry display value is produced. Nothing is
read from a parallel handwritten table, which is what allowed malformed values
such as ``AgNO`` before.

Every function is pure: no provider, environment, clock, random, or network
dependency, and no rounding of scientific quantities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from app.generated.chemistry_catalog import CATALOG
except ModuleNotFoundError:
    from backend.app.generated.chemistry_catalog import CATALOG

__all__ = [
    "CAPABILITY_BOUNDARY_KEYS",
    "HALF_AWAY_FROM_ZERO",
    "PHASE_LABELS",
    "REACTION_ARROW",
    "SUPERSCRIPT_MINUS",
    "SUPERSCRIPT_PLUS",
    "SUPPORTED_ROUNDING_MODES",
    "TERM_SEPARATOR",
    "ChemistryFormatter",
    "FORMATTER",
    "format_charge_superscript",
    "format_fixed",
    "format_formula_terms",
    "format_phase",
    "format_subscript",
    "normalize_pair_key",
    "round_half_away_from_zero",
]

_SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
_SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"

SUPERSCRIPT_PLUS = "\u207a"
SUPERSCRIPT_MINUS = "\u207b"
REACTION_ARROW = "\u2192"
TERM_SEPARATOR = " + "

PHASE_LABELS: Mapping[str, str] = {"aq": "(aq)", "s": "(s)", "l": "(l)", "g": "(g)"}

CAPABILITY_BOUNDARY_KEYS: Mapping[str, str] = {
    "temperature": "messageKey",
    "pressure": "messageKey",
    "energy": "messageKey",
    "inverse": "messageKey",
    "visualization": "scaleDisclosureKey",
}


def _require_positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, received {value!r}")
    return value


def _map_digits(value: int, alphabet: str) -> str:
    return "".join(alphabet[int(digit)] for digit in str(value))


def format_subscript(count: int) -> str:
    """Renders a positive integer count as Unicode subscript digits."""
    return _map_digits(_require_positive_integer(count, "subscript count"), _SUBSCRIPT_DIGITS)


def format_charge_superscript(charge: int | None) -> str:
    """Renders an ion charge as a superscript magnitude and sign.

    The magnitude is omitted when it is one, so ``1`` becomes ``+`` and ``-2``
    becomes ``2-``.
    """
    if charge is None or charge == 0:
        return ""
    if not isinstance(charge, int) or isinstance(charge, bool):
        raise ValueError(f"charge must be an integer, received {charge!r}")
    sign = SUPERSCRIPT_PLUS if charge > 0 else SUPERSCRIPT_MINUS
    magnitude = abs(charge)
    if magnitude == 1:
        return sign
    return f"{_map_digits(magnitude, _SUPERSCRIPT_DIGITS)}{sign}"


def format_formula_terms(terms: Sequence[Mapping[str, Any]]) -> str:
    """Renders structural formula terms as a Unicode formula.

    A term is either an element with a count or a parenthesized group with a
    count. Counts of one stay implicit, which turns the structural ``Pb`` plus
    group(``N``, ``O3``) times two into ``Pb(NO3)2``.
    """
    if not isinstance(terms, Sequence) or isinstance(terms, str) or len(terms) == 0:
        raise TypeError("formula terms must be a non-empty sequence")

    parts: list[str] = []
    for term in terms:
        count = _require_positive_integer(term.get("count"), "formula term count")
        suffix = "" if count == 1 else format_subscript(count)
        group = term.get("group")
        if group is not None:
            parts.append(f"({format_formula_terms(group)}){suffix}")
            continue
        element = term.get("element")
        if not isinstance(element, str) or element == "":
            raise TypeError(f"formula term must declare an element or a group, received {term!r}")
        parts.append(f"{element}{suffix}")
    return "".join(parts)


def format_phase(phase: str) -> str:
    """Renders the phase annotation of an equation term."""
    label = PHASE_LABELS.get(phase)
    if label is None:
        raise ValueError(f"unknown phase {phase!r}")
    return label


def normalize_pair_key(reagent_a: str, reagent_b: str) -> str:
    """Order-independent reagent pair key: both identifiers sorted and joined."""
    return "|".join(sorted((reagent_a, reagent_b)))


HALF_AWAY_FROM_ZERO = "half_away_from_zero"
"""The only rounding mode the display layer implements.

It is stated explicitly rather than delegated to ``format`` or ``round``,
because the language defaults disagree: Python rounds half to even and
JavaScript's ``toFixed`` rounds the binary value, so the same quantity would
print differently on the two halves of the pair.
"""

SUPPORTED_ROUNDING_MODES: tuple[str, ...] = (HALF_AWAY_FROM_ZERO,)


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, received {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number, received {value!r}")
    return number


def _require_display_precision(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, received {value!r}")
    return value


def _significant_digits(magnitude: float) -> tuple[str, int]:
    """Shortest round-trip significant digits and the leading digit's exponent.

    Rounding is performed on these decimal digits so that both language halves
    round the same digit string rather than two different binary expansions.
    """
    text = repr(magnitude).lower()
    if "e" in text:
        mantissa_text, exponent_text = text.split("e")
        exponent = int(exponent_text)
    else:
        mantissa_text, exponent = text, 0
    integer_text, _, fraction_text = mantissa_text.partition(".")
    combined = f"{integer_text}{fraction_text}"
    digits = combined.lstrip("0")
    if digits == "":
        return "0", 0
    exponent += len(integer_text) - 1 - (len(combined) - len(digits))
    return digits.rstrip("0") or "0", exponent


def _increment_digits(digits: str) -> str:
    """Adds one to a digit string, growing it by a leading ``1`` on full carry."""
    out = list(digits)
    for index in range(len(out) - 1, -1, -1):
        if out[index] == "9":
            out[index] = "0"
            continue
        out[index] = str(int(out[index]) + 1)
        return "".join(out)
    return f"1{''.join(out)}"


def _split_positional(digits: str, exponent: int) -> tuple[str, str]:
    """Positional integer and fraction digits of the described magnitude."""
    point_position = exponent + 1
    if point_position <= 0:
        return "0", f"{'0' * -point_position}{digits}"
    if point_position >= len(digits):
        return f"{digits}{'0' * (point_position - len(digits))}", ""
    return digits[:point_position], digits[point_position:]


def format_fixed(value: float, precision: int, decimal_separator: str = ".") -> str:
    """Renders a finite number at a fixed number of decimal places.

    Rounding is half away from zero on the decimal digits. A value that rounds
    to zero loses its sign, so an excess amount of ``-1e-18`` prints as
    ``0.0000`` rather than as the nonsense ``-0.0000``.
    """
    number = _require_finite_number(value, "display value")
    _require_display_precision(precision, "display precision")

    negative = math.copysign(1.0, number) < 0
    digits, exponent = _significant_digits(abs(number))
    integer_digits, fraction_digits = _split_positional(digits, exponent)

    if len(fraction_digits) <= precision:
        fraction_digits = fraction_digits.ljust(precision, "0")
    elif fraction_digits[precision] >= "5":
        carried = _increment_digits(f"{integer_digits}{fraction_digits[:precision]}")
        cut = len(carried) - precision
        integer_digits, fraction_digits = carried[:cut], carried[cut:]
    else:
        fraction_digits = fraction_digits[:precision]

    integer_digits = integer_digits.lstrip("0") or "0"
    if not any(digit in "123456789" for digit in f"{integer_digits}{fraction_digits}"):
        negative = False

    fraction = f"{decimal_separator}{fraction_digits}" if precision > 0 else ""
    return f"{'-' if negative else ''}{integer_digits}{fraction}"


def round_half_away_from_zero(value: float, precision: int) -> float:
    """Rounds a value half away from zero to a number of decimal places.

    The result is a *display* value. It is deliberately not produced inside the
    evaluator: feeding it back into extent, excess, pH, or the species ledger is
    exactly the precision leak requirement 3.7 forbids.
    """
    return float(format_fixed(value, precision, "."))


class ChemistryFormatter:
    """Formats catalog chemistry for display.

    Unknown identifiers raise instead of returning a placeholder, so a drifted
    reference fails loudly rather than rendering a silently wrong formula.
    """

    def __init__(self, catalog: Mapping[str, Any] = CATALOG) -> None:
        if not isinstance(catalog, Mapping):
            raise TypeError("catalog must be a mapping")
        self._catalog = catalog
        self._locale = catalog.get("locale")
        self._messages = catalog.get("messages", {}).get(self._locale, {})
        display = catalog.get("model", {}).get("display")
        if not isinstance(display, Mapping):
            display = {
                "roundingMode": HALF_AWAY_FROM_ZERO,
                "decimalSeparator": ",",
                "precision": {
                    "amount_mol": 4,
                    "concentration_molL": 3,
                    "extent_mol": 4,
                    "pH": 2,
                    "pressure_atm": 1,
                    "speedFactor": 2,
                    "temperature_C": 0,
                    "volume_mL": 1,
                },
            }
        if display.get("roundingMode") not in SUPPORTED_ROUNDING_MODES:
            raise ValueError(f"unsupported display rounding mode {display.get('roundingMode')!r}")
        separator = display.get("decimalSeparator")
        if not isinstance(separator, str) or len(separator) != 1:
            raise ValueError(
                f"display decimal separator must be a single character, received {separator!r}"
            )
        self._display = display

    @property
    def catalog_version(self) -> str:
        return self._catalog["catalogVersion"]

    @property
    def locale(self) -> str:
        return self._locale

    # -- resolution ---------------------------------------------------------

    def _require_species(self, species_id: str) -> Mapping[str, Any]:
        species = self._catalog.get("species", {}).get(species_id)
        if species is None:
            raise ValueError(f"unknown species {species_id!r}")
        return species

    def _require_ion(self, ion_id: str) -> Mapping[str, Any]:
        ion = self._catalog.get("ions", {}).get(ion_id)
        if ion is None:
            raise ValueError(f"unknown ion {ion_id!r}")
        return ion

    def _require_definition(self, definition_id: str) -> Mapping[str, Any]:
        definition = self._catalog.get("species", {}).get(definition_id) or self._catalog.get(
            "ions", {}
        ).get(definition_id)
        if definition is None:
            raise ValueError(f"unknown species or ion {definition_id!r}")
        return definition

    def _require_reagent(self, reagent_id: str) -> Mapping[str, Any]:
        reagent = self._catalog.get("reagents", {}).get(reagent_id)
        if reagent is None:
            raise ValueError(f"unknown reagent {reagent_id!r}")
        return reagent

    def resolve_reaction(self, reference: str | Mapping[str, Any]) -> Mapping[str, Any]:
        """Resolves a reaction from a pair key, a reaction id, or the record."""
        if isinstance(reference, Mapping):
            return reference
        reactions = self._catalog.get("reactions", {})
        by_pair_key = reactions.get(reference)
        if by_pair_key is not None:
            return by_pair_key
        for reaction in reactions.values():
            if reaction["id"] == reference:
                return reaction
        raise ValueError(f"unknown reaction {reference!r}")

    def find_reaction(self, reagent_a: str, reagent_b: str) -> Mapping[str, Any] | None:
        """Resolves the modeled reaction for a reagent pair in either order."""
        self._require_reagent(reagent_a)
        self._require_reagent(reagent_b)
        return self._catalog.get("reactions", {}).get(normalize_pair_key(reagent_a, reagent_b))

    # -- formulas -----------------------------------------------------------

    def format_formula(self, definition_id: str) -> str:
        """Formats a species or ion formula, including an ion's charge."""
        definition = self._require_definition(definition_id)
        terms = format_formula_terms(definition["formulaTerms"])
        return f"{terms}{format_charge_superscript(definition.get('charge'))}"

    def format_species_formula(self, species_id: str) -> str:
        return format_formula_terms(self._require_species(species_id)["formulaTerms"])

    def format_ion_formula(self, ion_id: str) -> str:
        ion = self._require_ion(ion_id)
        terms = format_formula_terms(ion["formulaTerms"])
        return f"{terms}{format_charge_superscript(ion['charge'])}"

    def format_display_name(self, definition_id: str) -> str:
        return self._require_definition(definition_id)["displayName"]

    def describe_ion(self, ion_id: str) -> dict[str, Any]:
        """Display facts of one ion: formula, name, charge, kind, and color."""
        ion = self._require_ion(ion_id)
        return {
            "ionId": ion["id"],
            "formula": self.format_ion_formula(ion_id),
            "displayName": ion["displayName"],
            "charge": ion["charge"],
            "kind": ion["kind"],
            "color": ion["presentation"]["color"],
        }

    def format_reagent_formula(self, reagent_id: str) -> str:
        return self.format_species_formula(self._require_reagent(reagent_id)["speciesId"])

    def format_reagent_label(self, reagent_id: str) -> str:
        """Control-label form of a reagent: ``AgNO3 (Perak Nitrat)``."""
        reagent = self._require_reagent(reagent_id)
        name = self.format_display_name(reagent["speciesId"])
        return f"{self.format_reagent_formula(reagent_id)} ({name})"

    def list_reagents(self) -> tuple[dict[str, Any], ...]:
        """Every selectable reagent in catalog display order."""
        reagents = [
            reagent
            for reagent in self._catalog.get("reagents", {}).values()
            if reagent.get("selectable")
        ]
        reagents.sort(key=lambda reagent: reagent["displayOrder"])
        return tuple(
            {
                "reagentId": reagent["id"],
                "speciesId": reagent["speciesId"],
                "formula": self.format_reagent_formula(reagent["id"]),
                "displayName": self.format_display_name(reagent["speciesId"]),
                "label": self.format_reagent_label(reagent["id"]),
                "phase": reagent["phase"],
                "solutionColor": reagent["presentation"]["solutionColor"],
                "ions": tuple(
                    {
                        "ionId": term["ionId"],
                        "multiplicity": term["multiplicity"],
                        "formula": self.format_ion_formula(term["ionId"]),
                        "displayName": self.format_display_name(term["ionId"]),
                        "color": self._require_ion(term["ionId"])["presentation"]["color"],
                    }
                    for term in reagent["ions"]
                ),
            }
            for reagent in reagents
        )

    # -- equations ----------------------------------------------------------

    def format_term(self, term: Mapping[str, Any]) -> str:
        """Formats one equation term: ``2NaCl(aq)``, coefficients of one implicit."""
        coefficient = _require_positive_integer(term.get("coefficient"), "term coefficient")
        prefix = "" if coefficient == 1 else str(coefficient)
        return f"{prefix}{self.format_formula(term['speciesId'])}{format_phase(term['phase'])}"

    def format_equation_side(self, terms: Sequence[Mapping[str, Any]]) -> str:
        if not isinstance(terms, Sequence) or isinstance(terms, str) or len(terms) == 0:
            raise TypeError("an equation side must declare at least one term")
        return TERM_SEPARATOR.join(self.format_term(term) for term in terms)

    def _format_equation(
        self,
        reactants: Sequence[Mapping[str, Any]],
        products: Sequence[Mapping[str, Any]],
        energy_term: str | None = None,
        exothermic: bool = False,
    ) -> str:
        product_side = self.format_equation_side(products)
        if energy_term and exothermic:
            product_side = f"{product_side}{TERM_SEPARATOR}{energy_term}"
        return f"{self.format_equation_side(reactants)} {REACTION_ARROW} {product_side}"

    def format_molecular_equation(
        self, reference: str | Mapping[str, Any], energy_term: str | None = None
    ) -> str:
        """Balanced molecular equation of a reaction.

        ``energy_term`` appends a caller-supplied localized energy word to the
        product side of an exothermic reaction, keeping the qualitative energy
        label out of the balanced chemistry itself.
        """
        reaction = self.resolve_reaction(reference)
        return self._format_equation(
            reaction["molecularReactants"],
            reaction["molecularProducts"],
            energy_term,
            reaction.get("presentation", {}).get("energyLabel") == "exothermic",
        )

    def format_net_ionic_equation(
        self, reference: str | Mapping[str, Any], energy_term: str | None = None
    ) -> str:
        """Balanced net ionic equation of a reaction."""
        reaction = self.resolve_reaction(reference)
        return self._format_equation(
            reaction["netIonicReactants"],
            reaction["netIonicProducts"],
            energy_term,
            reaction.get("presentation", {}).get("energyLabel") == "exothermic",
        )

    def format_product_summary(self, reference: str | Mapping[str, Any]) -> str:
        """Product formulas without phase annotations, such as ``NaCl + H2O``."""
        reaction = self.resolve_reaction(reference)
        parts: list[str] = []
        for term in reaction["molecularProducts"]:
            coefficient = _require_positive_integer(term.get("coefficient"), "term coefficient")
            prefix = "" if coefficient == 1 else str(coefficient)
            parts.append(f"{prefix}{self.format_formula(term['speciesId'])}")
        return TERM_SEPARATOR.join(parts)

    def format_dissociation_equation(self, reagent_id: str) -> str:
        """Dissociation of an aqueous reagent, using catalog ion multiplicities."""
        reagent = self._require_reagent(reagent_id)
        reactants = [
            {"speciesId": reagent["speciesId"], "coefficient": 1, "phase": reagent["phase"]}
        ]
        products = [
            {
                "speciesId": term["ionId"],
                "coefficient": term["multiplicity"],
                "phase": reagent["phase"],
            }
            for term in reagent["ions"]
        ]
        return self._format_equation(reactants, products)

    def format_spectator_ions(self, reference: str | Mapping[str, Any]) -> tuple[str, ...]:
        """Spectator ion formulas of a reaction, in catalog order."""
        reaction = self.resolve_reaction(reference)
        return tuple(self.format_ion_formula(ion_id) for ion_id in reaction["spectatorIons"])

    def format_spectator_ion_list(self, reference: str | Mapping[str, Any]) -> str:
        return ", ".join(self.format_spectator_ions(reference))

    # -- localized model boundaries ----------------------------------------

    def resolve_message(self, message_key: str) -> str:
        """Resolves localized text for a declared message key."""
        text = self._messages.get(message_key)
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError(f"unknown {self._locale} message key {message_key!r}")
        return text

    def describe_phase_model(self, phase_model_name: str) -> dict[str, Any]:
        phase_model = self._catalog["model"]["phaseModels"].get(phase_model_name)
        if phase_model is None:
            raise ValueError(f"unknown phase model {phase_model_name!r}")
        return {
            "phaseModel": phase_model_name,
            "phases": tuple(phase_model["phases"]),
            "layered": phase_model["layered"],
            "messageKey": phase_model["messageKey"],
            "message": self.resolve_message(phase_model["messageKey"]),
        }

    def describe_no_modeled_reaction(self) -> dict[str, Any]:
        """Unsupported status and localized explanation for an unmatched pair.

        No product and no equation is reported, and the mixture stays in one
        homogeneous aqueous phase.
        """
        outcome = self._catalog["model"]["unsupportedOutcomes"].get("noModeledReaction")
        if outcome is None:
            raise ValueError("catalog declares no no_modeled_reaction outcome")
        return {
            "status": "unsupported",
            "category": outcome["category"],
            "reasonCode": outcome["reasonCode"],
            "messageKey": outcome["messageKey"],
            "message": self.resolve_message(outcome["messageKey"]),
            "phase": self.describe_phase_model(outcome["phaseModel"]),
            "molecularEquation": None,
            "netIonicEquation": None,
            "precipitate": None,
        }

    def describe_unsupported_ph(self, reason_code: str) -> dict[str, Any]:
        """Unsupported pH status and localized explanation for a declared reason."""
        reasons = self._catalog["model"]["capabilities"]["pH"]["unsupportedReasons"]
        reason = reasons.get(reason_code)
        if reason is None:
            raise ValueError(f"unknown pH unsupported reason code {reason_code!r}")
        return {
            "status": "unsupported",
            "capability": "pH",
            "reasonCode": reason_code,
            "messageKey": reason["messageKey"],
            "message": self.resolve_message(reason["messageKey"]),
            "pH": None,
        }

    def describe_outside_catalog_capability(self) -> dict[str, Any]:
        """Unsupported status and explanation outside declared catalog capability."""
        outcome = self._catalog["model"]["unsupportedOutcomes"].get("outsideCatalogCapability")
        if outcome is None:
            raise ValueError("catalog declares no outsideCatalogCapability outcome")
        return {
            "status": "unsupported",
            "reasonCode": outcome["reasonCode"],
            "messageKey": outcome["messageKey"],
            "message": self.resolve_message(outcome["messageKey"]),
        }

    def describe_capability_boundary(self, capability_name: str) -> dict[str, Any]:
        """Localized boundary statement of one declared model capability."""
        key_name = CAPABILITY_BOUNDARY_KEYS.get(capability_name)
        if key_name is None:
            raise ValueError(f"unknown capability boundary {capability_name!r}")
        capability = self._catalog["model"]["capabilities"].get(capability_name, {})
        message_key = capability.get(key_name)
        if not message_key:
            raise ValueError(f"capability {capability_name} declares no {key_name}")
        return {
            "capability": capability_name,
            "messageKey": message_key,
            "message": self.resolve_message(message_key),
        }

    def list_model_boundaries(self) -> dict[str, Any]:
        """Every localized model boundary the catalog declares, in a stable order."""
        ph_reasons = self._catalog["model"]["capabilities"]["pH"]["unsupportedReasons"]
        return {
            "capabilities": tuple(
                self.describe_capability_boundary(name) for name in sorted(CAPABILITY_BOUNDARY_KEYS)
            ),
            "phaseModels": tuple(
                self.describe_phase_model(name)
                for name in sorted(self._catalog["model"]["phaseModels"])
            ),
            "noModeledReaction": self.describe_no_modeled_reaction(),
            "outsideCatalogCapability": self.describe_outside_catalog_capability(),
            "pHUnsupportedReasons": tuple(
                self.describe_unsupported_ph(reason_code) for reason_code in sorted(ph_reasons)
            ),
        }

    # -- quantity display precision ----------------------------------------
    #
    # Display precision lives here and nowhere else. The evaluator keeps every
    # scientific value unrounded, and these projections are the single place a
    # rounded number is produced, so a rounded value has no path back into
    # extent, excess, pH, the species ledger, or any later calculation.

    def describe_display_quantity(self, quantity: str) -> dict[str, Any]:
        """Declared precision, unit, rounding mode, and separator of a quantity."""
        declared = self._display.get("quantities", {}).get(quantity)
        if declared is None:
            raise ValueError(f"catalog declares no display precision for {quantity!r}")
        return {
            "quantity": quantity,
            "precision": _require_display_precision(
                declared.get("precision"), f"display precision for {quantity}"
            ),
            "unit": declared.get("unit"),
            "roundingMode": self._display["roundingMode"],
            "decimalSeparator": self._display["decimalSeparator"],
        }

    def list_display_quantities(self) -> tuple[str, ...]:
        """Every quantity the catalog declares a display precision for."""
        return tuple(sorted(self._display.get("quantities", {})))

    def format_quantity_value(self, quantity: str, value: float) -> str:
        """Numeric display string of one quantity, without its unit."""
        declared = self.describe_display_quantity(quantity)
        return format_fixed(
            _require_finite_number(value, quantity),
            declared["precision"],
            declared["decimalSeparator"],
        )

    def format_quantity(self, quantity: str, value: float, with_unit: bool = True) -> str:
        """Display string of one quantity, with its unit when it declares one."""
        declared = self.describe_display_quantity(quantity)
        text = self.format_quantity_value(quantity, value)
        if with_unit and declared["unit"]:
            return f"{text} {declared['unit']}"
        return text

    def round_quantity_for_display(self, quantity: str, value: float) -> float:
        """Rounded display value of one quantity as a number.

        Exposed for display consumers that need a number rather than a string,
        such as a chart tick. It must never be written back into an experiment
        input or a later calculation.
        """
        return round_half_away_from_zero(
            _require_finite_number(value, quantity),
            self.describe_display_quantity(quantity)["precision"],
        )

    def format_ph(self, value: float | None) -> str | None:
        """Display string of a pH value, or ``None`` when the model reports none.

        An absent pH stays absent. No neutral ``~7.0`` is fabricated for a
        mixture the ideal strong-acid/strong-base model does not cover; the
        localized unsupported reason from ``describe_unsupported_ph`` is what
        explains it.
        """
        if value is None:
            return None
        return self.format_quantity_value("pH", value)

    def _format_optional_quantity(self, quantity: str, value: float | None) -> str | None:
        return None if value is None else self.format_quantity(quantity, value)

    def format_experiment_quantities(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Display projection of an evaluator result's quantities.

        The result itself is not touched: this returns strings beside it, so the
        unrounded domain values remain the only input to any later calculation.
        ``pH`` and ``speciesLedger`` project the honest ``None`` the evaluator
        reports until the pH model and the ion ledger exist.
        """
        if not isinstance(result, Mapping):
            raise TypeError("experiment result must be a mapping")
        ph = result.get("pH")
        return {
            "catalogVersion": result.get("catalogVersion", self.catalog_version),
            "extent_mol": self.format_quantity("extent_mol", result["extent_mol"]),
            "totalVolume_L": self.format_quantity("volume_L", result["totalVolume_L"]),
            "reactants": tuple(
                {
                    "reagentId": reactant["reagentId"],
                    "speciesId": reactant["speciesId"],
                    "concentration_molL": self.format_quantity(
                        "concentration_molL", reactant["concentration_molL"]
                    ),
                    "volume_mL": self.format_quantity("volume_mL", reactant["volume_mL"]),
                    "volume_L": self.format_quantity("volume_L", reactant["volume_L"]),
                    "amount_mol": self.format_quantity("amount_mol", reactant["amount_mol"]),
                    "consumed_mol": self._format_optional_quantity(
                        "amount_mol", reactant.get("consumed_mol")
                    ),
                    "remaining_mol": self._format_optional_quantity(
                        "amount_mol", reactant.get("remaining_mol")
                    ),
                }
                for reactant in result.get("reactants", ())
            ),
            "excessSpecies": tuple(
                {
                    "speciesId": excess["speciesId"],
                    "amount_mol": self.format_quantity("amount_mol", excess["amount_mol"]),
                }
                for excess in result.get("excessSpecies", ())
            ),
            "products": tuple(
                {
                    "speciesId": product["speciesId"],
                    "amount_mol": self.format_quantity("amount_mol", product["amount_mol"]),
                }
                for product in result.get("products", ())
            ),
            "pH": self.format_ph(ph.get("value") if isinstance(ph, Mapping) else None),
            "speciesLedger": result.get("speciesLedger"),
        }

    # -- reaction projection ------------------------------------------------

    def _describe_precipitate(self, reaction: Mapping[str, Any]) -> dict[str, Any] | None:
        species_id = reaction.get("presentation", {}).get("precipitateSpeciesId")
        if species_id is None:
            return None
        return {
            "speciesId": species_id,
            "formula": self.format_species_formula(species_id),
            "displayName": self.format_display_name(species_id),
            "color": reaction["presentation"]["precipitateColor"],
        }

    def _describe_reaction_ph(self, reaction: Mapping[str, Any]) -> dict[str, Any]:
        ph_model = reaction.get("pH")
        if ph_model:
            return {
                "supported": True,
                "modelId": ph_model["modelId"],
                "modelLabel": self._catalog["model"]["capabilities"]["pH"].get("modelLabel"),
                "reasonCode": None,
                "message": None,
            }
        unsupported = self.describe_unsupported_ph(reaction["pHUnsupportedReasonCode"])
        return {
            "supported": False,
            "modelId": None,
            "modelLabel": None,
            "reasonCode": unsupported["reasonCode"],
            "message": unsupported["message"],
        }

    def format_reaction_display(
        self, reagent_a: str, reagent_b: str, energy_term: str | None = None
    ) -> dict[str, Any]:
        """Single display projection of a reagent pair.

        Equation, product formula, phase model, appearance, and energy label all
        come from the one resolved reaction record, so a precipitate can never
        disagree with the equation that produced it. Unmatched supported pairs
        return the machine-readable unsupported status with its localized
        model-boundary explanation and no equation.
        """
        reaction = self.find_reaction(reagent_a, reagent_b)
        reagents = (
            self.format_reagent_formula(reagent_a),
            self.format_reagent_formula(reagent_b),
        )
        if reaction is None:
            presentation = self._catalog["model"]["unsupportedOutcomes"]["noModeledReaction"][
                "presentation"
            ]
            return {
                **self.describe_no_modeled_reaction(),
                "pairKey": normalize_pair_key(reagent_a, reagent_b),
                "reactionId": None,
                "displayName": None,
                "reagents": reagents,
                "spectatorIons": (),
                "productSummary": None,
                "appearance": presentation["appearance"],
                "energyLabel": presentation["energyLabel"],
                "solutionColor": presentation["solutionColor"],
                "pH": self.describe_unsupported_ph("PH_MODEL_NO_MODELED_REACTION"),
            }

        return {
            "status": "modeled",
            "category": reaction["category"],
            "pairKey": normalize_pair_key(reagent_a, reagent_b),
            "reactionId": reaction["id"],
            "displayName": reaction["displayName"],
            "reagents": reagents,
            "molecularEquation": self.format_molecular_equation(reaction, energy_term),
            "netIonicEquation": self.format_net_ionic_equation(reaction, energy_term),
            "productSummary": self.format_product_summary(reaction),
            "spectatorIons": self.format_spectator_ions(reaction),
            "phase": self.describe_phase_model(reaction["phaseModel"]),
            "precipitate": self._describe_precipitate(reaction),
            "appearance": reaction["presentation"]["appearance"],
            "energyLabel": reaction["presentation"]["energyLabel"],
            "solutionColor": reaction["presentation"]["solutionColor"],
            "pH": self._describe_reaction_ph(reaction),
        }


FORMATTER = ChemistryFormatter()
