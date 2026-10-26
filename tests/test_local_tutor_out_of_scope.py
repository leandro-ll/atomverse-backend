"""Property 36: Out-of-scope questions get the limitation statement only.

Validates: Requirements 15.4
- When a question falls outside catalog-derivable facts, the system returns
  the explicit Indonesian limitation statement without fabricating an equation or product name.
"""

from __future__ import annotations

import random
import pytest

from app.services.local_tutor import (
    LIMITATION_ID_TEXT,
    SUPPORTED_SUGGESTIONS,
    build_chat_fallback,
)
from app.generated.chemistry_catalog import CATALOG, CATALOG_VERSION


# Feature: chemistry-species-expansion, Property 36: Out-of-scope questions get the limitation statement only
def test_local_tutor_out_of_scope_property() -> None:
    rng = random.Random(3636)
    iterations = 100

    out_of_scope_prompts = [
        "Bagaimana reaksi redoks yang terjadi pada logam seng dan tembaga?",
        "Apakah ada pelepasan gas CO2 atau oksigen dari campuran ini?",
        "Jelaskan mekanisme reaksi hidrokarbon dan sintesis organik senyawa ini",
        "Bagaimana cara membuat baterai sel galvani dengan katoda dan anoda dari bahan ini?",
        "Berapa harga bahan kimia ini di toko atau pasar kimia?",
        "Siapa penemu reaksi ini dalam sejarah kimia?",
        "Bagaimana pengaruh cuaca dan iklim global terhadap reaksi ini?",
        "Reaksi redoks apa yang terjadi?",
        "Apakah ada gas yang keluar dari tabung reaksi?",
    ]

    for i in range(iterations):
        prompt = rng.choice(out_of_scope_prompts)
        lab_state = {
            "reagentA": "sub.agno3",
            "reagentB": "sub.nacl",
            "mA": 1.0,
            "vA": 50.0,
            "mB": 1.0,
            "vB": 50.0,
            "temp": 25.0,
        }

        reply = build_chat_fallback(prompt, lab_state, catalog=CATALOG)

        # 1. Scope must be unsupported
        assert reply.scope == "unsupported", f"Iter {i}: expected scope unsupported for prompt '{prompt}'"
        assert reply.degraded is True
        assert reply.catalog_version == CATALOG_VERSION
        assert reply.reason_code == "provider_unavailable"

        # 2. Text must match explicit limitation statement
        assert reply.text == LIMITATION_ID_TEXT, f"Iter {i}: expected exact limitation statement"

        # 3. No fabricated equation or product
        assert "->" not in reply.text
        assert "→" not in reply.text
        assert "AgCl" not in reply.text
        assert "endapan putih" not in reply.text

        # 4. Suggestions provided
        assert reply.suggestions == SUPPORTED_SUGGESTIONS
