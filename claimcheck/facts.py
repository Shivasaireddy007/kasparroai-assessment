"""Read-only get_brand_facts tool."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .data import FACT_VERSION, Fact, load_facts


class UnknownBrandError(LookupError):
    pass


@lru_cache(maxsize=1)
def _facts() -> tuple[Fact, ...]:
    return tuple(load_facts())


def known_brands() -> list[str]:
    return sorted({f.brand_id for f in _facts()})


def all_fact_ids() -> frozenset[str]:
    return frozenset(f.fact_id for f in _facts())


def get_brand_facts(brand_id: str) -> dict[str, Any]:
    if not isinstance(brand_id, str) or brand_id not in known_brands():
        raise UnknownBrandError(f"unknown brand_id: {brand_id!r}")
    rows = [
        {
            "fact_id": f.fact_id,
            "model_id": f.model_id,
            "field": f.field,
            "value": f.value,
            "value_type": f.value_type,
            "unit": f.unit,
            "condition": f.condition,
            "valid_from": f.valid_from.isoformat(),
            "valid_to": f.valid_to.isoformat(),
        }
        for f in _facts()
        if f.brand_id == brand_id
    ]
    return {"brand_id": brand_id, "fact_version": FACT_VERSION, "facts": rows}
