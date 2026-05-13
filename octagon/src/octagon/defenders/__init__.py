"""Defender registry."""

from octagon.defenders.internal_it.defender import InternalITDefender

REGISTRY: dict[str, type] = {
    "internal_it": InternalITDefender,
}


def get_defender(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown defender: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["InternalITDefender", "REGISTRY", "get_defender"]
