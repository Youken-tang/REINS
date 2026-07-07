"""Compatibility shim — code lives in :mod:`reins.runtime.resource_access`."""

from reins.runtime.resource_access import (  # noqa: F401
    ResourceAccess,
    SideEffectLevel,
    access_conflicts,
    normalize_component,
    resources_overlap,
)
