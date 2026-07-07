"""Small time formatting helpers."""

from __future__ import annotations


def format_duration_compact(seconds: float) -> str:
    """Format elapsed seconds for status bars and runtime digests."""
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        value = 0.0
    if value < 1:
        return f"{int(round(value * 1000))}ms"
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{int(round(value))}s"
    minutes = value / 60
    if minutes < 60:
        whole_minutes = int(minutes)
        seconds_part = int(value % 60)
        return f"{whole_minutes}m {seconds_part}s" if seconds_part else f"{whole_minutes}m"
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    if hours < 24:
        return f"{hours}h {remaining_minutes}m" if remaining_minutes else f"{hours}h"
    days = value / 86400
    return f"{days:.1f}d"
