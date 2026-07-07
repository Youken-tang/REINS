"""Slash command registry for interactive sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


CommandHandler = Callable[[list[str]], int | None]


@dataclass(frozen=True)
class CommandEntry:
    name: str
    handler: CommandHandler
    help: str
    aliases: tuple[str, ...] = ()


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandEntry] = {}
        self._aliases: dict[str, str] = {}

    def register(self, name: str, handler: CommandHandler, help: str, aliases: tuple[str, ...] = ()) -> None:
        canonical = _normalize(name)
        entry = CommandEntry(canonical, handler, help, tuple(_normalize(alias) for alias in aliases))
        self._commands[canonical] = entry
        for alias in entry.aliases:
            self._aliases[alias] = canonical

    def dispatch(self, name: str, args: list[str]) -> int | None:
        entry = self.get(name)
        if entry is None:
            raise KeyError(_normalize(name))
        return entry.handler(args)

    def get(self, name: str) -> CommandEntry | None:
        normalized = _normalize(name)
        canonical = self._aliases.get(normalized, normalized)
        return self._commands.get(canonical)

    def entries(self) -> list[CommandEntry]:
        return [self._commands[name] for name in sorted(self._commands)]

    def names(self) -> list[str]:
        names = set(self._commands)
        names.update(self._aliases)
        return sorted(names)


def _normalize(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"
