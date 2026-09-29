"""MSL sources assembled from labeled fragments. `#line` markers make
Metal's diagnostics name the fragment, e.g. `body:12:5: error: ...`,
instead of a line in the concatenated blob.
"""

from __future__ import annotations

import dataclasses

__all__ = [
    "DEFAULT_INCLUDES",
    "Fragment",
    "assemble",
]

# What nearly every kernel includes.
DEFAULT_INCLUDES: tuple[str, ...] = ("metal_stdlib",)


@dataclasses.dataclass(frozen=True)
class Fragment:
    """One labeled piece of MSL source; `label` is the file name in diagnostics."""

    label: str
    text: str


def _line_marker(label: str) -> str:
    escaped = label.replace("\\", "\\\\").replace('"', '\\"')
    return f'#line 1 "{escaped}"'


def assemble(*fragments: Fragment) -> str:
    """Concatenate `fragments`, each preceded by a `#line 1 "label"` marker."""
    parts: list[str] = []
    for frag in fragments:
        parts.append(_line_marker(frag.label))
        parts.append(frag.text)
    return "\n".join(parts) + "\n"
