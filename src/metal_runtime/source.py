"""MSL sources assembled from labeled fragments. `#line` markers make
Metal's diagnostics name the fragment, e.g. `body:12:5: error: ...`,
instead of a line in the concatenated blob.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

__all__ = [
    "DEFAULT_INCLUDES",
    "Fragment",
    "assemble",
    "build_source",
]

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


def build_source(
    *,
    includes: Sequence[str] = DEFAULT_INCLUDES,
    preludes: Sequence[Fragment] = (),
    body: str,
    body_label: str = "body",
) -> str:
    """`#include` lines for `includes`, then `preludes`, then `body`.

    `includes` replaces the default rather than extending it.
    """
    fragments = []
    if includes:
        include_text = "\n".join(f"#include <{name}>" for name in includes)
        fragments.append(Fragment("includes", include_text))
    fragments.extend(preludes)
    fragments.append(Fragment(body_label, body))
    return assemble(*fragments)
