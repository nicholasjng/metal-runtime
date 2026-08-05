"""Composing MSL kernel sources from labeled fragments instead of one
flat Python string, so Metal's own compiler diagnostics name the piece
that's actually wrong.

Reusable preludes are passed as `Fragment` values rather than looked up by
name, so what a source is built from is always visible at the call site.

The `#line` markers between fragments are useful: a syntax error in a fragment
labeled "body" is reported by Metal's own compiler as `body:12:5: error: ...`,
not as an offset into the full concatenated blob. This is what makes
splitting a kernel into includes/prelude/body fragments strictly better
than one f-string for debugging, not just shorter to read:

    >>> import metal_runtime as mr
    >>> src = mr.build_source(
    ...     body="kernel void f(device float* o [[buffer(0)]]) { o[0] = bogus; }",
    ... )
    >>> mr.Kernel(src, "f")
    Traceback (most recent call last):
        ...
    metal_runtime.CompileError: MSL compile error: body:1:53: error: use of
    undeclared identifier 'bogus'
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

# Headers `build_source` includes unless the caller overrides `includes`.
DEFAULT_INCLUDES: tuple[str, ...] = ("metal_stdlib",)


@dataclasses.dataclass(frozen=True)
class Fragment:
    """One labeled piece of MSL source. `label` becomes the file name
    Metal's compiler reports in diagnostics for lines inside `text`."""

    label: str
    text: str


def _line_marker(label: str) -> str:
    escaped = label.replace("\\", "\\\\").replace('"', '\\"')
    return f'#line 1 "{escaped}"'


def assemble(*fragments: Fragment) -> str:
    """Concatenate `fragments` into one MSL source, each preceded by a
    `#line 1 "label"` marker so Metal's compiler attributes diagnostics
    to the fragment they came from rather than to a flattened line count.
    """
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
    """Convenience over `assemble`: `#include <...>` lines for `includes`,
    then each prelude `Fragment` in order, then `body` labeled
    `body_label` (default `"body"`, matched by the module docstring's
    example -- override it if a caller wants a more specific label,
    e.g. the emitting kernel's name).

    `includes` defaults to `DEFAULT_INCLUDES` (just `metal_stdlib`), so
    the common case needs no include bookkeeping. Passing `includes`
    replaces the default rather than extending it: pass `()` for a body
    that does its own includes, or spell out the full list when adding
    headers, e.g. `includes=(*DEFAULT_INCLUDES, "metal_tensor")`.
    """
    fragments = []
    if includes:
        include_text = "\n".join(f"#include <{name}>" for name in includes)
        fragments.append(Fragment("includes", include_text))
    fragments.extend(preludes)
    fragments.append(Fragment(body_label, body))
    return assemble(*fragments)
