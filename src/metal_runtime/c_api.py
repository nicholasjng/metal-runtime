"""Locates the C-linkable FFI bridge (libmetal_runtime_c + c_api.h) shipped
in this package, for a build system to compile and link against directly.
Same role as `jax.ffi.include_dir()`.

`metal_runtime`'s `__path__` can span multiple roots (e.g. an editable dev
install: source tree + CMake build dir), so `importlib.resources.files(...)`
alone may return a directory that doesn't actually hold the file. Joining
the filename and checking `.is_file()` resolves to the root that does.
"""

import importlib.resources
import pathlib


def _resolved_parent(package: str, filename: str) -> str:
    candidate = importlib.resources.files(package).joinpath(filename)
    if not candidate.is_file():
        raise FileNotFoundError(f"{filename} not found under package {package!r}")
    return str(pathlib.Path(str(candidate)).parent)


def include_dir() -> str:
    """Directory containing c_api.h."""
    return _resolved_parent("metal_runtime.include", "c_api.h")


def library_dir() -> str:
    """Directory containing libmetal_runtime_c.dylib."""
    return _resolved_parent("metal_runtime", "libmetal_runtime_c.dylib")
