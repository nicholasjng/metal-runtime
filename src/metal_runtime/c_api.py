"""Locations of the shipped C API (c_api.h, libmetal_runtime.dylib)."""

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
    """Directory containing libmetal_runtime.dylib."""
    return _resolved_parent("metal_runtime", "libmetal_runtime.dylib")
