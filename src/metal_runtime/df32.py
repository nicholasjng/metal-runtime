import importlib.resources

PRELUDE = (
    importlib.resources.files("metal_runtime")
    .joinpath("df32.metal")
    .read_text(encoding="utf-8")
)
