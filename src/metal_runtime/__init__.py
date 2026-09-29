import contextlib
from collections.abc import Iterator

from metal_runtime._core import (
    Batch,
    Buffer,
    CaptureError,
    CompileError,
    DeviceError,
    DispatchError,
    FunctionNotFoundError,
    Kernel,
    MathMode,
    MetalError,
    PipelineBuildError,
    clear_library_cache,
    device_info,
    device_name,
    is_capturing,
    library_cache_limit,
    library_cache_size,
    run,
    set_library_cache_limit,
    start_capture,
    stop_capture,
    supported_dtypes,
)
from metal_runtime.source import (
    DEFAULT_INCLUDES,
    Fragment,
    assemble,
    build_source,
)

__all__ = [
    "DEFAULT_INCLUDES",
    "Batch",
    "Buffer",
    "Capture",
    "CaptureError",
    "CompileError",
    "DeviceError",
    "DispatchError",
    "Fragment",
    "FunctionNotFoundError",
    "Kernel",
    "MathMode",
    "MetalError",
    "PipelineBuildError",
    "__version__",
    "assemble",
    "build_source",
    "clear_library_cache",
    "device_info",
    "device_name",
    "is_capturing",
    "library_cache_limit",
    "library_cache_size",
    "run",
    "set_library_cache_limit",
    "start_capture",
    "stop_capture",
    "supported_dtypes",
]

__version__ = "0.1.0"


@contextlib.contextmanager
def Capture(path: str) -> Iterator[None]:
    """Capture dispatches on the runtime's Metal device to a GPU trace
    document at `path`. The capture stops even if the body raises."""
    start_capture(path)
    try:
        yield
    finally:
        stop_capture()
