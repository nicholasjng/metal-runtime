from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

class CompileError(Exception):
    pass

def device_name() -> str: ...

class Buffer:
    def __init__(
        self, array: Annotated[NDArray[numpy.float32], dict(order="C")]
    ) -> None: ...
    @staticmethod
    def zeros(shape: Sequence[int]) -> Buffer: ...
    def to_numpy(self) -> NDArray[numpy.float32]: ...

class Kernel:
    def __init__(self, msl_source: str, function_name: str) -> None: ...

def run(
    kernel: Kernel, grid_size: int, threadgroup_size: int, buffers: Sequence[Buffer]
) -> None: ...
