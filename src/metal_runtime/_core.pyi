import enum
import types
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Self

import jax
import mlx
import numpy
from numpy.typing import NDArray

class DeviceError(Exception): ...
class CompileError(Exception): ...
class FunctionNotFoundError(CompileError): ...
class DispatchError(Exception): ...

def device_name() -> str:
    """
    Name of the default Metal device.

    Returns
    -------
    str
    """

def device_info() -> dict:
    """
    Device capabilities and limits.

    Returns
    -------
    dict
        Keys: name, unified_memory, recommended_max_working_set_size,
        max_threads_per_threadgroup, max_threadgroup_memory_length,
        max_buffer_length, supports_non_uniform_threadgroups.
    """

def supported_dtypes() -> str:
    """
    Comma-separated list of dtype names Buffer accepts.

    Returns
    -------
    str
    """

def library_cache_size() -> int:
    """
    Number of compiled MSL libraries currently cached.

    Returns
    -------
    int
    """

def library_cache_limit() -> int:
    """
    Current cap on cached libraries.

    Returns
    -------
    int
        0 means unlimited.
    """

def set_library_cache_limit(limit: int) -> None:
    """
    Cap the library cache size, evicting least-recently-used entries.

    Parameters
    ----------
    limit : int
        0 disables eviction.
    """

def clear_library_cache() -> None:
    """Drop every cached library."""

def set_pipeline_cache_dir(path: str | None) -> None:
    """
    Enable or disable the persistent pipeline cache.

    Parameters
    ----------
    path : str or None
        Loads an existing archive at this path if one exists, else starts
        an empty one. None turns caching back off. Call once at startup,
        not concurrently with kernel construction.
    """

def pipeline_cache_dir() -> str | None:
    """
    Path of the active pipeline cache, if one is configured.

    Returns
    -------
    str or None
    """

def save_pipeline_cache() -> None:
    """
    Write the pipeline cache to its configured path.

    Raises
    ------
    RuntimeError
        No pipeline cache directory is set.
    """

class Buffer:
    def __init__(
        self,
        array: Annotated[NDArray, dict(order="C", device="cpu", writable=False)],
        dtype: str | None = None,
    ) -> None:
        """
        Upload a NumPy array into a new device buffer.

        Parameters
        ----------
        array : numpy.ndarray
            C-contiguous, any dtype Metal can address; float64 is rejected.
        dtype : str, optional
            Relabels the array's bytes rather than converting them (`.view`
            semantics). Element width must match.
        """

    @staticmethod
    def zeros(shape: Sequence[int], dtype: str = "float32") -> Buffer:
        """
        Allocate a zero-filled buffer without an upload.

        Parameters
        ----------
        shape : Sequence[int]
        dtype : str, optional
            Defaults to "float32".

        Returns
        -------
        Buffer
        """

    @staticmethod
    def empty(shape: Sequence[int], dtype: str = "float32") -> Buffer:
        """
        Allocate an uninitialized buffer: no upload, no zero-fill.

        Parameters
        ----------
        shape : Sequence[int]
        dtype : str, optional
            Defaults to "float32".

        Returns
        -------
        Buffer
        """

    def copy_from(
        self,
        array: Annotated[NDArray, dict(order="C", device="cpu", writable=False)],
        dtype: str | None = None,
    ) -> None:
        """
        Refill this buffer's allocation in place.

        Parameters
        ----------
        array : numpy.ndarray
            Byte count must match this buffer's; shape may differ.
        dtype : str, optional
            Relabels rather than converts, like the constructor.

        Raises
        ------
        ValueError
            Byte count or resolved dtype doesn't match this buffer.
        """

    def to_numpy(self, dtype: str | None = None) -> NDArray:
        """
        A live NumPy view of this buffer's memory. Not a copy.

        Parameters
        ----------
        dtype : str, optional
            Relabels the bytes on the way out; element width must match.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        TypeError
            dtype is bfloat16, which NumPy has no native dtype for.
        """

    def to_jax(self) -> jax.Array:
        """
        This buffer as a JAX array, via DLPack. Zero-copy.

        Returns
        -------
        jax.Array
        """

    def to_mlx(self) -> mlx.core.array:
        """
        This buffer as an MLX array, via DLPack. Zero-copy.

        Returns
        -------
        mlx.core.array
        """

    def __dlpack__(self, **kwargs) -> Any:
        """DLPack capsule for this buffer's memory. Zero-copy."""

    def __dlpack_device__(self) -> tuple:
        """
        DLPack device tuple: (kDLCPU, 0), since Metal's shared storage is host-addressable.
        """

    @property
    def shape(self) -> tuple:
        """Array shape."""

    @property
    def dtype(self) -> str:
        """dtype name, e.g. 'float32'."""

    @property
    def size(self) -> int:
        """Element count."""

    @property
    def nbytes(self) -> int:
        """Byte count."""

    def __len__(self) -> int:
        """Length of the first dimension."""

class MathMode(enum.StrEnum):
    """
    How much freedom the Metal compiler has to rewrite floating-point arithmetic.
    """

    SAFE = "safe"
    """
    IEEE semantics. Required for compensated summation and double-single arithmetic, whose error terms FAST is free to fold away.
    """

    RELAXED = "relaxed"
    """
    Permits reassociation like FAST (so it also deletes compensated arithmetic), but keeps infinities and NaNs well-defined instead of assuming they never occur.
    """

    FAST = "fast"
    """Metal's default: permits reassociation and flushes denormals."""

class Kernel:
    def __init__(
        self,
        msl_source: str,
        function_name: str,
        math_mode: MathMode = MathMode.FAST,
        defines: Mapping[str, str] = {},
        constants: dict = {},
    ) -> None:
        """
        Compile MSL source into a dispatchable kernel.

        Parameters
        ----------
        msl_source : str
        function_name : str
            Entry point within `msl_source`.
        math_mode : MathMode, optional
            Defaults to FAST.
        defines : Mapping[str, str], optional
            Preprocessor macros, part of the library cache key.
        constants : dict, optional
            Function constant values, part of the pipeline cache key.

        Raises
        ------
        CompileError
            msl_source doesn't compile.
        FunctionNotFoundError
            function_name isn't in msl_source.
        """

    @property
    def function_name(self) -> str:
        """Entry point name."""

    @property
    def math_mode(self) -> MathMode:
        """Compiled math mode."""

    @property
    def defines(self) -> dict[str, str]:
        """Preprocessor macros this kernel compiled with."""

    @property
    def constants(self) -> dict:
        """Function constant values this kernel was specialized with."""

    @property
    def max_threads_per_threadgroup(self) -> int:
        """This kernel's own ceiling, which can be below the device maximum."""

    @property
    def thread_execution_width(self) -> int:
        """SIMD width for this kernel."""

    @property
    def static_threadgroup_memory_length(self) -> int:
        """
        Bytes of threadgroup memory the kernel itself declares, before any dynamic threadgroup_memory passed to run().
        """

def run(
    kernel: Kernel,
    grid: int | Sequence[int] | Buffer,
    threadgroup: int | Sequence[int] | None = None,
    buffers: Sequence[Buffer | tuple[Buffer, int]] = [],
    scalars: Sequence[numpy.ndarray | numpy.generic] = [],
    threadgroup_memory: Sequence[int] = [],
    indirect_offset: int = 0,
) -> None:
    """
    Dispatch one kernel launch and block until it completes.

    Parameters
    ----------
    kernel : Kernel
    grid : int or Sequence[int] or Buffer
        Thread count per dimension, or a Buffer of three uint32
        threadgroup counts for an indirect dispatch.
    threadgroup : int or Sequence[int], optional
        Explicit threadgroup size. None lets the runtime choose.
    buffers : Sequence[Buffer or tuple[Buffer, int]], optional
        Bind in order; a (Buffer, offset) tuple binds at a byte offset
        into that buffer's allocation.
    scalars : Sequence[numpy.ndarray or numpy.generic], optional
        Bound inline with setBytes, after buffers.
    threadgroup_memory : Sequence[int], optional
        Byte size for each [[threadgroup(i)]] allocation.
    indirect_offset : int, optional
        Byte offset into `grid` when it is a Buffer.

    Raises
    ------
    ValueError
        The launch doesn't satisfy the kernel's own binding reflection.
    DispatchError
        The GPU aborted the command buffer.
    """

class Batch:
    def __init__(self, concurrent: bool = False) -> None:
        """
        Encode several launches into one command buffer.

        Parameters
        ----------
        concurrent : bool, optional
            Lets independent launches overlap on the GPU; ordering then only
            exists across barrier(). Defaults to False (serial).
        """

    def add(
        self,
        kernel: Kernel,
        grid: int | Sequence[int] | Buffer,
        threadgroup: int | Sequence[int] | None = None,
        buffers: Sequence[Buffer | tuple[Buffer, int]] = [],
        scalars: Sequence[numpy.ndarray | numpy.generic] = [],
        threadgroup_memory: Sequence[int] = [],
        indirect_offset: int = 0,
    ) -> None:
        """
        Encode one launch into this batch. Same parameters as run().

        Raises
        ------
        DispatchError
            This batch has already been committed.
        """

    def barrier(self) -> None:
        """
        Order buffer writes across a concurrent batch. Meaningless on a serial (default) batch.
        """

    def commit(self) -> None:
        """Close encoding and submit without blocking."""

    def wait(self) -> None:
        """
        Commit if needed, then block until the GPU is done.

        Raises
        ------
        DispatchError
            The command buffer faulted.
        """

    @property
    def gpu_time(self) -> float | None:
        """Device-side execution seconds for the whole batch, set by wait()."""

    def __enter__(self) -> Self:
        """Returns self."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """
        Waits on the batch if the body didn't raise; otherwise discards it without committing.
        """
