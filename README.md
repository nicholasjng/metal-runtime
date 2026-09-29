# metal-runtime

A small Metal GPU runtime for Python, built on Apple's
[metal-cpp](https://developer.apple.com/metal/cpp/).
Compiles MSL kernel source at runtime (no offline `metal`/`metallib` toolchain),
moves NumPy arrays into shared GPU buffers with zero-copy readback, and dispatches.

```python
import numpy as np

import metal_runtime as mr

source = """
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0f;
}
"""

array = np.arange(16, dtype=np.float32)
buffer = mr.Buffer(array)
kernel = mr.Kernel(source, "add_one")

mr.run(kernel, grid=16, buffers=[buffer])

print(buffer.to_numpy())  # [1. 2. 3. ... 16.]
```

`threadgroup` is optional; if not provided, it is set from the kernel's own occupancy limits.

## What it does

**Buffers.** `mr.Buffer(array)` uploads a C-contiguous host array (`bool`,
`int8`-`int64`, `uint8`-`uint64`, `float16`, `float32`, `bfloat16`; `float64`
is rejected). `.to_numpy()` reads back a live view of the same unified
memory, and buffers also export through DLPack. A `dtype=` argument on
either side relabels bytes instead of converting them, which is how types
NumPy can't hand across on its own (`ml_dtypes.bfloat16`) get through.

**Scalars.** Non-array kernel arguments go in `scalars`, bound after
`buffers`, and are checked against the kernel's declared width:

```python
mr.run(kernel, grid=n, buffers=[y, x], scalars=[np.float32(alpha), np.uint32(n)])
```

**Grids.** `grid` and `threadgroup` take an `int` or a 1- to 3-tuple.
`threadgroup_memory` takes byte sizes for `[[threadgroup(i)]]`. A buffer can
bind at an offset via a `(buffer, offset)` tuple to suballocate one arena.

**Indirect dispatch.** Passing a `Buffer` as `grid` reads threadgroup counts
from that buffer at dispatch time, so a kernel earlier in the same batch can
size the next launch without a host round trip.

**Launch validation.** Missing arguments and threadgroup-memory overruns are
caught before dispatch and raise a clear error, instead of a GPU fault or
process abort.

**Batching.** `Batch` encodes several launches into one command buffer,
amortizing per-launch overhead to a few µs:

```python
with mr.Batch() as batch:
    batch.add(k1, grid=n, buffers=[a, b])
    batch.add(k2, grid=n, buffers=[b, c])
# committed and waited on at exit; discarded if the body raises
```

`batch.commit()` is non-blocking, so a stepping loop can encode batch *n+1*
while *n* executes. `mr.Batch(concurrent=True)` lets independent launches
overlap on the GPU.

**GPU capture.** Capture dispatches on the runtime's Metal device to an Xcode
GPU trace document. The context manager stops capture even if the profiled
block raises:

```python
with mr.Capture("attention.gputrace"):
    run_flash_attention()
```

For manual control, use `mr.start_capture(path)`, `mr.stop_capture()`, and
`mr.is_capturing()`. `CaptureError` reports unsupported or failed capture
requests.

**Compile options.** `Kernel` takes `math_mode` and preprocessor `defines`.
`math_mode` defaults to `FAST` (Metal's default), which permits
reassociation and can silently optimize away compensated arithmetic like a
Kahan update. Use `SAFE` for anything relying on error-free transformations.

**Function constants.** `[[function_constant(i)]]` values bake in at
pipeline creation and reuse the compiled library, so specializing per block
size skips the MSL front end:

```python
kernel = mr.Kernel(source, "step", constants={"DECAY": 0.999, "N": n})
```

Required constants must be set and unknown names raise, validated against
the kernel's own reflection.

**Errors.** `CompileError` (with `FunctionNotFoundError` and
`PipelineBuildError` as subclasses) for bad MSL, a missing entry point, or a
failed pipeline build; `DispatchError` for a GPU-aborted command buffer;
`DeviceError` when there is no Metal device; a plain `MemoryError` when
Metal refuses an allocation.

**Introspection.** `mr.device_info()` reports device limits; per-kernel
limits are available on the `Kernel` object. Compiled libraries are cached
by source text, bounded by `mr.set_library_cache_limit()`. There is no
on-disk pipeline cache: macOS caches compiled pipelines across processes on
its own, and a binary archive measured no faster.

**C API.** `libmetal_runtime.dylib` ships in the package with `c_api.h`
(see `metal_runtime.c_api.include_dir()` and `library_dir()`), so a host
without Python or the GIL, such as an XLA FFI handler, can compile and
dispatch through the same runtime the Python extension uses.

`benchmarks/bench_overhead.py` measures the runtime's own overhead,
separated from GPU execution via `gpu_time`.

The extension is built free-threaded and is safe to use from several
threads on a free-threaded interpreter.

## Installation

With a `[tool.uv.sources]` entry in your `pyproject.toml`:

```toml
[tool.uv.sources]
metal-runtime = { git = "https://github.com/nicholasjng/metal-runtime" }
```

```console
$ uv add metal-runtime
```

Requires macOS with a Metal-capable GPU and Python 3.11+. The C++ extension
is compiled via CMake + nanobind on install; there's no pre-built wheel yet.

Apple Silicon is the tested target. Older GPUs without non-uniform
threadgroup support fall back to a dispatch path requiring `grid` to divide
evenly by `threadgroup`.

## Development

Package builds use standard build isolation. Configure a stable compilation
database for clangd after installing the development dependencies:

```console
$ uv sync
$ uv run --no-sync python scripts/configure-clangd.py
```

The database is copied from `build/clangd` to the repository root and refers
to nanobind in the project environment. Use the
`update_metal_runtime_core_stub` or `check_metal_runtime_core_stub` target in
that build tree after changing the native API.

## License

This project is licensed under the Apache-2.0 license.
