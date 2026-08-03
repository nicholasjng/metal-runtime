# metal-runtime

A small, LLVM-free Metal GPU runtime for Python, built on Apple's
[metal-cpp](https://developer.apple.com/metal/cpp/). Compiles MSL kernel source
at runtime (no offline `metal`/`metallib` toolchain), moves NumPy arrays into
shared GPU buffers with zero-copy readback, and dispatches. No MLIR, no XLA.

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

`threadgroup` is optional; left out, it is picked from the kernel's own
occupancy limits.

## What it does

**Buffers.** `mr.Buffer(array)` uploads any C-contiguous host array whose dtype
Metal can address: `bool`, `int8`-`int64`, `uint8`-`uint64`, `float16`,
`float32`, `bfloat16`. `float64` is rejected rather than narrowed, because
Metal has no `double` and a silent truncation is a precision loss you never see.
`mr.Buffer.zeros(shape, dtype)` allocates an output without an upload;
`mr.Buffer.empty` skips even the zero-fill, for outputs a kernel overwrites
entirely. `buffer.copy_from(array)` refills an existing allocation, which is
~3x cheaper than constructing a new `Buffer` per iteration. Readback through
`.to_numpy()` is a live view of the same unified memory, not a copy, and
buffers also export through DLPack — `np.from_dlpack(buffer)` (or
`jax.dlpack.from_dlpack`) consumes the same memory zero-copy.

Both `Buffer(array, dtype=...)` and `.to_numpy(dtype=...)` take an optional
dtype that *relabels* the bytes rather than converting them — `.view` semantics,
element width being the only thing that has to match. That's the way in for
element types NumPy can't hand across on its own, `ml_dtypes.bfloat16` in
particular, which exports through neither DLPack nor the buffer protocol:

```python
buf = mr.Buffer(arr.view(np.uint16), dtype="bfloat16")   # device bfloat* kernel
out = buf.to_numpy(dtype="uint16").view(ml_dtypes.bfloat16)
```

**Scalars.** Kernel arguments that aren't arrays go in `scalars`, are copied
inline with `setBytes`, and bind at the indices after `buffers`:

```python
mr.run(kernel, grid=n, buffers=[y, x], scalars=[np.float32(alpha), np.uint32(n)])
```

Scalars bind byte-for-byte against typed kernel arguments, so their width is
checked: a `np.float64` (numpy's default) against a `constant float&` would
silently read half a double, and is rejected with a pointer at `np.float32`.

**Grids.** `grid` and `threadgroup` take an `int` or a 1- to 3-tuple.
`threadgroup_memory` takes a list of byte sizes bound at `[[threadgroup(i)]]`.
A buffer can bind at a byte offset via a `(buffer, offset)` tuple, so several
logical arrays can suballocate one arena:

```python
mr.run(kernel, grid=n, buffers=[(arena, 0), (arena, n * 4)])
```

**Indirect dispatch.** Passing a `Buffer` as `grid` reads the three uint32
threadgroup counts from that buffer *when the GPU reaches the dispatch* (at
`indirect_offset`, default 0). A kernel earlier in the same batch can write
them, which is what adaptive algorithms need to size their next step without
a host round trip. The threadgroup must be explicit — the counts are
threadgroups, not threads:

```python
counts = mr.Buffer(np.array([blocks, 1, 1], dtype=np.uint32))
mr.run(kernel, grid=counts, threadgroup=256, buffers=[state])
```

**Launch validation.** Launches are checked against the compiler's own view of
the kernel: a `device`/`constant` argument or `[[threadgroup(i)]]` allocation
the launch doesn't cover raises a `ValueError` naming the argument, instead of
a GPU fault or silent garbage. Threadgroup memory is validated against the
device budget up front, because exceeding it downstream is a process abort
(Metal's validation layer), not a catchable error.

**Batching.** Each `run()` is one commit and one round-trip (~160 µs of
overhead on an M2). A `Batch` encodes several launches into a single command
buffer, which run in order and see each other's writes, amortizing that to a
few µs per launch:

```python
with mr.Batch() as batch:
    batch.add(k1, grid=n, buffers=[a, b])
    batch.add(k2, grid=n, buffers=[b, c])
# committed and waited on at exit; discarded if the body raises
```

`batch.commit()` hands the batch to the GPU *without* blocking, so a stepping
loop can encode batch *n+1* while *n* executes, then `wait()` on each in turn;
batches committed to the same queue run in order and see each other's writes.
After `wait()`, `batch.gpu_time` reports the device-side execution seconds —
what a benchmark should compare, wall clock minus this being pure host
overhead. `mr.Batch(concurrent=True)` lets independent launches overlap on the
GPU (useful for ensembles of small kernels); ordering then only exists across
an explicit `batch.barrier()`.

**Compile options.** `Kernel` takes a `math_mode` and preprocessor `defines`,
both part of the library cache key:

```python
kernel = mr.Kernel(source, "reduce", math_mode=mr.MathMode.SAFE,
                   defines={"BLOCK": "256"})
```

**Function constants.** MSL `[[function_constant(i)]]` values bake in at
*pipeline* creation, so unlike `defines` they reuse the compiled library — a
generator emitting one variant per block size skips the MSL front end
entirely. Plain `bool`/`int`/`float` coerce to the declared MSL type
(`constant uint N` accepts a plain `int`, range-checked); a numpy scalar pins
an exact width that must match the declaration:

```python
kernel = mr.Kernel(source, "step", constants={"DECAY": 0.999, "N": n})
```

Metal itself neither rejects a misspelled constant name nor an unset one (the
kernel would compute with an undefined value), so the runtime validates
against the function's own constant reflection: required constants must be
set, unknown names raise, and constants guarded by
`is_function_constant_defined` stay optional. Compiled pipelines are cached
per library, so re-creating the same `Kernel` costs well under a microsecond.

`math_mode` defaults to `FAST`, which is Metal's own default and permits
reassociation. **That silently deletes compensated arithmetic**: under `FAST`
(and `RELAXED`) the Kahan update `c = (t - s) - y` folds algebraically to zero
and is optimized out with no diagnostic, leaving you the naive sum. Anything
built on error-free transformations — Kahan/Neumaier accumulation,
double-single arithmetic — must be compiled with `SAFE`. `MathMode` is a
`StrEnum`, so members compare equal to `"safe"` / `"relaxed"` / `"fast"` and
survive a round-trip through config files.

**Errors.** `CompileError` for bad MSL, `FunctionNotFoundError` (a subclass, so
one `except CompileError` covers both) for a missing entry point,
`DispatchError` for a command buffer the GPU aborted, `DeviceError` when there
is no Metal device at all.

**Introspection.** `mr.device_info()` reports the device's limits, including
`max_threadgroup_memory_length` and `max_buffer_length`;
`kernel.max_threads_per_threadgroup`, `kernel.thread_execution_width` and
`kernel.static_threadgroup_memory_length` report the ones that apply to a
specific kernel. Compiled libraries are cached by source text, bounded by
`mr.set_library_cache_limit()` (default 256, `0` for unlimited) so a generator
emitting one specialization per shape doesn't accumulate them for the life of
the process.

`benchmarks/bench_overhead.py` measures what the runtime itself costs — launch
round-trips, batch amortization, overlap, compile/cache/specialization times —
separated from GPU execution via `gpu_time`. It runs on
[mew](https://mew.readthedocs.io/): `uv run mew run`, filtered with
`--tag dispatch|compile|memory`, and `mew compare` tracks regressions across
runs.

The extension is built free-threaded and is safe to use from several threads on
a free-threaded interpreter.

## Installation

With a `[tool.uv.sources]` entry in your `pyproject.toml`:

```toml
[tool.uv.sources]
metal-runtime = { git = "https://github.com/nicholasjng/metal-runtime" }
```

```console
$ uv add metal-runtime
```

Requires macOS with a Metal-capable GPU and Python 3.11+. The C++ extension is
compiled via CMake + nanobind on install; there's no pre-built wheel yet.

Apple Silicon is the tested target. Older GPUs without support for non-uniform
threadgroups (below Apple family 4 / Mac family 2) take a fallback dispatch path
that requires `grid` to divide evenly by `threadgroup`; check
`mr.device_info()["supports_non_uniform_threadgroups"]`.

On macOS 14 and older, where the `mathMode` compile option doesn't exist,
`math_mode` falls back to the legacy fast-math boolean: `SAFE` keeps IEEE
semantics, and `RELAXED` collapses into `FAST`.

A PyPI release is planned for the future.

## Project direction

This repo's goal, running JAX programs on the Apple GPU without building
MLIR/StableHLO from source, is laid out in [ROADMAP.md](ROADMAP.md).

## License

This project is licensed under the Apache-2.0 license.
