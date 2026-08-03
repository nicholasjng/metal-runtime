"""Launch-overhead benchmarks for metal-runtime.

Dispatch rows use `use_real_time`: GPU waits block without consuming CPU
time, so Google Benchmark's CPU-time default would misreport them.
`bench_batch_gpu` reports device-side time via `gpu_time`; its gap to
`bench_batch` is host and driver overhead.

Run with `uv run mew run`, filtered via `--tag dispatch|compile|memory`.
"""

import itertools

import mew
import numpy as np

import metal_runtime as mr

_TINY_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void step(device float* state [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    state[tid] = state[tid] * 0.999f + 0.001f;
}
"""

_CONSTANT_KERNEL = """
#include <metal_stdlib>
using namespace metal;

constant float DECAY [[function_constant(0)]];

kernel void step(device float* state [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    state[tid] = state[tid] * DECAY + 0.001f;
}
"""

N = 4096  # small on purpose: overhead should dominate, not bandwidth
STEPS = 200  # launches per Batch

_UNIQUE = itertools.count()  # cache-busting for the compile benchmarks


# --- dispatch --------------------------------------------------------------


@mew.benchmark(tags="dispatch", use_real_time=True, unit="us", min_warmup_time=0.1)
def bench_run(state: mew.State) -> None:
    """One blocking run(): commit + waitUntilCompleted per launch."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for _ in state:
        mr.run(kernel, grid=N, buffers=[buffer])


@mew.benchmark(tags="dispatch", use_real_time=True, unit="us", min_warmup_time=0.1)
def bench_batch(state: mew.State) -> None:
    """Per-launch wall time of a Batch of STEPS launches."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for n in state.batches(STEPS):
        with mr.Batch() as batch:
            for _ in range(n):
                batch.add(kernel, grid=N, buffers=[buffer])


@mew.benchmark(tags="dispatch", use_manual_time=True, unit="us", min_warmup_time=0.1)
def bench_batch_gpu(state: mew.State) -> None:
    """Per-launch device-side time of the same batch."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for n in state.batches(STEPS):
        batch = mr.Batch()
        for _ in range(n):
            batch.add(kernel, grid=N, buffers=[buffer])
        batch.wait()
        gpu_time = batch.gpu_time
        if gpu_time is None:
            state.skip_with_error("gpu_time unavailable after wait()")
            return
        state.set_iteration_time(gpu_time)


@mew.parametrize(
    [{"overlap": False}, {"overlap": True}],
    ids=["sequential", "overlapped"],
    tags="dispatch",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_two_batches(state: mew.State, overlap: bool) -> None:
    """Encode batch 2 while batch 1 runs, or wait in between."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])

    def encode() -> mr.Batch:
        batch = mr.Batch()
        for _ in range(STEPS):
            batch.add(kernel, grid=N, buffers=[buffer])
        return batch

    for _ in state.batches(2 * STEPS):
        if overlap:
            first = encode()
            first.commit()
            second = encode()
            second.commit()
            first.wait()
            second.wait()
        else:
            encode().wait()
            encode().wait()


# --- compile ---------------------------------------------------------------


@mew.benchmark(tags="compile", unit="us")
def bench_kernel_cold(state: mew.State) -> None:
    """Full MSL compile: unique source per iteration."""
    for _ in state:
        mr.Kernel(_TINY_KERNEL + f"// cold-{next(_UNIQUE)}\n", "step")


@mew.benchmark(tags="compile", unit="us")
def bench_kernel_cache_hit(state: mew.State) -> None:
    """Library and pipeline cache hit for an already-seen kernel."""
    source = _TINY_KERNEL + f"// warm-{next(_UNIQUE)}\n"
    mr.Kernel(source, "step")  # primed outside the timed loop
    for _ in state:
        mr.Kernel(source, "step")


@mew.benchmark(tags="compile", unit="us")
def bench_kernel_constants(state: mew.State) -> None:
    """New function constants: pipeline specialization, no MSL front end."""
    source = _CONSTANT_KERNEL + f"// constants-{next(_UNIQUE)}\n"
    mr.Kernel(source, "step", constants={"DECAY": 0.5})  # library compiled here
    for _ in state:
        mr.Kernel(source, "step", constants={"DECAY": 0.5 + 0.001 * next(_UNIQUE)})


@mew.benchmark(tags="compile", unit="us")
def bench_kernel_defines(state: mew.State) -> None:
    """New define per iteration: the full recompile that constants avoid."""
    for _ in state:
        mr.Kernel(
            _TINY_KERNEL + "// defines\n",
            "step",
            defines={"UNUSED": str(next(_UNIQUE))},
        )


# --- memory ----------------------------------------------------------------

_UPLOAD = np.zeros(1 << 22, dtype=np.float32)  # 16 MiB


@mew.benchmark(tags="memory", use_real_time=True, unit="us")
def bench_upload(state: mew.State) -> None:
    for _ in state:
        mr.Buffer(_UPLOAD)
    state.set_bytes_processed(state.iterations * _UPLOAD.nbytes)


@mew.benchmark(tags="memory", use_real_time=True, unit="us")
def bench_copy_from(state: mew.State) -> None:
    """Refill an existing allocation instead of constructing a Buffer."""
    target = mr.Buffer(_UPLOAD)
    for _ in state:
        target.copy_from(_UPLOAD)
    state.set_bytes_processed(state.iterations * _UPLOAD.nbytes)


@mew.parametrize(
    [{"init": "zeros"}, {"init": "empty"}],
    ids=["zeros", "empty"],
    tags="memory",
    use_real_time=True,
    unit="us",
)
def bench_alloc(state: mew.State, init: str) -> None:
    """Zero-filled vs uninitialized output allocation."""
    alloc = mr.Buffer.zeros if init == "zeros" else mr.Buffer.empty
    count = _UPLOAD.size
    for _ in state:
        alloc([count])
    state.set_bytes_processed(state.iterations * _UPLOAD.nbytes)
