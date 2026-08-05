"""Launch-overhead benchmarks for metal-runtime.

Dispatch rows use `use_real_time`: GPU waits block without consuming CPU
time, so Google Benchmark's CPU-time default would misreport them.

Run with `uv run mew run`, filtered via
`--tag dispatch|compile|memory|shape|consumer`.
"""

import itertools
import time

import mew
import numpy as np

import metal_runtime as mr

clock = time.perf_counter

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

_EMPTY_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void nothing(device float* state [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0xffffffffu) state[0] = 1.0f;  // never taken; keeps the binding used
}
"""

# A dependent FMA chain, so device time can be dialled up without touching memory.
# The loop is a linear recurrence a compiler could in principle close out;
# measured, it does not - device time scales with `iters`.
_WORK_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void work(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                 uint tid [[thread_position_in_grid]]) {
    float acc = (float)tid;
    for (uint i = 0; i < iters; ++i) acc = fma(acc, 1.0000001f, 1.0f);
    if (acc == -1.0f) out[0] = acc;  // never taken (acc only grows); keeps the loop alive
}
"""


# Every binding is read: an unused one is optimized out and would skip the
# reflection check that costs host time.
def _multi_buffer_kernel(count: int) -> str:
    params = ", ".join(f"device float* b{i} [[buffer({i})]]" for i in range(count))
    body = " + ".join(f"b{i}[tid]" for i in range(count))
    return f"""
#include <metal_stdlib>
using namespace metal;

kernel void multi({params}, uint tid [[thread_position_in_grid]]) {{
    b0[tid] = {body};
}}
"""


N = 4096  # small on purpose: overhead should dominate, not bandwidth
STEPS = 200  # launches per Batch

# palladium's flash-attention shapes, which the consumer rows use instead of N.
PALLADIUM_SHAPE = (1024, 64)
PALLADIUM_GRID = 1024 * 32
PALLADIUM_THREADGROUP = (32, 1, 1)

_QKV = [np.zeros(PALLADIUM_SHAPE, dtype=np.float32) for _ in range(3)]  # 256 KiB each
_UPLOAD = np.zeros(1 << 22, dtype=np.float32)  # 16 MiB
_UNIQUE = itertools.count()  # cache-busting for the compile benchmarks

_PHASES = ("create", "encode", "commit", "wait", "total")
_SUBMIT_PARTS = ("driver", "queued", "gpu", "host_wake", "wall")
_CALL_PHASES = ("inputs", "output_alloc", "dispatch", "readback", "total")


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
    """Per-launch device-side time of the same batch; the gap to
    `bench_batch` is host and driver overhead."""
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


@mew.parametrize(
    [{"phase": p} for p in _PHASES],
    ids=list(_PHASES),
    tags="dispatch",
    use_manual_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_phase(state: mew.State, phase: str) -> None:
    """Where one blocking dispatch's wall time goes. All four phases are
    timed in the same iteration, so they reconstruct `total`."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for _ in state:
        t0 = clock()
        batch = mr.Batch()
        t1 = clock()
        batch.add(kernel, grid=N, buffers=[buffer])
        t2 = clock()
        batch.commit()
        t3 = clock()
        batch.wait()
        t4 = clock()
        elapsed = {
            "create": t1 - t0,
            "encode": t2 - t1,
            "commit": t3 - t2,
            "wait": t4 - t3,
            "total": t4 - t0,
        }
        state.set_iteration_time(elapsed[phase])


@mew.benchmark(tags="dispatch", use_real_time=True, unit="us", min_warmup_time=0.1)
def bench_run_empty(state: mew.State) -> None:
    """Absolute floor: a kernel that does nothing, one thread, one buffer."""
    kernel = mr.Kernel(_EMPTY_KERNEL, "nothing")
    buffer = mr.Buffer.zeros([1])
    for _ in state:
        mr.run(kernel, grid=1, buffers=[buffer])


@mew.parametrize(
    [{"part": p} for p in _SUBMIT_PARTS],
    ids=list(_SUBMIT_PARTS),
    tags="dispatch",
    use_manual_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_submit_path(state: mew.State, part: str) -> None:
    """Commit-to-completion, split by the command buffer's own timestamps.

    Only `gpu` is work; `queued` and `host_wake` are what a batched or
    pipelined caller avoids.
    """
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for _ in state:
        batch = mr.Batch()
        batch.add(kernel, grid=N, buffers=[buffer])
        t0 = clock()
        batch.commit()
        batch.wait()
        wall = clock() - t0
        stamps = batch.timestamps
        if stamps is None:
            state.skip_with_error("timestamps unavailable after wait()")
            return
        driver = stamps["kernel_end"] - stamps["kernel_start"]
        queued = stamps["gpu_start"] - stamps["kernel_end"]
        gpu = stamps["gpu_end"] - stamps["gpu_start"]
        state.set_iteration_time(
            {
                "driver": driver,
                "queued": queued,
                "gpu": gpu,
                "host_wake": max(wall - (driver + queued + gpu), 0.0),
                "wall": wall,
            }[part]
        )


@mew.benchmark(tags="dispatch", use_real_time=True, unit="us", min_warmup_time=0.1)
def bench_batch_create(state: mew.State) -> None:
    """`Batch()` alone: command buffer, encoder, device capability queries."""
    for _ in state:
        mr.Batch()


@mew.benchmark(tags="dispatch", use_real_time=True, unit="us", min_warmup_time=0.1)
def bench_encode(state: mew.State) -> None:
    """Pure per-launch encode: no commit, so no GPU involvement at all."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for n in state.batches(STEPS):
        batch = mr.Batch()
        for _ in range(n):
            batch.add(kernel, grid=N, buffers=[buffer])


@mew.parametrize(
    [{"count": c} for c in (1, 2, 4, 8, 16)],
    ids=["1buf", "2buf", "4buf", "8buf", "16buf"],
    tags="shape",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_encode_buffers(state: mew.State, count: int) -> None:
    """Per-binding host tax. palladium binds four, so the 1->4 slope is the
    part of its per-call cost that scales with arity."""
    kernel = mr.Kernel(_multi_buffer_kernel(count), "multi")
    buffers = [mr.Buffer.zeros([N]) for _ in range(count)]
    for n in state.batches(STEPS):
        batch = mr.Batch()
        for _ in range(n):
            batch.add(kernel, grid=N, buffers=buffers)


@mew.parametrize(
    [{"explicit": False}, {"explicit": True}],
    ids=["derived", "explicit"],
    tags="shape",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_encode_threadgroup(state: mew.State, explicit: bool) -> None:
    """Passing a threadgroup vs letting the runtime derive one per launch."""
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    threadgroup = kernel.max_threads_per_threadgroup if explicit else None
    for n in state.batches(STEPS):
        batch = mr.Batch()
        for _ in range(n):
            batch.add(kernel, grid=N, threadgroup=threadgroup, buffers=[buffer])


@mew.parametrize(
    [{"count": c} for c in (0, 1, 4)],
    ids=["0scalar", "1scalar", "4scalar"],
    tags="shape",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_encode_scalars(state: mew.State, count: int) -> None:
    """Encode cost against inline `setBytes` scalar count."""
    params = ["device float* out [[buffer(0)]]"]
    params += [f"constant float& s{i} [[buffer({i + 1})]]" for i in range(count)]
    params.append("uint tid [[thread_position_in_grid]]")
    terms = " + ".join(["out[tid]", *(f"s{i}" for i in range(count))])
    source = f"""
#include <metal_stdlib>
using namespace metal;

kernel void withscalars({", ".join(params)}) {{
    out[tid] = {terms};
}}
"""
    kernel = mr.Kernel(source, "withscalars")
    buffer = mr.Buffer.zeros([N])
    scalars = [np.float32(i) for i in range(count)]
    for n in state.batches(STEPS):
        batch = mr.Batch()
        for _ in range(n):
            batch.add(kernel, grid=N, buffers=[buffer], scalars=scalars)


@mew.parametrize(
    [
        {"iters": i, "part": p}
        for i in (0, 200, 2000, 8000)
        for p in ("wall", "gpu", "gap")
    ],
    ids=[f"{i}iters_{p}" for i in (0, 200, 2000, 8000) for p in ("wall", "gpu", "gap")],
    tags="consumer",
    use_manual_time=True,
    unit="us",
    min_warmup_time=0.2,
)
def bench_overhead_vs_work(state: mew.State, iters: int, part: str) -> None:
    """Is the dispatch cost additive to the kernel's own runtime, or hidden?

    Read the `gap` rows (wall minus device time, differenced inside one iteration);
    flat as `iters` grows means pure addition. Do not subtract the `wall` and
    `gpu` rows from each other: this family is twelve rows that heat the GPU as
    they run, and un-interleaved it has already reported a `gpu` row above its own
    `wall` row.
    """
    kernel = mr.Kernel(_WORK_KERNEL, "work")
    output = mr.Buffer.zeros([1])  # only out[0] is ever written
    count = np.uint32(iters)
    for _ in state:
        batch = mr.Batch()
        batch.add(
            kernel,
            grid=PALLADIUM_GRID,
            threadgroup=PALLADIUM_THREADGROUP,
            buffers=[output],
            scalars=[count],
        )
        t0 = clock()
        batch.commit()
        batch.wait()
        wall = clock() - t0
        gpu = batch.gpu_time
        if gpu is None:
            state.skip_with_error("gpu_time unavailable after wait()")
            return
        state.set_iteration_time(
            {"wall": wall, "gpu": gpu, "gap": max(wall - gpu, 0.0)}[part]
        )


@mew.parametrize(
    [{"depth": d} for d in (1, 2, 4, 8, 16)],
    ids=["depth1", "depth2", "depth4", "depth8", "depth16"],
    tags="consumer",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.2,
)
def bench_pipeline_depth(state: mew.State, depth: int) -> None:
    """Per-dispatch cost with `depth` command buffers in flight.

    Submit latency is latency, not throughput: it overlaps across command
    buffers. depth=1 is what a blocking call pays, and the slope to
    depth=16 is what a caller able to defer its waits can recover.
    """
    kernel = mr.Kernel(_TINY_KERNEL, "step")
    buffer = mr.Buffer.zeros([N])
    for n in state.batches(depth):
        pending: list[mr.Batch] = []
        for _ in range(n):
            batch = mr.Batch()
            batch.add(kernel, grid=N, buffers=[buffer])
            batch.commit()
            pending.append(batch)
            if len(pending) == depth:
                for b in pending:
                    b.wait()
                pending.clear()
        for b in pending:
            b.wait()


@mew.parametrize(
    [{"phase": p} for p in _CALL_PHASES],
    ids=list(_CALL_PHASES),
    tags="consumer",
    use_manual_time=True,
    unit="us",
    min_warmup_time=0.2,
)
def bench_consumer_call(state: mew.State, phase: str) -> None:
    """palladium's per-call cost, attributed. Mirrors `BoundKernel.launch`:
    refill three inputs, allocate a fresh output, dispatch, hand back a view."""
    kernel = mr.Kernel(_multi_buffer_kernel(4), "multi")
    inputs = [mr.Buffer(a) for a in _QKV]
    for _ in state:
        t0 = clock()
        for buffer, array in zip(inputs, _QKV, strict=True):
            buffer.copy_from(array)
        t1 = clock()
        output = mr.Buffer.empty(list(PALLADIUM_SHAPE), dtype="float32")
        t2 = clock()
        batch = mr.Batch()
        batch.add(
            kernel,
            grid=PALLADIUM_GRID,
            threadgroup=PALLADIUM_THREADGROUP,
            buffers=[*inputs, output],
        )
        batch.commit()
        batch.wait()
        t3 = clock()
        output.to_numpy()
        t4 = clock()
        elapsed = {
            "inputs": t1 - t0,
            "output_alloc": t2 - t1,
            "dispatch": t3 - t2,
            "readback": t4 - t3,
            "total": t4 - t0,
        }
        state.set_iteration_time(elapsed[phase])


@mew.parametrize(
    [{"reuse": False}, {"reuse": True}],
    ids=["fresh_output", "reused_output"],
    tags="consumer",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.2,
)
def bench_consumer_output(state: mew.State, reuse: bool) -> None:
    """What output reuse would save. palladium cannot reuse one today
    because `to_numpy()` returns a live view."""
    kernel = mr.Kernel(_multi_buffer_kernel(4), "multi")
    inputs = [mr.Buffer(a) for a in _QKV]
    held = mr.Buffer.empty(list(PALLADIUM_SHAPE), dtype="float32")
    for _ in state:
        output = (
            held if reuse else mr.Buffer.empty(list(PALLADIUM_SHAPE), dtype="float32")
        )
        batch = mr.Batch()
        batch.add(
            kernel,
            grid=PALLADIUM_GRID,
            threadgroup=PALLADIUM_THREADGROUP,
            buffers=[*inputs, output],
        )
        batch.commit()
        batch.wait()


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


@mew.parametrize(
    [
        {"op": "empty"},
        {"op": "zeros"},
        {"op": "upload"},
        {"op": "copy_from"},
        {"op": "to_numpy"},
    ],
    ids=["empty", "zeros", "upload", "copy_from", "to_numpy"],
    tags="memory",
    use_real_time=True,
    unit="us",
    min_warmup_time=0.1,
)
def bench_buffer_ops(state: mew.State, op: str) -> None:
    """The same operations as the 16 MiB rows above, at the 256 KiB a real
    per-call launch moves, where fixed cost per operation still shows."""
    array = _QKV[0]
    shape = list(PALLADIUM_SHAPE)
    existing = mr.Buffer(array)
    if op == "empty":
        for _ in state:
            mr.Buffer.empty(shape, dtype="float32")
    elif op == "zeros":
        for _ in state:
            mr.Buffer.zeros(shape, dtype="float32")
    elif op == "upload":
        for _ in state:
            mr.Buffer(array)
    elif op == "copy_from":
        for _ in state:
            existing.copy_from(array)
    else:
        for _ in state:
            existing.to_numpy()
    if op != "to_numpy":
        state.set_bytes_processed(state.iterations * array.nbytes)
