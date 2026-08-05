"""Query blocking in a fused flash-attention kernel.

palladium's emitted flash-attention kernel puts one query row in each Pallas
program instance (`grid=seq_len`), so every instance streams all of K and V
past itself and every `dot_general` has m == 1. That forecloses
`simdgroup_matrix` and multiplies K/V traffic by the query count.

These are hand-written MSL kernels standing in for what the emitter would
produce if it blocked over queries too, each checked against a NumPy oracle
before timing.

These rows are only comparable when interleaved: run back to back on a warm
GPU they invert, one sequential pass putting `blocked8` at 1206us of device
time against `rowwise`'s 963us, the reverse of the rested ordering. Compare
with

    uv run mew run --tag attention -k bench_attention_gpu \\
        --repetitions 7 --random-interleaving

and read the medians. Prefer the `_gpu` rows: they exclude the ~130us
per-dispatch floor, which is fixed and would flatter the slower kernel.
"""

import mew
import numpy as np

import metal_runtime as mr

SEQ_LEN = 1024
HEAD_DIM = 64
SCALE = 1.0 / np.sqrt(HEAD_DIM)

# Both kernels launch threadgroup == SIMD_WIDTH, so `simd_*` reductions cover
# exactly one threadgroup and a simdgroup_barrier suffices.
SIMD_WIDTH = 32

BLOCK_KV = 32  # keys per streamed block, the same for every variant

# One SIMD group per query row. Lane l holds q[l] and q[l + 32], accumulates
# o[l] and o[l + 32], and every score costs a simd_sum across the group.
_ROWWISE = f"""
#include <metal_stdlib>
using namespace metal;

kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint gid [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {{
    const uint D = {HEAD_DIM};
    const uint SEQ = {SEQ_LEN};
    const uint row = gid;

    const float qa = Q[row * D + lane];
    const float qb = Q[row * D + lane + 32];

    float m = -INFINITY;
    float lsum = 0.0f;
    float oa = 0.0f;
    float ob = 0.0f;

    for (uint kk = 0; kk < SEQ; ++kk) {{
        float part = qa * K[kk * D + lane] + qb * K[kk * D + lane + 32];
        float s = simd_sum(part) * {SCALE}f;

        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float p = exp(s - m_new);

        lsum = lsum * corr + p;
        oa = oa * corr + p * V[kk * D + lane];
        ob = ob * corr + p * V[kk * D + lane + 32];
        m = m_new;
    }}

    O[row * D + lane] = oa / lsum;
    O[row * D + lane + 32] = ob / lsum;
}}
"""


# One SIMD group per `bq` query rows. Q is staged in threadgroup memory so all
# lanes can read any row; the k-outer / row-inner loops make each loaded K and
# V element serve all `bq` rows, which is the traffic saving being tested.
def _blocked_source(bq: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;

kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint gid [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {{
    const uint D = {HEAD_DIM};
    const uint SEQ = {SEQ_LEN};
    const uint BQ = {bq};
    const uint BK = {BLOCK_KV};
    const uint q0 = gid * BQ;

    threadgroup float Qs[{bq}][{HEAD_DIM}];
    threadgroup float Ps[{bq}][{BLOCK_KV}];

    for (uint t = lane; t < BQ * D; t += 32) {{
        Qs[t / D][t % D] = Q[(q0 + t / D) * D + t % D];
    }}
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float m[{bq}];
    float lsum[{bq}];
    float oa[{bq}];
    float ob[{bq}];
    for (uint r = 0; r < BQ; ++r) {{
        m[r] = -INFINITY;
        lsum[r] = 0.0f;
        oa[r] = 0.0f;
        ob[r] = 0.0f;
    }}

    for (uint jb = 0; jb < SEQ; jb += BK) {{
        // Scores: lane owns key column jb + lane. K is read once per lane and
        // reused across all BQ query rows.
        const uint kk = jb + lane;
        float acc[{bq}];
        for (uint r = 0; r < BQ; ++r) acc[r] = 0.0f;
        for (uint d = 0; d < D; ++d) {{
            float kv = K[kk * D + d];
            for (uint r = 0; r < BQ; ++r) acc[r] += Qs[r][d] * kv;
        }}

        for (uint r = 0; r < BQ; ++r) {{
            float s = acc[r] * {SCALE}f;
            float m_new = max(m[r], simd_max(s));
            float corr = exp(m[r] - m_new);
            float p = exp(s - m_new);
            lsum[r] = lsum[r] * corr + simd_sum(p);
            oa[r] *= corr;
            ob[r] *= corr;
            m[r] = m_new;
            Ps[r][lane] = p;
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // P @ V: lane owns head dims `lane` and `lane + 32`, so the V loads
        // are contiguous across the group and each serves all BQ rows.
        for (uint kkk = 0; kkk < BK; ++kkk) {{
            float v0 = V[(jb + kkk) * D + lane];
            float v1 = V[(jb + kkk) * D + lane + 32];
            for (uint r = 0; r < BQ; ++r) {{
                oa[r] += Ps[r][kkk] * v0;
                ob[r] += Ps[r][kkk] * v1;
            }}
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint r = 0; r < BQ; ++r) {{
        O[(q0 + r) * D + lane] = oa[r] / lsum[r];
        O[(q0 + r) * D + lane + 32] = ob[r] / lsum[r];
    }}
}}
"""


# A sweep, not a chosen value: blocking by n cuts K/V traffic by n but also
# divides the SIMD-group count by n, and SEQ_LEN rows do not divide far. 8
# measured fastest (2.0x device time over `rowwise`); 32 leaves 32 SIMD groups
# for 16 GPU cores and is 6x slower than `rowwise`.
BLOCK_SIZES = (2, 4, 8, 16, 32)

_VARIANTS: dict[str, tuple[str, int]] = {
    "rowwise": (_ROWWISE, SEQ_LEN * SIMD_WIDTH),
    **{
        f"blocked{bq}": (_blocked_source(bq), (SEQ_LEN // bq) * SIMD_WIDTH)
        for bq in BLOCK_SIZES
    },
}

# 2 matmuls, 2 flops each, over seq x seq x head_dim.
FLOPS = 4 * SEQ_LEN * SEQ_LEN * HEAD_DIM


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    shape = (SEQ_LEN, HEAD_DIM)
    return (
        rng.standard_normal(shape, dtype=np.float32),
        rng.standard_normal(shape, dtype=np.float32),
        rng.standard_normal(shape, dtype=np.float32),
    )


def _oracle(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = (q.astype(np.float64) @ k.astype(np.float64).T) * SCALE
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=1, keepdims=True)) @ v.astype(np.float64)


def _build(variant: str) -> tuple[mr.Kernel, int, list[mr.Buffer], mr.Buffer]:
    """Compiles, runs once, and checks against the oracle before any timing."""
    source, grid = _VARIANTS[variant]
    q, k, v = _inputs()
    kernel = mr.Kernel(source, "attention")
    buffers = [mr.Buffer(a) for a in (q, k, v)]
    output = mr.Buffer.zeros([SEQ_LEN, HEAD_DIM])

    mr.run(
        kernel,
        grid=grid,
        threadgroup=(SIMD_WIDTH, 1, 1),
        buffers=[*buffers, output],
    )
    got = output.to_numpy()
    want = _oracle(q, k, v)
    error = np.abs(got - want).max()
    if not np.isfinite(error) or error > 2e-4:
        raise AssertionError(f"{variant}: max abs error {error:g} against the oracle")
    return kernel, grid, buffers, output


@mew.parametrize(
    [{"variant": v} for v in _VARIANTS],
    ids=list(_VARIANTS),
    tags="attention",
    use_real_time=True,
    unit="us",
    min_warmup_time=1.0,
)
def bench_attention(state: mew.State, variant: str) -> None:
    """Whole-call wall time, the number palladium's reward function sees."""
    kernel, grid, buffers, output = _build(variant)
    for _ in state:
        mr.run(
            kernel,
            grid=grid,
            threadgroup=(SIMD_WIDTH, 1, 1),
            buffers=[*buffers, output],
        )
    state.set_counter("gflops", FLOPS / 1e9 * state.iterations)


@mew.parametrize(
    [{"variant": v} for v in _VARIANTS],
    ids=list(_VARIANTS),
    tags="attention",
    use_manual_time=True,
    unit="us",
    min_warmup_time=1.0,
)
def bench_attention_gpu(state: mew.State, variant: str) -> None:
    """Device-side time only, so the ~130us dispatch floor is excluded."""
    kernel, grid, buffers, output = _build(variant)
    for _ in state:
        batch = mr.Batch()
        batch.add(
            kernel,
            grid=grid,
            threadgroup=(SIMD_WIDTH, 1, 1),
            buffers=[*buffers, output],
        )
        batch.wait()
        gpu_time = batch.gpu_time
        if gpu_time is None:
            state.skip_with_error("gpu_time unavailable after wait()")
            return
        state.set_iteration_time(gpu_time)
