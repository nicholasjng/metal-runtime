import threading

import numpy as np
import pytest

import metal_runtime as mr

_ADD_ONE_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0f;
}
"""

_FILL_TID_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void fill_tid(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = float(tid);
}
"""

# Takes its scale and length as setBytes scalars rather than as buffers.
_AXPY_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void axpy(device float* y [[buffer(0)]], device const float* x [[buffer(1)]],
                 constant float& a [[buffer(2)]], constant uint& n [[buffer(3)]],
                 uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    y[tid] = a * x[tid] + y[tid];
}
"""

_FILL_2D_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void fill_2d(device float* out [[buffer(0)]], constant uint& width [[buffer(1)]],
                    uint2 pos [[thread_position_in_grid]]) {
    out[pos.y * width + pos.x] = float(pos.y * 100 + pos.x);
}
"""

# Sums a threadgroup's worth of elements through threadgroup memory, so it only
# gives the right answer if the threadgroup size is exactly what was requested.
_TG_REDUCE_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void tg_sum(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                   threadgroup float* scratch [[threadgroup(0)]],
                   uint tid [[thread_position_in_grid]],
                   uint ltid [[thread_position_in_threadgroup]],
                   uint tg_size [[threads_per_threadgroup]]) {
    scratch[ltid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (ltid < s) scratch[ltid] += scratch[ltid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ltid == 0) out[0] = scratch[0];
}
"""


def test_device_name_is_nonempty():
    assert mr.device_name()


def test_device_info_reports_capabilities():
    info = mr.device_info()
    assert info["name"]
    assert info["max_threads_per_threadgroup"] > 0
    assert isinstance(info["unified_memory"], bool)
    assert isinstance(info["supports_non_uniform_threadgroups"], bool)


def test_buffer_roundtrip():
    array = np.arange(16, dtype=np.float32)
    buffer = mr.Buffer(array)
    assert np.array_equal(buffer.to_numpy(), array)


def test_dispatch_add_one_kernel():
    array = np.arange(16, dtype=np.float32)
    buffer = mr.Buffer(array)
    kernel = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    mr.run(kernel, grid=16, threadgroup=16, buffers=[buffer])

    assert np.allclose(buffer.to_numpy(), array + 1.0)


def test_zeros_allocates_without_upload():
    buffer = mr.Buffer.zeros([4, 3])
    assert np.array_equal(buffer.to_numpy(), np.zeros((4, 3), dtype=np.float32))


def test_zeros_buffer_dispatches_as_kernel_output():
    buffer = mr.Buffer.zeros([8])
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")

    mr.run(kernel, grid=8, threadgroup=8, buffers=[buffer])

    assert np.array_equal(buffer.to_numpy(), np.arange(8, dtype=np.float32))


def test_invalid_msl_raises_compile_error():
    with pytest.raises(mr.CompileError):
        mr.Kernel("this is not valid msl {{{", "nope")


def test_missing_function_raises_function_not_found():
    with pytest.raises(mr.FunctionNotFoundError):
        mr.Kernel(_ADD_ONE_SOURCE, "no_such_kernel")


def test_function_not_found_is_catchable_as_compile_error():
    """A generator emitting both the source and the entry-point name gets both
    failure modes from one `except CompileError`."""
    assert issubclass(mr.FunctionNotFoundError, mr.CompileError)
    with pytest.raises(mr.CompileError):
        mr.Kernel(_ADD_ONE_SOURCE, "no_such_kernel")


def test_library_cache_reuses_compiled_kernel():
    """Same MSL source string, two Kernel objects: dispatching both should still
    work (exercises MetalRuntime's library cache without asserting object identity,
    which is an internal implementation detail)."""
    array = np.zeros(8, dtype=np.float32)
    buffer_a = mr.Buffer(array)
    buffer_b = mr.Buffer(array)

    kernel_a = mr.Kernel(_ADD_ONE_SOURCE, "add_one")
    kernel_b = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    mr.run(kernel_a, grid=8, buffers=[buffer_a])
    mr.run(kernel_b, grid=8, buffers=[buffer_b])

    assert np.allclose(buffer_a.to_numpy(), array + 1.0)
    assert np.allclose(buffer_b.to_numpy(), array + 1.0)


# --- buffer metadata ---------------------------------------------------------


def test_buffer_exposes_shape_dtype_and_size():
    buffer = mr.Buffer.zeros([4, 3], "float32")
    assert buffer.shape == (4, 3)
    assert buffer.dtype == "float32"
    assert buffer.size == 12
    assert buffer.nbytes == 48
    assert len(buffer) == 4
    assert repr(buffer) == "Buffer(shape=(4, 3), dtype='float32')"


# --- dtypes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
    ],
)
def test_dtype_roundtrip(dtype):
    array = np.arange(4).astype(dtype)
    buffer = mr.Buffer(array)
    assert buffer.dtype == dtype
    assert np.array_equal(buffer.to_numpy(), array)


def test_half_precision_kernel():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void dbl(device half* b [[buffer(0)]], uint t [[thread_position_in_grid]]) {
        b[t] *= 2.0h;
    }
    """
    array = np.arange(8, dtype=np.float16)
    buffer = mr.Buffer(array)
    mr.run(mr.Kernel(source, "dbl"), grid=8, buffers=[buffer])
    assert np.array_equal(buffer.to_numpy(), array * 2)


def test_int32_kernel():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void inc(device int* b [[buffer(0)]], uint t [[thread_position_in_grid]]) {
        b[t] += 7;
    }
    """
    array = np.arange(8, dtype=np.int32)
    buffer = mr.Buffer(array)
    mr.run(mr.Kernel(source, "inc"), grid=8, buffers=[buffer])
    assert np.array_equal(buffer.to_numpy(), array + 7)


def test_float64_is_rejected_rather_than_silently_narrowed():
    """Metal has no double, and quietly truncating to float32 would hide real
    precision loss behind an API that never mentions it."""
    with pytest.raises(ValueError, match="float64"):
        mr.Buffer(np.zeros(4))  # numpy's default dtype
    with pytest.raises(ValueError, match="float64"):
        mr.Buffer.zeros([4], "float64")


def test_unknown_dtype_name_lists_the_supported_ones():
    with pytest.raises(ValueError, match="unsupported dtype"):
        mr.Buffer.zeros([4], "quux")


def test_bfloat16_buffer_reports_a_useful_readback_error():
    buffer = mr.Buffer.zeros([4], "bfloat16")
    assert buffer.dtype == "bfloat16"
    with pytest.raises(TypeError, match="bfloat16"):
        buffer.to_numpy()


# --- dtype reinterpretation --------------------------------------------------


def test_dtype_override_reinterprets_rather_than_converts():
    """`.view` semantics: same bytes, different label. 1.5 as float32 is a
    specific bit pattern, and relabelling it must not touch it."""
    original = np.array([1.5, -2.25, 3.0, 0.0], dtype=np.float32)
    buffer = mr.Buffer(original.view(np.uint32), dtype="float32")
    assert buffer.dtype == "float32"
    assert np.array_equal(buffer.to_numpy(), original)


def test_dtype_override_rejects_a_width_mismatch():
    array = np.arange(4, dtype=np.uint16)
    with pytest.raises(ValueError, match="element width"):
        mr.Buffer(array, dtype="float32")


def test_to_numpy_dtype_reinterprets_on_the_way_out():
    original = np.array([1.5, -2.25], dtype=np.float32)
    buffer = mr.Buffer(original)
    assert np.array_equal(buffer.to_numpy(dtype="uint32"), original.view(np.uint32))


def test_to_numpy_dtype_rejects_a_width_mismatch():
    buffer = mr.Buffer.zeros([4], "float32")
    with pytest.raises(ValueError, match="byte elements"):
        buffer.to_numpy(dtype="uint8")


def test_bfloat16_survives_a_uint16_round_trip():
    """No ml_dtypes needed to check the bytes make it there and back."""
    bits = np.array([0x3F80, 0x4020, 0xC070, 0x42C8], dtype=np.uint16)
    buffer = mr.Buffer(bits, dtype="bfloat16")
    assert buffer.dtype == "bfloat16"
    assert np.array_equal(buffer.to_numpy(dtype="uint16"), bits)


def test_ml_dtypes_bfloat16_interop():
    """ml_dtypes arrays cross the boundary as their uint16 view: NumPy exports
    them through neither DLPack nor the buffer protocol, so nanobind can't see
    them directly. The relabelling is what makes the kernel side honest."""
    ml_dtypes = pytest.importorskip("ml_dtypes")

    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void dbl(device bfloat* b [[buffer(0)]], uint t [[thread_position_in_grid]]) {
        b[t] *= 2.0bf;
    }
    """
    array = np.array([1.0, 2.5, -3.75, 100.0], dtype=ml_dtypes.bfloat16)
    buffer = mr.Buffer(array.view(np.uint16), dtype="bfloat16")

    mr.run(mr.Kernel(source, "dbl"), grid=4, buffers=[buffer])

    result = buffer.to_numpy(dtype="uint16").view(ml_dtypes.bfloat16)
    assert np.array_equal(result, array * ml_dtypes.bfloat16(2))


def test_ml_dtypes_arrays_are_rejected_without_a_view():
    """Documents the upstream limitation rather than pretending it's ours: the
    failure is NumPy refusing to export, before nanobind gets a pointer."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    with pytest.raises(TypeError):
        mr.Buffer(np.arange(4, dtype=ml_dtypes.bfloat16))


# --- empty buffers -----------------------------------------------------------


def test_empty_array_allocates():
    """newBuffer(0) returns nil, so a zero-element shape used to fail as if the
    device were out of memory."""
    buffer = mr.Buffer(np.array([], dtype=np.float32))
    assert buffer.shape == (0,)
    assert buffer.to_numpy().shape == (0,)


def test_zeros_with_a_degenerate_axis():
    buffer = mr.Buffer.zeros([0, 3])
    assert buffer.shape == (0, 3)
    assert buffer.to_numpy().shape == (0, 3)


# --- scalars, grids, threadgroup memory --------------------------------------


def test_scalars_bind_after_buffers():
    x = np.arange(10, dtype=np.float32)
    y = mr.Buffer(np.ones(10, dtype=np.float32))

    mr.run(
        mr.Kernel(_AXPY_SOURCE, "axpy"),
        grid=10,
        buffers=[y, mr.Buffer(x)],
        scalars=[np.float32(2.0), np.uint32(10)],
    )

    assert np.allclose(y.to_numpy(), 2.0 * x + 1.0)


def test_two_dimensional_grid():
    out = mr.Buffer.zeros([3, 4])
    mr.run(
        mr.Kernel(_FILL_2D_SOURCE, "fill_2d"),
        grid=(4, 3),
        buffers=[out],
        scalars=[np.uint32(4)],
    )
    expected = np.array(
        [[y * 100 + x for x in range(4)] for y in range(3)], dtype=np.float32
    )
    assert np.array_equal(out.to_numpy(), expected)


def test_threadgroup_memory():
    x = np.arange(256, dtype=np.float32)
    out = mr.Buffer.zeros([1])
    mr.run(
        mr.Kernel(_TG_REDUCE_SOURCE, "tg_sum"),
        grid=256,
        threadgroup=256,
        buffers=[mr.Buffer(x), out],
        threadgroup_memory=[256 * 4],
    )
    assert out.to_numpy()[0] == pytest.approx(x.sum())


def test_threadgroup_defaults_to_a_workable_size():
    buffer = mr.Buffer.zeros([1000])
    mr.run(mr.Kernel(_FILL_TID_SOURCE, "fill_tid"), grid=1000, buffers=[buffer])
    assert np.array_equal(buffer.to_numpy(), np.arange(1000, dtype=np.float32))


# --- validation --------------------------------------------------------------


def test_zero_grid_is_rejected():
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    with pytest.raises(ValueError, match="grid"):
        mr.run(kernel, grid=0, buffers=[mr.Buffer.zeros([4])])


def test_zero_threadgroup_is_rejected():
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    with pytest.raises(ValueError, match="threadgroup"):
        mr.run(kernel, grid=4, threadgroup=0, buffers=[mr.Buffer.zeros([4])])


def test_oversized_threadgroup_raises_instead_of_clamping():
    """Silently shrinking the threadgroup changes what a kernel indexing
    threadgroup memory computes, and the caller sees only wrong numbers."""
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    too_big = kernel.max_threads_per_threadgroup + 1
    with pytest.raises(ValueError, match="threadgroup"):
        mr.run(kernel, grid=4, threadgroup=too_big, buffers=[mr.Buffer.zeros([4])])


def test_grid_dimensionality_is_bounded():
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        mr.run(kernel, grid=(1, 2, 3, 4), buffers=[mr.Buffer.zeros([4])])


def test_oversized_inline_scalar_points_at_buffers():
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    with pytest.raises(ValueError, match="setBytes"):
        mr.run(
            kernel,
            grid=4,
            buffers=[mr.Buffer.zeros([4])],
            scalars=[np.zeros(2000, dtype=np.float32)],
        )


def test_too_many_bindings_is_rejected():
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    buffer = mr.Buffer.zeros([4])
    with pytest.raises(ValueError, match="31"):
        mr.run(kernel, grid=4, buffers=[buffer] * 32, scalars=[np.uint32(1)])


def test_shape_overflow_is_rejected():
    with pytest.raises(ValueError, match="too large"):
        mr.Buffer.zeros([2**40, 2**40])


def test_kernel_reports_its_own_limits():
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")
    assert kernel.function_name == "fill_tid"
    assert kernel.max_threads_per_threadgroup > 0
    assert kernel.thread_execution_width > 0
    assert repr(kernel) == "Kernel(function_name='fill_tid')"


# --- batching ----------------------------------------------------------------


def test_batch_runs_launches_in_order():
    """One command buffer, five launches: the encoder is serial, so each add-one
    observes the previous one's writes."""
    buffer = mr.Buffer.zeros([4])
    kernel = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    with mr.Batch() as batch:
        for _ in range(5):
            batch.add(kernel, grid=4, buffers=[buffer])

    assert np.array_equal(buffer.to_numpy(), np.full(4, 5.0, dtype=np.float32))


def test_batch_is_discarded_when_the_body_raises():
    buffer = mr.Buffer.zeros([4])
    kernel = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    with pytest.raises(ValueError), mr.Batch() as batch:
        batch.add(kernel, grid=4, buffers=[buffer])
        raise ValueError("boom")

    assert np.array_equal(buffer.to_numpy(), np.zeros(4, dtype=np.float32))


def test_batch_rejects_adds_after_wait():
    buffer = mr.Buffer.zeros([4])
    kernel = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    batch = mr.Batch()
    batch.add(kernel, grid=4, buffers=[buffer])
    batch.wait()
    with pytest.raises(mr.DispatchError):
        batch.add(kernel, grid=4, buffers=[buffer])


def test_batch_wait_is_idempotent():
    buffer = mr.Buffer.zeros([4])
    batch = mr.Batch()
    batch.add(mr.Kernel(_ADD_ONE_SOURCE, "add_one"), grid=4, buffers=[buffer])
    batch.wait()
    batch.wait()
    assert np.array_equal(buffer.to_numpy(), np.ones(4, dtype=np.float32))


# --- library cache -----------------------------------------------------------


@pytest.fixture
def restore_cache_limit():
    limit = mr.library_cache_limit()
    yield
    mr.set_library_cache_limit(limit)


def _counting_source(i: int) -> str:
    return f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void f(device float* b [[buffer(0)]]) {{ b[0] = {i}.0f; }}
    """


def test_library_cache_evicts_beyond_its_limit(restore_cache_limit):
    mr.clear_library_cache()
    mr.set_library_cache_limit(4)
    for i in range(20):
        mr.Kernel(_counting_source(i), "f")
    assert mr.library_cache_size() == 4


def test_library_cache_limit_zero_disables_eviction(restore_cache_limit):
    mr.clear_library_cache()
    mr.set_library_cache_limit(0)
    for i in range(10):
        mr.Kernel(_counting_source(1000 + i), "f")
    assert mr.library_cache_size() == 10


def test_kernel_outlives_eviction_of_its_library(restore_cache_limit):
    """The cache hands out shared ownership, so evicting an entry can't pull the
    library out from under a pipeline already built from it."""
    mr.clear_library_cache()
    mr.set_library_cache_limit(1)
    kernel = mr.Kernel(_counting_source(99), "f")
    for i in range(5):
        mr.Kernel(_counting_source(2000 + i), "f")

    buffer = mr.Buffer.zeros([1])
    mr.run(kernel, grid=1, buffers=[buffer])
    assert buffer.to_numpy()[0] == 99.0


def test_clear_library_cache(restore_cache_limit):
    mr.Kernel(_ADD_ONE_SOURCE, "add_one")
    assert mr.library_cache_size() > 0
    mr.clear_library_cache()
    assert mr.library_cache_size() == 0


# --- concurrency -------------------------------------------------------------


def test_concurrent_compile_and_dispatch(restore_cache_limit):
    """The extension is built FREE_THREADED, so on a free-threaded interpreter
    these threads compile, dispatch and evict with no GIL between them."""
    mr.clear_library_cache()
    mr.set_library_cache_limit(8)  # small enough to force eviction under contention
    errors: list[BaseException] = []
    counter = iter(range(10_000))
    lock = threading.Lock()

    def worker():
        try:
            for _ in range(25):
                with lock:
                    i = next(counter)
                kernel = mr.Kernel(_counting_source(i), "f")
                buffer = mr.Buffer.zeros([1])
                mr.run(kernel, grid=1, buffers=[buffer])
                assert buffer.to_numpy()[0] == float(i)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors


# --- compile options ---------------------------------------------------------

# Sums 1.0 plus n copies of 2^-24. Each addend is exactly half an ulp of 1.0, so
# naive FP32 accumulation rounds every one away under round-half-to-even and the
# result stays exactly 1.0; the Kahan compensation term is the only thing that
# recovers them.
_KAHAN_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void kahan(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]]) {
    float s = 1.0f, c = 0.0f;
    const float x = 1.0f / 16777216.0f;   // 2^-24
    for (uint i = 0; i < n; ++i) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    out[0] = s;
}
"""

_KAHAN_N = 1_000_000
_KAHAN_EXACT = 1.0 + _KAHAN_N * 2.0**-24


def _run_kahan(math_mode) -> float:
    out = mr.Buffer.zeros([1])
    kernel = mr.Kernel(_KAHAN_SOURCE, "kahan", math_mode=math_mode)
    mr.run(kernel, grid=1, buffers=[out], scalars=[np.uint32(_KAHAN_N)])
    return float(out.to_numpy()[0])


def test_fast_math_deletes_compensated_summation():
    """Not a bug being tolerated -- a documented consequence. Fast math permits
    reassociation, under which `c = (t - s) - y` folds to zero and the
    compensation is optimized out with no diagnostic. This is why math_mode
    exists at all, and why the runtime passing nullptr compile options was a
    trap for anyone writing error-free transformations."""
    assert _run_kahan(mr.MathMode.FAST) == 1.0


def test_safe_math_preserves_compensated_summation():
    assert _run_kahan(mr.MathMode.SAFE) == pytest.approx(_KAHAN_EXACT, rel=1e-6)


def test_relaxed_math_also_reassociates():
    assert _run_kahan(mr.MathMode.RELAXED) == 1.0


def test_math_mode_members_carry_their_string_values():
    """MathMode is a StrEnum, so members compare equal to the spellings Metal's
    own documentation uses and survive a round-trip through config or JSON."""
    assert mr.MathMode.SAFE == "safe"
    assert mr.MathMode("safe") is mr.MathMode.SAFE
    assert _run_kahan(mr.MathMode("safe")) == pytest.approx(_KAHAN_EXACT, rel=1e-6)


def test_unknown_math_mode_is_rejected():
    with pytest.raises(ValueError, match="quux"):
        mr.MathMode("quux")


def test_math_mode_is_part_of_the_cache_key(restore_cache_limit):
    """The same source under two math modes is two libraries, not whichever
    compilation happened to land first."""
    mr.clear_library_cache()
    mr.Kernel(_KAHAN_SOURCE, "kahan", math_mode=mr.MathMode.FAST)
    assert mr.library_cache_size() == 1
    mr.Kernel(_KAHAN_SOURCE, "kahan", math_mode=mr.MathMode.SAFE)
    assert mr.library_cache_size() == 2
    mr.Kernel(_KAHAN_SOURCE, "kahan", math_mode=mr.MathMode.FAST)
    assert mr.library_cache_size() == 2  # a hit, not a third entry


_DEFINE_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void scaled(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = float(tid) * SCALE;
}
"""


def test_preprocessor_defines_specialize_one_source():
    for scale in (2.0, 10.0):
        buffer = mr.Buffer.zeros([8])
        kernel = mr.Kernel(_DEFINE_SOURCE, "scaled", defines={"SCALE": str(scale)})
        mr.run(kernel, grid=8, buffers=[buffer])
        assert np.array_equal(buffer.to_numpy(), np.arange(8, dtype=np.float32) * scale)


def test_defines_are_part_of_the_cache_key(restore_cache_limit):
    mr.clear_library_cache()
    mr.Kernel(_DEFINE_SOURCE, "scaled", defines={"SCALE": "2.0"})
    mr.Kernel(_DEFINE_SOURCE, "scaled", defines={"SCALE": "3.0"})
    assert mr.library_cache_size() == 2


def test_kernel_reports_its_compile_options():
    kernel = mr.Kernel(
        _DEFINE_SOURCE, "scaled", math_mode=mr.MathMode.SAFE, defines={"SCALE": "1.0"}
    )
    assert kernel.math_mode == mr.MathMode.SAFE
    assert kernel.defines == {"SCALE": "1.0"}


def test_missing_define_is_a_compile_error():
    with pytest.raises(mr.CompileError):
        mr.Kernel(_DEFINE_SOURCE, "scaled")  # SCALE never defined


# --- readback aliasing -------------------------------------------------------


def test_to_numpy_is_a_live_view_of_shared_memory():
    """Unified memory means readback is a view, not a copy: a dispatch after
    to_numpy() is visible through the array already handed out."""
    buffer = mr.Buffer.zeros([8])
    view = buffer.to_numpy()
    mr.run(mr.Kernel(_FILL_TID_SOURCE, "fill_tid"), grid=8, buffers=[buffer])
    assert np.array_equal(view, np.arange(8, dtype=np.float32))


def test_view_keeps_its_buffer_alive():
    view = mr.Buffer(np.arange(4, dtype=np.float32)).to_numpy()
    assert np.array_equal(view, np.arange(4, dtype=np.float32))
