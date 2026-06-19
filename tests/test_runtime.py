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


def test_device_name_is_nonempty():
    assert mr.device_name()


def test_buffer_roundtrip():
    array = np.arange(16, dtype=np.float32)
    buffer = mr.Buffer(array)
    assert np.array_equal(buffer.to_numpy(), array)


def test_dispatch_add_one_kernel():
    array = np.arange(16, dtype=np.float32)
    buffer = mr.Buffer(array)
    kernel = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    mr.run(kernel, 16, 16, [buffer])

    assert np.allclose(buffer.to_numpy(), array + 1.0)


def test_zeros_allocates_without_upload():
    buffer = mr.Buffer.zeros([4, 3])
    assert np.array_equal(buffer.to_numpy(), np.zeros((4, 3), dtype=np.float32))


def test_zeros_buffer_dispatches_as_kernel_output():
    buffer = mr.Buffer.zeros([8])
    kernel = mr.Kernel(_FILL_TID_SOURCE, "fill_tid")

    mr.run(kernel, 8, 8, [buffer])

    assert np.array_equal(buffer.to_numpy(), np.arange(8, dtype=np.float32))


def test_invalid_msl_raises_compile_error():
    with pytest.raises(mr.CompileError):
        mr.Kernel("this is not valid msl {{{", "nope")


def test_library_cache_reuses_compiled_kernel():
    """Same MSL source string, two Kernel objects: dispatching both should still
    work (exercises MetalRuntime's library cache without asserting object identity,
    which is an internal implementation detail)."""
    array = np.zeros(8, dtype=np.float32)
    buffer_a = mr.Buffer(array)
    buffer_b = mr.Buffer(array)

    kernel_a = mr.Kernel(_ADD_ONE_SOURCE, "add_one")
    kernel_b = mr.Kernel(_ADD_ONE_SOURCE, "add_one")

    mr.run(kernel_a, 8, 8, [buffer_a])
    mr.run(kernel_b, 8, 8, [buffer_b])

    assert np.allclose(buffer_a.to_numpy(), array + 1.0)
    assert np.allclose(buffer_b.to_numpy(), array + 1.0)
