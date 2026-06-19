"""Exercises libmetal_runtime_c through ctypes, the same way an XLA FFI
handler would: only the C ABI in c_api.h, no Python/GIL in the calls.
"""

import ctypes
import mmap
import os
import threading

import numpy as np
import pytest

from metal_runtime import c_api

_ADD_ONE_SOURCE = b"""
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0f;
}
"""


class MRLaunchDesc(ctypes.Structure):
    _fields_ = [
        ("pipeline", ctypes.c_void_p),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("buffer_offsets", ctypes.POINTER(ctypes.c_size_t)),
        ("buffer_count", ctypes.c_size_t),
        ("scalars", ctypes.POINTER(ctypes.c_void_p)),
        ("scalar_sizes", ctypes.POINTER(ctypes.c_size_t)),
        ("scalar_count", ctypes.c_size_t),
        ("threadgroup_memory", ctypes.POINTER(ctypes.c_size_t)),
        ("threadgroup_memory_count", ctypes.c_size_t),
        ("grid_x", ctypes.c_size_t),
        ("grid_y", ctypes.c_size_t),
        ("grid_z", ctypes.c_size_t),
        ("threadgroup_x", ctypes.c_size_t),
        ("threadgroup_y", ctypes.c_size_t),
        ("threadgroup_z", ctypes.c_size_t),
    ]


@pytest.fixture(scope="module")
def lib():
    path = os.path.join(c_api.library_dir(), "libmetal_runtime_c.dylib")
    handle = ctypes.CDLL(path)

    handle.mr_free_error_message.argtypes = [ctypes.c_char_p]

    handle.mr_compile_library.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.mr_compile_library.restype = ctypes.c_int

    handle.mr_get_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.mr_get_pipeline.restype = ctypes.c_int

    handle.mr_wrap_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.mr_wrap_buffer.restype = ctypes.c_int

    handle.mr_buffer_flush_to.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.mr_buffer_flush_to.restype = ctypes.c_int

    handle.mr_dispatch.argtypes = [
        ctypes.POINTER(MRLaunchDesc),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    handle.mr_dispatch.restype = ctypes.c_int

    handle.mr_release_buffer.argtypes = [ctypes.c_void_p]
    handle.mr_release_pipeline.argtypes = [ctypes.c_void_p]
    handle.mr_release_library.argtypes = [ctypes.c_void_p]

    return handle


# MRStatus values from c_api.h.
MR_OK = 0
MR_ERROR_COMPILE = 2
MR_ERROR_FUNCTION_NOT_FOUND = 3

# MRMathMode values from c_api.h.
MR_MATH_MODE_FAST = 2


def _compile(lib, source: bytes, math_mode: int = MR_MATH_MODE_FAST):
    err = ctypes.c_char_p()
    library = ctypes.c_void_p()
    status = lib.mr_compile_library(
        source, len(source), math_mode, ctypes.byref(library), ctypes.byref(err)
    )
    return status, library, err


def _get_pipeline(lib, library, function_name: bytes):
    err = ctypes.c_char_p()
    pipeline = ctypes.c_void_p()
    status = lib.mr_get_pipeline(
        library, function_name, ctypes.byref(pipeline), ctypes.byref(err)
    )
    return status, pipeline, err


def _dispatch_add_one(lib, pipeline, buffer, n: int):
    buffers_arr = (ctypes.c_void_p * 1)(buffer)
    offsets_arr = (ctypes.c_size_t * 1)(0)
    desc = MRLaunchDesc()
    desc.pipeline = pipeline
    desc.buffers = buffers_arr
    desc.buffer_offsets = offsets_arr
    desc.buffer_count = 1
    desc.scalar_count = 0
    desc.threadgroup_memory_count = 0
    desc.grid_x, desc.grid_y, desc.grid_z = n, 1, 1
    desc.threadgroup_x, desc.threadgroup_y, desc.threadgroup_z = n, 1, 1
    err = ctypes.c_char_p()
    status = lib.mr_dispatch(ctypes.byref(desc), ctypes.byref(err))
    return status, err


def test_compile_and_dispatch_round_trip_with_a_misaligned_buffer(lib):
    """A plain numpy allocation isn't page-aligned: exercises whichever
    wrap path mr_wrap_buffer picks for it (page-rounded zero-copy or, if
    that range isn't safely mappable, the owned-copy fallback); either
    way mr_buffer_flush_to must leave `array` correct."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == MR_OK, err.value

    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == MR_OK, err.value

    array = np.arange(16, dtype=np.float32)
    ptr = array.ctypes.data_as(ctypes.c_void_p)

    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(
        ptr, array.nbytes, ctypes.byref(buffer), ctypes.byref(err)
    )
    assert status == MR_OK, err.value

    status, err = _dispatch_add_one(lib, pipeline, buffer, 16)
    assert status == MR_OK, err.value

    err = ctypes.c_char_p()
    status = lib.mr_buffer_flush_to(buffer, ctypes.byref(err))
    assert status == MR_OK, err.value

    assert np.array_equal(array, np.arange(16, dtype=np.float32) + 1.0)

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_dispatch_writes_are_visible_without_flush_for_a_page_rounded_buffer(lib):
    """Proves the page-rounded wrap is actually zero-copy, not merely
    error-free: dispatches into a buffer with a realistic (64-byte,
    XLA-style) but non-page-aligned offset inside a page-aligned mmap
    region, then reads the array back *without* calling
    mr_buffer_flush_to. Only true memory aliasing makes that visible."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == MR_OK, err.value

    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == MR_OK, err.value

    region = mmap.mmap(-1, mmap.PAGESIZE)
    base = ctypes.addressof(ctypes.c_char.from_buffer(region))
    assert base % mmap.PAGESIZE == 0
    offset = 64  # SIMD-width aligned, deliberately not page-aligned
    n = 16
    array = np.frombuffer(region, dtype=np.float32, count=n, offset=offset)
    array[:] = np.arange(n, dtype=np.float32)
    ptr = ctypes.c_void_p(base + offset)

    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(
        ptr, array.nbytes, ctypes.byref(buffer), ctypes.byref(err)
    )
    assert status == MR_OK, err.value

    status, err = _dispatch_add_one(lib, pipeline, buffer, n)
    assert status == MR_OK, err.value

    # Deliberately no mr_buffer_flush_to call here.
    assert np.array_equal(array, np.arange(n, dtype=np.float32) + 1.0)

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_compile_and_dispatch_round_trip_with_a_page_aligned_buffer(lib):
    """An anonymous mmap is page-aligned by construction: exercises the
    zero-copy wrap path."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == MR_OK, err.value

    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == MR_OK, err.value

    size = mmap.PAGESIZE
    region = mmap.mmap(-1, size)
    array = np.frombuffer(region, dtype=np.float32, count=16)
    array[:] = np.arange(16, dtype=np.float32)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    assert addr % mmap.PAGESIZE == 0

    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(
        ctypes.c_void_p(addr), size, ctypes.byref(buffer), ctypes.byref(err)
    )
    assert status == MR_OK, err.value

    status, err = _dispatch_add_one(lib, pipeline, buffer, 16)
    assert status == MR_OK, err.value

    err = ctypes.c_char_p()
    status = lib.mr_buffer_flush_to(buffer, ctypes.byref(err))
    assert status == MR_OK, err.value

    assert np.array_equal(array, np.arange(16, dtype=np.float32) + 1.0)

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_invalid_msl_returns_compile_error_status(lib):
    bad_source = b"this is not valid msl {{{"
    status, _, err = _compile(lib, bad_source)
    assert status == MR_ERROR_COMPILE
    assert err.value
    lib.mr_free_error_message(err)


def test_missing_function_returns_function_not_found_status(lib):
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == MR_OK, err.value

    status, _, err = _get_pipeline(lib, library, b"no_such_kernel")
    assert status == MR_ERROR_FUNCTION_NOT_FOUND
    assert err.value
    lib.mr_free_error_message(err)
    lib.mr_release_library(library)


def test_repeated_compile_and_release_does_not_crash(lib):
    for _ in range(50):
        status, library, err = _compile(lib, _ADD_ONE_SOURCE)
        assert status == MR_OK, err.value
        status, pipeline, err = _get_pipeline(lib, library, b"add_one")
        assert status == MR_OK, err.value
        lib.mr_release_pipeline(pipeline)
        lib.mr_release_library(library)


def test_concurrent_dispatch_against_one_shared_pipeline_does_not_corrupt(lib):
    """XLA's CPU backend runs FFI handlers on its own compute thread pool
    (xla/ffi/api/c_api.h), with no trait to opt out of concurrent calls, so
    a real handler backing mr_dispatch has to survive many threads racing
    mr_get_pipeline on one shared MRLibrary and dispatching through the
    resulting MRPipeline at once. Each thread wraps and dispatches into its
    own buffer; correctness of the readback is what actually proves no
    shared cache/state got corrupted, not just the absence of a crash."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == MR_OK, err.value

    n_threads = 24
    n_iters = 20
    barrier = threading.Barrier(n_threads)
    errors: list[tuple[int, str]] = []
    errors_lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for it in range(n_iters):
                status, pipeline, err = _get_pipeline(lib, library, b"add_one")
                assert status == MR_OK, err.value

                n = 11 + tid
                array = np.arange(n, dtype=np.float32)
                ptr = array.ctypes.data_as(ctypes.c_void_p)

                err = ctypes.c_char_p()
                buffer = ctypes.c_void_p()
                status = lib.mr_wrap_buffer(
                    ptr, array.nbytes, ctypes.byref(buffer), ctypes.byref(err)
                )
                assert status == MR_OK, err.value

                status, err = _dispatch_add_one(lib, pipeline, buffer, n)
                assert status == MR_OK, err.value

                err = ctypes.c_char_p()
                status = lib.mr_buffer_flush_to(buffer, ctypes.byref(err))
                assert status == MR_OK, err.value

                expect = np.arange(n, dtype=np.float32) + 1.0
                if not np.array_equal(array, expect):
                    raise AssertionError(
                        f"tid={tid} it={it}: got {array[:5]}, want {expect[:5]}"
                    )

                lib.mr_release_buffer(buffer)
                lib.mr_release_pipeline(pipeline)
        except Exception as e:  # noqa: BLE001 - collected and re-raised on the main thread
            with errors_lock:
                errors.append((tid, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lib.mr_release_library(library)

    assert not errors, errors
