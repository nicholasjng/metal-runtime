import numpy as np
import pytest

import metal_runtime as mr

ADD_ONE_BODY = """\
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0f;
}
"""


def test_assemble_orders_fragments_and_inserts_line_markers():
    src = mr.assemble(
        mr.Fragment("first", "// one"),
        mr.Fragment("second", "// two"),
    )
    assert src == '#line 1 "first"\n// one\n#line 1 "second"\n// two\n'


def test_assemble_escapes_labels():
    src = mr.assemble(mr.Fragment('we"ird\\label', "// x"))
    assert src.startswith('#line 1 "we\\"ird\\\\label"\n')


def test_default_includes_are_metal_stdlib():
    assert mr.DEFAULT_INCLUDES == mr.Fragment("includes", "#include <metal_stdlib>")


def test_kernel_compiles_and_runs_with_default_includes():
    kernel = mr.Kernel(
        mr.assemble(mr.DEFAULT_INCLUDES, mr.Fragment("body", ADD_ONE_BODY)), "add_one"
    )
    buf = mr.Buffer(np.zeros(8, dtype=np.float32))
    mr.run(kernel, grid=(8, 1, 1), buffers=[buf])
    np.testing.assert_array_equal(buf.to_numpy(), np.ones(8, dtype=np.float32))


def test_compile_error_is_attributed_to_the_failing_fragment():
    src = mr.assemble(
        mr.DEFAULT_INCLUDES,
        mr.Fragment("helpers", "float broken(float x) { return y; }"),
        mr.Fragment("body", ADD_ONE_BODY),
    )
    with pytest.raises(mr.CompileError, match=r"helpers:1:"):
        mr.Kernel(src, "add_one")
