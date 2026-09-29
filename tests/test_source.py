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


def test_build_source_includes_metal_stdlib_by_default():
    src = mr.build_source(body="// body")
    assert src.startswith('#line 1 "includes"\n#include <metal_stdlib>\n')
    assert mr.DEFAULT_INCLUDES == ("metal_stdlib",)


def test_build_source_includes_override_replaces_default():
    src = mr.build_source(includes=("metal_tensor",), body="// body")
    assert "#include <metal_tensor>" in src
    assert "#include <metal_stdlib>" not in src


def test_build_source_empty_includes_omits_includes_fragment():
    src = mr.build_source(includes=(), body="// body")
    assert "#include" not in src
    assert src == '#line 1 "body"\n// body\n'


def test_build_source_fragment_order_is_includes_preludes_body():
    src = mr.build_source(
        includes=("metal_stdlib",),
        preludes=(mr.Fragment("helpers", "// helper"),),
        body="// body",
        body_label="my_kernel",
    )
    markers = [line for line in src.splitlines() if line.startswith("#line")]
    assert markers == [
        '#line 1 "includes"',
        '#line 1 "helpers"',
        '#line 1 "my_kernel"',
    ]


def test_kernel_compiles_and_runs_with_default_includes():
    kernel = mr.Kernel(mr.build_source(body=ADD_ONE_BODY), "add_one")
    buf = mr.Buffer(np.zeros(8, dtype=np.float32))
    mr.run(kernel, grid=(8, 1, 1), buffers=[buf])
    np.testing.assert_array_equal(buf.to_numpy(), np.ones(8, dtype=np.float32))


def test_compile_error_is_attributed_to_the_body_fragment():
    src = mr.build_source(
        body="kernel void f(device float* o [[buffer(0)]]) { o[0] = bogus; }",
    )
    with pytest.raises(mr.CompileError, match=r"body:1:"):
        mr.Kernel(src, "f")


def test_compile_error_is_attributed_to_a_prelude_fragment():
    src = mr.build_source(
        preludes=(mr.Fragment("helpers", "float broken(float x) { return y; }"),),
        body=ADD_ONE_BODY,
    )
    with pytest.raises(mr.CompileError, match=r"helpers:1:"):
        mr.Kernel(src, "add_one")
