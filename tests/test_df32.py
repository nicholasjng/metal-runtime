import operator
from collections.abc import Callable

import numpy as np
import pytest

import metal_runtime as mr
from metal_runtime import MathMode, df32

ArrayOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]
PairTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

N = 8192

KERNEL_TEMPLATE = """
kernel void k_{name}(device const float *a   [[buffer(0)]],
                     device const float *b   [[buffer(1)]],
                     device df32*        out [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid], b[tid]);
}}
"""


def _random_pairs(n, rng, *, spread=20, scale=20):
    """Random float32 pairs whose exponents differ by at most `spread` bits.

    Bounding the spread keeps the exact sum representable in float64, so the
    reference the tests compare against is itself exact. `scale` stays well
    inside float32's exponent range, so no denormals and no overflow.
    """
    exp = rng.integers(-scale, scale + 1, size=n)
    delta = rng.integers(0, spread + 1, size=n)
    mant_a = rng.uniform(1.0, 2.0, size=n)
    mant_b = rng.uniform(1.0, 2.0, size=n)
    sign_a = rng.choice([-1.0, 1.0], size=n)
    sign_b = rng.choice([-1.0, 1.0], size=n)
    a = np.ldexp(sign_a * mant_a, exp).astype(np.float32)
    b = np.ldexp(sign_b * mant_b, exp - delta).astype(np.float32)
    return a, b


def _by_descending_magnitude(a, b):
    """Reorder each pair so |a| >= |b|, as quick_two_sum requires."""
    swap = np.abs(a) < np.abs(b)
    return np.where(swap, b, a), np.where(swap, a, b)


def _as_is(a, b):
    return a, b


def _inputs(input_gen=_as_is, seed=42):
      return input_gen(*_random_pairs(N, np.random.default_rng(seed)))


def _limbs(name, a, b, math_mode):
    """Run one EFT kernel over `a`, `b`; return the (hi, lo) limbs widened to float64."""
    source = df32.PRELUDE + KERNEL_TEMPLATE.format(name=name)
    kernel = mr.Kernel(source, f"k_{name}", math_mode=math_mode)
    out = mr.Buffer.zeros([len(a), 2], "float32")
    mr.run(kernel, grid=len(a), buffers=[mr.Buffer(a), mr.Buffer(b), out])
    limbs = out.to_numpy().astype(np.float64)
    return limbs[:, 0], limbs[:, 1]


EFTS = [
    pytest.param(
        "quick_two_sum", operator.add, _by_descending_magnitude, id="quick_two_sum"
    ),
    pytest.param("two_sum", operator.add, _as_is, id="two_sum"),
    pytest.param("two_prod", operator.mul, _as_is, id="two_prod"),
]
SUMS = [p for p in EFTS if p.values[0] != "two_prod"]


@pytest.mark.parametrize(("kernel_name", "ref_op", "input_gen"), EFTS)
def test_safe_math_exact_eft(
    kernel_name: str, ref_op: ArrayOperator, input_gen: PairTransform
):
    a, b = _inputs(input_gen)
    ref = ref_op(a.astype(np.float64), b.astype(np.float64))

    hi, lo = _limbs(kernel_name, a, b, MathMode.SAFE)
    assert np.count_nonzero(lo) > N // 2
    assert np.array_equal(hi + lo, ref)


@pytest.mark.parametrize(("kernel_name", "ref_op", "input_gen"), SUMS)
def test_fast_math_destroys_compensation(
    kernel_name: str, ref_op: ArrayOperator, input_gen: PairTransform
):
    a, b = _inputs(input_gen)
    ref = ref_op(a.astype(np.float64), b.astype(np.float64))
    hi, lo = _limbs(kernel_name, a, b, MathMode.FAST)
    assert not np.array_equal(hi + lo, ref)


def test_two_prod_survives_fast_math():
    a, b = _inputs()
    ref = a.astype(np.float64) * b.astype(np.float64)
    hi, lo = _limbs("two_prod", a, b, MathMode.FAST)
    assert np.count_nonzero(lo) > N // 2
    assert np.array_equal(hi + lo, ref)
