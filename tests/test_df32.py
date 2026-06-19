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


def _random_float64(n, rng, *, lo_exp=-100, hi_exp=100):
    """Random float64s with magnitudes spanning 2**lo_exp .. 2**(hi_exp + 1).

    The exponent window is a parameter so that the subnormal boundary is
    something a test asks for rather than stumbles into. The defaults sit well
    inside float32's *normal* range (2**-126 .. 2**127), so `split` sees neither
    a subnormal nor an overflow unless a test says so.
    """
    mant = rng.uniform(1.0, 2.0, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    exp = rng.integers(lo_exp, hi_exp + 1, size=n)
    return np.ldexp(sign * mant, exp)


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


def _representable_pairs(n, rng, *, lo_exp=-100, hi_exp=100):
    """Normalised (hi, lo) float32 pairs, plus the exact float64 they sum to.

    Return `(hi, lo, x)` with `x == float64(hi) + float64(lo)` *exactly*, so
    `split(x)` is obliged to give back `(hi, lo)` bit for bit.

    |lo| needs bounding from both sides, for unrelated reasons:

    * Upper. At exactly 0.5*ulp(hi) the sum is a round-half-even tie, and at a
      power of two the gap *below* hi is only half the gap above -- so a negative
      lo near 0.5*ulp rounds onto the neighbour and fl32(hi + lo) is not hi.
    * Lower. lo carries a full 24-bit significand wherever it sits, so the
      smaller it is relative to ulp(hi), the wider the span from hi's leading bit
      to lo's trailing bit. Past 53 bits the float64 sum stops being exact and
      this generator's premise quietly fails.

    Drawing |lo|/ulp(hi) from a narrow band satisfies both at once.
    """
    hi = _random_float64(n, rng, lo_exp=lo_exp, hi_exp=hi_exp).astype(np.float32)
    ulp = np.abs(np.spacing(hi))
    sign = rng.choice([-1, 1], size=n)
    lo = (sign * ulp * rng.uniform(0.12, 0.24, size=n)).astype(np.float32)
    x = hi.astype(np.float64) + lo.astype(np.float64)
    return (hi, lo, x)


_IDENTITY_SOURCE = (
    df32.PRELUDE
    + """
kernel void df32_identity(device const df32* src [[buffer(0)]],
                          device df32* dst [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    dst[tid] = src[tid];
}
"""
)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_round_trip_within_bound():
    """join(split(x)) recovers x to ~48 bits, over normal float32 magnitudes.

    The bound is relative, not absolute. Note this holds only away from the
    subnormal range -- keep the exponent window inside the defaults.
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    rt = df32.join(df32.split(x))
    assert np.all(np.abs(rt - x) / np.abs(x) <= 2**-47)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_exact_for_representable_values():
    """A value that already IS a non-overlapping float32 pair must split back
    into exactly that pair -- no tolerance, bit for bit.

    This is the test that pins `split` and the Step 1 EFTs as consistent with
    each other, rather than each being self-consistently wrong.
    """
    hi, lo, x = _representable_pairs(N, np.random.default_rng(seed=42))
    assert np.all(df32.split(x) == np.stack([hi, lo], axis=-1))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_limbs_do_not_overlap():
    """|lo| <= 0.5 * ulp(hi) -- the invariant df_add and df_mul will consume.

    `np.spacing` gives the ulp but is *signed*, so it is negative for negative
    hi. The bound is tight in practice (worst observed ratio ~0.999998), so this
    is a real constraint rather than a slack one.
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    pairs = df32.split(x)
    hi, lo = pairs[..., 0], pairs[..., 1]
    assert np.all(np.abs(lo) <= 0.5 * np.abs(np.spacing(hi)))


@pytest.mark.parametrize("shape", [(0,), (7,), (3, 4), (2, 3, 5)])
def test_split_shape_dtype_and_layout(shape):
    """split maps (...) -> (..., 2) as C-contiguous float32, and join inverts
    the shape. Contiguity is what lets `mr.Buffer` take the result without a copy.
    """
    x = _random_float64(np.prod(shape), np.random.default_rng(seed=42)).reshape(shape)
    pairs = df32.split(x)
    assert pairs.flags.c_contiguous
    assert pairs.dtype == np.float32
    assert pairs.shape == (*shape, 2)
    rt = df32.join(pairs)
    assert rt.shape == shape
    assert np.all(np.abs(rt - x) <= 2**-47 * np.abs(x))


def test_split_accepts_the_boundary_value():
    """float32's largest finite value must be ACCEPTED, not rejected.

    A guard written with >= instead of > passes every other test in this file.
    """
    f32max = np.finfo(np.float32).max
    x = np.array([f32max, -f32max], dtype=np.float64)
    pairs = df32.split(x)
    assert pairs[0, 0] == np.float32(f32max)
    assert pairs[0, 1] == 0.0


def test_split_rejects_out_of_range_anywhere_in_the_array():
    """A finite value above float32's range raises ValueError.

    Put the offending value at a non-zero index of a multi-element array: a guard
    that collapses the array to a single truth value passes the one-element case.
    """
    x = np.array([0.1, 1e40], dtype=np.float64)
    with pytest.raises(ValueError, match="out of range for np.float32"):
        df32.split(x)


def test_join_rejects_non_pairs():
    """join requires a trailing dimension of exactly 2. Without the guard a
    (4, 3) array silently yields (4,), using columns 0 and 1 and dropping the third.
    """
    x = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="must be an input of shape"):
        df32.join(x)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_signed_zero_survives():
    """-0.0 must round-trip as -0.0.

    `0.0 == -0.0` is True and `np.array_equal([0.0], [-0.0])` is True, so an
    equality assertion cannot see a lost sign. Use `np.signbit`.
    """
    x = np.array([-0.0])
    rt = df32.join(df32.split(x))
    assert np.signbit(x.item()) == np.signbit(rt.item())


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_non_finite_passes_through_with_zero_lo(value):
    """inf and nan are representable in float32, so split must not poison them.

    hi carries the value and lo must be exactly zero. `np.array_equal` reports
    nan != nan, so the nan case needs `np.isnan`. The filterwarnings mark is
    load-bearing here: computing the residual as inf - inf raises a
    RuntimeWarning, which this turns into a failure.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(("value", "worst_rel"), [(1e-40, 1e-5), (1e-44, 1e-1)])
def test_subnormals_degrade_but_do_not_crash(value, worst_rel):
    """Below float32's smallest normal the 2**-47 claim does not hold.

    Assert the *limitation*: the error exceeds 2**-47 but stays under
    `worst_rel`. The deeper into the subnormal range, the fewer significand bits
    survive -- 1e-40 keeps roughly 18 of them, 1e-44 about two.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_idempotent():
    """split(join(split(x))) == split(x), bit for bit.

    Once a value is representable as a df32 pair, a further round trip must not
    move it. This is the property chained df_add/df_mul leans on.
    """
    raise NotImplementedError


def test_gpu_round_trip_proves_the_memory_layout():
    """(..., 2) float32 binds against `device df32*` with no reinterpretation.

    Everything above tests numpy arithmetic; only this tests the claim that the
    two layouts coincide. Upload split output, run `df32_identity`, read it back,
    and require the bytes to be unchanged. Keep the element count a multiple of
    256 so the dispatch does not round the grid up past the buffer.
    """
    raise NotImplementedError
