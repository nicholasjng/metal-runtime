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
    with pytest.raises(ValueError, match="must be an array of shape"):
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
    x = np.array([value], dtype=np.float64)
    pair = df32.split(x)
    if np.isnan(value):
        assert np.isnan(pair[0, 0])
    else:
        assert pair[0, 0] == value
    assert pair[0, 1] == 0.0


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(("value", "worst_rel"), [(1e-40, 1e-5), (1e-44, 1e-1)])
def test_subnormals_degrade_but_do_not_crash(value, worst_rel):
    """Below float32's smallest normal the 2**-47 claim does not hold.

    Assert the *limitation*: the error exceeds 2**-47 but stays under
    `worst_rel`. The deeper into the subnormal range, the fewer significand bits
    survive -- 1e-40 keeps roughly 18 of them, 1e-44 about two.
    """
    x = np.array([value], dtype=np.float64)
    rt = df32.join(df32.split(x))
    rel_err = np.abs(rt - x) / np.abs(x)
    assert 2**-47 < rel_err[0] < worst_rel


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_split_is_idempotent():
    """split(join(split(x))) == split(x), bit for bit.

    Once a value is representable as a df32 pair, a further round trip must not
    move it. This is the property chained df_add/df_mul leans on.
    """
    x = _random_float64(N, np.random.default_rng(seed=42))
    assert np.array_equal(df32.split(df32.join(df32.split(x))), df32.split(x))


def test_gpu_round_trip_proves_the_memory_layout():
    """(..., 2) float32 binds against `device df32*` with no reinterpretation.

    Everything above tests numpy arithmetic; only this tests the claim that the
    two layouts coincide. Upload split output, run `df32_identity`, read it back,
    and require the bytes to be unchanged. Keep the element count a multiple of
    256 so the dispatch does not round the grid up past the buffer.
    """
    x = _random_float64(256, np.random.default_rng(seed=42))
    pairs = df32.split(x)

    kernel = mr.Kernel(_IDENTITY_SOURCE, "df32_identity")
    src = mr.Buffer(pairs)
    dst = mr.Buffer.zeros(pairs.shape, "float32")
    mr.run(kernel, grid=len(x), buffers=[src, dst])

    assert np.array_equal(dst.to_numpy(), pairs)


DF32_BINOP_TEMPLATE = """
kernel void k_{name}(device const df32* a   [[buffer(0)]],
                     device const df32* b   [[buffer(1)]],
                     device df32*       out [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {{
    out[tid] = {name}(a[tid], b[tid]);
}}
"""


def _df32_binop(name, a_pairs, b_pairs, math_mode):
    """Run a df32(df32, df32) -> df32 kernel; return output pairs widened to float64.

    Mirrors `_limbs()` from step 1, but the inputs and output are already df32
    pairs rather than raw float32s -- `df_add`/`df_mul` operate on `(hi, lo)`,
    not on scalars.
    """
    source = df32.PRELUDE + DF32_BINOP_TEMPLATE.format(name=name)
    kernel = mr.Kernel(source, f"k_{name}", math_mode=math_mode)
    out = mr.Buffer.zeros(a_pairs.shape, "float32")
    mr.run(
        kernel,
        grid=len(a_pairs),
        buffers=[mr.Buffer(a_pairs), mr.Buffer(b_pairs), out],
    )
    return out.to_numpy().astype(np.float64)


def _random_df32_operands(n, rng, **kwargs):
    """Two independent arrays of valid, non-overlapping df32 pairs, plus the
    float64 values each one exactly represents.

    Built from `_random_float64` + `split` rather than drawn as raw (hi, lo)
    floats, so every operand already satisfies the non-overlap invariant
    `test_limbs_do_not_overlap` pins -- df_add only needs to preserve that
    property, not establish it.
    """
    a64 = _random_float64(n, rng, **kwargs)
    b64 = _random_float64(n, rng, **kwargs)
    return df32.split(a64), df32.split(b64), a64, b64


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_matches_float64_reference_under_safe_math():
    """max relative error of df_add vs a float64 `+` reference must be <= 2**-46
    over ~10k random operand pairs, under MathMode.SAFE.

    This is the plan's stated "done when" bound for df_add -- looser than the
    2**-47 split/join bound because df_add's own arithmetic adds a little
    additional rounding on top of the operands' own representation error.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_output_limbs_do_not_overlap():
    """The (hi, lo) pair df_add returns must itself satisfy |lo| <= 0.5*ulp(hi) --
    the same invariant test_limbs_do_not_overlap pins for split.

    This is the property that separates a correctly renormalized df_add from a
    "sloppy" one that produces overlapping limbs -- both can look accurate on a
    single application, but only the renormalized one survives being chained
    (see the next test).
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_stays_accurate_when_chained():
    """Summing a long chain of df32 values with repeated df_add calls must stay
    within the same relative-error bound as a single df_add, compared against an
    equivalent float64 running sum.

    A df_add that skips renormalization can look fine applied once and still
    quietly violate non-overlap -- an error that only compounds, and so only
    shows up, once results get fed back in as operands. This is the test that
    would catch that.
    """
    raise NotImplementedError


def test_df_add_destroys_precision_under_fast_math():
    """Same comparison as the SAFE-math test above, but compiled under
    MathMode.FAST: the relative error should blow past the 2**-46 bound,
    mirroring step 1's `test_fast_math_destroys_compensation` for the raw EFTs.

    Whether this actually fails depends on how df_add's error term is written --
    an expression tree reassociates away under FAST the same way two_sum's did,
    an fma()-routed one might not. That's worth finding out, not assuming.
    """
    raise NotImplementedError


# --- Step 4 -- df_mul -------------------------------------------------------


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_matches_float64_reference_under_safe_math():
    """max relative error of df_mul vs a float64 `*` reference must be <= 2**-46
    over ~10k random operand pairs, under MathMode.SAFE.

    Same shape as df_add's exactness test, built on `two_prod` for the leading
    term plus the cross terms rather than `two_sum` on the his/los.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_output_limbs_do_not_overlap():
    """The (hi, lo) pair df_mul returns must itself satisfy |lo| <= 0.5*ulp(hi).

    Same invariant as df_add's, now on the composed two_prod + cross-term
    result -- worth checking independently, since dropping the wrong cross term
    (see the plan's note on which of the four is conventionally negligible)
    could pass the accuracy test above while still overlapping.
    """
    raise NotImplementedError


def test_df_mul_destroys_precision_under_fast_math():
    """Same comparison as the SAFE-math test above, but compiled under
    MathMode.FAST.

    Step 1 measured that two_prod's own error term survives FAST, because it is
    routed through fma() rather than an expression tree. df_mul's cross terms
    are extra additions on top of that -- whether *those* survive FAST is a
    separate question this test answers by measurement.
    """
    raise NotImplementedError


# --- Step 5 -- Accuracy suite ------------------------------------------------


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_near_cancellation_stays_accurate():
    """df_add(a, b) for b ~= -a (catastrophic cancellation in plain float32)
    must still land within the same relative-error bound as the general case.

    This is the adversarial input float32 arithmetic alone gets most visibly
    wrong -- the whole point of carrying a `lo` limb is that the compensation
    term survives where plain addition would cancel it away.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_and_df_mul_handle_wide_exponent_spread():
    """Operands whose exponents differ far more than step 1's bounded band
    (which existed only to keep the float64 *reference* exact) must still
    round-trip through df_add/df_mul within the accuracy bound.

    Step 1 bounded the spread so the test oracle stayed exact; that constraint
    doesn't apply here since df32 arithmetic's correctness doesn't depend on
    float64 exactly representing the answer, only on the relative-error
    comparison being valid.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_mul_near_float32_range_limits():
    """Operands close to float32's max magnitude must not silently overflow to
    inf inside df_mul's cross-term arithmetic, and values close to the smallest
    normal must not silently flush to zero.

    df_mul composes several float32 multiplications and additions per output
    element; each is a place range could be clipped that a single `a * b`
    reference wouldn't reveal.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_df_add_and_df_mul_preserve_signed_zero():
    """+0.0 and -0.0 operands must produce the sign IEEE 754 dictates for `+`
    and `*` respectively, through both df_add and df_mul.

    Direct analogue of `test_signed_zero_survives` from step 2, one level up:
    split/join already had a signed-zero bug once in this arc: worth checking
    it doesn't reappear at the arithmetic layer.
    """
    raise NotImplementedError


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_chained_df_add_matches_float64_running_sum():
    """Accumulating a long chain of df32 values via repeated df_add calls must
    stay within the accuracy bound compared to an equivalent float64 running
    sum -- not just the single-application bound already tested in step 3.

    This is the test the plan calls out as the one that would catch a broken
    non-overlap invariant: a df_add that skips renormalization can look
    accurate applied once and still drift once its own output is fed back in
    as the next operand.
    """
    raise NotImplementedError


@pytest.mark.parametrize("math_mode", list(mr.MathMode))
@pytest.mark.parametrize("routine", ["df_add", "df_mul"])
def test_math_mode_behavior_is_pinned_per_routine(routine, math_mode):
    """Whether df_add/df_mul survive a given MathMode is a property of how each
    routine's error term is written (expression tree vs fma()), not a suite-wide
    constant -- so pin it explicitly per (routine, math_mode) pair rather than
    assuming it follows step 1's EFTs.

    As df_div/df_sqrt (step 7, stretch) are added, extend this table rather than
    writing a new ad hoc FAST-math test per routine.
    """
    raise NotImplementedError


# --- Step 6 -- Make the math-mode requirement unmissable --------------------


def test_prelude_rejects_non_safe_math_at_compile_time():
    """Compiling df32.PRELUDE (plus any kernel using it) under MathMode.FAST or
    MathMode.RELAXED must raise mr.CompileError, with a message that names the
    fix (i.e. mentions SAFE math mode) rather than a bare compiler error.

    This is the step whose whole justification is that `mr.Kernel(df32.PRELUDE
    + my_source)` -- math_mode defaulting to FAST -- compiles, runs, and returns
    silently wrong answers otherwise. A compile-time guard converts that into a
    loud failure naming the fix.
    """
    raise NotImplementedError


def test_df32_safe_kernel_helper_pins_math_mode():
    """Whatever helper step 6 introduces for constructing a df32-safe kernel
    (e.g. `df32.compile_options(...)` or a `df32.kernel(source, ...)` wrapper)
    must produce SAFE math mode without the caller having to remember to pass
    it explicitly.

    Exact shape depends on which helper form gets picked; update this stub's
    body to match once that's decided.
    """
    raise NotImplementedError
