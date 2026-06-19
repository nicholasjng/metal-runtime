# float32x2 (double-single) arithmetic in MSL

Working plan for the df32 arc. Tick items as you go; `.tutor/progress.md` tracks
which step is live.

## Context

Apple GPUs have no FP64 hardware. The only way to get more than 24 bits of
mantissa without collapsing performance is **double-single**: represent a value
as an unevaluated sum `hi + lo` of two float32s, giving ~48 mantissa bits (vs
FP64's 53) while keeping float32's exponent range. Costs ~10–20× on ALU for
add/mul; for bandwidth-bound kernels the real figure is closer to 2–3%, because
the extra arithmetic hides under memory latency.

**No C++ changes.** A hi/lo pair is ordinary float32 storage, which the runtime
already handles. The work is MSL text, host-side split/join in numpy, and a test
suite that proves the arithmetic does what it claims.

`math_mode` is the enabling piece: these routines are algebraically no-ops, so
reassociation is free to delete them. The exception is anything whose error term
goes through `fma()` — an intrinsic call, not an expression tree, and so out of
reach of the optimizer. Assume destruction under `FAST` unless measured otherwise.

Intended outcome: a validated `df32` prelude a Pallas→MSL generator can prepend
to kernels needing extra precision, plus tests that catch silent degradation to
plain float32.

## Ground rules

Tutor mode, balanced autonomy. **This plan deliberately omits algorithm bodies.**
Each step names signatures, invariants, and an exact definition of done; the
arithmetic is yours. Say `hint` for a nudge, `review` for feedback, or `spoil it`
on any individual step.

Exception: `quick_two_sum`, `two_sum`, and `two_prod` were written out earlier,
so Step 1 treats them as given and makes the *harness* the exercise.

## Layout

This ships. It lives in the package, not the test folder:

```
src/metal_runtime/
  df32.metal     # the MSL prelude text
  df32.py        # loads the prelude; host-side split/join
tests/
  test_df32.py   # all tests, importing from metal_runtime.df32
```

Two consequences of shipping it rather than keeping it in `tests/`:

- **It is public API.** Names are a commitment, `py.typed` means downstream type
  checkers see your annotations, and `uvx ty check` already covers `src`.
- **It resolves the open question** in `.tutor/progress.md` about whether a
  numerics prelude belongs here at all. ROADMAP.md says nothing upstream of
  "compile this MSL text and run it" lives in this repo, and a prelude sits right
  on that line. The decision is made; ROADMAP should get a sentence saying so.

The edit loop is unaffected: scikit-build-core's editable install puts `src/` on
`sys.path` directly, so new `.py` files are importable with no rebuild.

---

## Step 0 — Placement and packaging

Cheap, but do it first so every later step lands in its final home.

- [x] `src/metal_runtime/df32.metal` — empty for now, or a comment header.
- [x] `src/metal_runtime/df32.py` — loads it via
      `importlib.resources.files("metal_runtime").joinpath("df32.metal")`
      `.read_text(encoding="utf-8")` into a module-level `PRELUDE`. Pass the
      encoding explicitly: `Path.read_text` defaults to the *locale* encoding
      until PEP 686 lands in 3.15, and this is shipped code reading a shipped
      data file.
- [x] Verify the `.metal` lands in a built wheel:
      `uv build --wheel && unzip -l dist/*.whl | grep df32`.
      `wheel.packages = ["src/metal_runtime"]` should copy it, but confirm rather
      than assume — a missing data file fails only at import time, in someone
      else's environment.
- [x] Decide whether `df32` is re-exported from `__init__.py` or stays an
      explicit `from metal_runtime import df32` import. Prefer the latter: it
      keeps the top-level namespace about the runtime. Decided: stays explicit,
      `__init__.py` does not import it.

**Done when:** `uv run python -c "from metal_runtime import df32; print(len(df32.PRELUDE))"`
works and the file is in the wheel.

**Watch for — name it `.metal`, not `.msl`.** `identify` tags `.metal` (Apple's
own extension) as `metal`, and the clang-format hook's default `types_or` already
includes that tag; `.msl` matches no tag at all and silently falls out of every
filter. Narrowing `types_or` to `[c++, c]` drops the prelude just as effectively —
it needs `[c++, c, metal]`. Adding a `files:` regex does *not* rescue an unmatched
type: `files` and `types_or` are ANDed, so it can only narrow. clang-format itself
handles MSL fine and leaves `[[thread_position_in_grid]]` alone.

If the packaging check turns into a fight, fall back to a triple-quoted string
constant in `df32.py`. You lose syntax highlighting and gain nothing else; it is a
five-minute reversal in either direction.

## Step 1 — EFT exactness harness

Prove the three error-free transformations reconstruct their inputs exactly, and
that `FAST` destroys them.

- [ ] Put `quick_two_sum`, `two_sum`, `two_prod` in `df32.metal`. Write the error
      term in `two_prod` as an explicit `fma()`. Not because `SAFE` refuses to
      contract `a*b - p` — measured, it contracts it and the result is exact
      either way — but because `fma()` is the only form that also survives `FAST`.
      Written as a plain expression, `FAST` reassociates `a*b - a*b` to zero.
- [ ] One kernel per transformation; two float32 output buffers (`hi`, `lo`), or
      one `device float2*`. Compile with `mr.MathMode.SAFE`.
- [ ] Feed `quick_two_sum` inputs satisfying `|a| >= |b|` — arranging that for
      random inputs is the small puzzle.
- [ ] Positive test: `float64(hi) + float64(lo) == float64(a) + float64(b)`,
      exact `==`, over ~10k random pairs. Same shape for `two_prod` against `*`.
- [ ] Negative control: recompile the same source with `mr.MathMode.FAST` and
      assert the identity **fails for at least some inputs**. Not every input —
      when `a+b` is exact, `lo` is legitimately zero and both agree.
- [ ] The control applies to the **two sums only**. `two_prod` survives `FAST`
      (measured — see the devlog): its error term is an `fma()` intrinsic, not an
      expression tree, so reassociation has nothing to fold. Assert that as its
      own positive property rather than filing it as an exception.

**Done when:** both tests pass, and dropping `math_mode=SAFE` makes the positive
test fail.

**Watch for — test sharpness.** The comparison is *sound* for any input (both
sides are the same mathematical value, so they round identically), but it only
*detects* breakage when the exact sum fits in float64's 53 bits. That needs the
operands' exponent spread ≲ 29 bits. Draw random inputs from a bounded exponent
band so the test can actually see a wrong `lo`. `two_prod` has no such caveat: the
exact product of two float32s needs ≤ 48 bits and always fits. Keep inf/nan and
denormals out for now — Step 5 is where adversarial inputs belong.

## Step 2 — Host-side split/join

Now public API, so these get docstrings and annotations.

- [ ] `split(x: NDArray[np.float64]) -> NDArray[np.float32]` producing interleaved
      `(hi, lo)` pairs, shape `(..., 2)`. `float2` in MSL maps onto this directly
      and is 8-byte aligned, so it coalesces well.
- [ ] `join(pairs: NDArray[np.float32]) -> NDArray[np.float64]`, the inverse.
- [ ] Guard inputs outside float32's exponent range — they become `inf` in `hi`
      and the pair is meaningless. Raise, don't warn.
- [ ] Round-trip test: `|join(split(x)) - x| <= 2**-47 * |x|`.

**Watch for — this round-trip is lossy by design.** df32 carries ~48 mantissa bits
against float64's 53, so `join(split(x)) == x` is false in general. Asserting exact
equality here is the most likely way to waste an hour. Exact round-trip holds only
for values whose significand fits in 48 bits; that makes a good separate test case.

## Step 3 — `df_add`

- [ ] Signature: `df32 df_add(df32 a, df32 b)`.
- [ ] Compose from `two_sum` on the hi limbs and the lo limbs, then renormalize.
- [ ] Invariant the result must satisfy: `|lo| <= 0.5 * ulp(hi)` (non-overlapping
      limbs).
- [ ] Test: max relative error over ~10k random pairs vs a float64 reference.

**Done when:** relative error ≤ 2^-46.

**Watch for.** Read the error figure diagnostically: ~2^-24 means you have plain
float32 and something was optimized away or a limb is being dropped; ~2^-47 means
the accurate variant; in between means the "sloppy" variant, a legitimate
speed/accuracy trade but it should be a decision, not an accident. The
renormalization step is where accurate and sloppy diverge, and skipping it violates
non-overlap in a way that only shows up once operations are chained — so also test
`df_add` applied repeatedly, not just once.

## Step 4 — `df_mul`

- [ ] Signature: `df32 df_mul(df32 a, df32 b)`.
- [ ] Built on `two_prod` for the leading term, plus the cross terms.
- [ ] Same non-overlap invariant on the result.
- [ ] Test: max relative error over ~10k random pairs vs float64.

**Done when:** relative error ≤ 2^-46.

**Watch for.** One of the four product terms is negligible relative to the others
and is conventionally dropped — work out which, and confirm dropping it doesn't
move your measured error bound. Ordering matters when you fold the cross terms in:
adding them in decreasing magnitude preserves more bits.

## Step 5 — Accuracy suite

- [ ] Adversarial inputs: near-cancellation (`a ≈ -b`), wide exponent spreads,
      values near float32's range limits, and zero/signed-zero.
- [ ] A chained-operation test — accumulate a long sum in df32 and compare against
      float64. This is the test that catches a broken non-overlap invariant.
- [ ] Assert a specific ulp bound per operation rather than `np.allclose`, so a
      regression to float32 fails loudly instead of passing at default tolerance.
- [ ] Parametrize over `MathMode` so each routine's behaviour under `FAST` stays
      pinned as the suite grows. Note this is **per routine, not global**: whatever
      is built purely from `+`/`*` expression arithmetic is destroyed, whatever
      routes its error term through `fma()` is not. Step 1 established the split
      for the EFTs; `df_add` and `df_mul` need it measured, not assumed.

**Done when:** the suite fails if any single limb is dropped from `df_add` or
`df_mul`.

## Step 6 — Make the math-mode requirement unmissable

This step exists only because the prelude now ships. `CompileOptions.math_mode`
defaults to `FAST`, so the obvious first thing a user writes —

```python
mr.Kernel(df32.PRELUDE + my_source)
```

— compiles, runs, and returns silently wrong answers at float32 precision. That is
the worst possible failure mode for a numerics library, and it is entirely on us.

- [ ] Check whether Metal's compiler defines `__FAST_MATH__` under
      `MathModeFast`/`Relaxed` (it is clang-derived, so it probably does; verify,
      don't assume). If it does, an `#if defined(__FAST_MATH__)` / `#error` guard
      at the top of `df32.metal` converts silent wrong answers into a compile error
      naming the fix. One of the highest-value lines in the whole arc.
- [ ] Regardless: expose the requirement in Python. Either a
      `df32.compile_options(**overrides) -> mr.CompileOptions` helper that
      pins `math_mode=SAFE`, or a `df32.kernel(source, ...)` wrapper. A helper is
      cheaper to document than a paragraph nobody reads.
- [ ] Test that the guard fires: compiling `PRELUDE` under `FAST` should raise
      `mr.CompileError`, and the message should be actionable.

**Done when:** the wrong thing is hard to do by accident.

## Step 7 — Stretch: `df_div`, `df_sqrt`

Newton–Raphson refinement from a float32 seed. Meaningfully harder than add/mul —
convergence and the correction term both need care. Skip transcendentals entirely;
they are a project, not a step.

---

## Verification

No rebuild is needed at any point — there are no C++ changes, and `src/` is on
`sys.path` via the editable install, so new `.py`/`.metal` files are picked up live.

```
uv run pytest tests/test_df32.py -v      # per-step, as you go
uv run pytest tests/ -q                  # full suite, 69 existing + new
uvx prek run --all-files
uvx ty check                             # now covers df32.py -- it's in src/
```

If you do touch `src/*.cpp`, `uv sync --reinstall-package metal-runtime` first.

Sanity check the whole premise before Step 3 — this should print a nonzero `lo`,
and zero under `FAST`:

```
uv run python -c "..."   # your Step 1 kernel, run directly
```

## Per-step ritual

- [ ] Tick the step in `.tutor/progress.md` (`[>]` in progress, `[x]` done).
- [ ] Append a 3–6 line entry to `notes/devlog.md`: **Done** (what changed),
      **Why** (its role in the arc), optional **Note** (the one gotcha worth
      remembering). No code dumps.
