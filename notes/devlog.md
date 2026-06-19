# metal-runtime devlog

Lean per-card notes (what landed and why), written as each tutor card
completes. Newest at the bottom. For status checkboxes see `.tutor/progress.md`;
for the plan see `ROADMAP.md`.

## 1a-i: Add a live command queue   (2026-06-19)

**Done:** `metal_probe` now also creates an `MTL::CommandQueue` off the device via
`device->newCommandQueue()`, confirms it's non-null, prints it, and releases both
queue and device.
**Why:** the device is *what* runs work; the command queue is *how* work is
submitted to it, and every later kernel dispatch (card 1d) goes through a queue.
This is the smallest step proving the device→queue handoff before wrapping it
in RAII.
**Note:** the `if (!queue)` early-return leaks the device (returns before
`device->release()`). Harmless here since the process exits, but left as-is on
purpose: that fragility is the motivation for the RAII class in 1a-ii.

## 1a-ii: RAII class (device + queue)   (2026-06-19)

**Done:** added `MetalRuntime` (`runtime.h`/`runtime.cpp`): acquires device +
queue in the ctor, releases both (null-guarded) in the dtor, copy ops `=delete`d,
move ctor steals-then-nulls. `main` now lives entirely through it with no manual
`release()`. metal-cpp's `*_PRIVATE_IMPLEMENTATION` macros were split out into a
single impl TU `metal.cpp` (+ `metal.h` = pure includes, safe to include
anywhere); both added to the `metal_runtime` static lib in CMake.
**Why:** RAII makes cleanup run on *every* exit path, killing the 1a-i device
leak; `MetalRuntime` is the runtime core that buffers (1b), the MSL cache (1c) and
dispatch (1d) all hang off.
**Note:** `runtime.h` forward-declares `MTL::Device`/`CommandQueue` so the heavy
metal-cpp headers stay out of consumers, the "header firewall." Public API still
returns raw `MTL::*` pointers, so callers need `metal.h`; 1e's `device_name()`-style
helpers are the lever to hide that later.

## 1a-iii: Process-wide singleton accessor   (2026-06-19)

**Done:** `MetalRuntime& runtime()` returns a function-local `static MetalRuntime`
(Meyers singleton): lazily constructed on first call, one instance for the
process. `main` uses it and confirms identity via address comparison.
**Why:** the roadmap wants one device/queue "held for the process"; this is the
seam later step-1 components reach through without threading a `MetalRuntime&`
everywhere.
**Note:** the static's dtor releases the Metal objects at process exit, safe here
(`release` is just a refcount decrement). If other statics ever hold Metal handles
and depend on `runtime()`, destruction order is unspecified (static-destruction
fiasco); the leaky-singleton (`new`, never deleted) is the escape hatch then.

## 1b-1e: Buffers, MSL library cache, dispatch, nanobind surface   (2026-07-24, Claude)

**Done:** implemented as a batch, per the split agreed for the pagode/Pallas→MSL
follow-up (Claude does the runtime plumbing; the Pallas-jaxpr-walk + MSL codegen
part, cards `2a'`-`5'` below, is yours). `Buffer` (`buffer.h/.cpp`) wraps
`MTL::Buffer` with `ResourceStorageModeShared`, same RAII, move-only style as
`Library`/`ComputePipeline` (`MetalRuntime` itself is non-movable: it's only ever
constructed once, as the `runtime()` singleton). `Library` (`library.h/.cpp`)
compiles MSL source text at runtime via `newLibrary`; `MetalRuntime::library_for()`
caches instances keyed directly by the source string (references/pointers into
`std::unordered_map` survive rehashing, so a `Library*` handed out once stays
valid). `ComputePipeline` + `dispatch()`
(`dispatch.h/.cpp`) build a command buffer + compute encoder, bind buffers, pick
a threadgroup size clamped to the pipeline's own max, and block until done.
`bindings.cpp` wraps all of it for nanobind: `device_name()`, `Buffer(np.ndarray)`
/ `.to_numpy()` (upload copies in; readback is a zero-copy `nb::ndarray` view kept
alive by the owning `PyBuffer`, via `nb::find(*this)`), `Kernel(msl_source,
function_name)`, `run(kernel, grid_size, threadgroup_size, buffers)`. Verified
against real Metal hardware (M1 Pro): buffer round-trip and a compiled+dispatched
"add one" kernel both check out (`tests/test_runtime.py`).
**Why:** these four are needed regardless of which frontend eventually feeds
MSL text in (arbitrary-jaxpr per the original ROADMAP, or the narrower
Pallas-kernel-specific track this repo is actually aimed at right now): pure
Metal-API wrapping, not the part worth learning by hand.
**Note (autorelease):** Cocoa objects created via `NS::String::string(...)` or
error out-params are only reclaimed by an enclosing `NS::AutoreleasePool`; with
none active they silently leak for the process's lifetime instead of crashing.
Added `AutoreleaseScope` (RAII pool, in `metal.h`) around each such call site
(`Library`'s ctor/`function()`, `ComputePipeline`'s ctor, `dispatch()`) so a long
Python session dispatching many kernels doesn't leak per call.
**Note (build):** `_core` (the nanobind module) is gated behind
`METAL_RUNTIME_HAVE_NANOBIND` in `CMakeLists.txt`: a plain `cmake -S . -B build`
against system Python (no nanobind installed there) still configures and builds
`metal_runtime` + `metal_probe`; only `uv sync` (scikit-build-core, pointed at the
project `.venv` where nanobind is a build dependency) builds `_core`. Renamed the
bound method `numpy` → `to_numpy`: a same-named method shadows the `numpy` module
inside the generated `.pyi` stub's own class scope under static analysis (`ty`
flagged it), even though it wouldn't collide at runtime.

## Audit follow-up: correctness, dtypes, scalars, batching   (2026-07-27, Claude)

**Done:** acted on an audit of the whole runtime. Five real defects:
`MetalRuntime`'s library cache was mutated with no lock while the module is
built `FREE_THREADED` (a `std::mutex` now guards it, with compilation kept
*outside* the lock so threads building different kernels don't serialize);
`dispatch()` never checked `MTL::CommandBuffer::status()`, so a faulted command
buffer returned exactly like a successful one (`DispatchError` now); the ctor
never checked `CreateSystemDefaultDevice()` (`NoDeviceError` now); the upload
`nb::ndarray` had no `device::cpu` constraint, so a GPU-resident DLPack tensor
would have been `memcpy`d from a device pointer; and a zero grid was accepted.

Quirks: zero-element buffers used to fail as if out of memory (`newBuffer(0)`
returns nil — the allocation is rounded up to a byte, `size()` still reports 0);
`float64` is rejected with a message pointing at `.astype(numpy.float32)`
instead of being silently narrowed; an oversized `threadgroup` raises instead of
being silently clamped, which was quietly giving wrong answers to any kernel
indexing threadgroup memory; a missing entry point raises
`FunctionNotFoundError`, a subclass of `CompileError`, rather than a bare
`RuntimeError`.

Features, all aimed at what Pallas codegen will actually need: every
Metal-addressable dtype rather than f32 only, inline `setBytes` scalars (a
kernel taking an element count no longer costs a buffer allocation), 1-3D
grids, threadgroup memory, `Batch` (several launches, one command buffer, one
round-trip), a bounded LRU library cache, and device/kernel introspection so
callers can size a threadgroup instead of guessing. Plus `py.typed` (without it
PEP 561 says downstream type checkers ignore the shipped `_core.pyi` entirely),
a CI workflow, and 52 tests where there were 7.

**Why:** the runtime is about to acquire a real consumer, and the two
silent-corruption classes — a torn cache under free threading, a swallowed GPU
fault — are exactly the kind that surface as "the numbers are wrong" three
layers up in the codegen rather than as an error here.

**Note (cache lifetime):** `library_for()` now returns `shared_ptr<Library>`,
not `Library&`. With eviction in the picture a reference into the map can be
invalidated by another thread's insert, and `PyKernel` holds its library for as
long as it lives, so a pipeline can outlive its source's cache entry.

**Note (unverified):** the `DispatchError` path for a GPU-side fault is
reachable and correct by construction but was never observed firing. An M1 Pro
tolerates a several-GB out-of-bounds write from a kernel without faulting the
command buffer, and the other way to force the error — a runaway kernel hitting
the watchdog — isn't worth risking on a development machine. The batch
"already waited on" path exercises the same C++ → Python translation.

## Compile options on Kernel (math mode + defines)   (2026-07-27, Claude)

**Done:** `Library` takes a `CompileOptions` (math mode + preprocessor
defines) instead of passing `nullptr` to `newLibrary`, and the library cache
keys on `(options, source)` rather than source alone — length-prefixed so the
options blob can't be confused with the start of the source text. Exposed as
`Kernel(src, fn, math_mode=..., defines=...)`. `MathMode` is bound with
`nb::enum_<...>(m, "MathMode", nb::is_str())`, which produces a real
`enum.StrEnum`: typed and discoverable in the stub, while `math_mode="safe"`
still coerces, and `kernel.math_mode == "safe"` still holds on readback.

**Why:** Metal defaults to fast math, which permits reassociation, and passing
`nullptr` compile options meant callers had no way off it. Measured on an M1
Pro, summing 1.0 plus 10^6 copies of 2^-24 (each exactly half an ulp, so naive
FP32 rounds every one away):

| math mode | result | |
|---|---|---|
| `FAST` (old default) | `1.0` | compensation deleted |
| `RELAXED` | `1.0` | compensation deleted |
| `SAFE` | `1.0596046` | correct, exact is `1.0596046` |

So the Kahan term wasn't degraded, it was compiled out entirely — silently, and
for every kernel this runtime has ever dispatched. That matters directly for
the double-single (`float32x2`) work being scoped next, which is error-free
transformations end to end and is simply incorrect under anything but `SAFE`.
Three tests pin all three modes.

**Note (Python floor -> 3.11):** `nb::is_str()` makes stubgen emit
`class MathMode(enum.StrEnum)` unconditionally, and `enum.StrEnum` is 3.11+.
Under the old `requires-python = ">=3.10"` that cascaded into five spurious `ty`
errors: with the base class unresolved the members degrade to plain `str`
literals, so `MathMode.FAST` types as `Literal["fast"]` and stops matching its
own annotation. Bumped the floor to 3.11 rather than patching the stub.

Worth an upstream report: this is *not* a value-rendering bug — the emitted
stub is correct — but nanobind's runtime and its stubgen disagree about which
Pythons they support. `src/nb_enum.cpp` has an explicit `PY_VERSION_HEX <
0x030B0000` fallback that builds an equivalent `(str, Enum)` class on 3.10,
while `stubgen.py` emits the `enum.StrEnum` base with no `sys.version_info`
guard. So on 3.10 a StrEnum works at runtime but generates a stub referencing a
symbol that doesn't exist. A guarded base class, or matching the runtime's
fallback, would close the gap.

**Note (default):** left at `FAST` to match Metal, so no existing kernel
silently gets slower. The tradeoff is that the unsafe-for-compensation mode is
the one you get by not thinking about it — hence the README warning rather than
a quieter doc line.

## Dtype reinterpretation on Buffer (ml_dtypes interop)   (2026-07-27, Claude)

**Done:** `Buffer(array, dtype=...)` and `to_numpy(dtype=...)` both take an
optional dtype that *relabels* the bytes instead of converting them — NumPy's
`.view` semantics, with element width the only invariant enforced. So an
ml_dtypes `bfloat16` array crosses as its `uint16` view and is labelled
`bfloat16` on arrival, making `device bfloat*` on the kernel side honest.
Verified against real hardware: `bfloat` compiles and runs on an M1 Pro, and
`[1, 2.5, -3.75, 100] * 2.0bf` round-trips exactly.

**Why:** ml_dtypes is how the JAX ecosystem spells `bfloat16`, and this runtime
had no way to accept one. The blocker is upstream, not here — NumPy exports
these dtypes through *neither* DLPack (`BufferError: DLPack only supports
signed/unsigned integers, float and complex dtypes`) nor the buffer protocol
(`ValueError: cannot include dtype 'E' in a buffer`), so nanobind never gets a
pointer and the cast fails before any of our code runs. NumPy's error mentions
"dtypes registered by third-party packages", which suggests an opt-in path
ml_dtypes 0.5.4 hasn't taken; this may fix itself upstream, at which point the
override becomes a convenience rather than the only way in.

**Note (deferred — `to_numpy()` on a bfloat16 buffer):** it still raises, now
pointing at `to_numpy(dtype='uint16')`. Whether it should instead return a real
`ml_dtypes.bfloat16` array when that package is importable is **undecided**.
Arguments against: it would make the return type depend on whether an optional
package happens to be installed, and nanobind can't construct a dtype NumPy
doesn't know, so the array would have to be built on the Python side — meaning
a Python shim over `_core`, which the package currently doesn't have. Arguments
for: it is what a JAX-adjacent caller actually wants, and the uint16 dance is
noise. Revisit once there's a consumer with an opinion.

**Note (test dep):** `ml-dtypes` added to the `test` group only. The interop
tests `importorskip` it, so a checkout without it still passes.

**Note (ruff double-pin, fixed):** the audit flagged ruff being pinned twice —
`>=0.15.12` in the `zed` group and a separate rev in `.pre-commit-config.yaml` —
and it finally bit here. Both format the generated `_core.pyi`: the venv's ruff
via the CMake stub post-step, prek's via the hook. With the venv on 0.15.12 and
prek on 0.16.0 the stub oscillated between two layouts, so every build-then-lint
cycle reported a spurious diff. Bumped the group floor to `>=0.16.0` to match,
with a comment on both. The durable fix is the `repo: local` hook shape (invoke
the locked ruff instead of a second pinned copy), still not done.

**Note (build):** the nanobind gating described in the card above is unchanged;
CI now covers both halves of it, `uv sync` + pytest on three interpreters
(including free-threaded 3.14t) and a bare `cmake` build of `metal_probe` with
no nanobind in sight.

## df32 Step 1: `two_prod` survives fast math   (2026-07-28, Claude)

**Done:** measured, not assumed — under `MathMode.FAST` on this toolchain (Metal
32023, M1 Pro), `quick_two_sum` and `two_sum` lose their error terms as expected,
but `two_prod` returns a `lo` that is still *exactly* correct. The EFT identity
`hi + lo == a * b` holds bit-for-bit at FAST.

**Why:** `two_prod`'s error term is `fma(a, b, -p)` — an intrinsic call with
defined single-rounding semantics, not an expression tree. Fast math reassociates
`+`/`*` arithmetic; it has nothing to fold in a builtin. The two sums are pure
expression arithmetic, so `(a - (s - bb)) + (b - bb)` collapses to zero.

**Note:** this corrects the "FAST destroys everything" framing the plan carried
in three places (now fixed in `notes/float32x2.md`). The real rule is per routine:
error terms routed through `fma()` are safe, error terms built from `+`/`-` are
not. `df_add` and `df_mul` inherit the question rather than the answer — measure
each. It also means `two_prod` alone is *not* a canary for a wrong `math_mode`,
so the SAFE requirement still needs the Step 6 guard to be discoverable.

**Note (test shape):** the FAST tests assert *inequality*, so they are insensitive
to a dropped limb by construction — sabotaging `lo` to zero makes them pass. All
detection duty sits on the SAFE test plus its `count_nonzero(lo) > N // 2`
sensitivity guard. Verified by sabotage: 4 of 6 tests fail, and the 2 survivors
are exactly the FAST pair.

**Note (contraction is not reassociation):** found by sabotage — replacing the
`fma()` with a plain `a * b - p` and measuring all four combinations:

| `two_prod` error term | mode | nonzero `lo` | exact |
|---|---|---|---|
| `fma(a, b, -p)` | safe | 100% | yes |
| `fma(a, b, -p)` | fast | 100% | yes |
| `a * b - p`     | safe | 100% | yes |
| `a * b - p`     | fast | 0%   | **no** |

So `SAFE` *does* contract `a * b - p` into an fma on its own — the two are
separate knobs. `mathMode` governs reassociation; FP contraction stays on
regardless (clang's `-ffp-contract=fast` default, within a statement). Under
`FAST` the stronger reassociation gets there first and folds `a*b - a*b` to zero,
so contraction never applies.

This corrects the reason given in the plan for writing `fma()` explicitly. It is
*not* that SAFE refuses to contract — it doesn't refuse. It is that the explicit
intrinsic is the only form that also survives FAST. The prelude was right; the
justification was wrong. Plan line fixed.

## df32 Step 1 done: EFT exactness harness   (2026-07-28, Claude + NJ)

**Done:** `src/metal_runtime/df32.metal` ships `quick_two_sum`, `two_sum` and
`two_prod`; `df32.py` exposes the text as `PRELUDE`. `tests/test_df32.py` proves
all three reconstruct their inputs exactly under `MathMode.SAFE` — bit-exact
`==` against a float64 reference over 8192 pairs — and that `FAST` deletes the
compensation in the two sums. Six tests, composed from one `_limbs()` helper and
a `pytest.param` table.

**Why:** everything later in the arc (`df_add`, `df_mul`) is built from these
three, so a silent degradation here would surface as a plausible-looking accuracy
figure four steps downstream rather than as a failure.

**Note (the test that makes the suite worth anything):** inputs are drawn with a
bounded exponent spread (≤ 20 bits) and guarded by
`assert np.count_nonzero(lo) > N // 2`. Without the band, `float64(a) + float64(b)`
stops being exactly representable and the comparison passes for a correct *and*
a broken kernel alike. Measured: at spread 20, per-element detection of a dropped
`lo` is 91%; at spread 80 it falls to 64%, at 150 to 35%. The suite stays sound at
any spread — both sides are the same real number and round identically — what
degrades is sensitivity.

**Note (verified by sabotage, not by passing):** four deliberate breakages, each
caught — dropping `lo` (4 of 6 fail), downgrading `SAFE` to `FAST` (2 fail),
replacing the `fma()` with a plain expression (1 fail), and flipping `FAST` back
to `SAFE` in the destruction test (2 fail). A green suite proves nothing here; the
`in`-against-`ParameterSet` bug earlier in this step passed lint, `ty`, and all
three tests while silently skipping the entire negative control.
