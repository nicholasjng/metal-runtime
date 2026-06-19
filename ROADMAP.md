# metal-runtime — a lightweight Metal runtime

Goal: a **small, LLVM-free** C++ runtime over metal-cpp — compile MSL kernel
source at load time, move data in and out via shared buffers, dispatch — exposed
to Python via nanobind. This repo owns the runtime only; JAX/Pallas integration
that consumes it lives in a sibling repo (see "Where this fits" below).

This note used to also carry the JAX-integration plan (arbitrary-jaxpr walk,
codegen, `@metal_jit`). That's moved out: it came from a different angle (a PJRT
plugin build for transparent `jax.jit`) than the actual near-term target (a
Pallas-specific kernel), and it's work that's implemented in, and only makes
sense read alongside, the code that consumes this runtime — not this one.

## The key insight

The heavy MLIR/StableHLO build in a full PJRT-plugin route buys **one** thing:
parsing the StableHLO **bytecode** that `PJRT_Client_Compile` receives, *in
C++*. Everything else — the op graph, the buffers, the GPU dispatch — is
already light. So:

| Layer | Needs LLVM? |
|---|---|
| **Backend** (op graph → Apple GPU): metal-cpp buffers + MSL kernels | **No.** Emit MSL text, let Metal compile it at load time. |
| **Frontend** (JAX → op graph): | **Only if you parse bytecode in C++.** Do it in Python and it's free. |

The whole weight is bought by the PJRT "C++ must parse bytecode" constraint. This
repo drops it — the runtime only ever sees already-generated MSL text — so
whatever calls into it stays light too, regardless of which frontend it uses.

## Steps, in order

### 0. metal-cpp via CMake (done)

`CMakeLists.txt` pulls Apple's official metal-cpp with `FetchContent` at
configure time. `metal_probe` prints the system GPU to prove the toolchain.
This is the foundation the runtime builds on.

### 1. Runtime core (C++ / metal-cpp), exposed to Python (done)

A thin C++ runtime over metal-cpp:

- **Device / queue** — one `MTL::Device` + `MTL::CommandQueue`, held for the
  process.
- **Buffers** — `MTL::Buffer` with `ResourceStorageModeShared` so host and GPU
  share the bytes (zero-copy upload/readback for f32 arrays).
- **Library cache** — compile MSL **source strings at runtime** with
  `MTL::Device::newLibrary(source, ...)` (the NVRTC equivalent), keyed by the
  source string itself. No offline `metal`/`metallib` toolchain.
- **Dispatch** — bind buffers, set threadgroup sizes, `commit` + `waitUntilCompleted`.

Bound to Python with **nanobind**. The Python surface: `device_name()`,
`Buffer(np.ndarray)` / `.to_numpy()`, `Kernel(msl_source, function_name)`,
`run(kernel, grid_size, threadgroup_size, buffers)`. This is the entire contract
a consumer needs: generate MSL text, hand it a `Kernel` and some `Buffer`s, get
results back as NumPy. Implemented in `src/buffer.*`, `src/library.*`,
`src/dispatch.*`, `src/bindings.cpp`; see `tests/test_runtime.py`.

## Where this fits

Nothing upstream of "compile this MSL text and run it" belongs here. The
Pallas-kernel → MSL codegen path, the JAX-side tracing, and the user-facing
entry point that ties it to this runtime's `Kernel`/`Buffer`/`run` all live in
the `pagode` repo (being renamed `palladium`) — see its own `ROADMAP.md`. That
plan was reformulated from scratch against the actual JAX/Pallas source (Pallas
has no third-party backend extension point — verified, not assumed); read it
there rather than here.

An arbitrary-JAX-program frontend (not scoped to Pallas kernels) is still a
plausible future direction if the Pallas-specific track outgrows itself, but
there's no concrete plan for it right now — it would start from the same
"emit MSL text, let Metal's runtime compiler do the rest" contract this repo
already provides.

## Validate

Whatever consumes this runtime should diff its results against a CPU/JAX
reference on the same computation. This repo's own tests do the narrower
version of that: `tests/test_runtime.py` round-trips a NumPy buffer and checks
a dispatched kernel's output against the expected arithmetic directly.
