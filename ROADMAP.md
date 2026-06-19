# metal-runtime: a lightweight Metal runtime

A small, LLVM-free C++ runtime over metal-cpp: compile MSL kernel source at
load time, move data through shared buffers, dispatch. Exposed to Python via
nanobind. This repo owns the runtime only; JAX/Pallas integration lives in a
sibling repo (see "Where this fits").

The JAX-integration plan (arbitrary-jaxpr walk, codegen, `@metal_jit`) that
used to live here moved out: it came from a PJRT-plugin angle (transparent
`jax.jit`), not the actual near-term target (a specific Pallas kernel), and it
only makes sense read alongside the code that consumes this runtime.

## The key insight

A full PJRT-plugin route needs a heavy MLIR/StableHLO build for one reason:
parsing the StableHLO bytecode `PJRT_Client_Compile` receives, in C++.
Everything else is already light: the op graph, the buffers, the GPU dispatch.

| Layer | Needs LLVM? |
|---|---|
| **Backend** (op graph → Apple GPU): metal-cpp buffers + MSL kernels | **No.** Emit MSL text, let Metal compile it at load time. |
| **Frontend** (JAX → op graph) | **Only if you parse bytecode in C++.** Do it in Python and it's free. |

Drop the "C++ must parse bytecode" constraint and the whole pipeline stays
light, regardless of frontend. This runtime only ever sees already-generated
MSL text.

## Steps, in order

### 0. metal-cpp via CMake (done)

`CMakeLists.txt` pulls Apple's official metal-cpp with `FetchContent` at
configure time. `metal_probe` prints the system GPU to prove the toolchain.

### 1. Runtime core (C++ / metal-cpp), exposed to Python (done)

- **Device / queue**: one `MTL::Device` + `MTL::CommandQueue`, held for the
  process.
- **Buffers**: `MTL::Buffer` with `ResourceStorageModeShared`, host and GPU
  share the bytes, zero-copy upload/readback for f32 arrays.
- **Library cache**: `MTL::Device::newLibrary(source, ...)` (the NVRTC
  equivalent) compiles MSL source strings at runtime, keyed by the source
  string. No offline `metal`/`metallib` toolchain.
- **Dispatch**: bind buffers, set threadgroup sizes, `commit` +
  `waitUntilCompleted`.

Bound to Python with nanobind: `device_name()`, `Buffer(np.ndarray)` /
`.to_numpy()`, `Kernel(msl_source, function_name)`, `run(kernel, grid_size,
threadgroup_size, buffers)`. The whole contract a consumer needs: generate MSL
text, hand it a `Kernel` and some `Buffer`s, get results back as NumPy.
Implemented in `src/buffer.*`, `src/library.*`, `src/dispatch.*`,
`src/bindings.cpp`; see `tests/test_runtime.py`.

## Where this fits

Nothing upstream of "compile this MSL text and run it" belongs here. The
Pallas-kernel-to-MSL codegen path, the JAX-side tracing, and the entry point
tying it to `Kernel`/`Buffer`/`run` live in the `pagode` repo (being renamed
`palladium`), see its own `ROADMAP.md`, reformulated against the actual
JAX/Pallas source (Pallas has no third-party backend extension point).

An arbitrary-JAX-program frontend is still a plausible direction if the
Pallas-specific track outgrows itself, but there's no concrete plan for it. It
would start from the same "emit MSL text, let Metal's compiler do the rest"
contract this repo already provides.

## Validate

Whatever consumes this runtime should diff its results against a CPU/JAX
reference on the same computation. `tests/test_runtime.py` does the narrower
version: round-trips a NumPy buffer, checks a dispatched kernel's output
against the expected arithmetic directly.
