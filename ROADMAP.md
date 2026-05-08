# metal-runtime — a lightweight Metal runtime for JAX

Goal: run normal JAX programs on the Apple GPU through a **small, LLVM-free**
runtime built on metal-cpp, and plug it back into JAX.

This note records the steps in order, and where the BarraCUDA model fits. It is
the distillation of a companion experiment (`mlir-metal`) that already runs
`jax.jit(f)` on the Apple GPU through a full **PJRT C API plugin** — but at the
cost of building MLIR + StableHLO from source (~5.5 GB). That experiment taught
us exactly where the weight comes from, and how to avoid it here.

## The key insight

The heavy MLIR/StableHLO build in the PJRT route buys **one** thing: parsing the
StableHLO **bytecode** that `PJRT_Client_Compile` receives, *in C++*. Everything
else — the op graph, the buffers, the GPU dispatch — is already light. So:

| Layer | Needs LLVM? |
|---|---|
| **Backend** (op graph → Apple GPU): metal-cpp buffers + MSL kernels | **No.** Emit MSL text, let Metal compile it at load time. |
| **Frontend** (JAX → op graph): | **Only if you parse bytecode in C++.** Do it in Python and it's free. |

The whole weight is bought by the PJRT "C++ must parse bytecode" constraint. Drop
it — move the frontend into Python — and the entire pipeline is light.

## Steps, in order

### 0. metal-cpp via CMake (done)

`CMakeLists.txt` pulls Apple's official metal-cpp with `FetchContent` at
configure time (replacing `download_metal_cpp.py`). `metal_probe` prints the
system GPU to prove the toolchain. This is the foundation the runtime builds on.

### 1. Runtime core (C++ / metal-cpp), exposed to Python

A thin C++ runtime over metal-cpp:

- **Device / queue** — one `MTL::Device` + `MTL::CommandQueue`, held for the
  process.
- **Buffers** — `MTL::Buffer` with `ResourceStorageModeShared` so host and GPU
  share the bytes (zero-copy upload/readback for f32 arrays). Expose the host
  pointer to NumPy via the buffer protocol / DLPack.
- **Library cache** — compile MSL **source strings at runtime** with
  `MTL::Device::newLibrary(source, ...)` (the NVRTC equivalent), keyed by source
  hash. No offline `metal`/`metallib` toolchain.
- **Dispatch** — bind buffers, set threadgroup sizes, `commit` + `waitUntilCompleted`.

Bind it to Python with **nanobind** (small, fast, modern). The Python surface is
roughly: `device_name()`, `Buffer(np.ndarray) / .numpy()`, `Kernel(msl_source)`,
`run(kernel, grid, buffers)`. This is the "bring the runtime into a JAX project"
layer — once buffers speak DLPack/NumPy, JAX can hand arrays in and read results
back.

### 2. Python frontend — JAX program → op graph (no LLVM)

Get the program *without* touching bytecode in C++. Two routes, both zero-build:

- **jaxpr (lightest):** `jax.make_jaxpr(f)(*args)` → walk `.eqns`, map each
  primitive (`dot_general`, `transpose`, `mul`, `add`, `broadcast_in_dim`, …) to
  a runtime op. Pure Python; no MLIR at all.
- **StableHLO (closer to XLA):** `jax.extend.mlir` exposes jaxlib's **bundled**
  StableHLO Python bindings — walk the module in-process with **no build**. (The
  in-process MLIR that is unlinkable from C++ is fully usable from Python.) The
  companion repo's `stablehlo_compile.py` line-parser is already 80% of this.

Output a tiny op list (the companion repo's `.mprog` is a fine starting IR).

### 3. Codegen — op graph → MSL (the "GPUToMetal" step)

- **First:** one kernel per op (or lean on MPSGraph as a reference oracle).
- **Then:** fuse elementwise chains (`transpose * 2 + 1`) into a **single** MSL
  kernel emitted as text and compiled via step 1's library cache. Matmul stays a
  tiled MSL kernel (or MPS) feeding the fused tail.

This is the real "lightweight compiler": walk op list → print MSL → `newLibrary`.
No LLVM anywhere.

### 4. JAX integration — pick your transparency/weight tradeoff

| Approach | `jax.jit(f)` transparent? | LLVM build? | Effort |
|---|---|---|---|
| PJRT + built MLIR (companion repo) | ✅ | ✅ 5.5 GB | done, heavy |
| **`@metal_jit` jaxpr interpreter (here)** | ❌ custom decorator | ❌ | **low** |
| `jax.extend.mlir` StableHLO walk | ❌ custom call | ❌ | low–med |
| `jax.extend.ffi` custom call | partial (XLA-hosted) | ❌ | med |

Recommended for this repo: **`@metal_jit`** — `jax.make_jaxpr` → op list →
codegen → runtime. All code you own, zero build, and the natural home for your
own kernels. Keep the PJRT plugin as the "transparent but heavy" reference and as
a **correctness oracle**.

The `ffi` row is the interesting middle if you later want stock `jax.jit`: Metal
kernels as XLA custom calls, with XLA (CPU) doing fusion/scheduling and calling
out to Metal for leaf ops — but then XLA still hosts the graph, so it isn't
"Apple GPU end to end."

### 5. Validate

Diff every result against JAX on CPU (and/or the PJRT plugin) on the same
function. f32 matches to ~1e-6.

## Where BarraCUDA fits

BarraCUDA (`ext/BarraCUDA`) is the **existence proof and the architectural
template**, not a dependency on the JAX path:

- **Proof:** it compiles CUDA/HIP/Triton → AMD/NVIDIA/CPU **without LLVM**, doing
  its own instruction encoding. Crucially it ships an **Apple Metal backend**
  (`src/metal/`) that does exactly our step 3 — *"lowers to Metal Shading
  Language text; the Metal toolchain compiles that to AIR at load time."* That
  validates the whole MSL-text + runtime-compile approach (its emitter is ~830
  lines).
- **Template:** its structure — frontend → **BIR** (flat-array SSA, "no malloc,
  no hardware assumptions") → passes (cfold/dce/mem2reg/sroa) → backend emit — is
  a clean blueprint if our flat op-list outgrows itself and wants real
  optimization before MSL emission.
- **Not on the JAX path:** BarraCUDA's frontends are GPU-*source* languages
  (CUDA/HIP/Triton), not JAX/StableHLO, so we don't reuse its frontend or link
  it. Two ways it could still connect later: (a) borrow/port its Metal backend's
  MSL-emission tricks for step 3; (b) a *separate* experiment — run Triton
  kernels on Metal via BarraCUDA's Triton frontend + Metal backend, orthogonal to
  the JAX work.

Net: BarraCUDA tells us the lightweight target (MSL text, runtime-compiled) is
real and shows how to structure the mini-compiler; we supply the JAX→op-graph
frontend it doesn't have.
