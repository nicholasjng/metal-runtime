# Hand-rolling MLIR bindings for codegen from Python

A companion to [ROADMAP.md](ROADMAP.md). This is about the option of writing your
own Python bindings over MLIR — what that buys you, what it doesn't, and the
dialect-registration model that makes it confusing.

## First, the reframe that matters most

**Bindings ≠ no build.** Rolling your own MLIR bindings does *not* avoid building
MLIR — you're binding `libMLIRCAPI*.a`, which only exists after you've built MLIR
from source (the ~5.5 GB the PJRT experiment paid). So there are really only two
reasons to hand-roll bindings:

1. **Independence from JAX** — a codegen toolchain that doesn't import jaxlib.
2. **Your own dialect** — exposing a Metal/`GPUToMetal` dialect to Python.

If the goal is *lightweight*, the zero-build path is to reuse jaxlib's **already
built** bindings — they ship in the wheel:

```python
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo   # works today, no build
```

`jax.extend.mlir` is the blessed door to the same thing. So: **hand-roll bindings
only for reasons (1)/(2); otherwise use jaxlib's.**

## Two very different uses of "bindings"

Decide which you actually need — they need different (and differently sized) API
surfaces:

| Use | What you do | API surface |
|---|---|---|
| **Read** an incoming module | parse bytecode/text, walk ops, pull shapes/attrs | small (~15 calls) |
| **Build / transform** a module | construct ops, run passes, emit | larger (+pass manager, +types/attrs) |

For Metal **codegen**, the honest question is whether you build MLIR IR at all:

- If "codegen" = *walk a StableHLO/jaxpr graph and print MSL strings*, you only
  need the **Read** surface. You never construct an MLIR op. (This is what
  `mlir-metal`'s `stablehlo_lower` does, just in C++.)
- If "codegen" = *use MLIR's lowering infra* (StableHLO → Linalg → … → EmitC →
  MSL, with real passes), you need the **Build/transform** surface and probably a
  custom dialect. That's the heavyweight, "proper compiler" route.

Most lightweight Metal paths want the first. Keep that in mind below.

## The dialect-registration story

This is the part that's underexplained. MLIR separates three things:

1. **Context** (`MlirContext`) — owns all IR, types, attributes.
2. **Registry** (`MlirDialectRegistry`) — a *catalogue of dialect constructors*,
   not the dialects themselves.
3. **Dialect handle** (`MlirDialectHandle`) — a C token for one dialect,
   produced by a generated function `mlirGetDialectHandle__<namespace>__()`.

And two verbs that people conflate:

- **Register** = teach the context *how to construct* a dialect (put its
  constructor in the registry/context). Required before its ops/types/attrs can
  be created or parsed.
- **Load** = actually *instantiate* the dialect in the context. The parser
  lazy-loads registered dialects on demand; you can also eager-load.

The generated handle name is real and predictable. StableHLO declares:

```c
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Stablehlo, stablehlo);
// => MlirDialectHandle mlirGetDialectHandle__stablehlo__(void);
```

### The canonical sequence (C API)

```c
MlirContext ctx = mlirContextCreate();

// Option A: per-dialect, explicit (what you'd do for stablehlo + friends)
MlirDialectHandle h = mlirGetDialectHandle__stablehlo__();
mlirDialectHandleRegisterDialect(h, ctx);   // register the constructor
mlirDialectHandleLoadDialect(h, ctx);       // instantiate it now
// ...repeat for __chlo__, __vhlo__, __func__

// Option B: the whole upstream bundle at once
MlirDialectRegistry reg = mlirDialectRegistryCreate();
mlirRegisterAllDialects(reg);                       // upstream dialects only
mlirContextAppendDialectRegistry(ctx, reg);
mlirContextLoadAllAvailableDialects(ctx);           // eager-load everything
mlirDialectRegistryDestroy(reg);
```

`mlirRegisterAllDialects` covers *upstream* MLIR (func, arith, linalg, …) but
**not** StableHLO — that's out-of-tree, so you must add its handle yourself
(Option A). This is the same split we hit in C++.

### Unregistered dialects — and the bytecode trap

```c
mlirContextSetAllowUnregisteredDialects(ctx, true);
```

This lets the context hold ops whose dialect was never registered (stored
generically). It works for **textual** IR. It does **not** rescue **bytecode**
that uses a dialect's custom encoding — the reader needs the real dialect to
decode. This is exactly the `sdy` (Shardy) crash from the PJRT experiment:
`allowUnregisteredDialects` parsed past it in text but the bytecode reader still
failed with *"dialect 'sdy' does not implement the bytecode interface."* Lesson
for your bindings: **register every dialect that appears in the bytecode, or
strip it upstream** (we disabled Shardy in JAX).

### Building ops without per-op bindings

The thing that makes a thin binding viable: you build ops **generically by string
name** — no generated C++ op class, no per-op wrapper needed:

```c
MlirOperationState st = mlirOperationStateGet(
    mlirStringRefCreateFromCString("arith.addf"), loc);
mlirOperationStateAddOperands(&st, 2, operands);
mlirOperationStateAddResults(&st, 1, &f32);
mlirOperationStateAddAttributes(&st, 1, &attr);
MlirOperation op = mlirOperationCreate(&st);
```

The dialect must be **loaded** so the op is recognized/verified (or
allow-unregistered for throwaway ops). Types and attributes come cheaply from
strings too: `mlirTypeParseGet(ctx, "f32")`, `mlirAttributeParseGet(ctx,
"dense<2.0> : tensor<f32>")`. So a few dozen C API calls cover *all* dialects.

## The minimal surface to bind (nanobind)

For a **Read**-only codegen frontend, this is roughly the whole list:

- Context: `mlirContextCreate`, `mlirContextDestroy`,
  `mlirContextSetAllowUnregisteredDialects`
- Registration: `mlirGetDialectHandle__stablehlo__` / `__chlo__` / `__vhlo__` /
  `__func__`, `mlirDialectHandleRegisterDialect`, `mlirDialectHandleLoadDialect`
- Parse: `mlirModuleCreateParse` (handles text *and* bytecode by magic),
  `mlirModuleGetOperation`
- Walk: `mlirOperationGetNumResults`, `mlirOperationGetResult`,
  `mlirOperationGetNumOperands`, `mlirOperationGetOperand`,
  `mlirOperationGetName`, `mlirIdentifierStr`, region/block iteration
  (`mlirOperationGetFirstRegion`, `mlirRegionGetFirstBlock`,
  `mlirBlockGetFirstOperation`, `mlirOperationGetNextInBlock`)
- Inspect: `mlirValueGetType`, `mlirShapedTypeGetRank/GetDimSize`, attribute
  getters

A nanobind wrapper is small because `Mlir*` handles are just structs holding a
pointer — wrap each as an opaque Python object:

```cpp
#include <nanobind/nanobind.h>
#include "mlir-c/IR.h"
#include "stablehlo/integrations/c/StablehloDialect.h"
namespace nb = nanobind;

NB_MODULE(_mlir, m) {
  nb::class_<MlirContext>(m, "Context");          // opaque handle
  m.def("make_context", [] {
    MlirContext c = mlirContextCreate();
    MlirDialectHandle h = mlirGetDialectHandle__stablehlo__();
    mlirDialectHandleRegisterDialect(h, c);
    mlirDialectHandleLoadDialect(h, c);
    return c;
  });
  m.def("parse", [](MlirContext c, nb::bytes bc) {
    return mlirModuleCreateParse(
        c, mlirStringRefCreate((const char*)bc.data(), bc.size()));
  });
  // ...walk helpers returning Python lists of (name, shape, operands)
}
```

Link against the C API archives — for StableHLO you already built them:
`libStablehloCAPI.a`, `libChloCAPI.a`, `libVhloCAPI.a`, plus the MLIR CAPI libs
(`libMLIRCAPIIR.a`, …). The portable artifact JAX emits deserializes via the
StableHLO C API: `stablehloDeserializePortableArtifact(...)` (same function we
called from C++).

## Your own Metal dialect

If reason (2) is the driver — a real `GPUToMetal` dialect callable from Python —
registration costs more:

- Define the dialect in C++/TableGen, build it, and expose a C handle with the
  `MLIR_DECLARE/DEFINE_CAPI_DIALECT_REGISTRATION(Metal, metal)` macros so Python
  gets `mlirGetDialectHandle__metal__()`. Then it registers/loads exactly like
  StableHLO above.
- Only then can you `register → load → build metal.* ops` and run your lowering
  passes (bind `mlirPassManager*` for that).

The cheaper stand-in: skip the dialect, keep ops as **unregistered** (build by
name with allow-unregistered) — fine for prototyping, but you forfeit
verification, the bytecode round-trip, and pass infrastructure. For emitting MSL
you don't need a Metal dialect at all; MLIR's **EmitC** dialect already lowers to
C-like source if you want the in-tree route to MSL.

## Recommendation

- **Just want JAX → MSL, lightweight:** don't hand-roll. Use `jaxlib.mlir` /
  `jax.extend.mlir` (zero build) to walk StableHLO in Python, or skip MLIR and
  walk the jaxpr. Bind nothing.
- **Want a JAX-independent codegen tool:** hand-roll the small **Read** surface
  over the MLIR + StableHLO C API (you've already built the archives). ~40 calls,
  one nanobind TU.
- **Want real MLIR passes / your own Metal dialect:** build the dialect in C++,
  expose its C handle, bind the **Build/transform** surface + pass manager. This
  is the heavyweight route — worth it only once the lightweight path proves the
  idea.

In every case the dialect rule is the same: **register + load every dialect the
IR mentions before you parse or build it** — and remember bytecode won't tolerate
an unregistered dialect, only text will.
