# metal-runtime

A small, LLVM-free Metal GPU runtime for Python, built on Apple's
[metal-cpp](https://developer.apple.com/metal/cpp/). Compiles MSL kernel source
at runtime (no offline `metal`/`metallib` toolchain), moves NumPy `float32`
arrays into shared GPU buffers with zero-copy readback, and dispatches — no MLIR,
no XLA.

```python
import numpy as np

import metal_runtime as mr

source = """
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0f;
}
"""

array = np.arange(16, dtype=np.float32)
buffer = mr.Buffer(array)
kernel = mr.Kernel(source, "add_one")

mr.run(kernel, grid_size=16, threadgroup_size=16, buffers=[buffer])

print(buffer.to_numpy())  # [1. 2. 3. ... 16.]
```

## Installation

With a `[tool.uv.sources]` entry in your `pyproject.toml`:

```toml
[tool.uv.sources]
metal-runtime = { git = "https://github.com/nicholasjng/metal-runtime" }
```

```console
$ uv add metal-runtime
```

Requires macOS with Apple Silicon (or another Metal-capable GPU) and Python
3.10+. The C++ extension is compiled via CMake + nanobind on install; there's no
pre-built wheel yet.

A PyPI release is planned for the future.

## Project direction

This repo's goal — running JAX programs on the Apple GPU without building
MLIR/StableHLO from source — is laid out in [ROADMAP.md](ROADMAP.md).

## License

This project is licensed under the Apache-2.0 license.
