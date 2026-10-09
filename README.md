# tiny-cuda-nn-musa

`tiny-cuda-nn-musa` is the Moore Threads MUSA maintenance line of
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), a library for hash-grid
encodings and compact neural networks used in real-time neural graphics.

## Source and version

- Upstream project: tiny-cuda-nn
- Inherited version: `v2.0`
- Upstream baseline: `2e757bbe781db59c4980d389d7dccbf5edc09669`
- Maintainer: Moore Threads

The MUSA line adds MUSA headers, kernels, and runtime-compilation integration
while retaining the upstream CUDA implementation, encodings, networks,
bindings, samples, and tests.

## Requirements and build

Use CMake 3.18 or newer, a C++ compiler, a supported PyTorch installation for
the bindings, and the MUSA SDK/compiler when selecting the MUSA backend. A
typical library build is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Use `CMakeLists_musa.txt` and the MUSA binding options supplied by the source
when targeting Moore Threads hardware. External framework and toolkit versions
are deployment choices.

## Layout and contribution

Core headers and sources are under `include/` and `src/`; MUSA counterparts,
bindings, samples, and tests retain the upstream organization. Keep MUSA and
CUDA paths separate and preserve third-party notices.

Moore Threads maintains this source line. See [LICENSE](LICENSE) for the
applicable declaration.
