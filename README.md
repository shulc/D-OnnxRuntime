# D-OnnxRuntime

D bindings + a thin **static-linkable** C ABI shim over
[ONNX Runtime](https://onnxruntime.ai). Load an `.onnx` model once and run it
per call on a batch of feature rows, getting one score per row back — the
common "rank / score N items with a small model" case (rankers, classifiers
reduced to a score, regressors).

The C boundary mentions no ONNX Runtime, C++ or STL types — plain C scalars and
caller-owned buffers — so it binds cleanly from D via `extern(C)` and the shim
itself links as one static archive you can fold into a distributable binary.
ONNX Runtime is fetched as a small prebuilt release (CPU, dynamic library).

## Layout

```
include/onnxrt.h              C API (extern "C") — the contract
csrc/onnxrt_ort.cpp           backend=ort   real ONNX Runtime (default)
csrc/onnxrt_mock.c            backend=mock  deterministic fixture, no deps
csrc/onnxrt_stub.c            backend=stub  honest "not built", no deps
source/onnxrt/c.d             D extern(C) bindings, 1:1 with the header
source/onnxrt/backend.d       RAII OnnxSession + rank() (argmax + softmax)
examples/score_smoke.d        smoke test: create -> score -> destroy
tools/make_test_model.py      generates a tiny model.onnx (no deps)
cmake/FetchOnnxRuntime.cmake  downloads + verifies the ONNX Runtime release
CMakeLists.txt                builds build/libonnxrt.a
doc/DESIGN.md                 ABI, lifetime, error policy, fetch & link notes
```

## Build

```sh
dub build
```

`dub`'s `preBuildCommands-posix` runs CMake, which downloads the prebuilt ONNX
Runtime release for the host platform (pinned to **1.22.0** — the last release
covering all four targets, including Intel macOS), verifies its SHA-256,
extracts it under `build/onnxruntime/sdk/`, and builds the shim. The download is
small (linux-x64 ≈ 8 MB, macOS ≈ 25 MB, win-x64 ≈ 69 MB) and cached after the
first run. Output: `build/libonnxrt.a` (static shim) + the dynamic
`libonnxruntime` in `build/onnxruntime/sdk/lib/`.

Platforms: `linux-x64`, `win-x64`, `osx-x86_64`, `osx-arm64`. Requirements: a
C/C++ toolchain, CMake ≥ 3.18, and network access on the first build. To point
at a hand-placed SDK instead (e.g. a custom or static ONNX Runtime), configure
with `-DONNXRT_ORT_DIR=/path/to/sdk` (expects `include/` + `lib/`).

## Smoke test

```sh
python3 tools/make_test_model.py   # writes examples/model.onnx (ReduceSum)
dub run --config=score_smoke       # create -> score -> destroy, ends in `OK`
```

Expected with the generated model:

```
backend available: true
scores: [1.1, 0.9, 1]
argmax index: 0  confidence: 0.3672
OK
```

Without a model at `examples/model.onnx` the example reports the load error
cleanly and still proves the link.

## Backends

Select at CMake configure time (`-DONNXRT_BACKEND=`):

| value  | deps         | `backendAvailable()` | use |
|--------|--------------|----------------------|-----|
| `ort`  | ONNX Runtime | true  | default; real inference, built from the submodule |
| `mock` | none         | false | deterministic ABI/lifetime fixture for tests/CI |
| `stub` | none         | false | honest "no backend" — every call → `errNotBuilt` |

```sh
# fast dependency-free build of just the binding layer (tests / CI):
cmake -S . -B build -DONNXRT_BACKEND=mock && cmake --build build --target onnxrt
```

## Use

```d
import onnxrt.backend;

if (!backendAvailable()) { /* this build has no real ONNX Runtime */ }

auto session = OnnxSession("model.onnx");      // throws OnnxException on failure
float[] scores = session.score(input, rowCount); // input: [rows*features] row-major
auto r = rank(scores);                           // r.index = argmax, r.confidence in [0,1]
```

See `doc/DESIGN.md` for the model contract, ownership/lifetime rules, error
policy, and the from-source build/link details. License: MIT (ONNX Runtime is
also MIT; the runtime's license is separate from any model's license).
