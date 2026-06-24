# D-OnnxRuntime

D bindings + a thin **static-linkable** C ABI shim over
[ONNX Runtime](https://onnxruntime.ai). Load an `.onnx` model once and run it
per call on a batch of feature rows, getting one score per row back — the
common "rank / score N items with a small model" case (rankers, classifiers
reduced to a score, regressors).

The C boundary mentions no ONNX Runtime, C++ or STL types — plain C scalars and
caller-owned buffers — so it binds cleanly from D via `extern(C)` and the shim
itself links as one static archive you can fold into a distributable binary.
ONNX Runtime is built from source as a pinned git submodule.

## Layout

```
include/onnxrt.h              C API (extern "C") — the contract
csrc/onnxrt_ort.cpp           backend=ort   real ONNX Runtime (default)
csrc/onnxrt_mock.c            backend=mock  deterministic fixture, no deps
csrc/onnxrt_stub.c            backend=stub  honest "not built", no deps
source/onnxrt/c.d             D extern(C) bindings, 1:1 with the header
source/onnxrt/backend.d       RAII OnnxSession + rank() (argmax + softmax)
examples/score_smoke.d        smoke test: create -> score -> destroy
cmake/BuildOnnxRuntime.cmake  builds the submodule via ONNX Runtime's build.py
CMakeLists.txt                builds build/libonnxrt.a
extern/onnxruntime            git submodule, pinned to v1.27.0
doc/DESIGN.md                 ABI, lifetime, error policy, build & link notes
```

## First build

```sh
git submodule update --init --recursive extern/onnxruntime
dub build
```

`dub`'s `preBuildCommands-posix` initialises the submodule, then CMake builds
ONNX Runtime from source and the shim. **The first build is long** — ONNX
Runtime fetches its own dependencies (protobuf, abseil, …) and compiles
everything; it is incremental afterwards and the artifacts under
`build/onnxruntime/` are reused. Output: `build/libonnxrt.a` (our static shim) +
`build/onnxruntime/Release/libonnxruntime.{so,dylib}` (built from source).

Requirements: a C/C++ toolchain, CMake ≥ 3.18, Python 3 (ONNX Runtime's
`build.py`), and — first time — network access for its dependency fetch.

## Smoke test

```sh
dub run --config=score_smoke   # create -> score -> destroy, ends in `OK`
```

Drop an `.onnx` model at `examples/model.onnx` (input `[rows, features]` f32 →
output `[rows]` f32) to see real scores; without one the example reports the
load error cleanly and still proves the link.

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
