# D-OnnxRuntime

A thin, **static-linkable** C ABI shim + D bindings for ranking Vibe3D editor
candidates with [ONNX Runtime](https://onnxruntime.ai). It is the spike for
vibe3d task 0021 — the first concrete `AiModelBackend`: hand it a feature
matrix for the live candidate set, get back a score per candidate, derive an
argmax + confidence that maps onto `AiModelBackendPrediction`.

The C boundary mentions no ONNX types (plain scalars + caller-owned buffers),
so it binds cleanly from D and can be re-pointed at a second runtime later. The
default build has **zero external dependencies** — the real ONNX Runtime is an
opt-in backend and is never vendored into the repo.

## Layout

```
include/v3d_ai_onnx.h         C API (extern "C") — the contract
csrc/v3d_ai_onnx_stub.c       backend=stub  honest "not built", no deps
csrc/v3d_ai_onnx_mock.c       backend=mock  deterministic fixture, no deps (default)
csrc/v3d_ai_onnx_ort.cpp      backend=ort   real ONNX Runtime (sketch, opt-in)
source/v3d_ai_onnx/c.d        D extern(C) bindings, 1:1 with the header
source/v3d_ai_onnx/backend.d  RAII OnnxRanker + RankResult (argmax + softmax)
examples/rank_smoke.d         smoke test: create -> rank -> destroy
CMakeLists.txt                builds build/libv3daionnx.a
doc/DESIGN.md                 ABI, lifetime, error, feature mapping, macOS build
```

## Build & smoke test

```sh
dub build                    # CMake prebuild -> build/libv3daionnx.a, then D lib
dub run --config=rank_smoke  # exercises the full lifetime, ends in `OK`
```

`dub`'s `preBuildCommands-posix` runs CMake for you, so a plain `dub build` is
enough. Default backend is `mock`, so the smoke test scores a sample matrix:

```
abi: binding=1 linked=1
backend available: false
scores: [0.65, 0.85, 0.75]
argmax index: 1  confidence: 0.3672
OK
```

## Backends

Select at CMake configure time (`-DV3D_AI_ONNX_BACKEND=`):

| value  | deps        | `backend_available()` | use |
|--------|-------------|-----------------------|-----|
| `mock` | none        | 0 | default; deterministic ABI/lifetime fixture |
| `stub` | none        | 0 | honest "no AI" build — every call → `errNotBuilt` |
| `ort`  | ONNX Runtime| 1 | real inference (see DESIGN "macOS build & link") |

```sh
# real backend, against an out-of-tree ONNX Runtime SDK (never committed):
cmake -S . -B build -DV3D_AI_ONNX_BACKEND=ort -DV3D_ORT_ROOT=/path/to/onnxruntime
cmake --build build --target v3daionnx
```

## Consuming from another dub project

```json
"dependencies": { "d-onnxruntime": { "path": "../D-OnnxRuntime" } }
```

```d
import v3d_ai_onnx.backend;

if (!backendAvailable()) { /* fall back to the deterministic advisor */ }

auto ranker = OnnxRanker("model/ranker.onnx");   // throws OnnxException on fail
auto r = ranker.rank(features, candidateCount);  // features: [N*F] row-major
// r.index = argmax, r.confidence = softmax prob in [0,1], r.scores = raw
```

See `doc/DESIGN.md` for the `AiModelBackendPrediction` mapping and the feature
layout. License: MIT (ONNX Runtime is MIT; the runtime's license is separate
from any model's license).
