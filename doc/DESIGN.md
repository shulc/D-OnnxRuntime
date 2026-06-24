# D-OnnxRuntime — design

A thin C ABI shim over ONNX Runtime plus 1:1 D bindings and a small RAII layer.
The goal: load an `.onnx` model and score a batch of feature rows from D,
without leaking any ONNX Runtime / C++ / STL type across the boundary, and ship
the shim as one static archive.

## Surface

```
include/onnxrt.h              C ABI — the contract
csrc/onnxrt_ort.cpp           backend=ort   real ONNX Runtime (default)
csrc/onnxrt_mock.c            backend=mock  deterministic fixture, no deps
csrc/onnxrt_stub.c            backend=stub  honest "not built", no deps
source/onnxrt/c.d             extern(C) bindings, 1:1 with the header
source/onnxrt/backend.d       RAII OnnxSession + rank() (argmax+softmax)
```

## C ABI

```c
int  onnxrt_backend_available(void);   // 1 = real ORT, 0 = mock/stub
int  onnxrt_abi_version(void);         // bump on any breaking change

onnxrt_status_t onnxrt_create(const char* model_path,
                              onnxrt_session_t** out_session,
                              onnxrt_error_t* out_error);
void            onnxrt_destroy(onnxrt_session_t* session);
onnxrt_status_t onnxrt_score(onnxrt_session_t* session,
                             const float* input,
                             int feature_count,
                             int row_count,
                             float* scores_out,
                             onnxrt_error_t* out_error);
const char*     onnxrt_last_error(const onnxrt_session_t* session);
```

Design choices:

- **Status code return + optional `out_error` struct.** Every fallible call
  returns an `onnxrt_status_t`; `out_error` (optional, caller-owned) carries the
  human-readable detail. A code lets the caller branch (`errNotBuilt` vs
  `errModelLoad`) without string matching.
- **`backend_available()` / `abi_version()`.** The first lets a consumer fall
  back when no real runtime is linked; the second lets a binding reject a
  drifted library at startup.
- **Batch in one call.** `input` is a row-major `[row_count][feature_count]`
  matrix; `scores_out` is `row_count` floats. One inference call scores the
  whole batch.

### Model contract

```
input  0: float32 tensor, shape [row_count, feature_count]
output 0: float32 tensor, shape [row_count]   (one score per row)
```

`feature_count` is validated against the model input width at score time
(`errShape` on mismatch). A model emitting `[row_count, num_classes]` would need
an argmax/gather in the shim — kept single-output for legibility.

## Ownership & lifetime

- **Session**: created by `create`, freed by `destroy`. Opaque handle; on
  create failure `*out_session == NULL` (never half-built). All other entry
  points are NULL-tolerant. One session per thread — no internal locking.
- **Model path**: borrowed for `create` only; the model is loaded into the
  session, so the path need not outlive it.
- **Input / score buffers**: 100% caller-owned. `input` is read-only and must
  stay valid for the `score` call; `scores_out` is written in place. The shim
  never returns a buffer the caller must free.
- **Errors**: `out_error->message` is a fixed 256-byte caller-owned buffer the
  shim copies into — no allocation crosses the boundary. `last_error(session)`
  returns a pointer owned by the session, valid until the next call on it or
  `destroy`.

## Error handling

`onnxrt_status_t`: `ok=0`, `errInvalidArg`, `errModelLoad`, `errShape`
(feature width ≠ model input), `errInference`, `errNotBuilt` (non-ort build),
`errOom`. On any non-`ok` return, output buffers are left untouched. The D
wrapper turns non-`ok` into an `OnnxException` carrying the status code.

## D layer

`source/onnxrt/c.d` is a literal mirror of the header. `source/onnxrt/backend.d`
adds:

- `OnnxSession` — RAII handle (non-copyable), `score(input, rowCount) -> float[]`.
- `rank(scores) -> RankResult{index, confidence}` — argmax + numerically-stable
  softmax; a generic reduction for ranking / single-label classification.
  Non-finite scores are treated as `-inf` so a NaN can't win.
- `backendAvailable()`, `linkedAbiVersion()`.

## Building from source & linking

ONNX Runtime is a pinned git submodule (`extern/onnxruntime`, v1.27.0) built
from source by `cmake/BuildOnnxRuntime.cmake`, which drives ONNX Runtime's own
`tools/ci_build/build.py` via an `ExternalProject`:

- Built as a **shared** library (`libonnxruntime.{so,dylib}`) — the reliable
  path; fully static ONNX Runtime drags in dozens of split archives and is
  painful to link. **Our shim stays static** (`libonnxrt.a`), so only one shared
  object ships alongside the binary.
- `dub` wiring: `preBuildCommands-posix` runs `git submodule update --init`,
  then CMake; `lflags`/`libs` link `onnxrt` + `onnxruntime` from
  `build/onnxruntime/Release`. An rpath of `$ORIGIN` (Linux) /
  `@executable_path` (macOS) is added so a distributed binary finds
  `libonnxruntime` placed next to it.
- First build is long (ONNX Runtime fetches protobuf/abseil/… and compiles
  everything); incremental and cached thereafter.
- **macOS**: CPU is the portable default. The CoreML execution provider can be
  appended in `onnxrt_create` (`OrtSessionOptionsAppendExecutionProvider_CoreML`)
  for GPU/ANE offload — left out of the default path.

The `mock` / `stub` backends compile with zero external dependencies and exist
so the binding layer (and its unit tests) can build without the heavy ONNX
Runtime build — handy for CI.

## Open questions

- Model output contract: single score per row (assumed) vs `[rows, classes]` +
  gather.
- Fully-static ONNX Runtime link (single-file distribution) vs the current
  shared-lib + rpath approach — a binary-size / link-complexity tradeoff.
