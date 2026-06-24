# D-OnnxRuntime — shim design

Spike for **vibe3d task 0021** (`ai-onnx-runtime-shim-spike`). Goal: pin down
the C ABI a D shim needs to drive ONNX Runtime as the first `AiModelBackend`,
the ownership/lifetime rules, error handling, and the macOS build/link story —
without pulling an ONNX Runtime binary into the repo or wiring anything into
the app.

ONNX Runtime is taken as the "first backend" hypothesis from task 0020 (MIT
license, mature C API, good macOS/CoreML support, small candidate-ranking
models export cleanly to `.onnx`). Nothing in the ABI is ONNX-specific, so a
second backend (llama.cpp / OpenVINO) can sit behind the same surface.

## Surface at a glance

```
include/v3d_ai_onnx.h         C ABI — the contract
csrc/v3d_ai_onnx_stub.c       backend=stub  honest "not built", zero deps
csrc/v3d_ai_onnx_mock.c       backend=mock  deterministic fixture, zero deps
csrc/v3d_ai_onnx_ort.cpp      backend=ort   real ONNX Runtime (sketch)
source/v3d_ai_onnx/c.d        extern(C) bindings, 1:1 with the header
source/v3d_ai_onnx/backend.d  RAII OnnxRanker + RankResult (argmax+softmax)
```

## C ABI

```c
int  v3d_ai_onnx_backend_available(void);   // 1 = real ORT, 0 = stub/mock
int  v3d_ai_onnx_abi_version(void);         // bump on any breaking change

v3d_ai_onnx_status_t v3d_ai_onnx_create(const char* model_path,
                                        v3d_ai_onnx_session_t** out_session,
                                        v3d_ai_onnx_error_t* out_error);
void                 v3d_ai_onnx_destroy(v3d_ai_onnx_session_t* session);
v3d_ai_onnx_status_t v3d_ai_onnx_rank(v3d_ai_onnx_session_t* session,
                                      const float* features,
                                      int feature_count,
                                      int candidate_count,
                                      float* scores_out,
                                      v3d_ai_onnx_error_t* out_error);
const char*          v3d_ai_onnx_last_error(const v3d_ai_onnx_session_t*);
```

Differences from the shape suggested in the task, and why:

- **Status code return + `out_error` struct**, not just `out_error`. Every
  fallible call returns a `v3d_ai_onnx_status_t`; `out_error` (optional) carries
  the human-readable detail. A code lets the caller branch (`errNotBuilt` vs
  `errModelLoad`) without string matching.
- **`backend_available()` / `abi_version()`** added. The first feeds the model
  availability mapping below; the second lets a binding reject a drifted `.a`.
- **`feature_count` is per-candidate, `candidate_count` rows.** `features` is a
  row-major `[candidate_count][feature_count]` matrix; `scores_out` is
  `candidate_count` floats. One inference call ranks the whole candidate set.

## Ownership & lifetime

- **Session**: created by `create`, freed by `destroy`. Opaque handle; on
  create failure `*out_session == NULL` (never half-built). All other entry
  points are NULL-tolerant. One session per thread — no internal locking.
- **Model path**: borrowed for the duration of `create` only; the model is
  loaded into the session, so the path need not outlive it.
- **Feature / score buffers**: 100% caller-owned. `features` is read-only and
  must stay valid for the `rank` call; `scores_out` is written in place. The
  shim never hands back a buffer the caller must free.
- **Errors**: `out_error->message` is a fixed 256-byte caller-owned buffer the
  shim copies into — no allocation crosses the boundary. `last_error(session)`
  returns a pointer owned by the session, valid until the next call on it or
  `destroy`.

## Error handling

`v3d_ai_onnx_status_t`: `ok=0`, `errInvalidArg`, `errModelLoad`, `errShape`
(feature width ≠ model input), `errInference`, `errNotBuilt` (stub build),
`errOom`. On any non-`ok` return, output buffers are left untouched. The D
wrapper turns non-`ok` into an `OnnxException` carrying the status code.

## Mapping to `AiModelBackendPrediction`

The shim returns **raw** scores; the D side derives the prediction. `RankResult`
already does argmax + numerically-stable softmax, so vibe3d's backend is thin:

```d
// in vibe3d: source/ai/onnx_backend.d (sketch — not in this repo)
final class OnnxModelBackend : AiModelBackend {
    private OnnxRanker ranker;          // owns the session
    private bool ready;

    AiModelAvailability availability() const {
        return ready
            ? AiModelAvailability(AiModelStatus.ready)
            : AiModelAvailability(AiModelStatus.unavailable, "onnx model not loaded");
    }

    AiModelBackendPrediction predict(const ref AiInteractionContext ctx,
                                     const(AiCandidate)[] cands) const {
        auto feats = encodeFeatures(ctx, cands);     // [N*F] row-major
        auto r = ranker.rank(feats, cast(int)cands.length);
        AiModelBackendPrediction p;
        if (r.index < 0) return p;                   // present()==false -> fallback
        p.candidateIndex = r.index;
        p.candidateId    = cands[r.index].id;        // adapter re-checks id==index
        p.confidence     = r.confidence;             // softmax prob in [0,1]
        return p;
    }
}
```

`AiModelAdapter.decisionFromPrediction` then applies the existing conservative
gates unchanged: `confidence` must be finite and in `(0.75, 1.0]`
(`aiModelAdapterMinConfidence`), `candidateId` must match `candidates[index].id`,
and a `keepDefault` / default-winner candidate is rejected. So a low-confidence
or malformed model output degrades to the deterministic fallback exactly like
the runtime advisor path — the shim adds no new trust.

### Feature layout (proposed, F = 8 per candidate)

Encoded from `AiCandidate` + `AiInteractionContext`; finite, scale-stable:

| idx | feature                          | source |
|-----|----------------------------------|--------|
| 0   | `-screenDist` (∞ → large neg)    | candidate |
| 1   | `-worldDist` (∞ → large neg)     | candidate |
| 2   | `priorityFromCurrentRules`       | candidate |
| 3   | `isDefaultWinner`                | candidate |
| 4   | `isExplicitModifierChoice`       | candidate |
| 5   | `kind` (one-hot or ordinal)      | candidate |
| 6   | `elementKind` (ordinal)          | candidate |
| 7   | phase / modifier bits            | context |

Frozen alongside the exported model — `feature_count` is checked against the
model input width at `rank` time (`errShape` on mismatch). The exact column set
is a knob for the exporter/trainer tasks (0022+); the ABI is agnostic to it.

## macOS build & link

Static library on purpose, so the runtime folds into the app binary and there
is **no `.dylib` to ship or `@rpath` to chase** — the task's "dynamic library
path" concern dissolves under static linking.

1. `dub build` runs the CMake prebuild → `build/libv3daionnx.a`; dub links it
   via `lflags-posix -L$PACKAGE_DIR/build` + `libs-posix v3daionnx`.
2. **Real backend**: build ONNX Runtime as a static lib (or fetch the
   prebuilt static package), then
   `cmake -DV3D_AI_ONNX_BACKEND=ort -DV3D_ORT_ROOT=/path/to/onnxruntime`.
   The consumer adds `-L/path/to/onnxruntime/lib -lonnxruntime -lc++` to the
   link line (these belong to the final executable, not the `.a`). On Apple
   Silicon the CoreML execution provider can be appended in `create`
   (`OrtSessionOptionsAppendExecutionProvider_CoreML`) for GPU/ANE offload;
   CPU is the portable default and the only path in the sketch.
3. **No binary in the repo**: ONNX Runtime and `.onnx` models are gitignored;
   `V3D_ORT_ROOT` points at an out-of-tree SDK. Default builds use `mock`/`stub`
   and have **zero** external dependency.

Size/risk: a static `libonnxruntime.a` is large (tens of MB) and bloats the
binary; the alternative is dynamic-link + `@rpath` (the D-RadeonProRender
pattern) if binary size wins over single-file distribution. Tracked in 0020.

## Stub vs mock vs ort

- **stub** — every fallible call returns `errNotBuilt` with an explicit
  message; never fabricates a session or score. The honest "no AI" build.
- **mock** — deterministic linear scoring (`score = Σ featureₖ/(k+1)`), zero
  deps, `backend_available()==0`. A test fixture to exercise the ABI/lifetime
  end-to-end; transparent and reproducible, never wired into the app, so it
  introduces no false runtime behavior.
- **ort** — the real ONNX Runtime implementation (sketch), opt-in.

## Open questions / next tasks

- Model output contract: single score per row vs `[N, num_classes]` + gather.
  The sketch assumes one score per candidate.
- Final feature column set + normalisation — owned by the exporter/trainer
  tasks; freeze it together with the model.
- Static vs dynamic ONNX Runtime link decision (binary size) — task 0020.
- Wiring `OnnxModelBackend` into `AiModelAdapter` is intentionally out of scope
  here (the adapter seam from 0019 is the integration point).
