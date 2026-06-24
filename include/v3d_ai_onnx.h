// D-OnnxRuntime — C ABI shim for the Vibe3D AI candidate ranker.
//
// This header is the contract. ONNX Runtime is the first intended backend
// (see doc/DESIGN.md and vibe3d task 0020 "ONNX Runtime first"), but nothing
// in this surface mentions ONNX types: the boundary is plain C scalars and
// caller-owned buffers so it binds cleanly from D via extern(C) and can be
// re-pointed at a second runtime (llama.cpp / OpenVINO) without an ABI break.
//
// All entry points are extern "C". No C++ types, no STL, no exceptions cross
// the boundary; the session is an opaque handle. The shim never allocates a
// buffer the caller must free — error detail is copied into a caller-owned
// struct, scores are written into a caller-owned array.

#ifndef V3D_AI_ONNX_H
#define V3D_AI_ONNX_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque inference session. Created by v3d_ai_onnx_create, released by
// v3d_ai_onnx_destroy. Every other entry point is NULL-tolerant.
typedef struct v3d_ai_onnx_session v3d_ai_onnx_session_t;

// Return / error codes. 0 == success; everything else is a failure and
// leaves output buffers untouched. Mirrors v3d_ai_onnx_status in source/.
typedef enum v3d_ai_onnx_status {
    V3D_AI_ONNX_OK            = 0,
    V3D_AI_ONNX_ERR_INVALID_ARG = 1,  // null/zero arg, candidate_count <= 0, ...
    V3D_AI_ONNX_ERR_MODEL_LOAD  = 2,  // path missing / not a valid .onnx
    V3D_AI_ONNX_ERR_SHAPE       = 3,  // feature_count != model input width
    V3D_AI_ONNX_ERR_INFERENCE   = 4,  // runtime threw during Run()
    V3D_AI_ONNX_ERR_NOT_BUILT   = 5,  // dependency-free stub: no real backend
    V3D_AI_ONNX_ERR_OOM         = 6
} v3d_ai_onnx_status_t;

// Caller-owned error sink. Pass NULL if you don't want the detail. No
// ownership crosses the boundary: the shim memcpy's into `message`, which is
// always NUL-terminated. `code` holds a v3d_ai_onnx_status_t value.
typedef struct v3d_ai_onnx_error {
    int  code;
    char message[256];
} v3d_ai_onnx_error_t;

// 1 if this build links a real ONNX Runtime and can actually score a model;
// 0 for the dependency-free stub or the deterministic mock fixture. Upstream
// maps 1 -> AiModelStatus.ready, 0 -> AiModelStatus.unavailable.
int v3d_ai_onnx_backend_available(void);

// ABI revision of this header. Bump on any breaking change to the structs or
// signatures below so a binding can refuse a mismatched .a at startup.
int v3d_ai_onnx_abi_version(void);

// Build an inference session from an on-disk model.
//   model_path  — NUL-terminated UTF-8 path to the .onnx file.
//   out_session — receives the handle on success, NULL on failure.
//   out_error   — optional; filled on failure.
// Returns V3D_AI_ONNX_OK or an error code. On failure *out_session == NULL,
// so a session handle is never half-constructed. The model is mmap'd /
// copied into the session at create time; `model_path` need not outlive it.
v3d_ai_onnx_status_t v3d_ai_onnx_create(const char* model_path,
                                        v3d_ai_onnx_session_t** out_session,
                                        v3d_ai_onnx_error_t* out_error);

// Release a session. NULL-tolerant. The handle is dangling afterwards; any
// pointer returned by v3d_ai_onnx_last_error for it is invalidated too.
void v3d_ai_onnx_destroy(v3d_ai_onnx_session_t* session);

// Score `candidate_count` candidates in one inference call.
//   session         — a live handle from v3d_ai_onnx_create.
//   features        — row-major [candidate_count][feature_count] floats,
//                     caller-owned, read-only, must stay valid for the call.
//   feature_count   — features per candidate; must equal the model input
//                     width or V3D_AI_ONNX_ERR_SHAPE is returned.
//   candidate_count — number of candidate rows and scores.
//   scores_out      — caller-owned buffer of `candidate_count` floats. On
//                     success holds one raw score per candidate (higher =
//                     more likely the intended candidate). Scores are NOT
//                     normalised: the caller applies argmax + softmax and the
//                     confidence threshold (see DESIGN: maps onto
//                     AiModelBackendPrediction).
//   out_error       — optional; filled on failure.
// Returns V3D_AI_ONNX_OK or an error code. On failure scores_out is untouched.
// The session holds no per-call state, so the same handle may be reused across
// frames; it is NOT internally synchronised (one session per thread).
v3d_ai_onnx_status_t v3d_ai_onnx_rank(v3d_ai_onnx_session_t* session,
                                      const float* features,
                                      int feature_count,
                                      int candidate_count,
                                      float* scores_out,
                                      v3d_ai_onnx_error_t* out_error);

// Last error message bound to `session`. The returned pointer is owned by the
// session and stays valid until the next call on that session or destroy.
// Never returns NULL — an empty string means "no error recorded".
const char* v3d_ai_onnx_last_error(const v3d_ai_onnx_session_t* session);

#ifdef __cplusplus
}
#endif

#endif // V3D_AI_ONNX_H
