// D-OnnxRuntime — a thin C ABI shim over ONNX Runtime.
//
// Purpose: load an .onnx model once and run it per call on a batch of feature
// rows, getting one score per row back. That covers the common "rank / score N
// items with a small model" case (rankers, classifiers reduced to a score,
// regressors) without exposing any ONNX Runtime, C++ or STL type across the
// boundary — everything here is plain C scalars and caller-owned buffers, so
// it binds cleanly from D via extern(C).
//
// The model contract this shim assumes:
//   input  0: float32 tensor, shape [row_count, feature_count]
//   output 0: float32 tensor, shape [row_count]   (one score per row)
//
// The session is an opaque handle. The shim never returns a buffer the caller
// must free: error detail is copied into a caller-owned struct, scores are
// written into a caller-owned array.

#ifndef ONNXRT_H
#define ONNXRT_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque inference session. Created by onnxrt_create, released by
// onnxrt_destroy. Every other entry point is NULL-tolerant.
typedef struct onnxrt_session onnxrt_session_t;

// Return / error codes. 0 == success; everything else is a failure and leaves
// output buffers untouched.
typedef enum onnxrt_status {
    ONNXRT_OK              = 0,
    ONNXRT_ERR_INVALID_ARG = 1,  // null/zero arg, row_count <= 0, ...
    ONNXRT_ERR_MODEL_LOAD  = 2,  // path missing / not a valid .onnx
    ONNXRT_ERR_SHAPE       = 3,  // feature_count != model input width
    ONNXRT_ERR_INFERENCE   = 4,  // runtime threw during Run()
    ONNXRT_ERR_NOT_BUILT   = 5,  // dependency-free stub: no real backend
    ONNXRT_ERR_OOM         = 6
} onnxrt_status_t;

// Caller-owned error sink. Pass NULL if you don't want the detail. No
// ownership crosses the boundary: the shim copies into `message`, which is
// always NUL-terminated. `code` holds an onnxrt_status_t value.
typedef struct onnxrt_error {
    int  code;
    char message[256];
} onnxrt_error_t;

// 1 if this build links a real ONNX Runtime and can actually score a model;
// 0 for the dependency-free stub or the deterministic mock fixture.
int onnxrt_backend_available(void);

// ABI revision of this header. Bump on any breaking change to the structs or
// signatures below so a binding can refuse a mismatched library at startup.
int onnxrt_abi_version(void);

// Build an inference session from an on-disk model.
//   model_path  — NUL-terminated UTF-8 path to the .onnx file.
//   out_session — receives the handle on success, NULL on failure.
//   out_error   — optional; filled on failure.
// Returns ONNXRT_OK or an error code. On failure *out_session == NULL, so a
// handle is never half-constructed. The model is loaded into the session at
// create time; `model_path` need not outlive it.
onnxrt_status_t onnxrt_create(const char* model_path,
                              onnxrt_session_t** out_session,
                              onnxrt_error_t* out_error);

// Release a session. NULL-tolerant. The handle is dangling afterwards; any
// pointer returned by onnxrt_last_error for it is invalidated too.
void onnxrt_destroy(onnxrt_session_t* session);

// Score `row_count` feature rows in one inference call.
//   session       — a live handle from onnxrt_create.
//   input         — row-major [row_count][feature_count] floats, caller-owned,
//                   read-only, must stay valid for the call.
//   feature_count — features per row; must equal the model input width or
//                   ONNXRT_ERR_SHAPE is returned.
//   row_count     — number of rows and scores.
//   scores_out    — caller-owned buffer of `row_count` floats. On success
//                   holds one raw score per row (higher = stronger). Scores are
//                   NOT normalised: the caller applies argmax / softmax /
//                   thresholding as it sees fit.
//   out_error     — optional; filled on failure.
// Returns ONNXRT_OK or an error code. On failure scores_out is untouched. The
// session holds no per-call state, so one handle may be reused across calls; it
// is NOT internally synchronised (use one session per thread).
onnxrt_status_t onnxrt_score(onnxrt_session_t* session,
                             const float* input,
                             int feature_count,
                             int row_count,
                             float* scores_out,
                             onnxrt_error_t* out_error);

// Last error message bound to `session`. The returned pointer is owned by the
// session and stays valid until the next call on it or destroy. Never returns
// NULL — an empty string means "no error recorded".
const char* onnxrt_last_error(const onnxrt_session_t* session);

#ifdef __cplusplus
}
#endif

#endif // ONNXRT_H
