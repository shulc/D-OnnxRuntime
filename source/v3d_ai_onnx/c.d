/// Raw D bindings for the D-OnnxRuntime C shim.
///
/// Mirrors include/v3d_ai_onnx.h exactly — one struct per opaque handle, one
/// extern(C) declaration per entry point, no D-side wrapping. The ergonomic
/// RAII layer lives in v3d_ai_onnx.backend so this raw surface stays available
/// to callers that manage session lifetime against their own arena/pool.
module v3d_ai_onnx.c;

extern (C) @nogc nothrow:

/// Opaque inference session. Created via v3d_ai_onnx_create, freed via
/// v3d_ai_onnx_destroy. Every other entry point tolerates a null handle.
struct v3d_ai_onnx_session_t;

/// Return / error codes. `ok` (0) is success; anything else leaves output
/// buffers untouched.
enum v3d_ai_onnx_status_t : int {
    ok            = 0,
    errInvalidArg = 1, /// null/zero arg, candidate_count <= 0, ...
    errModelLoad  = 2, /// path missing / not a valid .onnx
    errShape      = 3, /// feature_count != model input width
    errInference  = 4, /// runtime threw during Run()
    errNotBuilt   = 5, /// dependency-free stub: no real backend linked
    errOom        = 6,
}

/// Caller-owned error sink (see header). `message` is always NUL-terminated;
/// no ownership crosses the boundary.
struct v3d_ai_onnx_error_t {
    int code;
    char[256] message;
}

/// 1 if a real ONNX Runtime is linked; 0 for the stub / mock fixture.
int v3d_ai_onnx_backend_available();

/// ABI revision of the linked shim; compare against the value baked into the
/// binding (`abiVersion`) to reject a mismatched .a.
int v3d_ai_onnx_abi_version();

/// Build a session from an on-disk model. On failure `*out_session` is null.
v3d_ai_onnx_status_t v3d_ai_onnx_create(const(char)* model_path,
                                        v3d_ai_onnx_session_t** out_session,
                                        v3d_ai_onnx_error_t* out_error);

/// Release a session. Null-tolerant.
void v3d_ai_onnx_destroy(v3d_ai_onnx_session_t* session);

/// Score `candidate_count` rows of a row-major [candidate_count][feature_count]
/// feature matrix into `scores_out` (caller-owned, candidate_count floats).
/// Scores are raw (un-normalised) — caller does argmax + softmax + threshold.
v3d_ai_onnx_status_t v3d_ai_onnx_rank(v3d_ai_onnx_session_t* session,
                                      const(float)* features,
                                      int feature_count,
                                      int candidate_count,
                                      float* scores_out,
                                      v3d_ai_onnx_error_t* out_error);

/// Last error bound to `session`; valid until the next call on it or destroy.
/// Never null (empty string == no error).
const(char)* v3d_ai_onnx_last_error(const(v3d_ai_onnx_session_t)* session);

/// ABI revision this binding was written against. Keep in lockstep with the
/// V3D_AI_ONNX_ABI macro in include/v3d_ai_onnx.h.
enum int abiVersion = 1;
