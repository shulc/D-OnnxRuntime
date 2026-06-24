/// Raw D bindings for the D-OnnxRuntime C shim.
///
/// Mirrors include/onnxrt.h exactly — one struct per opaque handle, one
/// extern(C) declaration per entry point, no D-side wrapping. The ergonomic
/// RAII layer lives in onnxrt.backend so this raw surface stays available to
/// callers that manage session lifetime against their own arena/pool.
module onnxrt.c;

extern (C) @nogc nothrow:

/// Opaque inference session. Created via onnxrt_create, freed via
/// onnxrt_destroy. Every other entry point tolerates a null handle.
struct onnxrt_session_t;

/// Return / error codes. `ok` (0) is success; anything else leaves output
/// buffers untouched.
enum onnxrt_status_t : int {
    ok            = 0,
    errInvalidArg = 1, /// null/zero arg, row_count <= 0, ...
    errModelLoad  = 2, /// path missing / not a valid .onnx
    errShape      = 3, /// feature_count != model input width
    errInference  = 4, /// runtime threw during Run()
    errNotBuilt   = 5, /// dependency-free stub: no real backend linked
    errOom        = 6,
}

/// Caller-owned error sink (see header). `message` is always NUL-terminated;
/// no ownership crosses the boundary.
struct onnxrt_error_t {
    int code;
    char[256] message;
}

/// 1 if a real ONNX Runtime is linked; 0 for the stub / mock fixture.
int onnxrt_backend_available();

/// ABI revision of the linked shim; compare against `abiVersion` to reject a
/// mismatched library.
int onnxrt_abi_version();

/// Build a session from an on-disk model. On failure `*out_session` is null.
onnxrt_status_t onnxrt_create(const(char)* model_path,
                              onnxrt_session_t** out_session,
                              onnxrt_error_t* out_error);

/// Release a session. Null-tolerant.
void onnxrt_destroy(onnxrt_session_t* session);

/// Score `row_count` rows of a row-major [row_count][feature_count] input into
/// `scores_out` (caller-owned, row_count floats). Scores are raw
/// (un-normalised) — caller does argmax / softmax / thresholding.
onnxrt_status_t onnxrt_score(onnxrt_session_t* session,
                             const(float)* input,
                             int feature_count,
                             int row_count,
                             float* scores_out,
                             onnxrt_error_t* out_error);

/// Last error bound to `session`; valid until the next call on it or destroy.
/// Never null (empty string == no error).
const(char)* onnxrt_last_error(const(onnxrt_session_t)* session);

/// ABI revision this binding was written against. Keep in lockstep with the
/// ONNXRT_ABI macro in include/onnxrt.h.
enum int abiVersion = 1;
