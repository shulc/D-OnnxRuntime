// Dependency-free stub backend.
//
// Selected by `-DV3D_AI_ONNX_BACKEND=stub`. Links nothing external, so the
// repo (and anything that vendors this .a) builds with zero ONNX dependency.
// Every fallible call fails with V3D_AI_ONNX_ERR_NOT_BUILT and an explicit
// message — it never fabricates a session or a score, so a caller can't
// mistake the stub for a working model (no false runtime behavior; see the
// spike scope in vibe3d task 0021).

#include "v3d_ai_onnx.h"

#include <string.h>

#define V3D_AI_ONNX_ABI 1

static const char k_not_built[] =
    "ONNX backend not compiled in (stub build); rebuild with "
    "-DV3D_AI_ONNX_BACKEND=ort and a real onnxruntime";

static v3d_ai_onnx_status_t fail_not_built(v3d_ai_onnx_error_t* out_error) {
    if (out_error) {
        out_error->code = V3D_AI_ONNX_ERR_NOT_BUILT;
        // strncpy + explicit terminator: message is a fixed 256-byte buffer.
        strncpy(out_error->message, k_not_built, sizeof(out_error->message) - 1);
        out_error->message[sizeof(out_error->message) - 1] = '\0';
    }
    return V3D_AI_ONNX_ERR_NOT_BUILT;
}

int v3d_ai_onnx_backend_available(void) { return 0; }

int v3d_ai_onnx_abi_version(void) { return V3D_AI_ONNX_ABI; }

v3d_ai_onnx_status_t v3d_ai_onnx_create(const char* model_path,
                                        v3d_ai_onnx_session_t** out_session,
                                        v3d_ai_onnx_error_t* out_error) {
    (void)model_path;
    if (out_session) *out_session = NULL;
    return fail_not_built(out_error);
}

void v3d_ai_onnx_destroy(v3d_ai_onnx_session_t* session) { (void)session; }

v3d_ai_onnx_status_t v3d_ai_onnx_rank(v3d_ai_onnx_session_t* session,
                                      const float* features,
                                      int feature_count,
                                      int candidate_count,
                                      float* scores_out,
                                      v3d_ai_onnx_error_t* out_error) {
    (void)session; (void)features; (void)feature_count;
    (void)candidate_count; (void)scores_out;
    return fail_not_built(out_error);
}

const char* v3d_ai_onnx_last_error(const v3d_ai_onnx_session_t* session) {
    (void)session;
    return k_not_built;
}
