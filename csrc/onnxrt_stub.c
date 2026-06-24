// Dependency-free stub backend.
//
// Selected by `-DONNXRT_BACKEND=stub`. Links nothing external, so the binding
// layer builds and tests without ONNX Runtime. Every fallible call fails with
// ONNXRT_ERR_NOT_BUILT and an explicit message — it never fabricates a session
// or a score, so a caller can't mistake the stub for a working model. Useful
// for CI that wants to compile the D side without the heavy ORT source build.

#include "onnxrt.h"

#include <string.h>

#define ONNXRT_ABI 1

static const char k_not_built[] =
    "ONNX backend not compiled in (stub build); rebuild with "
    "-DONNXRT_BACKEND=ort";

static onnxrt_status_t fail_not_built(onnxrt_error_t* out_error) {
    if (out_error) {
        out_error->code = ONNXRT_ERR_NOT_BUILT;
        strncpy(out_error->message, k_not_built, sizeof(out_error->message) - 1);
        out_error->message[sizeof(out_error->message) - 1] = '\0';
    }
    return ONNXRT_ERR_NOT_BUILT;
}

int onnxrt_backend_available(void) { return 0; }

int onnxrt_abi_version(void) { return ONNXRT_ABI; }

onnxrt_status_t onnxrt_create(const char* model_path,
                              onnxrt_session_t** out_session,
                              onnxrt_error_t* out_error) {
    (void)model_path;
    if (out_session) *out_session = NULL;
    return fail_not_built(out_error);
}

void onnxrt_destroy(onnxrt_session_t* session) { (void)session; }

onnxrt_status_t onnxrt_score(onnxrt_session_t* session,
                             const float* input,
                             int feature_count,
                             int row_count,
                             float* scores_out,
                             onnxrt_error_t* out_error) {
    (void)session; (void)input; (void)feature_count;
    (void)row_count; (void)scores_out;
    return fail_not_built(out_error);
}

const char* onnxrt_last_error(const onnxrt_session_t* session) {
    (void)session;
    return k_not_built;
}
