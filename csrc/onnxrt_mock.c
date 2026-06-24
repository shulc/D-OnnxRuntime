// Deterministic mock backend — an ABI/lifetime fixture, NOT a model.
//
// Selected by `-DONNXRT_BACKEND=mock`. Zero external dependencies and reports
// onnxrt_backend_available() == 0, so it is never mistaken for a real model.
// Its only job is to exercise the C ABI, the ownership rules and the
// score plumbing end-to-end without the heavy ONNX Runtime build — handy for
// unit tests and examples.
//
// Scoring is a transparent, fixed linear map over each row:
//   score[r] = sum_f input[r*feature_count + f] * (1 / (f + 1)).
// Lower feature indices weigh more. It encodes no domain semantics — it is
// deliberately dumb and reproducible.

#include "onnxrt.h"

#include <stdlib.h>
#include <string.h>

#define ONNXRT_ABI 1

struct onnxrt_session {
    char last_error[256];  // owned by the session; see onnxrt_last_error
};

static void set_error(struct onnxrt_session* s,
                      onnxrt_error_t* out_error,
                      onnxrt_status_t code,
                      const char* msg) {
    if (s) {
        strncpy(s->last_error, msg, sizeof(s->last_error) - 1);
        s->last_error[sizeof(s->last_error) - 1] = '\0';
    }
    if (out_error) {
        out_error->code = code;
        strncpy(out_error->message, msg, sizeof(out_error->message) - 1);
        out_error->message[sizeof(out_error->message) - 1] = '\0';
    }
}

int onnxrt_backend_available(void) { return 0; }

int onnxrt_abi_version(void) { return ONNXRT_ABI; }

onnxrt_status_t onnxrt_create(const char* model_path,
                              onnxrt_session_t** out_session,
                              onnxrt_error_t* out_error) {
    if (out_session) *out_session = NULL;
    if (!model_path || !out_session) {
        set_error(NULL, out_error, ONNXRT_ERR_INVALID_ARG,
                  "model_path and out_session must be non-NULL");
        return ONNXRT_ERR_INVALID_ARG;
    }
    // The mock does not read the file (no real model); any path is accepted so
    // the lifetime path can be exercised without shipping an .onnx fixture.
    struct onnxrt_session* s =
        (struct onnxrt_session*)calloc(1, sizeof(*s));
    if (!s) {
        set_error(NULL, out_error, ONNXRT_ERR_OOM, "out of memory");
        return ONNXRT_ERR_OOM;
    }
    *out_session = s;
    return ONNXRT_OK;
}

void onnxrt_destroy(onnxrt_session_t* session) {
    free(session);
}

onnxrt_status_t onnxrt_score(onnxrt_session_t* session,
                             const float* input,
                             int feature_count,
                             int row_count,
                             float* scores_out,
                             onnxrt_error_t* out_error) {
    if (!session) {
        set_error(NULL, out_error, ONNXRT_ERR_INVALID_ARG, "session is NULL");
        return ONNXRT_ERR_INVALID_ARG;
    }
    if (!input || !scores_out || feature_count <= 0 || row_count <= 0) {
        set_error(session, out_error, ONNXRT_ERR_INVALID_ARG,
                  "input/scores_out must be non-NULL and counts > 0");
        return ONNXRT_ERR_INVALID_ARG;
    }

    for (int r = 0; r < row_count; ++r) {
        const float* row = input + (size_t)r * (size_t)feature_count;
        float score = 0.0f;
        for (int f = 0; f < feature_count; ++f)
            score += row[f] * (1.0f / (float)(f + 1));
        scores_out[r] = score;
    }
    session->last_error[0] = '\0';
    return ONNXRT_OK;
}

const char* onnxrt_last_error(const onnxrt_session_t* session) {
    return session ? session->last_error : "";
}
