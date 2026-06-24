// Deterministic mock backend — an ABI/lifetime fixture, NOT a model.
//
// Selected by `-DV3D_AI_ONNX_BACKEND=mock` (the default for this spike repo so
// `dub build` / `dub run` exercise the full create -> rank -> destroy path).
// It has zero external dependencies and reports backend_available() == 0, so
// upstream treats it as AiModelStatus.unavailable: it is never mistaken for a
// real model and is never wired into the app. Its only job is to prove the C
// ABI shape, the ownership rules and the score->prediction plumbing.
//
// Scoring is a transparent, fixed linear map over each candidate's feature
// row: score[c] = sum_f features[c*feature_count + f] * (1 / (f + 1)). Lower
// feature indices weigh more. Callers that put a "closeness" signal in
// feature 0 (e.g. -screenDist) get a sensible argmax, but the map encodes no
// editor semantics — it is deliberately dumb and reproducible.

#include "v3d_ai_onnx.h"

#include <stdlib.h>
#include <string.h>

#define V3D_AI_ONNX_ABI 1

struct v3d_ai_onnx_session {
    char last_error[256];  // owned by the session; see v3d_ai_onnx_last_error
};

static void set_error(struct v3d_ai_onnx_session* s,
                      v3d_ai_onnx_error_t* out_error,
                      v3d_ai_onnx_status_t code,
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

int v3d_ai_onnx_backend_available(void) { return 0; }

int v3d_ai_onnx_abi_version(void) { return V3D_AI_ONNX_ABI; }

v3d_ai_onnx_status_t v3d_ai_onnx_create(const char* model_path,
                                        v3d_ai_onnx_session_t** out_session,
                                        v3d_ai_onnx_error_t* out_error) {
    if (out_session) *out_session = NULL;
    if (!model_path || !out_session) {
        set_error(NULL, out_error, V3D_AI_ONNX_ERR_INVALID_ARG,
                  "model_path and out_session must be non-NULL");
        return V3D_AI_ONNX_ERR_INVALID_ARG;
    }
    // The mock does not read the file (no real model); any path is accepted so
    // the lifetime path can be exercised without shipping an .onnx fixture.
    struct v3d_ai_onnx_session* s =
        (struct v3d_ai_onnx_session*)calloc(1, sizeof(*s));
    if (!s) {
        set_error(NULL, out_error, V3D_AI_ONNX_ERR_OOM, "out of memory");
        return V3D_AI_ONNX_ERR_OOM;
    }
    *out_session = s;
    return V3D_AI_ONNX_OK;
}

void v3d_ai_onnx_destroy(v3d_ai_onnx_session_t* session) {
    free(session);
}

v3d_ai_onnx_status_t v3d_ai_onnx_rank(v3d_ai_onnx_session_t* session,
                                      const float* features,
                                      int feature_count,
                                      int candidate_count,
                                      float* scores_out,
                                      v3d_ai_onnx_error_t* out_error) {
    if (!session) {
        set_error(NULL, out_error, V3D_AI_ONNX_ERR_INVALID_ARG,
                  "session is NULL");
        return V3D_AI_ONNX_ERR_INVALID_ARG;
    }
    if (!features || !scores_out || feature_count <= 0 || candidate_count <= 0) {
        set_error(session, out_error, V3D_AI_ONNX_ERR_INVALID_ARG,
                  "features/scores_out must be non-NULL and counts > 0");
        return V3D_AI_ONNX_ERR_INVALID_ARG;
    }

    for (int c = 0; c < candidate_count; ++c) {
        const float* row = features + (size_t)c * (size_t)feature_count;
        float score = 0.0f;
        for (int f = 0; f < feature_count; ++f)
            score += row[f] * (1.0f / (float)(f + 1));
        scores_out[c] = score;
    }
    session->last_error[0] = '\0';
    return V3D_AI_ONNX_OK;
}

const char* v3d_ai_onnx_last_error(const v3d_ai_onnx_session_t* session) {
    return session ? session->last_error : "";
}
