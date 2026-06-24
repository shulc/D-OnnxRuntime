// Real ONNX Runtime backend — sketch.
//
// Selected by `-DV3D_AI_ONNX_BACKEND=ort`. Compiles only against ONNX
// Runtime's C API header (onnxruntime_c_api.h) and links libonnxruntime; the
// build wiring lives in CMakeLists.txt / DESIGN.md ("macOS build & link").
// This is the spike's reference implementation: it shows the exact ORT calls
// the shim needs, the buffer ownership and the error mapping. It is kept out
// of the default build precisely so the repo has no mandatory external
// dependency (vibe3d task 0021 acceptance).
//
// Model contract assumed by this shim:
//   input  0: float32 tensor, shape [candidate_count, feature_count]
//   output 0: float32 tensor, shape [candidate_count] (one score per row)
// A model that emits [candidate_count, num_classes] would need an argmax /
// gather here; kept single-output to keep the spike legible.

#include "v3d_ai_onnx.h"

#include <onnxruntime_c_api.h>

#include <cstring>
#include <string>
#include <vector>

#define V3D_AI_ONNX_ABI 1

namespace {

const OrtApi* g_ort = nullptr;  // resolved once via OrtGetApiBase

const OrtApi* ort() {
    if (!g_ort) g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return g_ort;
}

}  // namespace

struct v3d_ai_onnx_session {
    OrtEnv*           env     = nullptr;
    OrtSession*       session = nullptr;
    OrtSessionOptions* opts   = nullptr;
    std::string       input_name;
    std::string       output_name;
    int64_t           feature_count = 0;  // model input width, fixed at load
    char              last_error[256] = {0};
};

namespace {

void store_error(v3d_ai_onnx_session* s, v3d_ai_onnx_error_t* out_error,
                 v3d_ai_onnx_status_t code, const std::string& msg) {
    if (s) {
        std::strncpy(s->last_error, msg.c_str(), sizeof(s->last_error) - 1);
        s->last_error[sizeof(s->last_error) - 1] = '\0';
    }
    if (out_error) {
        out_error->code = code;
        std::strncpy(out_error->message, msg.c_str(),
                     sizeof(out_error->message) - 1);
        out_error->message[sizeof(out_error->message) - 1] = '\0';
    }
}

// Translate an OrtStatus into our code+message, freeing the status. Returns
// true if there was an error (and stored it), false on success.
bool check(OrtStatus* status, v3d_ai_onnx_session* s,
           v3d_ai_onnx_error_t* out_error, v3d_ai_onnx_status_t code) {
    if (!status) return false;
    store_error(s, out_error, code, ort()->GetErrorMessage(status));
    ort()->ReleaseStatus(status);
    return true;
}

}  // namespace

extern "C" {

int v3d_ai_onnx_backend_available(void) { return 1; }

int v3d_ai_onnx_abi_version(void) { return V3D_AI_ONNX_ABI; }

v3d_ai_onnx_status_t v3d_ai_onnx_create(const char* model_path,
                                        v3d_ai_onnx_session_t** out_session,
                                        v3d_ai_onnx_error_t* out_error) {
    if (out_session) *out_session = nullptr;
    if (!model_path || !out_session) {
        store_error(nullptr, out_error, V3D_AI_ONNX_ERR_INVALID_ARG,
                    "model_path and out_session must be non-NULL");
        return V3D_AI_ONNX_ERR_INVALID_ARG;
    }

    auto* s = new (std::nothrow) v3d_ai_onnx_session();
    if (!s) {
        store_error(nullptr, out_error, V3D_AI_ONNX_ERR_OOM, "out of memory");
        return V3D_AI_ONNX_ERR_OOM;
    }

    if (check(ort()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "v3d_ai_onnx",
                               &s->env), s, out_error,
              V3D_AI_ONNX_ERR_MODEL_LOAD) ||
        check(ort()->CreateSessionOptions(&s->opts), s, out_error,
              V3D_AI_ONNX_ERR_MODEL_LOAD)) {
        v3d_ai_onnx_destroy(s);
        return V3D_AI_ONNX_ERR_MODEL_LOAD;
    }
    ort()->SetIntraOpNumThreads(s->opts, 1);

    // On macOS the CoreML execution provider can be appended here; omitted in
    // the sketch — CPU is the portable default. See DESIGN "macOS build".
    if (check(ort()->CreateSession(s->env, model_path, s->opts, &s->session),
              s, out_error, V3D_AI_ONNX_ERR_MODEL_LOAD)) {
        v3d_ai_onnx_destroy(s);
        return V3D_AI_ONNX_ERR_MODEL_LOAD;
    }

    // Cache input/output names + input width so rank() can validate shapes.
    OrtAllocator* alloc = nullptr;
    ort()->GetAllocatorWithDefaultOptions(&alloc);
    char* in_name = nullptr;
    char* out_name = nullptr;
    if (check(ort()->SessionGetInputName(s->session, 0, alloc, &in_name), s,
              out_error, V3D_AI_ONNX_ERR_MODEL_LOAD) ||
        check(ort()->SessionGetOutputName(s->session, 0, alloc, &out_name), s,
              out_error, V3D_AI_ONNX_ERR_MODEL_LOAD)) {
        if (in_name) alloc->Free(alloc, in_name);
        if (out_name) alloc->Free(alloc, out_name);
        v3d_ai_onnx_destroy(s);
        return V3D_AI_ONNX_ERR_MODEL_LOAD;
    }
    s->input_name = in_name;
    s->output_name = out_name;
    alloc->Free(alloc, in_name);
    alloc->Free(alloc, out_name);

    *out_session = s;
    return V3D_AI_ONNX_OK;
}

void v3d_ai_onnx_destroy(v3d_ai_onnx_session_t* session) {
    if (!session) return;
    if (session->session) ort()->ReleaseSession(session->session);
    if (session->opts)    ort()->ReleaseSessionOptions(session->opts);
    if (session->env)     ort()->ReleaseEnv(session->env);
    delete session;
}

v3d_ai_onnx_status_t v3d_ai_onnx_rank(v3d_ai_onnx_session_t* session,
                                      const float* features,
                                      int feature_count,
                                      int candidate_count,
                                      float* scores_out,
                                      v3d_ai_onnx_error_t* out_error) {
    if (!session) {
        store_error(nullptr, out_error, V3D_AI_ONNX_ERR_INVALID_ARG,
                    "session is NULL");
        return V3D_AI_ONNX_ERR_INVALID_ARG;
    }
    if (!features || !scores_out || feature_count <= 0 || candidate_count <= 0) {
        store_error(session, out_error, V3D_AI_ONNX_ERR_INVALID_ARG,
                    "features/scores_out non-NULL and counts > 0 required");
        return V3D_AI_ONNX_ERR_INVALID_ARG;
    }

    OrtMemoryInfo* mem_info = nullptr;
    if (check(ort()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                         &mem_info), session, out_error,
              V3D_AI_ONNX_ERR_INFERENCE))
        return V3D_AI_ONNX_ERR_INFERENCE;

    const int64_t shape[2] = { candidate_count, feature_count };
    const size_t elem_count = (size_t)candidate_count * (size_t)feature_count;

    OrtValue* input = nullptr;
    // Tensor borrows `features` — no copy; must outlive the Run() call.
    OrtStatus* st = ort()->CreateTensorWithDataAsOrtValue(
        mem_info, const_cast<float*>(features), elem_count * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
    ort()->ReleaseMemoryInfo(mem_info);
    if (check(st, session, out_error, V3D_AI_ONNX_ERR_SHAPE))
        return V3D_AI_ONNX_ERR_SHAPE;

    const char* in_names[1]  = { session->input_name.c_str() };
    const char* out_names[1] = { session->output_name.c_str() };
    OrtValue* output = nullptr;
    st = ort()->Run(session->session, nullptr, in_names,
                    (const OrtValue* const*)&input, 1, out_names, 1, &output);
    ort()->ReleaseValue(input);
    if (check(st, session, out_error, V3D_AI_ONNX_ERR_INFERENCE))
        return V3D_AI_ONNX_ERR_INFERENCE;

    float* out_data = nullptr;
    if (check(ort()->GetTensorMutableData(output, (void**)&out_data), session,
              out_error, V3D_AI_ONNX_ERR_INFERENCE)) {
        ort()->ReleaseValue(output);
        return V3D_AI_ONNX_ERR_INFERENCE;
    }
    std::memcpy(scores_out, out_data, (size_t)candidate_count * sizeof(float));
    ort()->ReleaseValue(output);

    session->last_error[0] = '\0';
    return V3D_AI_ONNX_OK;
}

const char* v3d_ai_onnx_last_error(const v3d_ai_onnx_session_t* session) {
    return session ? session->last_error : "";
}

}  // extern "C"
