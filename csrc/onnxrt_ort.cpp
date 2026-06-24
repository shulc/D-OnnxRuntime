// Real ONNX Runtime backend — the default.
//
// Selected by `-DONNXRT_BACKEND=ort` (the default). Compiles against ONNX
// Runtime's C API header (onnxruntime_c_api.h) from the vendored submodule and
// links libonnxruntime built from that same source — see CMakeLists.txt /
// doc/DESIGN.md ("Building from source").
//
// Model contract (see onnxrt.h):
//   input  0: float32 tensor, shape [row_count, feature_count]
//   output 0: float32 tensor, shape [row_count]   (one score per row)
// A model emitting [row_count, num_classes] would need an argmax/gather here;
// kept single-output to keep the shim legible.

#include "onnxrt.h"

#include <onnxruntime_c_api.h>

#include <cstring>
#include <new>
#include <string>

#define ONNXRT_ABI 1

namespace {

const OrtApi* g_ort = nullptr;  // resolved once via OrtGetApiBase

const OrtApi* ort() {
    if (!g_ort) g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return g_ort;
}

}  // namespace

struct onnxrt_session {
    OrtEnv*            env     = nullptr;
    OrtSession*        session = nullptr;
    OrtSessionOptions* opts    = nullptr;
    std::string        input_name;
    std::string        output_name;
    char               last_error[256] = {0};
};

namespace {

void store_error(onnxrt_session* s, onnxrt_error_t* out_error,
                 onnxrt_status_t code, const std::string& msg) {
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
bool check(OrtStatus* status, onnxrt_session* s, onnxrt_error_t* out_error,
           onnxrt_status_t code) {
    if (!status) return false;
    store_error(s, out_error, code, ort()->GetErrorMessage(status));
    ort()->ReleaseStatus(status);
    return true;
}

}  // namespace

extern "C" {

int onnxrt_backend_available(void) { return 1; }

int onnxrt_abi_version(void) { return ONNXRT_ABI; }

onnxrt_status_t onnxrt_create(const char* model_path,
                              onnxrt_session_t** out_session,
                              onnxrt_error_t* out_error) {
    if (out_session) *out_session = nullptr;
    if (!model_path || !out_session) {
        store_error(nullptr, out_error, ONNXRT_ERR_INVALID_ARG,
                    "model_path and out_session must be non-NULL");
        return ONNXRT_ERR_INVALID_ARG;
    }

    auto* s = new (std::nothrow) onnxrt_session();
    if (!s) {
        store_error(nullptr, out_error, ONNXRT_ERR_OOM, "out of memory");
        return ONNXRT_ERR_OOM;
    }

    if (check(ort()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxrt", &s->env),
              s, out_error, ONNXRT_ERR_MODEL_LOAD) ||
        check(ort()->CreateSessionOptions(&s->opts), s, out_error,
              ONNXRT_ERR_MODEL_LOAD)) {
        onnxrt_destroy(s);
        return ONNXRT_ERR_MODEL_LOAD;
    }
    if (check(ort()->SetIntraOpNumThreads(s->opts, 1), s, out_error,
              ONNXRT_ERR_MODEL_LOAD)) {
        onnxrt_destroy(s);
        return ONNXRT_ERR_MODEL_LOAD;
    }

    // On macOS the CoreML execution provider can be appended here for GPU/ANE
    // offload; omitted in the default path — CPU is portable. See DESIGN.
    if (check(ort()->CreateSession(s->env, model_path, s->opts, &s->session),
              s, out_error, ONNXRT_ERR_MODEL_LOAD)) {
        onnxrt_destroy(s);
        return ONNXRT_ERR_MODEL_LOAD;
    }

    // Cache input/output names so onnxrt_score can name the I/O bindings.
    OrtAllocator* alloc = nullptr;
    if (check(ort()->GetAllocatorWithDefaultOptions(&alloc), s, out_error,
              ONNXRT_ERR_MODEL_LOAD)) {
        onnxrt_destroy(s);
        return ONNXRT_ERR_MODEL_LOAD;
    }
    char* in_name = nullptr;
    char* out_name = nullptr;
    if (check(ort()->SessionGetInputName(s->session, 0, alloc, &in_name), s,
              out_error, ONNXRT_ERR_MODEL_LOAD) ||
        check(ort()->SessionGetOutputName(s->session, 0, alloc, &out_name), s,
              out_error, ONNXRT_ERR_MODEL_LOAD)) {
        if (in_name) alloc->Free(alloc, in_name);
        if (out_name) alloc->Free(alloc, out_name);
        onnxrt_destroy(s);
        return ONNXRT_ERR_MODEL_LOAD;
    }
    s->input_name = in_name;
    s->output_name = out_name;
    alloc->Free(alloc, in_name);
    alloc->Free(alloc, out_name);

    *out_session = s;
    return ONNXRT_OK;
}

void onnxrt_destroy(onnxrt_session_t* session) {
    if (!session) return;
    if (session->session) ort()->ReleaseSession(session->session);
    if (session->opts)    ort()->ReleaseSessionOptions(session->opts);
    if (session->env)     ort()->ReleaseEnv(session->env);
    delete session;
}

onnxrt_status_t onnxrt_score(onnxrt_session_t* session,
                             const float* input,
                             int feature_count,
                             int row_count,
                             float* scores_out,
                             onnxrt_error_t* out_error) {
    if (!session) {
        store_error(nullptr, out_error, ONNXRT_ERR_INVALID_ARG,
                    "session is NULL");
        return ONNXRT_ERR_INVALID_ARG;
    }
    if (!input || !scores_out || feature_count <= 0 || row_count <= 0) {
        store_error(session, out_error, ONNXRT_ERR_INVALID_ARG,
                    "input/scores_out non-NULL and counts > 0 required");
        return ONNXRT_ERR_INVALID_ARG;
    }

    OrtMemoryInfo* mem_info = nullptr;
    if (check(ort()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                         &mem_info), session, out_error,
              ONNXRT_ERR_INFERENCE))
        return ONNXRT_ERR_INFERENCE;

    const int64_t shape[2] = { row_count, feature_count };
    const size_t elem_count = (size_t)row_count * (size_t)feature_count;

    OrtValue* in_value = nullptr;
    // Tensor borrows `input` — no copy; must outlive the Run() call.
    OrtStatus* st = ort()->CreateTensorWithDataAsOrtValue(
        mem_info, const_cast<float*>(input), elem_count * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_value);
    ort()->ReleaseMemoryInfo(mem_info);
    if (check(st, session, out_error, ONNXRT_ERR_SHAPE))
        return ONNXRT_ERR_SHAPE;

    const char* in_names[1]  = { session->input_name.c_str() };
    const char* out_names[1] = { session->output_name.c_str() };
    OrtValue* out_value = nullptr;
    st = ort()->Run(session->session, nullptr, in_names,
                    (const OrtValue* const*)&in_value, 1, out_names, 1,
                    &out_value);
    ort()->ReleaseValue(in_value);
    if (check(st, session, out_error, ONNXRT_ERR_INFERENCE))
        return ONNXRT_ERR_INFERENCE;

    float* out_data = nullptr;
    if (check(ort()->GetTensorMutableData(out_value, (void**)&out_data),
              session, out_error, ONNXRT_ERR_INFERENCE)) {
        ort()->ReleaseValue(out_value);
        return ONNXRT_ERR_INFERENCE;
    }
    std::memcpy(scores_out, out_data, (size_t)row_count * sizeof(float));
    ort()->ReleaseValue(out_value);

    session->last_error[0] = '\0';
    return ONNXRT_OK;
}

const char* onnxrt_last_error(const onnxrt_session_t* session) {
    return session ? session->last_error : "";
}

}  // extern "C"
