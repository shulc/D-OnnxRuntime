/// Ergonomic D layer over the raw C shim.
///
/// `OnnxSession` is an RAII handle: it owns one `onnxrt_session_t`, frees it in
/// the destructor, and turns the C status codes into D exceptions. `rank`
/// turns a raw score array into an argmax + softmax confidence — a generic
/// convenience for ranking / single-label classification on top of `score`.
module onnxrt.backend;

import std.exception : enforce;
import std.math : exp, isFinite;
import std.string : fromStringz, toStringz;

import onnxrt.c;

/// Thrown by `OnnxSession` when the C shim reports a non-`ok` status. Carries
/// the raw status code so callers can distinguish e.g. `errNotBuilt` (no real
/// backend in this build) from `errModelLoad` (bad path).
class OnnxException : Exception {
    onnxrt_status_t status;
    this(onnxrt_status_t status, string msg,
         string file = __FILE__, size_t line = __LINE__) {
        super(msg, file, line);
        this.status = status;
    }
}

/// Argmax + softmax over a score array: `index` is the winning row,
/// `confidence` its softmax probability in [0, 1].
struct RankResult {
    int index = -1;
    float confidence = 0.0f;
}

/// True if this build links a real ONNX Runtime (vs the stub / mock fixture).
bool backendAvailable() {
    return onnxrt_backend_available() != 0;
}

/// ABI revision of the linked shim. Compare against `onnxrt.c.abiVersion` at
/// startup; a mismatch means the library and the bindings drifted.
int linkedAbiVersion() {
    return onnxrt_abi_version();
}

/// Owns a single inference session. Non-copyable; move or pass by ref.
struct OnnxSession {
    private onnxrt_session_t* session_;

    @disable this(this);

    /// Load a model. Throws `OnnxException` on failure (e.g. `errNotBuilt` for
    /// a stub build, `errModelLoad` for a bad path).
    this(string modelPath) {
        onnxrt_error_t err;
        const status = onnxrt_create(modelPath.toStringz, &session_, &err);
        if (status != onnxrt_status_t.ok) {
            session_ = null;
            throw new OnnxException(status, errorText(err));
        }
    }

    ~this() {
        if (session_ !is null) {
            onnxrt_destroy(session_);
            session_ = null;
        }
    }

    /// True while the session handle is live.
    bool valid() const { return session_ !is null; }

    /// Score `rowCount` rows. `input` is a row-major [rowCount][featureCount]
    /// matrix; its length must be an exact multiple of `rowCount`. Returns one
    /// raw score per row. Throws `OnnxException` on shim error.
    float[] score(const(float)[] input, int rowCount) {
        enforce(session_ !is null, "score() on a destroyed OnnxSession");
        enforce(rowCount > 0, "rowCount must be > 0");
        enforce(input.length % rowCount == 0,
                "input length must be a multiple of rowCount");
        const featureCount = cast(int)(input.length / rowCount);
        enforce(featureCount > 0, "featureCount must be > 0");

        auto scores = new float[rowCount];
        onnxrt_error_t err;
        const status = onnxrt_score(session_, input.ptr, featureCount,
                                    rowCount, scores.ptr, &err);
        if (status != onnxrt_status_t.ok)
            throw new OnnxException(status, errorText(err));
        return scores;
    }

    /// Score then reduce to argmax + softmax confidence.
    RankResult rank(const(float)[] input, int rowCount) {
        return .rank(score(input, rowCount));
    }

    /// Last error message bound to this session ("" if none).
    string lastError() const {
        if (session_ is null) return "";
        return fromStringz(onnxrt_last_error(session_)).idup;
    }
}

/// argmax + numerically-stable softmax over raw scores. Non-finite scores are
/// treated as -inf so a NaN can't win; an all-non-finite set yields index -1,
/// confidence 0.
RankResult rank(const(float)[] scores) {
    RankResult r;
    if (scores.length == 0) return r;

    int best = -1;
    float bestScore = -float.infinity;
    foreach (i, s; scores) {
        if (!s.isFinite) continue;
        if (best < 0 || s > bestScore) {
            best = cast(int)i;
            bestScore = s;
        }
    }
    if (best < 0) return r;

    double denom = 0.0;
    foreach (s; scores) {
        if (!s.isFinite) continue;
        denom += exp(cast(double)s - cast(double)bestScore);
    }
    r.index = best;
    r.confidence = denom > 0.0 ? cast(float)(1.0 / denom) : 0.0f;
    return r;
}

private string errorText(ref const onnxrt_error_t err) {
    // message is a fixed NUL-terminated buffer; copy out the live portion.
    return fromStringz(err.message.ptr).idup;
}

unittest {
    // Pure rank() logic — no shim needed, runs on any build.
    auto r = rank([1.0f, 3.0f, 2.0f]);
    assert(r.index == 1);
    assert(r.confidence > 0.0f && r.confidence <= 1.0f);

    auto one = rank([5.0f]);
    assert(one.index == 0 && one.confidence > 0.99f);

    assert(rank([]).index == -1);

    auto nan = rank([float.nan, float.nan]);
    assert(nan.index == -1);
}
