/// Ergonomic D layer over the raw C shim.
///
/// `OnnxRanker` is an RAII handle: it owns one `v3d_ai_onnx_session_t`, frees
/// it in the destructor, and turns the C status codes into D exceptions. It
/// also packages the score->prediction step (argmax + softmax confidence) so a
/// consumer can fill a vibe3d `AiModelBackendPrediction` directly — see
/// doc/DESIGN.md "Mapping to AiModelBackendPrediction".
///
/// This module deliberately does NOT import anything from vibe3d: the shim is
/// a standalone package. The vibe3d side implements `AiModelBackend` by
/// holding an `OnnxRanker` and translating its `RankResult` (sketch in DESIGN).
module v3d_ai_onnx.backend;

import std.exception : enforce;
import std.math : exp, isFinite;
import std.string : fromStringz, toStringz;

import v3d_ai_onnx.c;

/// Thrown by `OnnxRanker` when the C shim reports a non-`ok` status. Carries
/// the raw status code so callers can distinguish e.g. `errNotBuilt` (no real
/// backend in this build) from `errModelLoad` (bad path).
class OnnxException : Exception {
    v3d_ai_onnx_status_t status;
    this(v3d_ai_onnx_status_t status, string msg,
         string file = __FILE__, size_t line = __LINE__) {
        super(msg, file, line);
        this.status = status;
    }
}

/// Result of ranking one candidate set. `scores` are the raw shim outputs;
/// `index`/`confidence` are the derived argmax + softmax that map onto
/// `AiModelBackendPrediction.{candidateIndex, confidence}`.
struct RankResult {
    int index = -1;          /// argmax over scores; -1 only if empty
    float confidence = 0.0f; /// softmax probability of `index`, in [0, 1]
    float[] scores;          /// one raw score per candidate, in input order
}

/// True if this build links a real ONNX Runtime (vs the stub / mock fixture).
/// Maps to vibe3d's `AiModelStatus.ready` (true) vs `.unavailable` (false).
bool backendAvailable() {
    return v3d_ai_onnx_backend_available() != 0;
}

/// ABI revision of the linked shim. Compare against `v3d_ai_onnx.c.abiVersion`
/// at startup; a mismatch means the `.a` and the bindings drifted.
int linkedAbiVersion() {
    return v3d_ai_onnx_abi_version();
}

/// Owns a single inference session. Non-copyable; move or pass by ref.
struct OnnxRanker {
    private v3d_ai_onnx_session_t* session_;

    @disable this(this);

    /// Load a model. Throws `OnnxException` on failure (e.g. `errNotBuilt`
    /// for a stub build, `errModelLoad` for a bad path).
    this(string modelPath) {
        v3d_ai_onnx_error_t err;
        const status = v3d_ai_onnx_create(modelPath.toStringz, &session_, &err);
        if (status != v3d_ai_onnx_status_t.ok) {
            session_ = null;
            throw new OnnxException(status, errorText(err));
        }
    }

    ~this() {
        if (session_ !is null) {
            v3d_ai_onnx_destroy(session_);
            session_ = null;
        }
    }

    /// True while the session handle is live.
    bool valid() const { return session_ !is null; }

    /// Rank `candidateCount` candidates. `features` is a row-major
    /// [candidateCount][featureCount] matrix; its length must be an exact
    /// multiple of `candidateCount`. Throws `OnnxException` on shim error.
    RankResult rank(const(float)[] features, int candidateCount) {
        enforce(session_ !is null, "rank() on a destroyed OnnxRanker");
        enforce(candidateCount > 0, "candidateCount must be > 0");
        enforce(features.length % candidateCount == 0,
                "features length must be a multiple of candidateCount");
        const featureCount = cast(int)(features.length / candidateCount);
        enforce(featureCount > 0, "featureCount must be > 0");

        auto scores = new float[candidateCount];
        v3d_ai_onnx_error_t err;
        const status = v3d_ai_onnx_rank(session_, features.ptr, featureCount,
                                        candidateCount, scores.ptr, &err);
        if (status != v3d_ai_onnx_status_t.ok)
            throw new OnnxException(status, errorText(err));

        return summarise(scores);
    }

    /// Last error message bound to this session ("" if none).
    string lastError() const {
        if (session_ is null) return "";
        return fromStringz(v3d_ai_onnx_last_error(session_)).idup;
    }
}

private string errorText(ref const v3d_ai_onnx_error_t err) {
    // message is a fixed NUL-terminated buffer; copy out the live portion.
    return fromStringz(err.message.ptr).idup;
}

/// argmax + softmax over raw scores. Non-finite scores are treated as -inf so
/// a NaN row can't win; an all-non-finite set yields index -1, confidence 0.
private RankResult summarise(float[] scores) {
    RankResult r;
    r.scores = scores;
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
    if (best < 0) return r; // no finite score

    // Numerically stable softmax; confidence is the winning probability.
    double denom = 0.0;
    foreach (s; scores) {
        if (!s.isFinite) continue;
        denom += exp(cast(double)s - cast(double)bestScore);
    }
    r.index = best;
    r.confidence = denom > 0.0 ? cast(float)(1.0 / denom) : 0.0f;
    return r;
}

unittest {
    // Pure summarise() logic — no shim needed, runs on any build.
    auto r = summarise([1.0f, 3.0f, 2.0f]);
    assert(r.index == 1);
    assert(r.confidence > 0.0f && r.confidence <= 1.0f);

    auto tie = summarise([5.0f]);
    assert(tie.index == 0);
    assert(tie.confidence > 0.99f); // single candidate -> ~1.0

    auto empty = summarise([]);
    assert(empty.index == -1 && empty.confidence == 0.0f);

    auto nan = summarise([float.nan, float.nan]);
    assert(nan.index == -1);
}
