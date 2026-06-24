/// Smoke test for the D-OnnxRuntime shim.
///
/// Exercises the full create -> rank -> destroy lifetime and the
/// score -> prediction mapping through the D RAII wrapper. With the default
/// `mock` backend it scores a small feature matrix deterministically; with the
/// `stub` backend it confirms create fails cleanly with `errNotBuilt`. Either
/// way it ends in `OK`, so it doubles as a build sanity check.
///
///   dub run --config=rank_smoke
module rank_smoke;

import std.stdio;

import v3d_ai_onnx.backend;
import v3d_ai_onnx.c : abiVersion, v3d_ai_onnx_status_t;

int main() {
    writefln("abi: binding=%d linked=%d", abiVersion, linkedAbiVersion());
    enforceAbi();

    writefln("backend available: %s", backendAvailable());

    // Three candidates, two features each (row-major). Feature 0 is the
    // dominant signal under the mock's weighting; candidate 1 should win.
    const candidateCount = 3;
    const float[] features = [
        0.2f, 0.9f,   // candidate 0
        0.8f, 0.1f,   // candidate 1  <- highest feature-0
        0.5f, 0.5f,   // candidate 2
    ];

    try {
        auto ranker = OnnxRanker("model/ranker.onnx");
        auto result = ranker.rank(features, candidateCount);
        writefln("scores: %s", result.scores);
        writefln("argmax index: %d  confidence: %.4f",
                 result.index, result.confidence);
        assert(result.index == 1, "mock weighting should pick candidate 1");
        assert(result.confidence > 0.0f && result.confidence <= 1.0f);
    } catch (OnnxException e) {
        if (e.status == v3d_ai_onnx_status_t.errNotBuilt) {
            // Expected on a stub build: no real backend linked.
            writefln("stub build, no backend: %s", e.msg);
        } else {
            stderr.writefln("unexpected shim error (%s): %s", e.status, e.msg);
            return 1;
        }
    }

    writeln("OK");
    return 0;
}

private void enforceAbi() {
    if (linkedAbiVersion() != abiVersion) {
        stderr.writefln("ABI mismatch: binding %d vs linked %d",
                        abiVersion, linkedAbiVersion());
        assert(false);
    }
}
