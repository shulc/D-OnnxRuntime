/// Smoke test for the D-OnnxRuntime shim.
///
/// Exercises the full create -> score -> destroy lifetime through the RAII
/// wrapper. It needs a model at examples/model.onnx with input
/// [rows, features] f32 -> output [rows] f32; without one (or on a stub build)
/// it reports the error cleanly. Either way it ends in `OK`, so it doubles as
/// a build/link sanity check.
///
///   dub run --config=score_smoke
module score_smoke;

import std.stdio;

import onnxrt.backend;
import onnxrt.c : abiVersion, onnxrt_status_t;

int main() {
    writefln("abi: binding=%d linked=%d", abiVersion, linkedAbiVersion());
    if (linkedAbiVersion() != abiVersion) {
        stderr.writefln("ABI mismatch: binding %d vs linked %d",
                        abiVersion, linkedAbiVersion());
        return 1;
    }
    writefln("backend available: %s", backendAvailable());

    // Three rows, two features each (row-major).
    const rowCount = 3;
    const float[] input = [
        0.2f, 0.9f,
        0.8f, 0.1f,
        0.5f, 0.5f,
    ];

    try {
        auto session = OnnxSession("examples/model.onnx");
        auto scores = session.score(input, rowCount);
        auto r = rank(scores);
        writefln("scores: %s", scores);
        writefln("argmax index: %d  confidence: %.4f", r.index, r.confidence);
    } catch (OnnxException e) {
        // Expected without a model file (errModelLoad) or on a stub build
        // (errNotBuilt): the link is still proven good.
        writefln("no inference (%s): %s", e.status, e.msg);
    }

    writeln("OK");
    return 0;
}
