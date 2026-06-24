#!/usr/bin/env python3
"""Generate a tiny valid model.onnx with no third-party deps.

The model is a single ReduceSum over the feature axis: input X [N, F] f32 ->
output Y [N] f32 (one score per row = sum of its features). That matches the
shim's model contract and lets examples/score_smoke.d show a real inference.

Writes examples/model.onnx. Pure-stdlib protobuf encoder so it runs anywhere
(no onnx / onnxruntime Python package needed).

  python3 tools/make_test_model.py
"""
import os
import struct


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def tag(field: int, wire: int) -> bytes:
    return varint((field << 3) | wire)


def f_varint(field: int, value: int) -> bytes:
    return tag(field, 0) + varint(value)


def f_bytes(field: int, payload: bytes) -> bytes:
    return tag(field, 2) + varint(len(payload)) + payload


def f_str(field: int, s: str) -> bytes:
    return f_bytes(field, s.encode("utf-8"))


# --- TypeProto for a float tensor with the given symbolic dims ---------------
def value_info(name: str, dims: list[str]) -> bytes:
    shape = b"".join(f_bytes(1, f_str(2, d)) for d in dims)  # Dimension.dim_param
    tensor_type = f_varint(1, 1) + f_bytes(2, shape)         # elem_type=FLOAT, shape
    type_proto = f_bytes(1, tensor_type)                     # TypeProto.tensor_type
    return f_str(1, name) + f_bytes(2, type_proto)           # ValueInfoProto


# --- AttributeProto helpers --------------------------------------------------
def attr_ints(name: str, values: list[int]) -> bytes:
    body = f_str(1, name)
    body += b"".join(f_varint(8, v) for v in values)  # ints
    body += f_varint(20, 7)                            # type = INTS
    return body


def attr_int(name: str, value: int) -> bytes:
    return f_str(1, name) + f_varint(3, value) + f_varint(20, 2)  # i, type=INT


def main() -> None:
    # ReduceSum(axes=[1], keepdims=0): X[N,F] -> Y[N]. (opset 11: axes is an
    # attribute; opset 13+ moved it to an input.)
    node = (
        f_str(1, "X")            # input
        + f_str(2, "Y")          # output
        + f_str(3, "reduce")     # name
        + f_str(4, "ReduceSum")  # op_type
        + f_bytes(5, attr_ints("axes", [1]))
        + f_bytes(5, attr_int("keepdims", 0))
    )

    graph = (
        f_bytes(1, node)                          # node
        + f_str(2, "g")                           # name
        + f_bytes(11, value_info("X", ["N", "F"]))  # input
        + f_bytes(12, value_info("Y", ["N"]))       # output
    )

    opset = f_varint(2, 11)  # OperatorSetIdProto.version (domain "" default)

    model = (
        f_varint(1, 7)               # ir_version
        + f_str(2, "onnxrt-test")    # producer_name
        + f_bytes(7, graph)          # graph
        + f_bytes(8, opset)          # opset_import
    )

    out = os.path.join(os.path.dirname(__file__), "..", "examples", "model.onnx")
    out = os.path.abspath(out)
    with open(out, "wb") as fh:
        fh.write(model)
    print(f"wrote {out} ({len(model)} bytes)")


if __name__ == "__main__":
    main()
