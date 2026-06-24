# Builds the C ABI shim as a static library: build/libv3daionnx.a
#
# Backend is selected at configure time; the dub preBuild hook drives this.
#   -DV3D_AI_ONNX_BACKEND=mock   (default) deterministic fixture, no deps
#   -DV3D_AI_ONNX_BACKEND=stub             honest "not built", no deps
#   -DV3D_AI_ONNX_BACKEND=ort              real ONNX Runtime (needs the SDK)
#
# Static lib so consumers can fold it straight into a distributable binary
# (no .dylib/.so to ship alongside) — see doc/DESIGN.md "macOS build & link".

cmake_minimum_required(VERSION 3.16)
project(v3d_ai_onnx C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(V3D_AI_ONNX_BACKEND "mock" CACHE STRING "Shim backend: stub | mock | ort")
set_property(CACHE V3D_AI_ONNX_BACKEND PROPERTY STRINGS stub mock ort)

if(V3D_AI_ONNX_BACKEND STREQUAL "ort")
    add_library(v3daionnx STATIC csrc/v3d_ai_onnx_ort.cpp)
    target_compile_features(v3daionnx PRIVATE cxx_std_17)

    # ONNX Runtime SDK location: -DV3D_ORT_ROOT=/path/to/onnxruntime
    # with include/onnxruntime_c_api.h and lib/libonnxruntime.a beneath it.
    if(NOT DEFINED V3D_ORT_ROOT)
        message(FATAL_ERROR "ort backend needs -DV3D_ORT_ROOT=<onnxruntime sdk>")
    endif()
    target_include_directories(v3daionnx PRIVATE "${V3D_ORT_ROOT}/include")
    # The consumer (dub lflags / DESIGN notes) is responsible for putting
    # ${V3D_ORT_ROOT}/lib on the link line — a static lib does not embed it.
elseif(V3D_AI_ONNX_BACKEND STREQUAL "stub")
    add_library(v3daionnx STATIC csrc/v3d_ai_onnx_stub.c)
else()
    add_library(v3daionnx STATIC csrc/v3d_ai_onnx_mock.c)
endif()

target_include_directories(v3daionnx PUBLIC include)

# Land the archive at the build root so dub's `-L$PACKAGE_DIR/build` finds it.
set_target_properties(v3daionnx PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

message(STATUS "v3d_ai_onnx backend: ${V3D_AI_ONNX_BACKEND}")
