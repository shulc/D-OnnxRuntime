# Builds the C ABI shim as a static library: build/libonnxrt.a
#
# Backend is selected at configure time:
#   -DONNXRT_BACKEND=ort   (default) link real ONNX Runtime, downloaded as a
#                          prebuilt release (see cmake/FetchOnnxRuntime.cmake)
#   -DONNXRT_BACKEND=mock            deterministic fixture, no deps (tests/CI)
#   -DONNXRT_BACKEND=stub            honest "not built", no deps (tests/CI)
#
# The shim itself is a static lib so it folds into a distributable binary. With
# the ort backend the consumer also links the dynamic libonnxruntime fetched
# under build/onnxruntime/sdk/lib — see doc/DESIGN.md "Fetching & linking".

cmake_minimum_required(VERSION 3.18)
project(onnxrt C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ONNXRT_BACKEND "ort" CACHE STRING "Shim backend: ort | mock | stub")
set_property(CACHE ONNXRT_BACKEND PROPERTY STRINGS ort mock stub)

if(ONNXRT_BACKEND STREQUAL "ort")
    # Fetches a prebuilt ONNX Runtime release and exposes:
    #   ONNXRT_ORT_INCLUDE_DIR  — dir holding onnxruntime_c_api.h
    #   ONNXRT_ORT_LIB_DIR      — dir holding the dynamic libonnxruntime
    include(${CMAKE_SOURCE_DIR}/cmake/FetchOnnxRuntime.cmake)

    add_library(onnxrt STATIC csrc/onnxrt_ort.cpp)
    target_compile_features(onnxrt PRIVATE cxx_std_17)
    target_include_directories(onnxrt PRIVATE "${ONNXRT_ORT_INCLUDE_DIR}")
elseif(ONNXRT_BACKEND STREQUAL "mock")
    add_library(onnxrt STATIC csrc/onnxrt_mock.c)
else()
    add_library(onnxrt STATIC csrc/onnxrt_stub.c)
endif()

target_include_directories(onnxrt PUBLIC include)

# Land the archive at the build root so dub's `-L$PACKAGE_DIR/build` finds it.
set_target_properties(onnxrt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

message(STATUS "onnxrt backend: ${ONNXRT_BACKEND}")
