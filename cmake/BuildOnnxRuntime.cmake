# Builds ONNX Runtime from the extern/onnxruntime submodule via its own
# build.py, as a shared library (the reliable, well-trodden path — full static
# linking of ONNX Runtime drags in dozens of split archives). Our shim stays a
# static lib; only libonnxruntime ships as a shared object next to the binary.
#
# First build is long (ONNX Runtime fetches protobuf/abseil/... and compiles
# everything); it is incremental afterwards and the artifacts under
# build/onnxruntime/ are reused. Exposes ONNXRT_ORT_INCLUDE_DIR + the
# onnxruntime_build target for the parent CMakeLists to depend on.

include(ExternalProject)

set(ORT_SRC "${CMAKE_SOURCE_DIR}/extern/onnxruntime")
set(ORT_BUILD "${CMAKE_BINARY_DIR}/onnxruntime")
set(ORT_LIB_DIR "${ORT_BUILD}/${CMAKE_BUILD_TYPE}")

# The C API header lives in the source tree (not generated).
set(ONNXRT_ORT_INCLUDE_DIR "${ORT_SRC}/include/onnxruntime/core/session")

if(NOT EXISTS "${ONNXRT_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR
        "extern/onnxruntime is empty — run:\n"
        "  git submodule update --init --recursive extern/onnxruntime\n"
        "(the dub preBuild hook does this automatically).")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ORT_SHARED
    "${ORT_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}onnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Prefer Ninja for the inner build when available; fall back to the default.
find_program(NINJA_EXE ninja)
set(ORT_GEN_ARG "")
if(NINJA_EXE)
    set(ORT_GEN_ARG --cmake_generator Ninja)
endif()

ExternalProject_Add(onnxruntime_build
    SOURCE_DIR        "${ORT_SRC}"
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE    0
    BUILD_COMMAND
        ${Python3_EXECUTABLE} "${ORT_SRC}/tools/ci_build/build.py"
            --build_dir "${ORT_BUILD}"
            --config ${CMAKE_BUILD_TYPE}
            --parallel
            --skip_tests
            --skip_submodule_sync
            --build_shared_lib
            --compile_no_warning_as_error
            ${ORT_GEN_ARG}
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${ORT_SHARED}"
    USES_TERMINAL_BUILD 1)

# Surface the link location for the parent / consumer (dub lflags mirror this).
set(ONNXRT_ORT_LIB_DIR "${ORT_LIB_DIR}")
message(STATUS "onnxruntime: build from source -> ${ORT_SHARED}")
