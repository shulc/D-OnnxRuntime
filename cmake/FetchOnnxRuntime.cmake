# Downloads a prebuilt ONNX Runtime release (CPU, dynamic library) and exposes
# its headers + lib to the parent CMakeLists. No source build, no submodule —
# just a small per-platform tarball (8-70 MB) fetched + verified + extracted at
# configure time and cached under build/onnxruntime/sdk.
#
# Pinned to 1.22.0: the last release that ships all four target platforms,
# including Intel macOS (osx-x86_64), which 1.23+ dropped. The shim uses ONNX
# Runtime's stable C API, so the exact version is not load-bearing.
#
# Exposes: ONNXRT_ORT_INCLUDE_DIR, ONNXRT_ORT_LIB_DIR.
# Escape hatch: -DONNXRT_ORT_DIR=/path/to/sdk (with include/ + lib/) to use a
# hand-placed SDK (e.g. a custom or statically-built ONNX Runtime).

set(ONNXRT_VERSION "1.22.0" CACHE STRING "ONNX Runtime release version")

if(ONNXRT_ORT_DIR)
    set(ONNXRT_ORT_INCLUDE_DIR "${ONNXRT_ORT_DIR}/include")
    set(ONNXRT_ORT_LIB_DIR "${ONNXRT_ORT_DIR}/lib")
    message(STATUS "onnxruntime: using ONNXRT_ORT_DIR=${ONNXRT_ORT_DIR}")
    return()
endif()

# --- pick the release asset for this platform --------------------------------
set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _arch MATCHES "x86_64|AMD64")
    set(_asset "onnxruntime-linux-x64-${ONNXRT_VERSION}.tgz")
elseif(APPLE AND _arch MATCHES "arm64|aarch64")
    set(_asset "onnxruntime-osx-arm64-${ONNXRT_VERSION}.tgz")
elseif(APPLE AND _arch MATCHES "x86_64")
    set(_asset "onnxruntime-osx-x86_64-${ONNXRT_VERSION}.tgz")
elseif(WIN32 AND _arch MATCHES "x86_64|AMD64")
    set(_asset "onnxruntime-win-x64-${ONNXRT_VERSION}.zip")
else()
    message(FATAL_ERROR
        "No prebuilt ONNX Runtime ${ONNXRT_VERSION} asset for "
        "${CMAKE_SYSTEM_NAME}/${_arch}. Pass -DONNXRT_ORT_DIR=<sdk> to use a "
        "hand-placed SDK, or -DONNXRT_BACKEND=mock for a deps-free build.")
endif()

# --- known SHA-256s (1.22.0 only) --------------------------------------------
set(_sha "")
if(ONNXRT_VERSION STREQUAL "1.22.0")
    if(_asset MATCHES "linux-x64")
        set(_sha "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3")
    elseif(_asset MATCHES "osx-arm64")
        set(_sha "cab6dcbd77e7ec775390e7b73a8939d45fec3379b017c7cb74f5b204c1a1cc07")
    elseif(_asset MATCHES "osx-x86_64")
        set(_sha "e4ec94a7696de74fb1b12846569aa94e499958af6ffa186022cfde16c9d617f0")
    elseif(_asset MATCHES "win-x64")
        set(_sha "174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da")
    endif()
else()
    message(WARNING "No pinned SHA-256 for ONNX Runtime ${ONNXRT_VERSION}; "
                    "downloading ${_asset} without integrity check.")
endif()

set(_ort_root "${CMAKE_BINARY_DIR}/onnxruntime")
set(_sdk "${_ort_root}/sdk")
set(ONNXRT_ORT_INCLUDE_DIR "${_sdk}/include")
set(ONNXRT_ORT_LIB_DIR "${_sdk}/lib")

# --- fetch + verify + extract (idempotent) -----------------------------------
if(NOT EXISTS "${ONNXRT_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    set(_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VERSION}/${_asset}")
    set(_archive "${_ort_root}/${_asset}")
    message(STATUS "onnxruntime: downloading ${_url}")

    set(_dl_args "${_url}" "${_archive}" SHOW_PROGRESS STATUS _dl TLS_VERIFY ON)
    if(_sha)
        list(APPEND _dl_args EXPECTED_HASH "SHA256=${_sha}")
    endif()
    file(DOWNLOAD ${_dl_args})
    list(GET _dl 0 _code)
    if(NOT _code EQUAL 0)
        list(GET _dl 1 _msg)
        file(REMOVE "${_archive}")
        message(FATAL_ERROR "onnxruntime download failed: ${_msg}")
    endif()

    set(_extract "${_ort_root}/extract")
    file(REMOVE_RECURSE "${_extract}")
    file(MAKE_DIRECTORY "${_extract}")
    file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract}")

    # The tarball holds a single top dir (onnxruntime-<plat>-<ver>); promote it
    # to a stable path so include/lib locations don't carry the platform name.
    file(GLOB _top LIST_DIRECTORIES true "${_extract}/*")
    list(GET _top 0 _topdir)
    file(REMOVE_RECURSE "${_sdk}")
    file(RENAME "${_topdir}" "${_sdk}")
    file(REMOVE_RECURSE "${_extract}")
    file(REMOVE "${_archive}")
endif()

if(NOT EXISTS "${ONNXRT_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR "onnxruntime: extracted SDK is missing the C API header")
endif()

message(STATUS "onnxruntime: ${ONNXRT_VERSION} (${_asset}) -> ${_sdk}")
