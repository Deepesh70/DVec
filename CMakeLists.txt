cmake_minimum_required(VERSION 3.20)
project(dvec_daemon VERSION 0.1.0 LANGUAGES C CXX)

# Enforce C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. Locate ONNX Runtime
# Can be overridden via -DONNXRUNTIME_ROOT_DIR=<path> or $ENV{ORT_DIR}
set(ONNXRUNTIME_ROOT_DIR "$ENV{ORT_DIR}" CACHE PATH "Root directory of the ONNX Runtime distribution")

# Automatic default detection for Windows user environment
if (NOT ONNXRUNTIME_ROOT_DIR AND WIN32)
    if (EXISTS "C:/Users/aadit/Downloads/onnxruntime-win-x64-1.30.1/onnxruntime-win-x64-1.30.1")
        set(ONNXRUNTIME_ROOT_DIR "C:/Users/aadit/Downloads/onnxruntime-win-x64-1.30.1/onnxruntime-win-x64-1.30.1")
    endif()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT_DIR}/include
    REQUIRED
    DOC "Path to onnxruntime_cxx_api.h"
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT_DIR}/lib ${ONNXRUNTIME_ROOT_DIR}/lib/x64
    REQUIRED
    DOC "Path to onnxruntime library (.lib or .so)"
)

message(STATUS "Found ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# 2. Define Executable Target
add_executable(dvec_daemon
    cpp/main.cpp
    cpp/daemon.hpp
    cpp/audio_device.hpp
    cpp/audio_processor.hpp
    cpp/onnx_model.hpp
    cpp/spsc_ring_buffer.hpp
)

target_include_directories(dvec_daemon PRIVATE
    cpp
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# 3. Compiler Options & Optimization
if (MSVC)
    target_compile_options(dvec_daemon PRIVATE /O2 /W3 /permissive-)
else()
    target_compile_options(dvec_daemon PRIVATE
        -O3
        -Wall
        -Wextra
        -Wno-stringop-overflow
    )
endif()

# 4. Link Libraries
target_link_libraries(dvec_daemon PRIVATE
    ${ONNXRUNTIME_LIBRARY}
)

# Platform-specific audio & system libraries
if (WIN32)
    target_link_libraries(dvec_daemon PRIVATE ole32 winmm)
elseif (APPLE)
    find_library(COREAUDIO_LIB CoreAudio REQUIRED)
    find_library(AUDIOUNIT_LIB AudioUnit REQUIRED)
    target_link_libraries(dvec_daemon PRIVATE ${COREAUDIO_LIB} ${AUDIOUNIT_LIB})
elseif (UNIX)
    target_link_libraries(dvec_daemon PRIVATE pthread dl m asound)
endif()

# 5. Post-Build: Copy onnxruntime.dll to Output Directory (Windows)
if (WIN32)
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS ${ONNXRUNTIME_ROOT_DIR}/lib ${ONNXRUNTIME_ROOT_DIR}/bin
        DOC "Path to onnxruntime.dll"
    )
    if (ONNXRUNTIME_DLL)
        add_custom_command(TARGET dvec_daemon POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:dvec_daemon>
            COMMENT "Copying onnxruntime.dll to executable directory..."
        )
    endif()
endif()
