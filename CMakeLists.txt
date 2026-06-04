cmake_minimum_required(VERSION 3.20)
project(MoeLocalEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# Ensure warnings are treated as defects (errors) if preferred, but we will useWall, Wextra, Wpedantic
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -Werror)
endif()

find_package(Threads REQUIRED)

# Check for ONNX Runtime (Vitis AI EP uses standard ONNX Runtime APIs)
set(ONNXRUNTIME_SEARCH_PATHS "C:/Program Files/RyzenAI/1.7.1/onnxruntime")
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    list(APPEND ONNXRUNTIME_SEARCH_PATHS "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h 
    PATHS ${ONNXRUNTIME_SEARCH_PATHS} /usr/include /usr/local/include /opt/onnxruntime/include
    PATH_SUFFIXES include include/onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARY 
    NAMES onnxruntime 
    PATHS ${ONNXRUNTIME_SEARCH_PATHS} /usr/lib /usr/local/lib /opt/onnxruntime/lib
    PATH_SUFFIXES lib bin
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
    add_compile_definitions(HAS_ONNXRUNTIME)
else()
    message(WARNING "ONNX Runtime not found. Building with CPU emulation fallback (no mock sleep timers).")
endif()

# Shared library target for the Python ctypes wrapper
add_library(kovidhar SHARED
    src/memory_manager.cpp
    src/expert_cache.cpp
    src/router_thread.cpp
    src/prefetch_thread.cpp
    src/benchmark.cpp
    src/inference_api.cpp
    src/visualizer_server.cpp
)
target_include_directories(kovidhar PUBLIC src)
target_compile_definitions(kovidhar PRIVATE BUILDING_MOE_ENGINE)
target_link_libraries(kovidhar PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

if(WIN32)
    target_link_libraries(kovidhar PRIVATE ws2_32)
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    target_include_directories(kovidhar PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(kovidhar PUBLIC ${ONNXRUNTIME_LIBRARY})
endif()

# Main executable target for the performance benchmark
add_executable(moe_benchmark
    src/main.cpp
)
target_link_libraries(moe_benchmark PRIVATE kovidhar Threads::Threads)

# Visualizer server target
add_executable(moe_visualizer_server
    src/visualizer_server.cpp
)
target_compile_definitions(moe_visualizer_server PRIVATE BUILDING_MOE_ENGINE)
target_link_libraries(moe_visualizer_server PRIVATE Threads::Threads)

if(WIN32)
    target_link_libraries(moe_visualizer_server PRIVATE ws2_32)
    
    # Copy ONNX Runtime and Vitis AI EP DLLs to build directory post-build
    add_custom_command(TARGET moe_benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/Program Files/RyzenAI/1.7.1/onnxruntime/bin"
        $<TARGET_FILE_DIR:moe_benchmark>
    )
    add_custom_command(TARGET moe_visualizer_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/Program Files/RyzenAI/1.7.1/onnxruntime/bin"
        $<TARGET_FILE_DIR:moe_visualizer_server>
    )
endif()
