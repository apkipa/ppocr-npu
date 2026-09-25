cmake_minimum_required(VERSION 3.24)
project(ppocr_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/O2 /W3 /EHsc /utf-8 /std:c++17)
endif()

# Locate Level Zero SDK
if(NOT DEFINED ENV{LEVEL_ZERO_V1_SDK_PATH})
    set(LEVEL_ZERO_V1_SDK_PATH "C:/Program Files/LevelZeroSDK/1.32.0")
else()
    set(LEVEL_ZERO_V1_SDK_PATH "$ENV{LEVEL_ZERO_V1_SDK_PATH}")
endif()

set(LEVEL_ZERO_EXT_ROOT "D:/MyData/Documents/GitHub/ExternalRepos/intel/level-zero-npu-extensions")
set(NPU_DRIVER_API_ROOT "D:/MyData/Documents/GitHub/ExternalRepos/intel/linux-npu-driver/firmware/include/api")

find_library(LEVEL_ZERO_LOADER ze_loader PATHS "${LEVEL_ZERO_V1_SDK_PATH}/lib" REQUIRED)

add_executable(ppocr_npu
    src/main.cpp
    src/npu_engine.cpp
    src/wic_decoder.cpp
    src/postprocess.cpp
    src/clipper.cpp
)

target_include_directories(ppocr_npu PRIVATE
    "src"
    "${LEVEL_ZERO_V1_SDK_PATH}/include/level_zero"
    "${LEVEL_ZERO_EXT_ROOT}"
    "${NPU_DRIVER_API_ROOT}"
)

target_link_libraries(ppocr_npu PRIVATE
    "${LEVEL_ZERO_LOADER}"
    windowscodecs.lib
    ole32.lib
    user32.lib
    gdi32.lib
)
