cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files are UTF-8 without BOM. MSVC otherwise decodes them using the
# system code page (936/GBK on zh-CN Windows), which pairs the trailing byte of
# a Chinese comment with the following '\n' and swallows the next line.
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ---- OpenCV ----
# Default to the location produced by the official Windows self-extracting
# installer (extracts to C:\opencv\build). Override with -DOpenCV_DIR=... if needed.
if(NOT OpenCV_DIR)
    set(_ocv_hints "C:/opencv/build" "C:/opencv/opencv/build" "D:/opencv/build" "D:/opencv/opencv/build")
    find_file(_ocv_cfg OpenCVConfig.cmake PATHS ${_ocv_hints} NO_DEFAULT_PATH)
    if(_ocv_cfg)
        get_filename_component(OpenCV_DIR "${_ocv_cfg}" DIRECTORY)
    else()
        set(OpenCV_DIR "C:/opencv/build" CACHE PATH "Path to OpenCV build directory")
    endif()
endif()
find_package(OpenCV REQUIRED)

# ---- Eigen (header-only) ----
# Prefer a system-installed Eigen3, otherwise fall back to ./include
# (eigen-3.4.0's Eigen directory has been placed at Code/include/Eigen).
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    if(NOT EIGEN3_INCLUDE_DIR)
        set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Eigen include dir (parent of Eigen/)")
    endif()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Rasterizer
    main.cpp rasterizer.cpp rasterizer.hpp global.hpp
    Triangle.cpp Triangle.hpp Texture.cpp Texture.hpp
    Shader.hpp OBJ_Loader.h)

target_link_libraries(Rasterizer ${OpenCV_LIBS})
