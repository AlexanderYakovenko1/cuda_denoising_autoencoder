cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

project(my_cuda_autoencoder LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

find_package(ZLIB REQUIRED) # cnpy requirement
add_executable(denoiser main.cu include/cnpy.cpp include/stb_image.h include/stb_image_write.h layers.cu layers.cuh)

include_directories(${ZLIB_INCLUDE_DIRS})
target_link_libraries(denoiser ${ZLIB_LIBRARIES})

set_target_properties(denoiser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
