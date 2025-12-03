cmake_minimum_required(VERSION 4.0)
project(cuda_labs CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_labs main.cu)

set_target_properties(cuda_labs PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
