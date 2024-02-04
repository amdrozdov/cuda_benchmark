cmake_minimum_required(VERSION 3.22)
project(cuda_ex CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_ex main.cu
        utils.cu
        compute.cu
        benchmark.cu
        utils.h
        compute.h
        benchmark.h
)

set_target_properties(cuda_ex PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)