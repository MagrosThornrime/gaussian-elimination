cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_ARCHITECTURES "native")

project(gauss LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(gauss
        main.cu
        src/transaction.cu
        src/graphs.cpp
        src/foata.cpp
        src/elimination.cu
        include/transaction.cuh
        include/graphs.h
        include/foata.h
        include/elimination.cuh
)