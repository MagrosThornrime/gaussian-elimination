cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_ARCHITECTURES "native")

project(gauss LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(gauss
        main.cu
        src/transaction.cpp
        src/graphs.cpp
        src/algorithm.cpp
        include/transaction.h
        include/graphs.h
        include/algorithm.h
)