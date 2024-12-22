cmake_minimum_required(VERSION 3.21)
project(gauss)

set(CMAKE_CXX_STANDARD 20)

add_executable(gauss
        main.cpp
        src/transaction.cpp
        src/graphs.cpp
        src/algorithm.cpp
        include/transaction.h
        include/graphs.h
        include/algorithm.h)
