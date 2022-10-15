cmake_minimum_required(VERSION 3.23)
project(autodiff)

set(CMAKE_CXX_STANDARD 14)

add_library(autodiff INTERFACE)
target_include_directories(autodiff INTERFACE .)

add_subdirectory(tests)