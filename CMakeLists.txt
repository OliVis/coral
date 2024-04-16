cmake_minimum_required(VERSION 3.18)
project(minimal C CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(tensorflow/tensorflow/lite EXCLUDE_FROM_ALL)

find_library(LIBEDGETPU edgetpu REQUIRED)

add_executable(minimal minimal.cc)

target_link_libraries(minimal tensorflow-lite ${LIBEDGETPU})
