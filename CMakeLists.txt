cmake_minimum_required(VERSION 3.18)
project(minimal C CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/tensorflow/tensorflow/lite EXCLUDE_FROM_ALL)

add_library(edgetpu SHARED IMPORTED GLOBAL)
set_target_properties(edgetpu PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external/libedgetpu/out/direct/k8/libedgetpu.so.1.0)

add_executable(minimal minimal.cc)

target_link_libraries(minimal tensorflow-lite edgetpu)
