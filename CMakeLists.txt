cmake_minimum_required(VERSION 3.18)
project(coral C CXX)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR "src")
set(INCLUDE_DIR "include")

find_package(Threads REQUIRED)

add_subdirectory(tensorflow/tensorflow/lite EXCLUDE_FROM_ALL)

find_library(LIBEDGETPU edgetpu REQUIRED)

find_library(LIBRTLSDR rtlsdr REQUIRED)

add_executable(coral)
target_sources(coral PRIVATE
  ${SOURCE_DIR}/main.cpp
  ${SOURCE_DIR}/interpreter.cpp
  ${SOURCE_DIR}/rtlsdr.cpp
  ${SOURCE_DIR}/buffer.cpp
)
target_include_directories(coral PRIVATE ${INCLUDE_DIR})

target_link_libraries(coral PRIVATE
  ${CMAKE_THREAD_LIBS_INIT}
  tensorflow-lite
  ${LIBEDGETPU}
  ${LIBRTLSDR}
)
