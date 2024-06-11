cmake_minimum_required(VERSION 3.18)
project(coral C CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

set(TENSORFLOW_VERSION v2.16.1)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
find_library(LIBEDGETPU edgetpu REQUIRED)
find_library(LIBRTLSDR rtlsdr REQUIRED)

FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue
  GIT_TAG master
)

FetchContent_MakeAvailable(readerwriterqueue)

FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG ${TENSORFLOW_VERSION}
  SOURCE_SUBDIR tensorflow/lite
)

FetchContent_MakeAvailable(tensorflow)

add_executable(coral)

target_sources(coral PRIVATE
  ${SOURCE_DIR}/main.cpp
  ${SOURCE_DIR}/interpreter.cpp
  ${SOURCE_DIR}/rtlsdr.cpp
)

target_include_directories(coral PRIVATE ${INCLUDE_DIR})

target_link_libraries(coral PRIVATE
  ${CMAKE_THREAD_LIBS_INIT}
  ${LIBEDGETPU}
  ${LIBRTLSDR}
  readerwriterqueue
  tensorflow-lite
)

target_compile_options(coral PRIVATE -Wall -Wextra)

file(GLOB SCRIPTS ${CMAKE_SOURCE_DIR}/scripts/*.py)

file(COPY ${SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR})
