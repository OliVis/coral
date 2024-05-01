cmake_minimum_required(VERSION 3.18)
project(coral C CXX)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

set(SCRIPT_NAME "fft_model.py")

find_package(Threads REQUIRED)
find_library(LIBEDGETPU edgetpu REQUIRED)
find_library(LIBRTLSDR rtlsdr REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/tensorflow/tensorflow/lite EXCLUDE_FROM_ALL)

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
  ${LIBEDGETPU}
  ${LIBRTLSDR}
  tensorflow-lite
)

configure_file(
  ${CMAKE_SOURCE_DIR}/${SCRIPT_NAME}
  ${CMAKE_BINARY_DIR}/${SCRIPT_NAME}
  COPYONLY
)
