cmake_minimum_required(VERSION 3.12)
project(hpc-final-project)
set(mandelbrot-serial MandelbrotSerial)
add_executable(MandelbrotSerial
  ${mandelbrot-serial}/main.cpp
  ${mandelbrot-serial}/bmp.c
  ${mandelbrot-serial}/bmp.h
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(MandelbrotSerial PRIVATE /W4 /permissive-)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(MandelbrotSerial PRIVATE -Wall -Wextra -pedantic)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(MandelbrotSerial PRIVATE -Wall -Wextra -pedantic)
endif()
add_executable(MandelbrotSerialTest
  ${mandelbrot-serial}/test.cpp
)
include(FetchContent)
FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googleTest)
target_link_libraries(MandelbrotSerialTest gtest_main)