cmake_minimum_required(VERSION 3.12)
project(hpc-final-project)
set(mandelbrot-serial MandelbrotSerial)
set(mandelbrot-threaded MandelbrotThreaded)
set(mandelbrot-cuda MandelbrotGPU)
set(mandelbrot-dist MandelbrotDistributed)
add_executable(MandelbrotSerial
  ${mandelbrot-serial}/main.cpp
  ${mandelbrot-serial}/bmp.c
  ${mandelbrot-serial}/bmp.h
)
add_executable(MandelbrotThreaded
	${mandelbrot-threaded}/main.cpp
	${mandelbrot-threaded}/bmp.c
	${mandelbrot-threaded}/bmp.h
)
add_executable(MandelbrotDistributed
  ${mandelbrot-dist}/main.cpp
  ${mandelbrot-dist}/bmp.c
  ${mandelbrot-dist}/bmp.h
)
find_package(CUDA REQUIRED)
cuda_add_executable(MandelbrotCuda
  ${mandelbrot-cuda}/main.cu
  ${mandelbrot-cuda}/bmp.c
  ${mandelbrot-cuda}/bmp.h
)
target_link_libraries(MandelbrotCuda ${CUDA_LIBRARIES})
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(MandelbrotSerial PRIVATE /W4 /permissive-)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(MandelbrotSerial PRIVATE -Wall -Wextra -pedantic)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(MandelbrotSerial PRIVATE -Wall -Wextra -pedantic)
endif()
# add_executable(MandelbrotSerialTest
#   ${mandelbrot-serial}/test.cpp
# )
# include(FetchContent)
# FetchContent_Declare(
# googletest
# GIT_REPOSITORY https://github.com/google/googletest.git
# GIT_TAG v1.14.0
# )
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googleTest)
# target_link_libraries(MandelbrotSerialTest gtest_main)