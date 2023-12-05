cmake_minimum_required(VERSION 3.12)
project(hpc-final-project)
set(bmp_srcs bmp/bmp.c bmp/bmp.h)
set(compute_srcs computePixel.h computePixel.c)
set(compute_cuda_srcs computePixel.cuh)
set(defs defs.h)
set(timer_srcs timer.c timer.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
##############################################
add_executable(MandelbrotSerial
  MandelbrotSerial/main.cpp
  ${bmp_srcs}
  ${compute_srcs}
  ${timer_srcs}
)
target_include_directories(MandelbrotSerial PRIVATE bmp/)
##############################################
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_executable(MandelbrotThreaded
MandelbrotThreaded/main.cpp
${bmp_srcs}
${compute_srcs}
${timer_srcs}
)
target_link_libraries(MandelbrotThreaded Threads::Threads)
target_include_directories(MandelbrotThreaded PRIVATE bmp/)
##############################################
find_package(MPI REQUIRED)
add_executable(MandelbrotDistributed
  MandelbrotDistributed/main.cpp
  ${bmp_srcs}
  ${compute_srcs}
  ${timer_srcs}
)
target_include_directories(MandelbrotDistributed PRIVATE bmp/)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(MandelbrotDistributed ${MPI_CXX_LIBRARIES})
##############################################
find_package(CUDA REQUIRED)
cuda_add_executable(MandelbrotCuda
  MandelbrotGPU/main.cu
  ${bmp_srcs}
  ${compute_cuda_srcs}
  ${timer_srcs}
)
target_link_libraries(MandelbrotCuda ${CUDA_LIBRARIES})
target_include_directories(MandelbrotCuda PRIVATE bmp/)
##############################################
cuda_add_executable(MandelbrotGPUDistributed
  MandelbrotGpuDistributed/main.cu
  ${bmp_srcs}
  ${compute_cuda_srcs}
  ${timer_srcs}
)
target_link_libraries(MandelbrotGPUDistributed ${CUDA_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_include_directories(MandelbrotGPUDistributed PRIVATE bmp/)
##############################################
add_executable(Validate
validationFunc.cpp
)
##############################################
