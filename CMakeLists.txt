cmake_minimum_required(VERSION 3.12)
project(hpc-final-project)
set(bmp_srcs bmp/bmp.c bmp/bmp.h)
##############################################
add_executable(MandelbrotSerial
  MandelbrotSerial/main.cpp
  ${bmp_srcs}
)
target_include_directories(MandelbrotSerial PRIVATE bmp/)
##############################################
add_executable(MandelbrotThreaded
	MandelbrotThreaded/main.cpp
  ${bmp_srcs}
  )
target_include_directories(MandelbrotThreaded PRIVATE bmp/)
##############################################
add_executable(MandelbrotDistributed
  MandelbrotDistributed/main.cpp
  ${bmp_srcs}
)
target_include_directories(MandelbrotDistributed PRIVATE bmp/ "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
target_link_libraries(MandelbrotDistributed PRIVATE
"C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib"
"C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpifec.lib"
"C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpifmc.lib"
)
target_link_libraries(MandelbrotDistributed ${MPI_C_LIBRARIES})
##############################################
# find_package(CUDA REQUIRED)
# cuda_add_executable(MandelbrotCuda
#   ${mandelbrot-cuda}/main.cu
#   ${bmp_srcs}
# )
# target_link_libraries(MandelbrotCuda ${CUDA_LIBRARIES})
# target_include_directories(MandelbrotCuda PRIVATE bmp/)
##############################################