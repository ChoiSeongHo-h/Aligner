cmake_minimum_required(VERSION 2.8)
project(Aligner)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(/opt/pylon/include)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(/usr/local/cuda-11.4/include)

link_directories(/opt/pylon/lib)

# SETUP FOR CPP FILES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# COMPILE CU FILES
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_53,code=sm_53; -std=c++11")
CUDA_COMPILE(CudaSupporter CudaSupporter.cu)

# COMPILE AND LINK
cuda_add_executable(
	Aligner
	${CMAKE_CURRENT_SOURCE_DIR}/AlignerLauncher.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Aligner.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PatternMatching.cpp
	${CudaSupporter}
	${CMAKE_CURRENT_SOURCE_DIR}/MemorySharing.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Grabber.cpp
#	${CMAKE_CURRENT_SOURCE_DIR}/Webcam.cpp
)

target_link_libraries(
	Aligner
	${OpenCV_LIBS}
	pylonutility
	pylonbase
	GCBase_gcc_v3_1_Basler_pylon
)
