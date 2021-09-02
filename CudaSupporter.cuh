#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AlignerConsts.h"

namespace cudaSupporter
{
	namespace kernel
	{
		__global__ void Get1stMoment(const uchar* sceneData, uchar* moment1stData, const int sceneW, const int sceneH
			, const ushort objectLen);
		__global__ void GetBinaryMap(const uchar* sceneData, const uchar* moment1stData, uchar* binaryMapData, const int sceneW
			, const int sceneH, const ushort objectLen, const float* objectMoment);
		__global__ void GetAdaptiveBinaryMap(uchar* binaryMapData, const uchar* moment1stData, const int sceneW, const int sceneH
			, const ushort objectLen);
		__global__ void GetRMat(const uchar* originalSceneData, const ushort originalSceneW, const int* statsData
			, const uchar statsLen, const ushort objectLen, const ushort radius, const uint* circleSumVecParam
			, const uchar* binaryMapData, const ushort binaryMapW, float* rMat);
		__device__ int SumCircleDevice(const uchar* originalSceneData, const ushort originalSceneW, int centerX, int centerY, int radius);
		__global__ void SortRXY(const float* rMat, const ushort rMatW, const ushort rMatH, float* outR, ushort* outX
			, ushort* outY);
	}

	void LaunchGet1stMoment(const dim3& dimGridFilter1, const dim3& dimBlockFilter1, const uint sharedMemSizeFilter1
		, const uchar* sceneData, uchar* moment1stData, const int sceneW, const int sceneH, const ushort objectLen);
	void LaunchGetBinaryMap(const dim3& dimGridFilter1, const dim3& dimBlockFilter1, const uint sharedMemSizeFilter1
		, const uchar* sceneData, const uchar* moment1stData, uchar* binaryMapData, const int sceneW, const int sceneH
		, const ushort objectLen, const float* objectMoment);
	void LaunchGetAdaptiveBinaryMap(const dim3& dimGridFilter1, const dim3& dimBlockFilter1, uchar* binaryMapData, const uchar* moment1stData
		, const int sceneW, const int sceneH, const ushort objectLen);
	void LaunchGetRMat(const dim3& dimGridFilter2, const dim3& dimBlockFilter2, const uint sharedMemSizeFilter2
		, const uchar* originalSceneData, const ushort originalSceneW, const int* statsData, const uchar statsLen
		, const ushort objectLen, const ushort radius, const uint* circleSumVecParam, const uchar* binaryMapData
		, const ushort binaryMapW, float* rMat);
	void LaunchSortRXY(const dim3& dimGridFilter3, const dim3& dimBlockFilter3, const float* rMat, const ushort rMatW, const ushort rMatH
		, float* outR, ushort* outX, ushort* outY);
}