#include "CudaSupporter.cuh"

__device__ int cudaSupporter::kernel::SumCircleDevice(const uchar* originalSceneData, const ushort originalSceneW, int centerX
	, int centerY, int radius)
{
	//center val
	if (radius == 0)
		return originalSceneData[centerX + centerY * originalSceneW];

	int iX = 0, iY = radius;
	int iD = 1 - radius;
	int iDeltaE = 3;
	int iDeltaSE = -2 * radius + 5;

	int value = 0;
	value += originalSceneData[(centerX)+(centerY + iY) * originalSceneW];
	value += originalSceneData[(centerX)+(centerY - iY) * originalSceneW];
	value += originalSceneData[(centerX + iY) + (centerY)*originalSceneW];
	value += originalSceneData[(centerX - iY) + (centerY)*originalSceneW];

	//E+W+S+W val
	if (radius == 1)
		return value;

	int accessedY = 0;
	while (iY > iX)
	{
		if (iD < 0)
		{
			iD += iDeltaE;
			iDeltaE += 2;
			iDeltaSE += 2;
		}
		else
		{
			iD += iDeltaSE;
			iDeltaE += 2;
			iDeltaSE += 4;
			iY--;
		}
		iX++;

		accessedY = centerY + iY;

		value += originalSceneData[(centerX + iX) + (centerY + iY) * originalSceneW];
		value += originalSceneData[(centerX - iX) + (centerY + iY) * originalSceneW];
		value += originalSceneData[(centerX + iX) + (centerY - iY) * originalSceneW];
		value += originalSceneData[(centerX - iX) + (centerY - iY) * originalSceneW];

		if (accessedY == centerY + iX)
			break;

		value += originalSceneData[(centerX + iY) + (centerY + iX) * originalSceneW];
		value += originalSceneData[(centerX - iY) + (centerY + iX) * originalSceneW];
		value += originalSceneData[(centerX + iY) + (centerY - iX) * originalSceneW];
		value += originalSceneData[(centerX - iY) + (centerY - iX) * originalSceneW];

		if (accessedY - 1 == centerY + iX)
			break;
	}
	return value;
}

__global__ void cudaSupporter::kernel::SortRXY(const float* rMat, const ushort rMatW, const ushort rMatH, float* outR
	, ushort* outX
	, ushort* outY)
{
	//set shared mem
	__shared__ float rArr[NUM_THREADS_PER_BLOCK_SORT_RXY];
	__shared__ ushort xArr[NUM_THREADS_PER_BLOCK_SORT_RXY];
	__shared__ ushort yArr[NUM_THREADS_PER_BLOCK_SORT_RXY];

	const uint blockOffsetX = LEN_BLOCKS_SORT_RXY * blockIdx.x;
	const uint blockOffsetY = LEN_BLOCKS_SORT_RXY * blockIdx.y;

	ushort x = blockOffsetX + threadIdx.x % LEN_BLOCKS_SORT_RXY;
	ushort y = blockOffsetY + threadIdx.x / LEN_BLOCKS_SORT_RXY;

	rArr[threadIdx.x] = rMat[x + y * rMatW];
	xArr[threadIdx.x] = x;
	yArr[threadIdx.x] = y;

	__syncthreads();


	//set arr to top 256 R (out of 1024 pixels)
	ushort arrIdx = threadIdx.x;
	while (arrIdx < NUM_PIXELS_4_FINDNG_POINT)
	{
		arrIdx += NUM_THREADS_PER_BLOCK_SORT_RXY;
		x = blockOffsetX + arrIdx % LEN_BLOCKS_SORT_RXY;
		y = blockOffsetY + arrIdx / LEN_BLOCKS_SORT_RXY;

		if (x >= rMatW || y >= rMatH)
			continue;

		if (rArr[threadIdx.x] < rMat[x + y * rMatW])
		{
			rArr[threadIdx.x] = rMat[x + y * rMatW];
			xArr[threadIdx.x] = x;
			yArr[threadIdx.x] = y;
		}
	}

	__syncthreads();


	//reduction for highest R
	ushort reductionIter = NUM_THREADS_PER_BLOCK_SORT_RXY / 2;
	while (reductionIter)
	{
		reductionIter /= 2;
		if (threadIdx.x < reductionIter)
		{
			arrIdx = threadIdx.x + reductionIter;
			x = blockOffsetX + arrIdx % LEN_BLOCKS_SORT_RXY;
			y = blockOffsetY + arrIdx / LEN_BLOCKS_SORT_RXY;

			if (x >= rMatW || y >= rMatH)
				continue;

			if (rArr[threadIdx.x] < rArr[arrIdx])
			{
				rArr[threadIdx.x] = rArr[arrIdx];
				xArr[threadIdx.x] = xArr[arrIdx];
				yArr[threadIdx.x] = yArr[arrIdx];
			}
		}
		__syncthreads();
	}

	__syncthreads();


	//set output
	if (threadIdx.x == 0)
	{
		outR[blockIdx.x + gridDim.x * blockIdx.y] = rArr[0];
		outX[blockIdx.x + gridDim.x * blockIdx.y] = xArr[0];
		outY[blockIdx.x + gridDim.x * blockIdx.y] = yArr[0];
	}

	__syncthreads();
}

__global__ void cudaSupporter::kernel::GetRMat(const uchar* originalSceneData, const ushort originalSceneW, const int* statsData
	, const uchar statsLen, const ushort objectLen, const ushort radius, const uint* circleSumVecParam
	, const uchar* binaryMapData, const ushort binaryMapW, float* rMat)
{
	//choice pixels for sumcircle
	uint tIdOnCandidate = blockIdx.x;
	int candidateIdx = -1;
	ushort x = 0;
	ushort y = 0;
	for (int statsRow = STAT_COLS; statsRow < STAT_COLS * statsLen; statsRow += STAT_COLS)
	{
		if (statsData[CANDIDATE_W + statsRow] >= MAX_CANDIDATE_LEN_RATIO * objectLen
			|| statsData[CANDIDATE_H + statsRow] >= MAX_CANDIDATE_LEN_RATIO * objectLen)
			continue;

		uint candidateSize = int(statsData[CANDIDATE_W + statsRow] / ZOOM_BINARYMAP) * int(statsData[CANDIDATE_H + statsRow] / ZOOM_BINARYMAP);
		candidateIdx += candidateSize;
		if (blockIdx.x <= candidateIdx)
		{
			int candidateW = statsData[CANDIDATE_W + statsRow] / ZOOM_BINARYMAP;
			x = int(statsData[CANDIDATE_X + statsRow] / ZOOM_BINARYMAP) + tIdOnCandidate % candidateW;
			y = int(statsData[CANDIDATE_Y + statsRow] / ZOOM_BINARYMAP) + tIdOnCandidate / candidateW;
			break;
		}
		else
			tIdOnCandidate -= candidateSize;
	}

	uint binaryMapIdx = int(x * ZOOM_BINARYMAP) + int(y * ZOOM_BINARYMAP) * binaryMapW;
	if (binaryMapData[binaryMapIdx] == 0)
		return;

	__syncthreads();


	//set shared mem
	extern __shared__ float circleSumVecDevice[];
	uint* circleSumVec4DeviceMean = (uint*)&circleSumVecDevice[radius + 1];
	float* circleSumVecHost = (float*)&circleSumVec4DeviceMean[radius + 1];
	uint* circleSumVec4HostMean = (uint*)&circleSumVecHost[radius + 1];
	float* circleSumVecDeviceHost = (float*)&circleSumVec4HostMean[radius + 1];

	circleSumVec4DeviceMean[threadIdx.x] = SumCircleDevice(originalSceneData, originalSceneW, x, y, threadIdx.x);
	circleSumVec4DeviceMean[radius - threadIdx.x] = SumCircleDevice(originalSceneData, originalSceneW, x, y, radius - threadIdx.x);

	circleSumVecDevice[threadIdx.x] = circleSumVec4DeviceMean[threadIdx.x];
	circleSumVecDevice[radius - threadIdx.x] = circleSumVec4DeviceMean[radius - threadIdx.x];

	circleSumVec4HostMean[threadIdx.x] = circleSumVecParam[threadIdx.x];
	circleSumVec4HostMean[radius - threadIdx.x] = circleSumVecParam[radius - threadIdx.x];

	circleSumVecHost[threadIdx.x] = circleSumVec4HostMean[threadIdx.x];
	circleSumVecHost[radius - threadIdx.x] = circleSumVec4HostMean[radius - threadIdx.x];

	__syncthreads();


	//find appropriate 2^n for reduction
	uint numReductionIter = (blockDim.x & (0xFF - 1));
	numReductionIter |= numReductionIter >> 1;
	numReductionIter |= numReductionIter >> 2;
	numReductionIter |= numReductionIter >> 4;
	numReductionIter |= numReductionIter >> 8;
	numReductionIter |= numReductionIter >> 16;
	numReductionIter = ++numReductionIter >> 1;


	//sum out of numReductionIter
	if (threadIdx.x < numReductionIter)
	{
		for (int r = numReductionIter; threadIdx.x + r <= radius; r += numReductionIter)
		{
			circleSumVec4DeviceMean[threadIdx.x] += circleSumVec4DeviceMean[threadIdx.x + r];
			circleSumVec4HostMean[threadIdx.x] += circleSumVec4HostMean[threadIdx.x + r];
		}
		__syncthreads();
	}

	__syncthreads();


	//reduction for mean
	uint reductionIter = numReductionIter / 2;
	while (reductionIter)
	{
		reductionIter /= 2;
		if (threadIdx.x < reductionIter)
		{
			circleSumVec4DeviceMean[threadIdx.x] += circleSumVec4DeviceMean[threadIdx.x + reductionIter];
			circleSumVec4HostMean[threadIdx.x] += circleSumVec4HostMean[threadIdx.x + reductionIter];
		}
		__syncthreads();
	}

	__syncthreads();


	//get mean
	__shared__ float circleMeanDevice;
	__shared__ float circleMeanHost;
	if (threadIdx.x == 0)
	{
		circleMeanDevice = (float)circleSumVec4DeviceMean[0] / (radius + 1);
		circleMeanHost = (float)circleSumVec4HostMean[0] / (radius + 1);
	}
	__syncthreads();
	__syncthreads();


	//get dev
	bool isNotCenter = (threadIdx.x != radius - threadIdx.x);

	circleSumVecDevice[threadIdx.x] -= circleMeanDevice;
	circleSumVecHost[threadIdx.x] -= circleMeanHost;
	if (isNotCenter)
	{
		circleSumVecDevice[radius - threadIdx.x] -= circleMeanDevice;
		circleSumVecHost[radius - threadIdx.x] -= circleMeanHost;
	}

	__syncthreads();


	//set some values for R
	circleSumVecDeviceHost[threadIdx.x] = circleSumVecDevice[threadIdx.x] * circleSumVecHost[threadIdx.x];
	circleSumVecDeviceHost[radius - threadIdx.x] = circleSumVecDevice[radius - threadIdx.x] * circleSumVecHost[radius - threadIdx.x];

	__syncthreads();


	circleSumVecDevice[threadIdx.x] = circleSumVecDevice[threadIdx.x] * circleSumVecDevice[threadIdx.x];
	if (isNotCenter)
		circleSumVecDevice[radius - threadIdx.x] = circleSumVecDevice[radius - threadIdx.x] * circleSumVecDevice[radius - threadIdx.x];

	__syncthreads();


	circleSumVecHost[threadIdx.x] = circleSumVecHost[threadIdx.x] * circleSumVecHost[threadIdx.x];
	if (isNotCenter)
		circleSumVecHost[radius - threadIdx.x] = circleSumVecHost[radius - threadIdx.x] * circleSumVecHost[radius - threadIdx.x];

	__syncthreads();


	//sum out of numReductionIter
	if (threadIdx.x < numReductionIter)
	{
		for (int r = numReductionIter; threadIdx.x + r <= radius; r += numReductionIter)
		{
			circleSumVecDevice[threadIdx.x] += circleSumVecDevice[threadIdx.x + r];
			circleSumVecDeviceHost[threadIdx.x] += circleSumVecDeviceHost[threadIdx.x + r];
			circleSumVecHost[threadIdx.x] += circleSumVecHost[threadIdx.x + r];
		}
		__syncthreads();
	}

	__syncthreads();


	//reduction for R
	reductionIter = numReductionIter / 2;
	while (reductionIter)
	{
		reductionIter /= 2;
		if (threadIdx.x < reductionIter)
		{
			circleSumVecDevice[threadIdx.x] += circleSumVecDevice[threadIdx.x + reductionIter];
			circleSumVecDeviceHost[threadIdx.x] += circleSumVecDeviceHost[threadIdx.x + reductionIter];
			circleSumVecHost[threadIdx.x] += circleSumVecHost[threadIdx.x + reductionIter];
		}
		__syncthreads();
	}

	__syncthreads();


	//get R
	if (threadIdx.x == 0)
		rMat[x + y * originalSceneW] = circleSumVecDeviceHost[0] / sqrtf(circleSumVecDevice[0] * circleSumVecHost[0]);

	__syncthreads();
}

__global__ void cudaSupporter::kernel::Get1stMoment(const uchar* sceneData, uchar* moment1stData, const int sceneW, const int sceneH
	, const ushort objectLen)
{
	//set some values.
	const int workOffset = (objectLen - 1) / 2;
	const int tIdOnblock = blockDim.x * threadIdx.y + threadIdx.x;
	const bool isObjectLenEven = (objectLen % 2 == 0);
	const int shareMapLen = NUM_THREADS_PER_BLOCK_LINE + 2 * workOffset + isObjectLenEven;
	const int numThreadsOnBlock = blockDim.x * blockDim.y;
	const int shareMapOffsetX = -workOffset + blockIdx.x * blockDim.x;
	const int shareMapOffsetY = -workOffset + blockIdx.y * blockDim.y;

	//set share memory
	extern __shared__ float shareMem[];
	for (int shareMapIdOnBlock = tIdOnblock; shareMapIdOnBlock < shareMapLen * shareMapLen; shareMapIdOnBlock += numThreadsOnBlock)
	{
		const int shareMapX = shareMapIdOnBlock % shareMapLen + shareMapOffsetX;
		const int shareMapY = shareMapIdOnBlock / shareMapLen + shareMapOffsetY;
		const int shareMapId = shareMapX + shareMapY * sceneW;
		const int shareMemId = shareMapX - shareMapOffsetX + (shareMapY - shareMapOffsetY) * shareMapLen;

		shareMem[shareMemId] = (shareMapX < 0 || shareMapX >= sceneW || shareMapY < 0 || shareMapY >= sceneH) ?
			0 : sceneData[shareMapId];
	}

	__syncthreads();


	//Set the row and col value for each thread.
	const int rowOnScene = blockIdx.y * blockDim.y + threadIdx.y;
	const int colOnScene = blockIdx.x * blockDim.x + threadIdx.x;

	//get mean
	int temp4Loop = 0;
	for (int j = -workOffset; j <= workOffset + isObjectLenEven; j++)
	{
		for (int i = -workOffset; i <= workOffset + isObjectLenEven; i++)
		{
			int shareMemId = colOnScene + i - shareMapOffsetX + (rowOnScene + j - shareMapOffsetY) * shareMapLen;
			temp4Loop += shareMem[shareMemId];
		}
	}
	temp4Loop = round((float)temp4Loop / (float)(objectLen * objectLen));
	if (temp4Loop > UCHAR_MAX)
		temp4Loop = UCHAR_MAX;

	const int tIdOnScene = colOnScene + rowOnScene * sceneW;
	if (tIdOnScene < sceneH * sceneW)
		moment1stData[tIdOnScene] = temp4Loop;

	__syncthreads();
}

__global__ void cudaSupporter::kernel::GetBinaryMap(const uchar* sceneData, const uchar* moment1stData, uchar* binaryMapData
	, const int sceneW, const int sceneH, const ushort objectLen, const float* objectMoment)
{
	//set some values.
	const int workOffset = (objectLen - 1) / 2;
	const int tIdOnblock = blockDim.x * threadIdx.y + threadIdx.x;
	const bool isObjectLenEven = (objectLen % 2 == 0);
	const int shareMapLen = NUM_THREADS_PER_BLOCK_LINE + 2 * workOffset + isObjectLenEven;
	const int numThreadsOnBlock = blockDim.x * blockDim.y;
	const int shareMapOffsetX = -workOffset + blockIdx.x * blockDim.x;
	const int shareMapOffsetY = -workOffset + blockIdx.y * blockDim.y;
	float sceneMoment[NUM_MOMOENTS];

	//set share memory
	extern __shared__ float shareMem[];
	for (int shareMapIdOnBlock = tIdOnblock; shareMapIdOnBlock < shareMapLen * shareMapLen; shareMapIdOnBlock += numThreadsOnBlock)
	{
		const int shareMapX = shareMapIdOnBlock % shareMapLen + shareMapOffsetX;
		const int shareMapY = shareMapIdOnBlock / shareMapLen + shareMapOffsetY;
		const int shareMapId = shareMapX + shareMapY * sceneW;
		const int shareMemId = shareMapX - shareMapOffsetX + (shareMapY - shareMapOffsetY) * shareMapLen;

		shareMem[shareMemId] =
			(shareMapX < 0 || shareMapX >= sceneW || shareMapY < 0 || shareMapY >= sceneH) ?
			0 : sceneData[shareMapId];
	}

	__syncthreads();


	//get 3rd moment
	for (int shareMapIdOnBlock = tIdOnblock; shareMapIdOnBlock < shareMapLen * shareMapLen; shareMapIdOnBlock += numThreadsOnBlock)
	{
		const int shareMapX = shareMapIdOnBlock % shareMapLen + shareMapOffsetX;
		const int shareMapY = shareMapIdOnBlock / shareMapLen + shareMapOffsetY;

		if (shareMapX < 0 || shareMapX >= sceneW || shareMapY < 0 || shareMapY >= sceneH)
			continue;

		const int shareMapId = shareMapX + shareMapY * sceneW;
		const int shareMemId = shareMapX - shareMapOffsetX + (shareMapY - shareMapOffsetY) * shareMapLen;

		shareMem[shareMemId] = shareMem[shareMemId] - moment1stData[shareMapId];
		shareMem[shareMemId] = powf(shareMem[shareMemId], int(3));
	}

	const int rowOnScene = blockIdx.y * blockDim.y + threadIdx.y;
	const int colOnScene = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();


	float temp4Loop = 0;
	for (int j = -workOffset; j <= workOffset + isObjectLenEven; j++)
	{
		for (int i = -workOffset; i <= workOffset + isObjectLenEven; i++)
		{
			int shareMemId = colOnScene + i - shareMapOffsetX + (rowOnScene + j - shareMapOffsetY) * shareMapLen;
			temp4Loop += shareMem[shareMemId];
		}
	}

	const bool isDevPos = temp4Loop > 0;
	sceneMoment[2] = powf(abs(temp4Loop) / (objectLen * objectLen), (1.0 / 3));
	sceneMoment[2] = isDevPos ? sceneMoment[2] : -sceneMoment[2];

	__syncthreads();


	//get 2nd moment
	for (int shareMapIdOnBlock = tIdOnblock; shareMapIdOnBlock < shareMapLen * shareMapLen; shareMapIdOnBlock += numThreadsOnBlock)
	{
		const int shareMapX = shareMapIdOnBlock % shareMapLen + shareMapOffsetX;
		const int shareMapY = shareMapIdOnBlock / shareMapLen + shareMapOffsetY;

		if (shareMapX < 0 || shareMapX >= sceneW || shareMapY < 0 || shareMapY >= sceneH)
			continue;

		const int shareMemId = shareMapX - shareMapOffsetX + (shareMapY - shareMapOffsetY) * shareMapLen;

		shareMem[shareMemId] = powf(abs(shareMem[shareMemId]), (2.0 / 3));
	}

	__syncthreads();


	temp4Loop = 0;
	for (int j = -workOffset; j <= workOffset + isObjectLenEven; j++)
	{
		for (int i = -workOffset; i <= workOffset + isObjectLenEven; i++)
		{
			int shareMemId = colOnScene + i - shareMapOffsetX + (rowOnScene + j - shareMapOffsetY) * shareMapLen;
			temp4Loop += shareMem[shareMemId];
		}
	}
	sceneMoment[1] = sqrtf(temp4Loop / (objectLen * objectLen));

	__syncthreads();


	//get 4th moment
	for (int shareMapIdOnBlock = tIdOnblock; shareMapIdOnBlock < shareMapLen * shareMapLen; shareMapIdOnBlock += numThreadsOnBlock)
	{
		const int shareMapX = shareMapIdOnBlock % shareMapLen + shareMapOffsetX;
		const int shareMapY = shareMapIdOnBlock / shareMapLen + shareMapOffsetY;

		if (shareMapX < 0 || shareMapX >= sceneW || shareMapY < 0 || shareMapY >= sceneH)
			continue;

		const int shareMemId = shareMapX - shareMapOffsetX + (shareMapY - shareMapOffsetY) * shareMapLen;

		shareMem[shareMemId] *= shareMem[shareMemId];
	}

	__syncthreads();


	temp4Loop = 0;
	for (int j = -workOffset; j <= workOffset + isObjectLenEven; j++)
	{
		for (int i = -workOffset; i <= workOffset + isObjectLenEven; i++)
		{
			int shareMemId = colOnScene + i - shareMapOffsetX + (rowOnScene + j - shareMapOffsetY) * shareMapLen;
			temp4Loop += shareMem[shareMemId];
		}
	}
	sceneMoment[3] = powf(temp4Loop / (objectLen * objectLen), 1.0 / 4);

	__syncthreads();


	//get 1st moment
	const int tIdOnScene = colOnScene + rowOnScene * sceneW;
	sceneMoment[0] = moment1stData[tIdOnScene];

	__syncthreads();


	//get mse -> d
	float d = 0;
	for (int i = 0; i < NUM_MOMOENTS; i++)
	{
		float temp = objectMoment[i] - sceneMoment[i];
		d += temp * temp;
	}
	d = sqrtf(d);
	d = 1 / (d + 1);
	d = d * 255;
	d = d > UCHAR_MAX ? UCHAR_MAX : d;
	__syncthreads();


	if (tIdOnScene < sceneW * sceneH)
		binaryMapData[tIdOnScene] = d;

	__syncthreads();
}

__global__ void cudaSupporter::kernel::GetAdaptiveBinaryMap(uchar* binaryMapData, const uchar* moment1stData, const int sceneW, const int sceneH
	, const ushort objectLen)
{
	//set some values.
	const int workOffset = (objectLen - 1) / 2;
	const bool isObjectLenEven = (objectLen % 2 == 0);

	//Set the row and col value for each thread.
	const int rowOnScene = blockIdx.y * blockDim.y + threadIdx.y;
	const int colOnScene = blockIdx.x * blockDim.x + threadIdx.x;
	const int tIdOnScene = colOnScene + rowOnScene * sceneW;

	short binaryMapElement = binaryMapData[tIdOnScene];
	short moment1stElement = moment1stData[tIdOnScene];
	if (tIdOnScene < sceneW* sceneH)
	{
		binaryMapElement -= (moment1stElement - ADAPTIVE_THRESHOLD_C);
		binaryMapData[tIdOnScene] = binaryMapElement > 0 ? UCHAR_MAX : 0;
	}

	if ((tIdOnScene < sceneW * sceneH) &&
		((colOnScene < workOffset) || (colOnScene + workOffset + isObjectLenEven >= sceneW)
			|| (rowOnScene < workOffset) || (rowOnScene + workOffset + isObjectLenEven >= sceneH)))
		binaryMapData[tIdOnScene] = 0;

	__syncthreads();
}

void cudaSupporter::LaunchGet1stMoment(const dim3& dimGridFilter1, const dim3& dimBlockFilter1, const uint sharedMemSizeFilter1
	, const uchar* sceneData, uchar* moment1stData, const int sceneW, const int sceneH, const ushort objectLen)
{
	cudaSupporter::kernel::Get1stMoment << <dimGridFilter1, dimBlockFilter1, sharedMemSizeFilter1 >> >
		(sceneData, moment1stData, sceneW, sceneH, objectLen);
}

void cudaSupporter::LaunchGetBinaryMap(const dim3& dimGridFilter1, const dim3& dimBlockFilter1, const uint sharedMemSizeFilter1
	, const uchar* sceneData, const uchar* moment1stData, uchar* binaryMapData, const int sceneW, const int sceneH
	, const ushort objectLen, const float* objectMoment)
{
	cudaSupporter::kernel::GetBinaryMap << <dimGridFilter1, dimBlockFilter1, sharedMemSizeFilter1 >> >
		(sceneData, moment1stData, binaryMapData, sceneW, sceneH, objectLen, objectMoment);
}

void cudaSupporter::LaunchGetRMat(const dim3& dimGridFilter2, const dim3& dimBlockFilter2, const uint sharedMemSizeFilter2
	, const uchar* originalSceneData, const ushort originalSceneW, const int* statsData, const uchar statsLen
	, const ushort objectLen, const ushort radius, const uint* circleSumVecParam, const uchar* binaryMapData
	, const ushort binaryMapW, float* rMat)
{
	cudaSupporter::kernel::GetRMat << <dimGridFilter2, dimBlockFilter2, sharedMemSizeFilter2 >> >
		(originalSceneData, originalSceneW, statsData, statsLen, objectLen, radius, circleSumVecParam, binaryMapData, binaryMapW, rMat);
}

void cudaSupporter::LaunchSortRXY(const dim3& dimGridFilter3, const dim3& dimBlockFilter3, const float* rMat, const ushort rMatW
	, const ushort rMatH, float* outR, ushort* outX, ushort* outY)
{
	cudaSupporter::kernel::SortRXY << < dimGridFilter3, dimBlockFilter3 >> > (rMat, rMatW, rMatH, outR, outX, outY);
}

void cudaSupporter::LaunchGetAdaptiveBinaryMap(const dim3& dimGridFilter1, const dim3& dimBlockFilter1, uchar* binaryMapData, const uchar* moment1stData
	, const int sceneW, const int sceneH, const ushort objectLen)
{
	cudaSupporter::kernel::GetAdaptiveBinaryMap << < dimGridFilter1, dimBlockFilter1 >> > (binaryMapData, moment1stData, sceneW, sceneH, objectLen);
}




