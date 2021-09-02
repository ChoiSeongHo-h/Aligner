#include "PatternMatching.h"

int PatternMatcher::SumCircleHost(cv::Mat& image, const ushort centerX, const ushort centerY, const ushort radius)
{
	uchar* dot;

	if (radius == 0)
	{
		dot = image.ptr<uchar>(centerY);
		return dot[centerX];
	}

	int iX = 0, iY = radius;
	int iD = 1 - radius;
	int iDeltaE = 3;
	int iDeltaSE = -2 * radius + 5;

	int value = 0;

	dot = image.ptr<uchar>(centerY + iY);
	value += dot[centerX];

	dot = image.ptr<uchar>(centerY - iY);
	value += dot[centerX];

	dot = image.ptr<uchar>(centerY);
	value += dot[centerX + iY];
	value += dot[centerX - iY];

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

		dot = image.ptr<uchar>(centerY + iY);
		value += dot[centerX + iX] + dot[centerX - iX];
		accessedY = centerY + iY;

		dot = image.ptr<uchar>(centerY - iY);
		value += dot[centerX + iX] + dot[centerX - iX];

		if (accessedY == centerY + iX)
			break;

		dot = image.ptr<uchar>(centerY + iX);
		value += dot[centerX + iY] + dot[centerX - iY];

		dot = image.ptr<uchar>(centerY - iX);
		value += dot[centerX + iY] + dot[centerX - iY];

		if (accessedY - 1 == centerY + iX)
			break;
	}
	return value;
}

void PatternMatcher::SetObjectROI(const cv::Mat& originalObject, const cv::Mat& object, ushort& objectLen, ushort& originalObjectLen
	, cv::Mat& objectROI)
{
	using namespace cv;

	if (object.rows > object.cols)
	{
		objectLen = object.cols;
		originalObjectLen = originalObject.cols;
		objectROI = object(Rect(0, (object.rows - objectLen) / 2, objectLen, objectLen));
	}
	else
	{
		objectLen = object.rows;
		originalObjectLen = originalObject.rows;
		objectROI = object(Rect((object.cols - objectLen) / 2, 0, objectLen, objectLen));
	}
}

void PatternMatcher::SetSceneROI(const cv::Mat& scene, const ushort& objectLen, ushort& offset, cv::Rect& sceneROI, bool& isObjectLenEven)
{
	using namespace cv;

	offset = (objectLen - 1) / 2;
	isObjectLenEven = objectLen % 2 == 0;
	ushort sceneROIWidth = scene.cols - offset - isObjectLenEven;
	ushort sceneROIHeight = scene.rows - offset - isObjectLenEven;
	sceneROI = Rect(offset, offset, sceneROIWidth, sceneROIHeight);
}

void PatternMatcher::GetObjectMoment(const cv::Mat& objectROI, float* objectMoment)
{
	objectMoment[0] = (float)mean(objectROI)[0];
	for (int i = 2; i < STAT_COLS; i++)
	{
		cv::Mat objectTempMat;
		objectROI.convertTo(objectTempMat, CV_64FC1);
		objectTempMat -= objectMoment[0];
		pow(objectTempMat, i, objectTempMat);
		objectMoment[i - 1] = (float)mean(objectTempMat)[0];
		if (objectMoment[i - 1] > 0)
			objectMoment[i - 1] = (float)pow(objectMoment[i - 1], 1.0 / i);
		else
			objectMoment[i - 1] = -(float)pow(-objectMoment[i - 1], 1.0 / i);
	}
}

void PatternMatcher::GpuFree(const GpuPtr& gpuPtr)
{
	cudaFree(gpuPtr.devSceneData);
	cudaFree(gpuPtr.devMoment1stData);
	cudaFree(gpuPtr.devBinaryMapData);
	cudaFree(gpuPtr.devObjectMoment);
	cudaFree(gpuPtr.devOriginalSceneData);
	cudaFree(gpuPtr.devStatsData);
	cudaFree(gpuPtr.devCircleSumVecHost);
	cudaFree(gpuPtr.devRMat);
	cudaFree(gpuPtr.devOutR);
	cudaFree(gpuPtr.devOutX);
	cudaFree(gpuPtr.devOutY);
}

void PatternMatcher::GpuMalloc(const GpuPtr& gpuPtr, const uint originalScenePixels, const uint scenePixels, const ushort radius
	, const uint numCandidates)
{
	cudaMalloc((void**)&gpuPtr.devSceneData, scenePixels * sizeof(uchar));
	cudaMalloc((void**)&gpuPtr.devMoment1stData, scenePixels * sizeof(uchar));
	cudaMalloc((void**)&gpuPtr.devBinaryMapData, scenePixels * sizeof(uchar));
	cudaMalloc((void**)&gpuPtr.devObjectMoment, NUM_MOMOENTS * sizeof(float));
	cudaMalloc((void**)&gpuPtr.devOriginalSceneData, originalScenePixels * sizeof(uchar));
	cudaMalloc((void**)&gpuPtr.devStatsData, STAT_COLS * MAX_STATS_LEN * sizeof(int));
	cudaMalloc((void**)&gpuPtr.devCircleSumVecHost, (radius + 1) * sizeof(uint));
	cudaMalloc((void**)&gpuPtr.devRMat, originalScenePixels * sizeof(float));
	cudaMalloc((void**)&gpuPtr.devOutR, numCandidates * sizeof(float));
	cudaMalloc((void**)&gpuPtr.devOutX, numCandidates * sizeof(ushort));
	cudaMalloc((void**)&gpuPtr.devOutY, numCandidates * sizeof(ushort));
}

void PatternMatcher::GpuUpload(const GpuPtr& gpuPtr, const cv::Mat& scene, const uint scenePixels, const cv::Mat& binaryMap
	, const float* objectMoment, const cv::Mat& originalScene, const uint originalScenePixels, std::vector<uint>& circleSumVecHost, const ushort radius
	, const std::vector<float>& outR, const std::vector<ushort>& outX, const std::vector<ushort>& outY, const uint numCandidates)
{
	cudaMemcpy(gpuPtr.devSceneData, scene.data, scenePixels * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemset((void**)&gpuPtr.devMoment1stData, 0, scenePixels * sizeof(uchar));
	cudaMemcpy(gpuPtr.devBinaryMapData, binaryMap.data, scenePixels * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPtr.devObjectMoment, objectMoment, NUM_MOMOENTS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPtr.devOriginalSceneData, originalScene.data, originalScenePixels * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPtr.devCircleSumVecHost, circleSumVecHost.data(), (radius + 1) * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemset(gpuPtr.devRMat, 0, originalScenePixels * sizeof(float));
	cudaMemcpy(gpuPtr.devOutR, outR.data(), numCandidates * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPtr.devOutX, outX.data(), numCandidates * sizeof(ushort), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPtr.devOutY, outY.data(), numCandidates * sizeof(ushort), cudaMemcpyHostToDevice);
}

void PatternMatcher::GetAdaptiveBinaryMap(const cv::Mat& scene, const ushort offset, const bool isObjectLenEven, const GpuPtr& gpuPtr
	, const ushort objectLen, const cv::Mat& binaryMap)
{
	dim3 dimGridFilter1(
		(scene.cols + NUM_THREADS_PER_BLOCK_LINE - 1) / NUM_THREADS_PER_BLOCK_LINE,
		(scene.rows + NUM_THREADS_PER_BLOCK_LINE - 1) / NUM_THREADS_PER_BLOCK_LINE
	);
	dim3 dimBlockFilter1(NUM_THREADS_PER_BLOCK_LINE, NUM_THREADS_PER_BLOCK_LINE);
	uint sharedMemSizeFilter1 = (NUM_THREADS_PER_BLOCK_LINE + 2 * offset + isObjectLenEven) * (NUM_THREADS_PER_BLOCK_LINE + 2 * offset + isObjectLenEven) * sizeof(float);

	cudaSupporter::LaunchGet1stMoment(dimGridFilter1, dimBlockFilter1, sharedMemSizeFilter1, gpuPtr.devSceneData, gpuPtr.devMoment1stData, scene.cols
		, scene.rows, objectLen);
	cudaSupporter::LaunchGetBinaryMap(dimGridFilter1, dimBlockFilter1, sharedMemSizeFilter1, gpuPtr.devSceneData, gpuPtr.devMoment1stData, gpuPtr.devBinaryMapData
		, scene.cols, scene.rows, objectLen, gpuPtr.devObjectMoment);

	cudaSupporter::LaunchGet1stMoment(dimGridFilter1, dimBlockFilter1, sharedMemSizeFilter1, gpuPtr.devBinaryMapData, gpuPtr.devMoment1stData, scene.cols
		, scene.rows, objectLen);

	cudaSupporter::LaunchGetAdaptiveBinaryMap(dimGridFilter1, dimBlockFilter1, gpuPtr.devBinaryMapData, gpuPtr.devMoment1stData, scene.cols, scene.rows, objectLen);

	cudaMemcpy(binaryMap.data, gpuPtr.devBinaryMapData, (binaryMap.cols) * (binaryMap.rows) * sizeof(uchar), cudaMemcpyDeviceToHost);
}

void PatternMatcher::GpuUploadStats(const cv::Mat& binaryMap, const ushort objectLen, const GpuPtr& gpuPtr, uint& statsLen
	, uint& sizeFilter1Passed)
{
	cv::Mat labels, stats, centroids;
	statsLen = connectedComponentsWithStats(binaryMap, labels, stats, centroids);

	int numFilter1Passed = 0;
	for (uint i = 1; i < statsLen; i++)
	{
		int* p = stats.ptr<int>(i);
		if (p[2] >= 2 * objectLen || p[3] >= 2 * objectLen)
			continue;
		numFilter1Passed++;
		sizeFilter1Passed += uint(p[2] / ZOOM_BINARYMAP) * uint(p[3] / ZOOM_BINARYMAP);
	}

	if (numFilter1Passed > MAX_STATS_LEN)
	{
		std::cout << "stats len too long : " << numFilter1Passed << std::endl;
		cudaFree(gpuPtr.devStatsData);
		cudaMalloc((void**)&gpuPtr.devStatsData, 5 * statsLen * sizeof(int));
	}
	cudaMemcpy(gpuPtr.devStatsData, stats.data, 5 * statsLen * sizeof(int), cudaMemcpyHostToDevice);
}

void PatternMatcher::GetRMap(const uint sizeFilter1Passed, const ushort radius, const GpuPtr& gpuPtr, const cv::Mat& originalScene
	, const uint statsLen, const ushort objectLen, const cv::Mat& scene)
{
	dim3 dimGridFilter2(sizeFilter1Passed);
	dim3 dimBlockFilter2(radius / 2 + 1);
	uint sharedMemSizeFilter2 = 5 * (radius + 1) * sizeof(float);
	cudaSupporter::LaunchGetRMat(dimGridFilter2, dimBlockFilter2, sharedMemSizeFilter2, gpuPtr.devOriginalSceneData, originalScene.cols, gpuPtr.devStatsData
		, statsLen, objectLen, radius, gpuPtr.devCircleSumVecHost, gpuPtr.devBinaryMapData, scene.cols, gpuPtr.devRMat);
}

void PatternMatcher::SortRXY(const cv::Mat& originalScene, const GpuPtr& gpuPtr, std::vector<float>& outR, std::vector<ushort>& outX
	, std::vector<ushort>& outY, const uint numCandidates, uint& maxRIdx)
{
	dim3 dimGridFilter3(
		(originalScene.cols + LEN_BLOCKS_SORT_RXY - 1) / LEN_BLOCKS_SORT_RXY,
		(originalScene.rows + LEN_BLOCKS_SORT_RXY - 1) / LEN_BLOCKS_SORT_RXY
	);
	dim3 dimBlockFilter3(NUM_THREADS_PER_BLOCK_SORT_RXY);
	cudaSupporter::LaunchSortRXY(dimGridFilter3, dimBlockFilter3, gpuPtr.devRMat, originalScene.cols, originalScene.rows, gpuPtr.devOutR, gpuPtr.devOutX, gpuPtr.devOutY);

	cudaMemcpy(outR.data(), gpuPtr.devOutR, numCandidates * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(outX.data(), gpuPtr.devOutX, numCandidates * sizeof(ushort), cudaMemcpyDeviceToHost);
	cudaMemcpy(outY.data(), gpuPtr.devOutY, numCandidates * sizeof(ushort), cudaMemcpyDeviceToHost);

	maxRIdx = 0;
	for (uint i = 0; i < numCandidates; i++)
		if (outR[maxRIdx] < outR[i])
			maxRIdx = i;
}

void PatternMatcher::SetObjectInfo()
{
	//Set object ROI and length of short axit
	ushort originalObjectLen;
	cv::Mat objectROI;
	SetObjectROI(this->originalObject, this->object, this->objectInfo.objectLen, originalObjectLen, objectROI);

	//Calc object Moment
	GetObjectMoment(objectROI, objectInfo.objectMoment);

	//Calc object circlesum
	this->objectInfo.radius = (originalObjectLen - 1) / 2;
	this->objectInfo.circleSumVecHost.resize(this->objectInfo.radius + 1);
	for (int i = 0; i <= objectInfo.radius; i++)
		this->objectInfo.circleSumVecHost[i] = SumCircleHost(this->originalObject, this->originalObject.cols / 2, this->originalObject.rows / 2, i);

	//set values
	this->objectInfo.offset = (objectInfo.objectLen - 1) / 2;
	this->objectInfo.isObjectLenEven = this->objectInfo.objectLen % 2 == 0;
}

cv::Point PatternMatcher::FindRXY(const cv::Rect& sceneROI)
{
	this->originalScene = this->originalScene(sceneROI).clone();
	cv::resize(this->originalScene, this->scene, cv::Size(), ZOOM_BINARYMAP, ZOOM_BINARYMAP, cv::INTER_AREA);

	//set params to upload
	cv::Mat binaryMap = cv::Mat::zeros(scene.size(), CV_8UC1);
	const uint numCandidates = ((this->originalScene.cols + LEN_BLOCKS_SORT_RXY - 1) / LEN_BLOCKS_SORT_RXY)
		* ((this->originalScene.rows + LEN_BLOCKS_SORT_RXY - 1) / LEN_BLOCKS_SORT_RXY);
	std::vector<float> outR(numCandidates);
	std::vector<ushort> outX(numCandidates);
	std::vector<ushort> outY(numCandidates);


	//if too big, remalloc
	uint scenePixels = (scene.cols) * (scene.rows);
	uint originalScenePixels = (this->originalScene.rows) * (this->originalScene.cols);
	if (scenePixels > NUM_MALLOCED_SCENE_PIXELS || originalScenePixels > NUM_STANDARD_ORIGINAL_SCENE_PIXELS
		|| this->objectInfo.radius > LEN_MALLOCED_RADIUS || NUM_MALLOCED_CANDIDATES > numCandidates)
	{
		GpuFree(this->gpuPtr);
		GpuMalloc(this->gpuPtr, originalScenePixels, scenePixels, this->objectInfo.radius, numCandidates);
	}


	//gpuUpload
	GpuUpload(this->gpuPtr, scene, scenePixels, binaryMap, this->objectInfo.objectMoment, this->originalScene, originalScenePixels
		, this->objectInfo.circleSumVecHost, this->objectInfo.radius, outR, outX, outY, numCandidates);


	//Filter1, getBinaryMap
	GetAdaptiveBinaryMap(scene, this->objectInfo.offset, this->objectInfo.isObjectLenEven, this->gpuPtr, this->objectInfo.objectLen, binaryMap);

	//get & upload stats
	uint sizeFilter1Passed = 0;
	uint statsLen = 0;
	GpuUploadStats(binaryMap, this->objectInfo.objectLen, this->gpuPtr, statsLen, sizeFilter1Passed);


	//Filter2, getRMap
	GetRMap(sizeFilter1Passed, this->objectInfo.radius, this->gpuPtr, this->originalScene, statsLen, this->objectInfo.objectLen, scene);


	//Filter3, sort RXY
	uint maxRIdx = 0;
	SortRXY(this->originalScene, this->gpuPtr, outR, outX, outY, numCandidates, maxRIdx);

	std::printf("%f\n", outR[maxRIdx]);

	return cv::Point(outX[maxRIdx], outY[maxRIdx]);
}

PatternMatcher::PatternMatcher()
{
	GpuMalloc(this->gpuPtr);
}

PatternMatcher::~PatternMatcher()
{
	GpuFree(this->gpuPtr);
}





