#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "CudaSupporter.cuh"
#include "AlignerConsts.h"


class PatternMatcher
{
private:
	struct GpuPtr
	{
		uchar* devSceneData;
		uchar* devMoment1stData;
		uchar* devBinaryMapData;
		float* devObjectMoment;
		uchar* devOriginalSceneData;
		int* devStatsData;
		uint* devCircleSumVecHost;
		float* devRMat;
		float* devOutR;
		ushort* devOutX;
		ushort* devOutY;
	};
	struct ObjectInfo
	{
		ushort objectLen, radius, offset;
		float objectMoment[NUM_MOMOENTS];
		std::vector<uint> circleSumVecHost;
		bool isObjectLenEven;
	};

	GpuPtr gpuPtr;
	ObjectInfo objectInfo;
	cv::Mat scene;

	int SumCircleHost(cv::Mat& image, const ushort centerX, const ushort centerY, const ushort radius);
	void SetObjectROI(const cv::Mat& originalObject, const cv::Mat& object, ushort& objectLength, ushort& originalObjectLen, cv::Mat& objectROI);
	void SetSceneROI(const cv::Mat& scene, const ushort& objectLen, ushort& offset, cv::Rect& sceneROI, bool& isObjectLenEven);
	void GetObjectMoment(const cv::Mat& objectROI, float* objectMoment);
	void GpuFree(const GpuPtr& gpuPtr);
	void GpuMalloc(const GpuPtr& gpuPtr, const uint originalScenePixels = NUM_STANDARD_ORIGINAL_SCENE_PIXELS, const uint scenePixels = NUM_MALLOCED_SCENE_PIXELS
		, const ushort radius = LEN_MALLOCED_RADIUS, const uint numCandidates = NUM_MALLOCED_CANDIDATES);
	void GpuUpload(const GpuPtr& gpuPtr, const cv::Mat& scene, const uint scenePixels, const cv::Mat& binaryMap
		, const float* objectMoment, const cv::Mat& originalScene, const uint originalScenePixels, std::vector<uint>& circleSumVecHost, const ushort radius
		, const std::vector<float>& outR, const std::vector<ushort>& outX, const std::vector<ushort>& outY, const uint numCandidates);
	void GetAdaptiveBinaryMap(const cv::Mat& scene, const ushort offset, const bool isObjectLenEven, const GpuPtr& gpuPtr, const ushort objectLen
		, const cv::Mat& binaryMap);
	void GpuUploadStats(const cv::Mat& binaryMap, const ushort objectLen, const GpuPtr& gpuPtr, uint& statsLen, uint& sizeFilter1Passed);
	void GetRMap(const uint sizeFilter1Passed, const ushort radius, const GpuPtr& gpuPtr, const cv::Mat& originalScene, const uint statsLen
		, const ushort objectLen, const cv::Mat& scene);
	void SortRXY(const cv::Mat& originalScene, const GpuPtr& gpuPtr, std::vector<float>& outR, std::vector<ushort>& outX, std::vector<ushort>& outY
		, const uint numCandidates, uint& maxRIdx);

public:
	cv::Mat originalObject;
	cv::Mat object;
	cv::Mat originalScene;

	PatternMatcher();
	~PatternMatcher();
	cv::Point FindRXY(const cv::Rect& sceneROI);
	void SetObjectInfo();
};
