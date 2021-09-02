#pragma once

enum StdImgInfo
{
	CAM_H = 720,
	CAM_W = 1280,
};

enum ProjectId
{
	SCENE0_DATA_SHARER_ID = 2333,
	SCENE1_DATA_SHARER_ID = 2343,
	OBJECT0_DATA_SHARER_ID = 2344,
	OBJECT1_DATA_SHARER_ID = 2345,
	INTER_STATUS_SHARER_ID = 2334,
	OBJECT_X0_Q_SHARER_ID = 2335,
	OBJECT_X0_R_SHARER_ID = 2336,
	OBJECT_Y0_Q_SHARER_ID = 2337,
	OBJECT_Y0_R_SHARER_ID = 2338,
	OBJECT_X1_Q_SHARER_ID = 2339,
	OBJECT_X1_R_SHARER_ID = 2340,
	OBJECT_Y1_Q_SHARER_ID = 2341,
	OBJECT_Y1_R_SHARER_ID = 2342,
};

enum PythonStatus
{
	CPP_STANDBY = 0,
	GRABBING_SCENE0 = 1,
	GRABBING_SCENE1 = 2,
	INSPECTION = 3,
	SETTING_SCENE0_ROI = 4,
	SETTING_SCENE1_ROI = 5,
	CPP_WORK_DONE = 6,
};

enum CudaConsts
{
	NUM_THREADS_PER_BLOCK_SORT_RXY = 256,
	LEN_BLOCKS_SORT_RXY = 32,
	NUM_THREADS_PER_BLOCK_LINE = 16,
	NUM_PIXELS_4_FINDNG_POINT = 1024,
	MAX_STATS_LEN = 20,
	ADAPTIVE_THRESHOLD_C = 2,
	STAT_COLS = 5,
	MAX_CANDIDATE_LEN_RATIO = 2,
	CANDIDATE_W = 2,
	CANDIDATE_H = 3,
	CANDIDATE_X = 0,
	CANDIDATE_Y = 1,
	NUM_MOMOENTS = 4,
	EXTRA_ROI_LEN_RATIO = 1,
};

const float ZOOM_GLOBAL = (float)1;
const float ZOOM_BINARYMAP = (float)0.25;
const ushort STANDARD_ORIGINAL_SCENE_H = (ushort)round(CAM_H * ZOOM_GLOBAL);
const ushort STANDARD_ORIGINAL_SCENE_W = (ushort)round(CAM_W * ZOOM_GLOBAL);
const uint NUM_STANDARD_ORIGINAL_SCENE_PIXELS = STANDARD_ORIGINAL_SCENE_H * STANDARD_ORIGINAL_SCENE_W;
const uint NUM_MALLOCED_SCENE_PIXELS = (uint)(round(STANDARD_ORIGINAL_SCENE_H * ZOOM_BINARYMAP)
	* round(STANDARD_ORIGINAL_SCENE_W * ZOOM_BINARYMAP));
const ushort LEN_MALLOCED_RADIUS = (ushort)(STANDARD_ORIGINAL_SCENE_H - 1) / 2;
const uint NUM_MALLOCED_CANDIDATES = (uint)(((STANDARD_ORIGINAL_SCENE_W + LEN_BLOCKS_SORT_RXY - 1) / LEN_BLOCKS_SORT_RXY)
	* ((STANDARD_ORIGINAL_SCENE_H + LEN_BLOCKS_SORT_RXY - 1) / LEN_BLOCKS_SORT_RXY));

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;