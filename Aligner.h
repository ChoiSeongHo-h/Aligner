#pragma once

#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <sys/wait.h>

#include "PatternMatching.h"
#include "Grabber.h"
#include "MemorySharing.h"
#include "AlignerConsts.h"
#include "kbhit.h"
//#include "Webcam.h"

class Aligner
{
private:
    struct MemSharerStruct
    {
        ImageMemorySharer* scene0DataSharer;
        ImageMemorySharer* scene1DataSharer;
        ImageMemorySharer* object0DataSharer;
        ImageMemorySharer* object1DataSharer;
        ByteMemorySharer* interStatusSharer;
        ByteMemorySharer* objectX0QSharer;
        ByteMemorySharer* objectX0RSharer;
        ByteMemorySharer* objectY0QSharer;
        ByteMemorySharer* objectY0RSharer;
        ByteMemorySharer* objectX1QSharer;
        ByteMemorySharer* objectX1RSharer;
        ByteMemorySharer* objectY1QSharer;
        ByteMemorySharer* objectY1RSharer;
    };

    MemSharerStruct memSharerStruct;
    Grabber cam0;
    PatternMatcher matcher0;
    PatternMatcher matcher1;
    cv::Mat scene0PointingMat;
    cv::Mat scene1PointingMat;

    void GrabScene(Grabber& cam, const MemSharerStruct& memSharerStruct, const int& idx);
    void SetSceneROI(const MemSharerStruct& memSharerStruct, const cv::Mat& scenePointingMat, PatternMatcher& matcher, cv::Rect sceneROI
        , const int& idx);

public:
    Aligner();
    ~Aligner();
    void Launch();
};