#pragma once

#include <pylon/PylonIncludes.h>
#include "opencv2/opencv.hpp"

class Grabber
{
private:
    Pylon::CInstantCamera* camera;
    Pylon::CImageFormatConverter formatConverter;
    Pylon::CPylonImage pylonImage;
    Pylon::CGrabResultPtr ptrGrabResult;

public:
    cv::Mat grabbedMat;

    Grabber();
    ~Grabber();
    void Shot();
};