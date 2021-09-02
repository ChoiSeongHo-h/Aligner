#pragma once

#include <opencv2/opencv.hpp>

class Grabber
{
private:
    cv::VideoCapture* cap;

public:
    cv::Mat grabbedMat;

    Grabber();
    ~Grabber();
    void Shot();
};
