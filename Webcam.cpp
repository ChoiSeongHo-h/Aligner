#include "Webcam.h"

Grabber::Grabber()
{
	cap = new cv::VideoCapture(0);
}

Grabber::~Grabber()
{
	cap->release();
	delete cap;
}

void Grabber::Shot()
{
	cv::Mat tempMat;
	*cap >> tempMat;
	cvtColor(tempMat, grabbedMat, cv::COLOR_BGR2GRAY);
}