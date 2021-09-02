#include "Grabber.h"

Grabber::Grabber()
{
    using namespace Pylon;

    PylonInitialize();
    (this->camera) = new CInstantCamera(CTlFactory::GetInstance().CreateFirstDevice());
    (*(this->camera)).MaxNumBuffer = 1;
    (*(this->camera)).StartGrabbing();
    (this->formatConverter).OutputPixelFormat = PixelType_Mono8;
}

Grabber::~Grabber()
{
    using namespace Pylon;

    (*(this->camera)).StopGrabbing();
    delete (this->camera);
    PylonTerminate();
}

void Grabber::Shot()
{
    using namespace Pylon;
    using namespace cv;

    (*(this->camera)).RetrieveResult(500, (this->ptrGrabResult), TimeoutHandling_ThrowException);
    if ((this->ptrGrabResult)->GrabSucceeded())
    {
        (this->formatConverter).Convert((this->pylonImage), (this->ptrGrabResult));
        grabbedMat = Mat((this->ptrGrabResult)->GetHeight(), (this->ptrGrabResult)->GetWidth(), CV_8UC1, (uchar*)((this->pylonImage).GetBuffer()));
    }
}
