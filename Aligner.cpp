#include "Aligner.h"

Aligner::Aligner()
{
    (this->memSharerStruct).scene0DataSharer = new ImageMemorySharer(SCENE0_DATA_SHARER_ID, NUM_STANDARD_ORIGINAL_SCENE_PIXELS);
    (this->memSharerStruct).scene1DataSharer = new ImageMemorySharer(SCENE1_DATA_SHARER_ID, NUM_STANDARD_ORIGINAL_SCENE_PIXELS);
    (this->memSharerStruct).object0DataSharer = new ImageMemorySharer(OBJECT0_DATA_SHARER_ID, NUM_STANDARD_ORIGINAL_SCENE_PIXELS);
    (this->memSharerStruct).object1DataSharer = new ImageMemorySharer(OBJECT1_DATA_SHARER_ID, NUM_STANDARD_ORIGINAL_SCENE_PIXELS);

    (this->memSharerStruct).interStatusSharer = new ByteMemorySharer(INTER_STATUS_SHARER_ID);
    (this->memSharerStruct).objectX0QSharer = new ByteMemorySharer(OBJECT_X0_Q_SHARER_ID);
    (this->memSharerStruct).objectX0RSharer = new ByteMemorySharer(OBJECT_X0_R_SHARER_ID);
    (this->memSharerStruct).objectY0QSharer = new ByteMemorySharer(OBJECT_Y0_Q_SHARER_ID);
    (this->memSharerStruct).objectY0RSharer = new ByteMemorySharer(OBJECT_Y0_R_SHARER_ID);
    (this->memSharerStruct).objectX1QSharer = new ByteMemorySharer(OBJECT_X1_Q_SHARER_ID);
    (this->memSharerStruct).objectX1RSharer = new ByteMemorySharer(OBJECT_X1_R_SHARER_ID);
    (this->memSharerStruct).objectY1QSharer = new ByteMemorySharer(OBJECT_Y1_Q_SHARER_ID);
    (this->memSharerStruct).objectY1RSharer = new ByteMemorySharer(OBJECT_Y1_R_SHARER_ID);

    (this->scene0PointingMat).create(STANDARD_ORIGINAL_SCENE_H, STANDARD_ORIGINAL_SCENE_W, CV_8UC1);
    (this->scene1PointingMat).create(STANDARD_ORIGINAL_SCENE_H, STANDARD_ORIGINAL_SCENE_W, CV_8UC1);
    (this->scene0PointingMat).data = (this->memSharerStruct).scene0DataSharer->getPtr();
    (this->scene1PointingMat).data = (this->memSharerStruct).scene1DataSharer->getPtr();
}

Aligner::~Aligner()
{
    delete (this->memSharerStruct).scene0DataSharer;
    delete (this->memSharerStruct).scene1DataSharer;
    delete (this->memSharerStruct).object0DataSharer;
    delete (this->memSharerStruct).object1DataSharer;

    delete (this->memSharerStruct).interStatusSharer;
    delete (this->memSharerStruct).objectX0QSharer;
    delete (this->memSharerStruct).objectX0RSharer;
    delete (this->memSharerStruct).objectY0QSharer;
    delete (this->memSharerStruct).objectY0RSharer;
    delete (this->memSharerStruct).objectX1QSharer;
    delete (this->memSharerStruct).objectX1RSharer;
    delete (this->memSharerStruct).objectY1QSharer;
    delete (this->memSharerStruct).objectY1RSharer;
}

void Aligner::Launch()
{
    cv::Rect scene0ROI;
    cv::Rect scene1ROI;
    cv::Point p0, p1;

    //press key
    while (!kbhit())
    {
        int condition = (this->memSharerStruct).interStatusSharer->ReadByte();
        switch (condition)
        {
        case GRABBING_SCENE0:
            GrabScene((this->cam0), (this->memSharerStruct), GRABBING_SCENE0);
            break;

        case GRABBING_SCENE1:
            GrabScene((this->cam0), (this->memSharerStruct), GRABBING_SCENE1);
            break;

        case INSPECTION:
            (this->matcher0).originalScene = (this->scene0PointingMat);
            (this->matcher1).originalScene = (this->scene1PointingMat);
            p0 = (this->matcher0).FindRXY(scene0ROI);
            p1 = (this->matcher1).FindRXY(scene1ROI);
            printf("%d %d, %d %d\n", p0.x + scene0ROI.x, p0.y + scene0ROI.y, p1.x + scene1ROI.x, p1.y + scene1ROI.y);
            (this->memSharerStruct).interStatusSharer->WriteByte(CPP_STANDBY);
            break;

        case SETTING_SCENE0_ROI:
            SetSceneROI((this->memSharerStruct), (this->scene0PointingMat), (this->matcher0), scene0ROI, SETTING_SCENE0_ROI);
            break;

        case SETTING_SCENE1_ROI:
            SetSceneROI((this->memSharerStruct), (this->scene1PointingMat), (this->matcher1), scene1ROI, SETTING_SCENE1_ROI);
            break;

        default:
            break;
        }
    }
}

void Aligner::GrabScene(Grabber& cam, const MemSharerStruct& memSharerStruct, const int& idx)
{
    cam.Shot();
    if (cam.grabbedMat.cols <= 1 || cam.grabbedMat.rows <= 1)
        return;

    ImageMemorySharer* sceneDataSharer;
    if (idx == GRABBING_SCENE0)
        sceneDataSharer = (this->memSharerStruct).scene0DataSharer;
    else if (idx == GRABBING_SCENE1)
        sceneDataSharer = (this->memSharerStruct).scene1DataSharer;
    else return;

    cv::Mat tempMat;
    cv::resize(cam.grabbedMat, tempMat, cv::Size(), ZOOM_GLOBAL, ZOOM_GLOBAL, cv::INTER_AREA);
    memcpy(sceneDataSharer->getPtr(), tempMat.data, NUM_STANDARD_ORIGINAL_SCENE_PIXELS);

    (this->memSharerStruct).interStatusSharer->WriteByte(CPP_WORK_DONE);
}

void Aligner::SetSceneROI(const MemSharerStruct& memSharerStruct, const cv::Mat& scenePointingMat, PatternMatcher& matcher
    , cv::Rect sceneROI, const int& idx)
{
    ImageMemorySharer* objectDataSharer;
    if (idx == SETTING_SCENE0_ROI)
        objectDataSharer = (this->memSharerStruct).object0DataSharer;
    else if (idx == SETTING_SCENE1_ROI)
        objectDataSharer = (this->memSharerStruct).object1DataSharer;
    else return;

    ushort x0, y0, x1, y1, tempVar, h, w;
    x0 = (this->memSharerStruct).objectX0QSharer->ReadByte() * UCHAR_MAX + (this->memSharerStruct).objectX0RSharer->ReadByte();
    y0 = (this->memSharerStruct).objectY0QSharer->ReadByte() * UCHAR_MAX + (this->memSharerStruct).objectY0RSharer->ReadByte();
    x1 = (this->memSharerStruct).objectX1QSharer->ReadByte() * UCHAR_MAX + (this->memSharerStruct).objectX1RSharer->ReadByte();
    y1 = (this->memSharerStruct).objectY1QSharer->ReadByte() * UCHAR_MAX + (this->memSharerStruct).objectY1RSharer->ReadByte();

    if (x0 > x1)
    {
        tempVar = x0;
        x0 = x1;
        x1 = tempVar;
    }
    if (y0 > y1)
    {
        tempVar = y0;
        y0 = y1;
        y1 = tempVar;
    }
    h = y1 - y0;
    w = x1 - x0;
    uint roiSize = h * w * sizeof(uchar);
    if (roiSize == 0)
        (this->memSharerStruct).interStatusSharer->WriteByte(CPP_STANDBY);

    cv::Mat tempMat = scenePointingMat(cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)));
    memcpy(objectDataSharer->getPtr(), tempMat.clone().data, roiSize);

    (this->memSharerStruct).interStatusSharer->WriteByte(CPP_WORK_DONE);

    matcher.originalObject.create(tempMat.rows, tempMat.cols, CV_8UC1);
    matcher.originalObject.data = objectDataSharer->getPtr();
    resize(matcher.originalObject, matcher.object, cv::Size(), ZOOM_BINARYMAP, ZOOM_BINARYMAP, cv::INTER_AREA);
    matcher.SetObjectInfo();

    ushort dw = ushort(EXTRA_ROI_LEN_RATIO * w);
    ushort dh = ushort(EXTRA_ROI_LEN_RATIO * h);

    x0 = x0 - dw;
    y0 = y0 - dh;
    x1 = x1 + dw;
    y1 = y1 + dh;
    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    x1 = x1 >= scenePointingMat.cols ? scenePointingMat.cols : x1;
    y1 = y1 >= scenePointingMat.rows ? scenePointingMat.rows : y1;

    sceneROI = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}
