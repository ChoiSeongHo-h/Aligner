from django.shortcuts import render
from django.http import StreamingHttpResponse

from django.conf import settings as glb
from django.conf import settings as consts
from enum import Enum

class AlignerConsts(Enum):
    CPP_STANDBY = 0
    GRABBING_SCENE0 = 1
    GRABBING_SCENE1 = 2
    INSPECTION = 3
    SETTING_SCENE0_ROI = 4
    SETTING_SCENE1_ROI = 5
    CPP_WORK_DONE = 6
    UCHAR_MAX = 255

def GetHtmlArgs():
    return {'scene0URI': glb.SCENE0.ReadURI(), 'scene1URI': glb.SCENE1.ReadURI(), 'object0URI': glb.OBJECT0.ReadURI(), 'object1URI': glb.OBJECT1.ReadURI()}

def WaitChangeImage(memSharer, h = 0, w = 0):
    while(glb.INTER_STATUS.ReadByte() != AlignerConsts.CPP_WORK_DONE.value) :
        pass
    memSharer.SetURI(h, w)
    glb.INTER_STATUS.WriteByte(AlignerConsts.CPP_STANDBY.value)

def ViewHome(request):
    glb.IS_ACCESSED = False
    status = list(request.GET.keys())
    if status == []:
        return render(request, 'aligner_home.html',  GetHtmlArgs())
    
    status = status[0]
    if status == '0':
        glb.INTER_STATUS.WriteByte(AlignerConsts.GRABBING_SCENE0.value)
        WaitChangeImage(glb.SCENE0)
    elif status == '1':
        glb.INTER_STATUS.WriteByte(AlignerConsts.GRABBING_SCENE1.value)
        WaitChangeImage(glb.SCENE1)
    elif status == 'inspect':
        glb.INTER_STATUS.WriteByte(AlignerConsts.INSPECTION.value)
    return render(request, 'aligner_home.html',  GetHtmlArgs())


def SetObject(x, y, id):
    if glb.IS_ACCESSED == False:
        glb.OBJECT_X0_Q.WriteByte(x//AlignerConsts.UCHAR_MAX.value)
        glb.OBJECT_X0_R.WriteByte(x%AlignerConsts.UCHAR_MAX.value)
        glb.OBJECT_Y0_Q.WriteByte(y//AlignerConsts.UCHAR_MAX.value)
        glb.OBJECT_Y0_R.WriteByte(y%AlignerConsts.UCHAR_MAX.value)
        glb.IS_ACCESSED = True
    else :
        glb.OBJECT_X1_Q.WriteByte(x//AlignerConsts.UCHAR_MAX.value)
        glb.OBJECT_X1_R.WriteByte(x%AlignerConsts.UCHAR_MAX.value)
        glb.OBJECT_Y1_Q.WriteByte(y//AlignerConsts.UCHAR_MAX.value)
        glb.OBJECT_Y1_R.WriteByte(y%AlignerConsts.UCHAR_MAX.value)
        glb.IS_ACCESSED = False
        x0 = glb.OBJECT_X0_Q.ReadByte()*AlignerConsts.UCHAR_MAX.value+glb.OBJECT_X0_R.ReadByte()
        y0 = glb.OBJECT_Y0_Q.ReadByte()*AlignerConsts.UCHAR_MAX.value+glb.OBJECT_Y0_R.ReadByte()

        h = abs(y-y0)
        w = abs(x-x0)
        if id == 0:
            tempStatus = AlignerConsts.SETTING_SCENE0_ROI.value
            tempMemSharer = glb.OBJECT0
        elif id == 1:
            tempStatus = AlignerConsts.SETTING_SCENE1_ROI.value
            tempMemSharer = glb.OBJECT1

        glb.INTER_STATUS.WriteByte(tempStatus)
        WaitChangeImage(tempMemSharer, h, w)

def Set0(request):
    status = list(request.GET.keys())
    if status == []:
        return render(request, 'aligner_home.html',  GetHtmlArgs())

    (x, y) = status[0].split(',')
    SetObject(int(x), int(y), 0)
    return render(request, 'aligner_home.html',  GetHtmlArgs())


def Set1(request):
    status = list(request.GET.keys())
    if status == []:
        return render(request, 'aligner_home.html',  GetHtmlArgs())

    (x, y) = status[0].split(',')
    SetObject(int(x), int(y), 1)
    return render(request, 'aligner_home.html',  GetHtmlArgs())