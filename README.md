# Aligner
Vision Align Software with CUDA, C++ and Python on Nvidia Jetson Nano


An aligner is a device that uses two cameras to correct a rotated and shifted object.
Each camera detects rotation and shifting of the region of interest.
This software locates the changed region of interest.

OpenCV's template matching is vulnerable to rotation and cannot be used.
So I implemented an algorithm using GPU.

CUDA, C++ (Calculation)  <----- share memory -----> Python(Web viewer, Django)

------------------------------------------------

CMakeLists.txt -> Edit cuda version

mkdir build
cd build
cmake ..
make
./Aligner

------------------------------------------------

structure(CUDA, C++) :
AlignerLauncher
--Aligner
----PatternMatching
------CudaSupporter
----Grabber
----MemorySharing
----//Webcam
AlignerConsts
kbhit

<----- share memory -----> Python(Web viewer, Django)

------------------------------------------------

AlignerLauncher : Launch C++ and Python

Aligner : C++ class, set and free share memory, set and free camera, image process, communicate with python

PatternMatching : Manage GPU memory, launch CUDA kernel, image process

CudaSupporter : Image process

Grabber : Pylon camera class

MemorySharing : Memory share with python

Webcam : If you have no pylon camera, activate this

AlignerConsts : Consts

kbhit : Key press check
