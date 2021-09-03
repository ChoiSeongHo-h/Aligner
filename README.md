# Aligner

Vision Aligning Software with CUDA, C++ and Python on Nvidia Jetson Nano


An aligner is a device that uses two cameras to correct a rotated and shifted object.

Each camera detects rotation and shifting of the region of interest.

This software locates the changed region of interest.


OpenCV's template matching is vulnerable to rotation and cannot be used.

So I implemented an algorithm using GPU.


CUDA, C++ (Calculation)  <----- share memory -----> Python(Web viewer, Django)


------------------------------------------------


(CMakeLists.txt -> Edit cuda version)

mkdir build

cd build

cmake ..

make

./Aligner


------------------------------------------------


Idea of finding a rotated object

1. In a circular region, an object rotated about the midpoint of the region has the same mean, variance, skewness, and kurtosis as compared to the original.

(the mean, variance, skewness, and kurtosis are called moments vectors)

2. The circular integral of an object is maintained as it rotates around the center of the circle

At 2, circular integration is slow because it is difficult to utilize the GPU's shared memory and caching.

At 1, Specify a rectangular area instead of a circular area, Can utilize the GPU's shared memory and caching, which is fast.

However, it is inaccurate because it is not a circular region.

Therefore, the moment vector is found in the rectangular area to quickly determine the candidate group. After that, the correct coordinates are returned by using 2.


------------------------------------------------


Algorithm progress (CUDA, C++, image process)

1. Grab Scene

2. Set object(pattern)

3. Compute object's info (1st~4th moments vector, circular integral vector with a radius dimension, ...)

4. Specify the ROI(search area, near the object) in the scene

5. Grab rotated and shifted Scene

6. Search object in ROI(using CUDA)

6.1. moments vector comparison

6.1.1. Compute the moments vector for each pixel in the ROI

6.1.2. Compare the moments vector of the object with the moments vector of the pixels in the ROI.

6.1.3. A candidate group is extracted by applying an adaptive threshold.

6.2. circular integral vector comparison

6.2.1. Calculate a circular integral vector with a radius dimension for the candidates.

6.2.2. Compare the circular integral vector of the object with the circular integral vector of the pixels in the ROI. get correlation coefficient

6.2.3. Extract the coordinates with the highest correlation coefficient


------------------------------------------------


structure(CUDA, C++) :

AlignerLauncher

----Aligner

--------PatternMatching

------------CudaSupporter

--------Grabber

--------MemorySharing

--------//Webcam

--------kbhit

AlignerConsts



<----- shared memory -----> Python(Web viewer, Django)


------------------------------------------------


AlignerLauncher : Launch C++ and Python

Aligner : Set and free shared memory, set and free camera, image process, communicate with python

PatternMatching : Manage GPU memory, launch CUDA kernel, image process

CudaSupporter : Image process

Grabber : Pylon camera class

MemorySharing : Memory share with python

Webcam : If you have no pylon camera, activate this

AlignerConsts : Consts

kbhit : Key press check
