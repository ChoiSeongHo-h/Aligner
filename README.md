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

1. In a circular region, an object rotated about the midpoint of the region has the same mean, variance, skewness, and kurtosis as compared to the original(the mean, variance, skewness, and kurtosis are called moments vectors).

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

PatternMatching : Manage GPU memory, launch CUDA kernel, image process using OpenCV

CudaSupporter : Image process, use CUDA to find the moment vector and circular integral vector for each pixel.

Grabber : Pylon camera class, set and dispose camera and grab

MemorySharing : Memory share with python. Sharing memory in bytes. Images are shared as a one-dimensional byte array. 
Numbers greater than 255 are shared by the number's quotient and remainder.

Webcam : If you have no pylon camera, activate this and modify CMakeLists

AlignerConsts : Consts

kbhit : Key press check function


------------------------------------------------


![슬라이드1](https://user-images.githubusercontent.com/72921481/131960995-5ba56429-e7fe-4ef9-8433-94862865c6be.JPG)
![슬라이드2](https://user-images.githubusercontent.com/72921481/131961001-6ca9769a-fae9-4a34-9a41-63c38e88da90.JPG)
![슬라이드3](https://user-images.githubusercontent.com/72921481/131961003-2bf304bd-980c-4aea-9447-1a54c5ed4f94.JPG)
![슬라이드4](https://user-images.githubusercontent.com/72921481/131961005-48bc1803-9df0-461d-b561-c93189b2201a.JPG)
![슬라이드5](https://user-images.githubusercontent.com/72921481/131961006-96c288f3-e933-4f10-8d35-ebac8d3bbbb6.JPG)
![슬라이드6](https://user-images.githubusercontent.com/72921481/131961009-8324bc10-7dae-4cf0-9a43-3b895614e53f.JPG)
![슬라이드7](https://user-images.githubusercontent.com/72921481/131961012-63a01cf2-4155-4fa9-9423-0bc3528b6042.JPG)
![슬라이드8](https://user-images.githubusercontent.com/72921481/131961016-9677104e-3bfd-429c-b351-76f46f3cf2e4.JPG)
![슬라이드9](https://user-images.githubusercontent.com/72921481/131961018-8d8362d5-d8b5-430c-82e9-8b94c9c1f200.JPG)
![슬라이드10](https://user-images.githubusercontent.com/72921481/131961020-4e01cf49-7686-4e93-b87c-bdc06085be91.JPG)
![슬라이드11](https://user-images.githubusercontent.com/72921481/131961021-a854689f-577d-4b42-ae11-0106bbaaf9ba.JPG)
![슬라이드12](https://user-images.githubusercontent.com/72921481/131961023-018390a5-1981-44b0-8a51-6b4c61994834.JPG)
![슬라이드13](https://user-images.githubusercontent.com/72921481/131961026-6a993d25-f914-4589-ad69-c773225d8eb1.JPG)
![슬라이드14](https://user-images.githubusercontent.com/72921481/131961028-05dd43e5-cbbe-4649-a4e4-d02e18c3ffe0.JPG)
![슬라이드15](https://user-images.githubusercontent.com/72921481/131961030-be76dff9-fc5d-48d3-a69d-51ae0df26d87.JPG)
![슬라이드16](https://user-images.githubusercontent.com/72921481/131961032-7e6eca95-6c14-46b0-8f75-b837cc8e9b3c.JPG)
![슬라이드17](https://user-images.githubusercontent.com/72921481/131961033-948bfc79-f99c-4447-ba83-4e6d714e62a1.JPG)
![슬라이드18](https://user-images.githubusercontent.com/72921481/131961034-fee1f6de-40f6-4b6d-8d70-b7c87f9dacae.JPG)
![슬라이드19](https://user-images.githubusercontent.com/72921481/131961036-37206b1c-94ba-4282-8ef0-0fb0357b4396.JPG)
![슬라이드20](https://user-images.githubusercontent.com/72921481/131961039-57cb19f2-ef97-403e-91a4-bff7d6d44a86.JPG)
![슬라이드21](https://user-images.githubusercontent.com/72921481/131961040-9c2c5179-f6c2-4db6-aff9-d8eca6f9d791.JPG)
![슬라이드22](https://user-images.githubusercontent.com/72921481/131961041-6b7295b1-59d2-41c5-b728-b92490a04008.JPG)
![슬라이드23](https://user-images.githubusercontent.com/72921481/131961046-3d1517c2-418d-4c0d-a914-29bafb061ed6.JPG)
![슬라이드24](https://user-images.githubusercontent.com/72921481/131961048-26a8f6ee-f79b-4962-aa33-15bfde31ed4c.JPG)
![슬라이드25](https://user-images.githubusercontent.com/72921481/131961050-67d683dd-bff9-4a9e-a5d5-dd0c161353d5.JPG)
![슬라이드26](https://user-images.githubusercontent.com/72921481/131961052-545da30d-5d11-4cbf-a2de-c5a488e7b47b.JPG)
![슬라이드27](https://user-images.githubusercontent.com/72921481/131961053-f1a4867f-aed5-4f33-a978-6b74a2385dba.JPG)
![슬라이드28](https://user-images.githubusercontent.com/72921481/131961055-61cc4b75-2801-46e5-9184-883b13f440f8.JPG)


