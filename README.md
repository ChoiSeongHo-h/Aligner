# Aligner
## Overview
Circuit Aligner with CUDA
- **Rotation-invariant template matching** considering GPU architecture
- Providing a **web frontend** with Django
- Available **at 1/10 the price** of traditional products

## 1 Rotation-invariant template matching considering GPU architecture
Accelerating **rotation-invariant template matching** with **CUDA parallel algorithms** while considering **GPU architecture**

### 1.1 Motivation
![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/8637c327-4f86-447e-9f36-2d58a526a526)
- Red crosshairs: Characteristic parts of aligned circuits saved in the aligner
- Blue crosshairs: Characteristic parts of the misaligned circuit that entered the aligner

Rotation-invariant template matching is necessary to compute the amount of rotation and translation of a circuit. 

The user wants to align the misaligned circuit by rotating and translating it in two dimensions. The aligned circuit already has the characteristic patches of the aligned circuit stored. The aligner finds the same patches in the misaligned circuit and calculates the amount of rotation and translation of the circuit based on the degree of misalignment between the patches.

### 1.2 Template matching overview
![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/b12f0802-473a-4d2d-ae37-5b8a19c2a0fb)

Template matching performs the following coarse to fine search:
1. Coarse search: Fast search based on pixel statistics, taking advantage of the GPU architecture.
2. Fine search: Precise search using circular sum

### 1.3 Step 1: Coarse Search
Fast search based on pixel statistics, taking advantage of the GPU architecture.
A coarse search is performed as follows:
1. Find the 1st-4th order moments of a reference patch.
2. Construct an ROI around a reference patch in a misaligned circuit image.
3. Traverse all pixels in the ROI to calculate the 1st-4th order moments. The region where the moments are calculated is centred on the pixel being traversed and has the same size as the reference patch.
4. Compare the reference patch moment vector to the moment vector of all patches in the ROI.
5. Extract candidate regions using an adaptive threshold.

#### 1.3.1 1st to 4th Order Moments
![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/766b1c7f-d576-4d15-894c-78d1a3e3f256)

The 1st to 4th order moments represent the mean, variance, skewness, and kurtosis of the pixels in the patch. These features are invariant to rotation or translation of the patch. For example, a patch with a mean of 3 will still have a mean of 3 if you move it slightly to the right.


#### 1.3.2 Grid, Block, and Thread Structures and Shared Memory
![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/4a4fddc5-d8ab-4ac8-9759-01fb8f81bb24)

The unit for performing kernels (functions) on the GPU is a grid. A grid is made up of blocks. Blocks are composed of threads. Threads within the same block can access shared memory and share data. Shared memory is very fast, so it should be actively used to improve performance.

![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/7eb25eff-6ea2-4bc0-9c91-f6c010b5384f)


In Step 1, the aligner's grid consists of (16 columns/16 rows/16) blocks. Each block consists of (16 x 16) threads. Each thread is responsible for one patch to compute moments.

Within each block, threads share a pixel area of (16+margin x 16+margin). Due to the limitations of the Jetson Nano, it is not possible to allocate a large amount of shared memory.

For speed, all GPU memory, including shared memory, is pre-allocated and the image is uploaded.

### 1.4 Step 2: Fine Search
![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/9043638e-e927-44c6-80f0-1a1472cf250d)

The aligner accumulates pixel values along a circular path. Each radius of the circle has a cumulative value. With n radiuses, an n-dimensional vector of accumulated values can be constructed. This vector encodes the rotation-invariant features of the image.

1. Find the circular sum of the reference patches.
2. Perform a circular sum in parallel similar to Step1 on the candidates that passed the adaptive threshold in Step1. Each thread is responsible for two circular paths.
3. Apply NCC between the reference and candidate vectors to pick the pixel with the maximum value.

#### 1.4.1 Algorithms for accelerating parallel computing
As mentioned in Step 2, each thread is responsible for two circular paths. One thread is responsible for the k th circle and the n-k th circle to reduce bottlenecks between threads.

Additionally, a reduction algorithm is used to extract maxima and sums to accelerate the process.

## 2 Providing a web frontend with Django
![image](https://github.com/ChoiSeongHo-h/Aligner/assets/72921481/ef185b03-c6a6-4db1-aed5-9baf36ac70ea)

Django provides a web interface. Users can access the aligner from any device, not just a PC. In addition to PC, the responsive web is also conveniently accessible on mobile.

The sorter backend and Django frontend send and receive data through shared memory. For image sharing, a reference-only object is used to minimise latency.

## 3 Available at 1/10 the price of traditional products

At the time the aligner was developed, previous circuit alignment PCs cost around $1,000. However, this aligner, based on Jetson Nano, costs around $100. Despite the low price, there is no reduction in performance (search time < 50ms).

## 4 File Description
- AlignerLauncher: main()
- Aligner: sets up and tears down shared memory, sets up and tears down cameras, processes images, and communicates with the web.
- PatternMatching: Manages GPU memory, runs the CUDA kernel, and performs image processing using OpenCV.
- CudaSupporter: Image processing, finding the moment vector and circular sum vector for each pixel using CUDA.
- Grabber: Pylon camera classes, setting up and destroying cameras and grabbers
- MemorySharing: Sharing memory with the web.
- Webcam: Enables the webcam if there is no Pylon camera.
- AlignerConsts: Constants
- kbhit: checks for key presses

## 5 Presentation slides


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
![슬라이드27](https://user-images.githubusercontent.com/72921481/131961903-39c26955-3a64-4157-ae3c-486d2d423800.JPG)
![슬라이드28](https://user-images.githubusercontent.com/72921481/131961904-7a07db63-4818-4cdf-bc60-045b2390aeb7.JPG)
![슬라이드29](https://user-images.githubusercontent.com/72921481/131961907-ce865d66-367b-4971-9832-0fa5dd292406.JPG)
