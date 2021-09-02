# Aligner
Vision Align Software with CUDA, C++ and Python on Nvidia Jetson Nano


C++, CUDA (Calculation)  <--- shared memory ---> Python(Web, Django)

------------------------------------------------

CMakeLists.txt -> Edit cuda version

mkdir build
cd build
cmake ..
make
./Aligner

------------------------------------------------

structure :
AlignerLauncher
--Aligner
----PatternMatching
------CudaSupporter
----Grabber
----MemorySharing
----//Webcam

AlignerConsts
kbhit


AlignerLauncher : Launch C++ Python
Aligner : C++ class, 
