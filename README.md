# Parallel-Point-Cloud-Registration
This is a our team's project for course 15618 Parallel Computing Architecture and Programming

For our final report, check Final Report.pdf.

All cource code(not including libraries) and data sets are provided in PICP folder.

To run our program, you need to download Eigen 3.1.3(or above) and boost 1.62.0(or above) and save them in picp/library

Folder strucure should like:


pcip-|-data
     |-library-|-boost_1_62_0
     |         |-eigen
     |-src-|-icp.cpp
     |     |-icp.cu
     |     |-kdTree.cpp
     |-Makefile
     
Then you are good to make runnable file.

Evironment requirement:

GCC 5.0 or above

OpenMP

CUDA 7.5

Hope you enjoy it!
