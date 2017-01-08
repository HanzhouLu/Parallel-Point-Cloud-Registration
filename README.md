# Parallel-Point-Cloud-Registration
15-618 Project
This is a our team's project for course 15618 Parallel Computing Architecture and Programming
For our final report, check Final Report.pdf.
All cource code(not including libraries) and data sets are provided in PICP folder.

To run our program, you need to download Eigen 3.1.3(or above) and boost 1.62.0(or above) and save them in picp/library

Folder strucure should like:

pcip-|-data\n
     |-library-|-boost_1_62_0\n
     |         |-eigen\n
     |-src-|-icp.cpp\n
     |     |-icp.cu\n
     |     |-kdTree.cpp\n
     |-Makefile\n
     
Then you are good to make runnable file.

Evironment requirement:

GCC 5.0 or above

OpenMP

CUDA 7.5

Hope you enjoy it!
