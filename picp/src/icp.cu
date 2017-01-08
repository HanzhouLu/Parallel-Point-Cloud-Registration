/**
 *  CMU 15418 Fall 2016 PICP project
 *  Author: Hanzhou Lu, Yujie Wei
 *  Date: Nov 21, 2016
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <atomic>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <assert.h>


#include <Eigen/Dense>
#include <omp.h>
#include <cuda.h>

#include "kdtree.cpp"

typedef std::chrono::high_resolution_clock Clock;
#define PI 3.14159265
#define DEBUG


// Catch the cuda error
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif


using Eigen::MatrixXf;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::VectorXf;
using Eigen::Map;
using std::ofstream;
using std::cout;
using std::endl;

void outputPointCloud(std::string name, MatrixXf & matrix);
KdTree* plyParser(FILE *plyFile, int numOfVertexes, std::vector<float> &vertexes, int& nP);
void cudaFindNearest(int numBlocks, int threadsPerBlock, float *P, float *Q, int nP, int nQ, int* tree, int* layer, int* kd_tree_para, float *Q_select, int *min_index_device);
__global__ void kernelIterativeClosestPoint(float *P, float *Q, int nP, int nQ, int pointsPerThread, int* tree, int* layer, int* kd_tree_para, float *Q_select_device, int *min_index_device);

__device__ int cuGetTree(int *tree, int index);
__device__ void cuSetTree(int *tree, int index, int value);
__device__ void cuPositionToCoordinate(int* layer, int position, int sumMax, int *coordinate);
__device__ int cuPositionInTree(int x, int y, int z, int *kd_tree_para);
__device__ float cuMinDistancePointToBox(float* point, int* box);
__device__ float cuFindNearestNode(int* tree, int* layer, int* kd_tree_para, float x, float y, float z, int* returnValue);
__device__ float cuKdTreeSearch(int* tree, int* layer, int* kd_tree_para, float* target, int* left, int* boundaryBox, int* hyperPlan, int currLayer, int node, float minDistance, int* nearestNode);


int main(){

    /* Read from file */
    int dimension = 3;
    int nIter = 100;

    /* Start from origianal cloud P and ref cloud Q */
    std::vector<float> vertexes = std::vector<float>(0);
    int nP;
    FILE* input = fopen("data/vertex.txt", "r");
    KdTree* vertexTree = plyParser(input, 40256, vertexes, nP);
    cout << nP << endl;
    fclose(input);

    if(nP == 0){
        std::cout << "Point cloud error.\n" << std::endl;
        return 0;
    }
    std::cout << vertexes.size() << std::endl;
    int nQ = nP;

    MatrixXf P(dimension, nP);
    MatrixXf Q(dimension, nQ);

    /* Construct the Q from the point cloud file, Q is a 3 * nQ matrix. */
    for(int i=0; i<nQ; i++){
        for(int j=0; j<3; j++){
            int index = i * 3 + j;
            Q(j, i) = vertexes[index];
        }
    }

    /* Initialize transformation */
    Matrix3f R = Matrix3f::Zero();
    R(0,0) = 1;
    float angleAlongX = 30.0;
    R(1,1) = cos(angleAlongX * PI / 180);
    R(1,2) = -sin(angleAlongX * PI / 180);
    R(2,1) = -R(1,2);
    R(2,2) = R(1,1);


    Vector3f T = Vector3f::Zero();
    T(1) = 5.0; // Move along y

    /* Synthetic point cloud P, the goal is to learn a transfromation that transfrom P to Q */
    P = R * Q + T.replicate(1, nQ);

    /* Data on Host */
    int p_size = sizeof(float) * nP * 3;
    int q_size = sizeof(float) * nQ * 3;

    /* Construct the flaten data */
    float* Q_float = Q.data();
    float* P_float = P.data();

    /* Malloc space to store the result from CUDA */
    float* Q_select = (float *)malloc(p_size);

    /* Data on Device */
    float* P_device;
    float* Q_device;
    float* Q_select_device;
    int* min_index_device;
    float* min_dist_device;
    int *kd_tree_para;
    int *tree_device;
    int *layer_device;

    /* Compressed K-d tree args*/
    int necessary[5];
    necessary[0] = vertexTree->maxX;
    necessary[1] = vertexTree->maxY;
    necessary[2] = vertexTree->maxZ;
    necessary[3] = vertexTree->sumMax;
    necessary[4] = vertexTree->branches;

    /* Malloc space in CUDA */
    cudaMalloc(&P_device, p_size);
    cudaMalloc(&Q_device, q_size);
    cudaMalloc(&Q_select_device, p_size); /* Size should be the same as P */
    cudaMalloc(&min_index_device, sizeof(int) * nP);
    cudaMalloc(&min_dist_device, sizeof(float) * nP);
    cudaMalloc(&kd_tree_para, sizeof(int) * 5);

    /* k-d tree */
    int treeSize = sizeof(int) * ((pow(2, vertexTree->maxX) * pow(2, vertexTree->maxY) * pow(2, vertexTree->maxZ) * 2) / 32 + 1);
    int layerSize = sizeof(int) * vertexTree->sumMax;
    cudaMalloc(&tree_device, treeSize);
    cudaMalloc(&layer_device, layerSize);

    /* Copy the data to Q since it won't change */
    cudaMemcpy(Q_device, Q_float, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(kd_tree_para, necessary, sizeof(int) * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(tree_device, vertexTree->tree, treeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(layer_device, vertexTree->layer, layerSize, cudaMemcpyHostToDevice);


    /* Temp Result */
    R = Matrix3f::Identity();
    T = Vector3f::Zero();
    
    int numBlocks = 64;
    int threadsPerBlock = 64;

    /* Find match */
    VectorXf match = VectorXf::Zero(nP);
    VectorXf minDist = VectorXf::Zero(nP);

    
    std::cout << "Searching for match...\n";

    VectorXf MSE = VectorXf::Zero(nIter); // Minimum Square Error
    float acceptRate = 1;
    int acceptSize = (int)(nP * acceptRate);

    MatrixXf QShrink(dimension, acceptSize);
    MatrixXf PShrink(dimension, acceptSize);

    /* Start optimizing */
    for(int iter = 0; iter < nIter; iter ++){
        auto t1 = Clock::now();
        std::cout << "\n\nIter # " << iter << std::endl; 
        std::array<std::atomic<int>, 200> bucket = {};

        // /* KD Tree search */
        MatrixXf QSelect = MatrixXf(dimension, nP); 
        for(int i = 0; i < nP; i++){
            // std::cout << "searching " << i << "th point in kdtree." << std::endl;
            int nearestPoint[3];
            // std::cout << P(0,i) << ", " << P(1,i) << ", " << P(2,i) << std::endl;
            float dist = (float)vertexTree->findNearestNode(P(0, i), P(1, i), P(2, i), nearestPoint);
            // if (dist > maxDist) maxDist = dist;
            std::atomic_fetch_add(&bucket[(int)dist], 1);
            minDist(i) = dist;
            QSelect(0, i) = nearestPoint[0];
            QSelect(1, i) = nearestPoint[1];
            QSelect(2, i) = nearestPoint[2];
        }
        int count = 0;
        int limit = 0;
        for (int i = 0; i < 200; i++) {
            count += bucket[i];
            if (count > acceptSize) {
                limit = i;

                break;
            }
        }
        int j = 0;
        for (int i = 0; i < nP; i++) {
            if (minDist(i) < limit + 1) {
                PShrink(0, j) = P(0, i);
                PShrink(1, j) = P(1, i);
                PShrink(2, j) = P(2, i);
                QShrink(0, j) = QSelect(0, i);
                QShrink(1, j) = QSelect(1, i);
                QShrink(2, j) = QSelect(2, i);
            }
            j++;
            if (j == acceptSize) break;
        }

        PShrink = P;
        QShrink = QSelect;

        /* CUDA */
        /* Copy the latest P to device */
        auto cuda_t1 = Clock::now();

        /* Copy the P from host to device */
        cudaMemcpy(P_device, P_float, p_size, cudaMemcpyHostToDevice);
        
        /* Find the nearest point via CUDA */
        cudaFindNearest(numBlocks, threadsPerBlock, P_device, Q_device, nP, nQ, tree_device, layer_device, kd_tree_para, Q_select_device, min_index_device);
        /* Copy the select P */
        cudaMemcpy(Q_select, Q_select_device, p_size, cudaMemcpyDeviceToHost);

        auto cuda_t2 = Clock::now();
        std::cout << "Compute Delta t2-t1: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(cuda_t2 - cuda_t1).count()
              << " nanoseconds" << std::endl;

        QSelect = Map<MatrixXf>(Q_select, 3, nP);

        /* Calculate the min square distance */
        minDist = minDist.array();
        MSE(iter) = minDist.sum();
        std::cout << "MSE:" << minDist.sum() <<"\n"<< std::endl;
        if(MSE(iter) < 10.0)
            break;

        /* Find the best R & T iteratively */
        VectorXf weights = VectorXf::Ones(acceptSize);
        weights = weights / weights.sum();
        
        /* Select according columns from Q */

        //std::cout << QSelect << std::endl;
        VectorXf QBar = QShrink * weights;
        MatrixXf QMark = QShrink - QBar.replicate(1, QShrink.cols());

        QMark = QMark.cwiseProduct(weights.transpose().replicate(dimension, 1));

        VectorXf PBar = PShrink * weights;
        MatrixXf PMark = PShrink - PBar.replicate(1, PShrink.cols());
        //PMark = PMark.cwiseProduct(weights.transpose().replicate(dimension, 1));
        MatrixXf N = PMark * QMark.transpose();
        
        /* SVD decomposition */
        Eigen::JacobiSVD<Matrix3f> svd( N, Eigen::ComputeFullV | Eigen::ComputeFullU );
        Matrix3f U = svd.matrixU();
        Matrix3f V = svd.matrixV();

        float det = (U * V.transpose()).determinant();
        Vector3f diagVec(1.0, 1.0, det);
        R = V * diagVec.asDiagonal() * U.transpose();
        T = QBar - R * PBar;
        std::cout<<"R: \n" << R <<"\n" << std::endl; 
        std::cout<<"T: \n" << T <<"\n" << std::endl; 

        P = R * P + T.replicate(1, nP);
        auto t2 = Clock::now();
        std::cout << "Overall Delta t2-t1: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
              << " nanoseconds" << std::endl;

        // free((void *)bucket);
    }
    free(vertexTree);
    cudaFree(P_device);
    cudaFree(Q_device);
    cudaFree(Q_select_device); /* Size should be the same as P */
    cudaFree(min_index_device);
    cudaFree(&min_dist_device);
    cudaFree(kd_tree_para);
    cudaFree(tree_device);
    cudaFree(layer_device);
    outputPointCloud("data/rabbit_000_x30.txt", Q);
    outputPointCloud("data/rabbit_000_x30T.txt", P);
    //std::cout << "Q: \n" << Q <<"\n" << std::endl; 

}

void cudaFindNearest(int numBlocks, int threadsPerBlock, float *P, float *Q, int nP, int nQ, int* tree, int* layer, int* kd_tree_para, float *Q_select, int *min_index_device){
    /* Assign points to each thread */
    int pointsPerThread = (nP + numBlocks * threadsPerBlock - 1) / (numBlocks * threadsPerBlock);

    printf("%d\n", pointsPerThread);
    kernelIterativeClosestPoint<<<numBlocks, threadsPerBlock>>>(P, Q, nP, nQ, pointsPerThread, tree, layer, kd_tree_para, Q_select, min_index_device);
    cudaCheckError(cudaThreadSynchronize());

}


/* Use a kernel to find  */
__global__ void kernelIterativeClosestPoint(float *P, float *Q, int nP, int nQ, int pointsPerThread, int* tree, int* layer, int* kd_tree_para, float *Q_select_device, int *min_index_device){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int treeSize = sizeof(int) * ((powf(2, kd_tree_para[0]) * powf(2, kd_tree_para[1]) * powf(2, kd_tree_para[2]) * 2) / 32 + 1);

    int layerSize = sizeof(int) * kd_tree_para[3];

    if(idx == 0){
        printf("TreeSize: %d\n", treeSize);
        printf("LayerSize: %d\n", layerSize);
    }
    if(idx == 0){
        printf("cudaTree:   %d, %d\n", tree[0], tree[treeSize/sizeof(int) - 1]);
        printf("cudaLayer:  %d, %d\n", layer[0], layer[layerSize/sizeof(int) - 1]);
        printf("max:        %d, %d, %d, %d, %d\n", kd_tree_para[0], kd_tree_para[1], kd_tree_para[2], kd_tree_para[3], kd_tree_para[4]);
        // printf("%f\n", P[3 * nP]);
    }
    /* Each thread handle a part of the points in P */

    for(int i = 0; i < pointsPerThread; i++){
        /* Handle exceptions */
        int pIdx = idx * pointsPerThread + i; // The location in P
        if(pIdx < nP){
            /* For each point in Q */
            float minDist = FLT_MAX; // Change this later
            int minIndex = -1;
            int pValIdx = pIdx*3;
            for(int j = 0; j < nQ; j++){
                int qValIdx = j*3;
                int dx = P[pValIdx] - Q[qValIdx];
                int dy = P[pValIdx + 1] - Q[qValIdx + 1];
                int dz = P[pValIdx + 2] - Q[qValIdx + 2];
                int dist = dx*dx + dy*dy + dz*dz;
                dist = sqrtf(dist);
                /* Update the nearest point */
                if(dist < minDist){
                    minDist = dist;
                    minIndex = j;
                }
            }
            min_index_device[pIdx] = minIndex;
        }
    }

    /* Copy the data to Qselect */
    for(int i = 0; i < pointsPerThread; i++){
        int pIdx = idx * pointsPerThread + i;
        if(pIdx < nP){
            int qIdx = min_index_device[pIdx];
            int qValIdx = qIdx * 3;
            Q_select_device[pIdx*3] = Q[qValIdx];
            Q_select_device[pIdx*3+1] = Q[qValIdx+1];
            Q_select_device[pIdx*3+2] = Q[qValIdx+2];
        }
    }
} 

__device__ int cuGetTree(int *tree, int index){
    int result = (tree[(index + 1) / 32] >> ((index + 1) % 32)) & 0x1;
    return result;
}

__device__ void cuSetTree(int *tree, int index, int value){
    tree[(index + 1) / 32] += value << ((index + 1) % 32);
}

__device__ void cuPositionToCoordinate(int* layer, int position, int sumMax, int *coordinate) {
    int currIndex = 0;
    while (currIndex < sumMax) {
        coordinate[layer[currIndex]] <<= 1;
        coordinate[layer[currIndex]] += (position >> (sumMax - currIndex - 1)) & 0x1;
        currIndex++;
    }
}  

__device__ int cuPositionInTree(int x, int y, int z, int *kd_tree_para) {
    int position = 0;
    int xLeft = kd_tree_para[0];
    int yLeft = kd_tree_para[1];
    int zLeft = kd_tree_para[2];
    while (xLeft + yLeft + zLeft > 0) {
        if (xLeft != 0) {
            position = position << 1;
            position += (x >> (xLeft - 1)) & 0x1;
            xLeft--;
        }
        if (yLeft != 0) {
            position = position << 1;
            position += (y >> (yLeft - 1)) & 0x1;
            yLeft--;
        }
        if (zLeft != 0) {
            position = position << 1;
            position += (z >> (zLeft - 1)) & 0x1;
            zLeft--;
        }
    }
    return position;
}

__device__ float cuMinDistancePointToBox(float* point, int* box) {
    // printf("*\n");
    int dx = (point[0] < box[0]) ? box[0] - point[0] : (box[1] < point[0] ? box[1] - point[0] : 0);
    int dy = (point[1] < box[2]) ? box[2] - point[1] : (box[3] < point[1] ? box[3] - point[1] : 0);
    int dz = (point[2] < box[4]) ? box[4] - point[2] : (box[5] < point[2] ? box[5] - point[2] : 0);
    float result = sqrtf(powf(dx, 2.0) + powf(dy, 2.0) + powf(dz, 2.0));
    return result;
}

__device__ float cuFindNearestNode(int* tree, int* layer, int* kd_tree_para, float x, float y, float z, int* returnValue) {
    int maxX = kd_tree_para[0];
    int maxY = kd_tree_para[1];
    int maxZ = kd_tree_para[2];
    int left[3];
    left[0] = maxX;
    left[1] = maxY;
    left[2] = maxZ;
    int hyperPlan[3];
    hyperPlan[0] = 0;
    hyperPlan[1] = 0;
    hyperPlan[2] = 0;
    float target[3];
    target[0] = x;
    target[1] = y;
    target[2] = z;
    int boundaryBox[6];
    boundaryBox[0] = 0; boundaryBox[1] = (int)powf(2.0, maxX);
    boundaryBox[2] = 0; boundaryBox[3] = (int)powf(2.0, maxY);
    boundaryBox[4] = 0; boundaryBox[5] = (int)powf(2.0, maxZ);
    return cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, -1, 0, FLT_MAX, returnValue);
}

__device__ float cuKdTreeSearch(int* tree, int* layer, int* kd_tree_para, float* target, int* left, int* boundaryBox, int* hyperPlan, int currLayer, int node, float minDistance, int* nearestNode) {
    if (node >= kd_tree_para[4] - 1) {
        
        int coordinate[3];
        cuPositionToCoordinate(layer, node - kd_tree_para[4] + 1, kd_tree_para[3], coordinate);
        float distance = sqrtf(powf(coordinate[0] - target[0], 2.0) + 
                               powf(coordinate[1] - target[1], 2.0) +
                               powf(coordinate[2] - target[2], 2.0));
        if (distance < minDistance) {
            minDistance = distance;
            nearestNode[0] = coordinate[0];
            nearestNode[1] = coordinate[1];
            nearestNode[2] = coordinate[2];
        }
        return minDistance;
    } else {
        currLayer = (currLayer + 1) % 3;
        while (left[currLayer % 3] == 0) currLayer = (currLayer + 1) % 3;
        left[currLayer]--;
        
        if (cuGetTree(tree, node * 2 + 1) == 0) {
            int prevBoundaryMin = boundaryBox[currLayer * 2];
            boundaryBox[currLayer * 2] = hyperPlan[currLayer] + (1 << left[currLayer]);
            if (cuMinDistancePointToBox(target, boundaryBox) <= minDistance) {
                hyperPlan[currLayer] += 1 << left[currLayer];
                minDistance = cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 2, minDistance, nearestNode);
                hyperPlan[currLayer] -= 1 << left[currLayer];
            }
            boundaryBox[currLayer * 2] = prevBoundaryMin;
            left[currLayer]++;
            return minDistance;
        } else if (cuGetTree(tree, node * 2 + 2) == 0) {
            int prevBoundaryMax = boundaryBox[currLayer * 2 + 1];
            boundaryBox[currLayer * 2 + 1] = hyperPlan[currLayer] + (1 << left[currLayer]);
            if (cuMinDistancePointToBox(target, boundaryBox) <= minDistance) {
                minDistance = cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 1, minDistance, nearestNode);
            }
            boundaryBox[currLayer * 2 + 1] = prevBoundaryMax;
            left[currLayer]++;
            return minDistance;
        }
        int prevBoundaryMin = boundaryBox[currLayer * 2];
        int prevBoundaryMax = boundaryBox[currLayer * 2 + 1];

        if (target[currLayer] < hyperPlan[currLayer] + (1 << left[currLayer])) {


            boundaryBox[currLayer * 2 + 1] = hyperPlan[currLayer] + (1 << left[currLayer]);
            if (cuMinDistancePointToBox(target, boundaryBox) <= minDistance) {
                minDistance = cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 1, minDistance, nearestNode);
            }
            boundaryBox[currLayer * 2 + 1] = prevBoundaryMax;
            boundaryBox[currLayer * 2] = hyperPlan[currLayer] + (1 << left[currLayer]);
            if (cuMinDistancePointToBox(target, boundaryBox) <= minDistance) {
                hyperPlan[currLayer] += 1 << left[currLayer];
                minDistance = cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 2, minDistance, nearestNode);
                hyperPlan[currLayer] -= 1 << left[currLayer];
            }
            boundaryBox[currLayer * 2] = prevBoundaryMin;
            left[currLayer]++;
            return minDistance;
        } else {
            boundaryBox[currLayer * 2] = hyperPlan[currLayer] + (1 << left[currLayer]);
            if (cuMinDistancePointToBox(target, boundaryBox) <= minDistance) {
                hyperPlan[currLayer] += 1 << left[currLayer];
                minDistance = cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 2, minDistance, nearestNode);
                hyperPlan[currLayer] -= 1 << left[currLayer];
            }
            boundaryBox[currLayer * 2] = prevBoundaryMin;
            boundaryBox[currLayer * 2 + 1] = hyperPlan[currLayer] + (1 << left[currLayer]);
            if (cuMinDistancePointToBox(target, boundaryBox) <= minDistance) {
                minDistance = cuKdTreeSearch(tree, layer, kd_tree_para, target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 1, minDistance, nearestNode);
            }
            boundaryBox[currLayer * 2 + 1] = prevBoundaryMax;
            left[currLayer]++;
            return minDistance;
        }

    }
}



void outputPointCloud(std::string name, MatrixXf & matrix){
    ofstream OFile(name);
    for(int i = 0; i < matrix.cols(); i++){
        for(int j=0; j<matrix.rows(); j++){
            OFile << matrix(j,i) << " ";
        }
        OFile << std::endl;
    }
    OFile.close();
}



KdTree* plyParser(FILE *plyFile, int numOfVertexes, std::vector<float> &vertexes, int& nP) {
    int scaleX = 300;
    int scaleY = 300;
    int scaleZ = 300;

    FILE *costOutput = fopen("./trunc.txt", "w");
    float* temp = (float *)malloc(sizeof(int) * 3 * numOfVertexes);


    float minX, maxX;
    float minY, maxY;
    float minZ, maxZ;
    fscanf(plyFile, "%f %f %f \n", &minX, &minY, &minZ);
    maxX = minX;
    maxY = minY;
    maxZ = minZ;
    for (int i = 0; i < numOfVertexes - 1; i++) {
        float x, y, z;
        fscanf(plyFile, "%f %f %f \n", &x, &y, &z);

        minX = minX < x ? minX : x;
        minY = minY < y ? minY : y;
        minZ = minZ < z ? minZ : z;
        maxX = maxX > x ? maxX : x;
        maxY = maxY > y ? maxY : y;
        maxZ = maxZ > z ? maxZ : z;

        temp[3 * i + 0] = x;
        temp[3 * i + 1] = y;
        temp[3 * i + 2] = z;
    }

    float deltaX = maxX - minX;
    float deltaY = maxY - minY;
    float deltaZ = maxZ - minZ;

    int sizeX = (int)ceil(deltaX * scaleX) + 1;
    int sizeY = (int)ceil(deltaY * scaleY) + 1;
    int sizeZ = (int)ceil(deltaZ * scaleZ) + 1;
    int numOfGrids = 0;

    int* grids = (int *)malloc(sizeof(int) * (sizeX) * (sizeY) * (sizeZ));
    for (int i = 0; i < numOfVertexes; i++) {
        int x, y, z;
        x = (int)ceil((temp[3 * i + 0] - minX) * scaleX);
        y = (int)ceil((temp[3 * i + 1] - minY) * scaleY);
        z = (int)ceil((temp[3 * i + 2] - minZ) * scaleZ);

        if (grids[x * sizeY * sizeZ + y * sizeZ + z] == 0) {
            grids[x * sizeY * sizeZ + y * sizeZ + z] = 1;
            numOfGrids++;
        }
    }
    free(temp);

    vertexes.resize(3 * numOfGrids);

    int currGrid = 0;

    for (int x = 0; x < sizeX; x++) {
        for (int y = 0; y < sizeY; y++) {
            for (int z = 0; z < sizeZ; z++) {
                if (grids[x * sizeY * sizeZ + y * sizeZ + z] == 1) {
                    vertexes[currGrid * 3 + 0] = (float)x;
                    vertexes[currGrid * 3 + 1] = (float)y;
                    vertexes[currGrid * 3 + 2] = (float)z;
                    currGrid++;
                    fprintf(costOutput, "%f %f %f \n", (float)x,(float)y,(float)z);
                }
            }
        }
    }
    KdTree* vertexTree = new KdTree(grids, sizeX, sizeY, sizeZ);
    cout << "vertexTree pointer: " << (vertexTree == NULL) << std::endl;
    free(grids);
    fclose(costOutput);
    nP = numOfGrids;
    return vertexTree;
}



