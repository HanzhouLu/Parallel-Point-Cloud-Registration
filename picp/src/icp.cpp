/**
 *  CMU 15418 Fall 2016 PICP project
 *  Author: Hanzhou Lu, Yujie Wei
 *  Date: Dec 14, 2016
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
#include <Eigen/Dense>
#include <omp.h>
#include "kdtree.cpp"

typedef std::chrono::high_resolution_clock Clock;
#define PI 3.14159265
#define BUN000_FILE_SIZE 40256
#define TOP3_FILE_SEZE 36023

/* Accept rate of bucket sort, should in range(0, 1) */
#define ACCEPT_RATE 0.95

/* Number of itrations */
#define ITERATIONS 200

#ifndef OPENMP
    #define OPENMP
#endif

using Eigen::MatrixXf;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::VectorXf;
using std::ofstream;
using std::cout;
using std::endl;

void outputPointCloud(std::string name, MatrixXf & matrix);
KdTree* plyParser(FILE *plyFile, int numOfVertexes, std::vector<float> &vertexes, int& nP);
void plyReader(FILE *plyFile, int numOfVertexes, std::vector<float> &vertexes);

int main(){
    /* Read from file */
    int dimension = 3;
    int nIter = ITERATIONS;

    /* Start from origianal cloud P and ref cloud Q */
    std::vector<float> vertexes = std::vector<float>(0);
    std::vector<float> vertexes_back = std::vector<float>(0);
    int nP, nQ;
    FILE* input0 = fopen("data/bun000_vertex.txt", "r");
    KdTree* vertexTree = plyParser(input0, BUN000_FILE_SIZE, vertexes, nP);
    FILE* input1 = fopen("data/top3_vertex.txt", "r");
    plyReader(input1, TOP3_FILE_SEZE, vertexes_back);
    nQ = TOP3_FILE_SEZE; 
    cout << nP << endl;
    cout << nQ << endl;
    fclose(input0);

    if(nP == 0){
        std::cout << "Point cloud error.\n" << std::endl;
        return 0;
    }
    nP = nQ;
    MatrixXf P(dimension, nP);
    MatrixXf Q(dimension, nQ);

    /* Construct the Q from the point cloud file, Q is a 3 * nQ matrix. */
    #ifdef OPENMP
    omp_set_num_threads(8);
    #pragma omp parallel for collapse(2)
    #endif
    for(int i=0; i<nQ; i++){
        for(int j=0; j<3; j++){
            int index = i * 3 + j;
            Q(j, i) = vertexes_back[index];
        }
    }

    /* Initialize transformation */
    Matrix3f R = Matrix3f::Identity();
    Vector3f T = Vector3f::Zero();
    P = Q;

    /* Find match */
    VectorXf match = VectorXf::Zero(nP);
    VectorXf minDist = VectorXf::Zero(nP);
    std::cout << "searching for match...\n";
    VectorXf MSE = VectorXf::Zero(nIter);

    float acceptRate = ACCEPT_RATE;
    int acceptSize = (int)(nP * acceptRate);

    MatrixXf QShrink(dimension, acceptSize);
    MatrixXf PShrink(dimension, acceptSize);

    /* Start optimizing */
    for(int iter = 0; iter < nIter; iter ++){
        auto t1 = Clock::now();
        std::cout << "\nIter # " << iter << std::endl; 
        std::array<std::atomic<int>, 200> bucket = {};

        /* KD Tree search */
        MatrixXf QSelect = MatrixXf(dimension, nP);
        float maxDist = 0.0;

        #ifdef OPENMP
        #pragma omp parallel for
        #endif  
        for(int i = 0; i < nP; i++){
            int nearestPoint[3];
            float dist = (float)vertexTree->findNearestNode(P(0, i), P(1, i), P(2, i), nearestPoint);
            if (dist > maxDist) maxDist = dist;
            std::atomic_fetch_add(&bucket[(int)dist], 1);
            minDist(i) = dist;
            QSelect(0, i) = nearestPoint[0];
            QSelect(1, i) = nearestPoint[1];
            QSelect(2, i) = nearestPoint[2];
        }
        printf("maxDist = %f\n", maxDist);

        /* Find how many buckets should be accepted */
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
        VectorXf QBar = QShrink * weights;
        MatrixXf QMark = QShrink - QBar.replicate(1, QShrink.cols());

        QMark = QMark.cwiseProduct(weights.transpose().replicate(dimension, 1));

        VectorXf PBar = PShrink * weights;
        MatrixXf PMark = PShrink - PBar.replicate(1, PShrink.cols());
        MatrixXf N = PMark * QMark.transpose();
        
        /* SVD decomposition */
        Eigen::JacobiSVD<Matrix3f> svd( N, Eigen::ComputeFullV | Eigen::ComputeFullU );
        Matrix3f U = svd.matrixU();
        Matrix3f V = svd.matrixV();

        float det = (U * V.transpose()).determinant();
        Vector3f diagVec(1.0, 1.0, det);
        R = V * diagVec.asDiagonal() * U.transpose();
        T = QBar - R * PBar;

        P = R * P + T.replicate(1, nP);
        auto t2 = Clock::now();
        std::cout << "Delta t2-t1: " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
              << " nanoseconds" << std::endl;
    }
    free(vertexTree);
    outputPointCloud("result_top.ply", P);

}

void outputPointCloud(std::string name, MatrixXf & matrix){
    ofstream OFile(name);
    OFile << "ply\n\
format ascii 1.0\n\
obj_info is_cyberware_data 1\n\
obj_info is_mesh 0\n\
obj_info is_warped 0\n\
obj_info is_interlaced 1\n\
obj_info num_cols 512\n\
obj_info num_rows 400\n\
obj_info echo_rgb_offset_x 0.013000\n\
obj_info echo_rgb_offset_y 0.153600\n\
obj_info echo_rgb_offset_z 0.172000\n\
obj_info echo_rgb_frontfocus 0.930000\n\
obj_info echo_rgb_backfocus 0.012660\n\
obj_info echo_rgb_pixelsize 0.000010\n\
obj_info echo_rgb_centerpixel 232\n\
obj_info echo_frames 512\n\
obj_info echo_lgincr 0.000500\n\
element vertex 36023\n\
property float x\n\
property float y\n\
property float z\n\
element range_grid 204800\n\
property list uchar int vertex_indices\nend_header\n";

    for(int i = 0; i < matrix.cols(); i++){
        for(int j=0; j<matrix.rows(); j++){
            OFile << matrix(j,i) << " ";
        }
        OFile << std::endl;
    }
    OFile.close();
}

void plyReader(FILE *plyFile, int numOfVertexes, std::vector<float> &vertexes) {
    vertexes.resize(3 * numOfVertexes);
    for (int i = 0; i < numOfVertexes; i++) {
        float x, y, z;
        fscanf(plyFile, "%f %f %f \n", &x, &y, &z);
        vertexes[i * 3 + 0] = -1000 * y;
        vertexes[i * 3 + 1] = 1000 * x;
        vertexes[i * 3 + 2] = 1000 * z;

    }
    fclose(plyFile);
}

KdTree* plyParser(FILE *plyFile, int numOfVertexes, std::vector<float> &vertexes, int& nP) {
    int scaleX = 1000;
    int scaleY = 1000;
    int scaleZ = 1000;


    // string plyHeader = ;

    FILE *costOutput = fopen("./result_front.ply", "w");
    fprintf(costOutput, "ply\n\
format ascii 1.0\n\
obj_info is_cyberware_data 1\n\
obj_info is_mesh 0\n\
obj_info is_warped 0\n\
obj_info is_interlaced 1\n\
obj_info num_cols 512\n\
obj_info num_rows 400\n\
obj_info echo_rgb_offset_x 0.013000\n\
obj_info echo_rgb_offset_y 0.153600\n\
obj_info echo_rgb_offset_z 0.172000\n\
obj_info echo_rgb_frontfocus 0.930000\n\
obj_info echo_rgb_backfocus 0.012660\n\
obj_info echo_rgb_pixelsize 0.000010\n\
obj_info echo_rgb_centerpixel 232\n\
obj_info echo_frames 512\n\
obj_info echo_lgincr 0.000500\n\
element vertex 21551\n\
property float x\n\
property float y\n\
property float z\n\
element range_grid 204800\n\
property list uchar int vertex_indices\nend_header\n");
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
    printf("k-d treesize = [%d %d %d]\n",sizeX, sizeY, sizeZ );
    cout << "vertexTree pointer: " << (vertexTree == NULL) << std::endl;
    free(grids);
    fclose(costOutput);
    nP = numOfGrids;
    return vertexTree;
}