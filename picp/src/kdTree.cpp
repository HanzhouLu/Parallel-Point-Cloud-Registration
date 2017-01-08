#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string>
#include <iostream>
#include <fstream>

using std::ofstream;

class KdTree {
    public:
        int* tree;
        int* layer;
        int maxX;
        int maxY;
        int maxZ;
        int sumMax;
        int branches;
        void buildLayer();
        double kdTreeSearch(double*, int*, int*, int*, int, int, double, int*);
        double minDistancePointToBox(double*, int*);
        int getTree(int);
        void setTree(int, int);

    public:
        KdTree(int*, int, int, int);

        double findNearestNode(double, double, double, int*);

        int positionInTree(int, int, int);

        int* positionToCoordinate(int);
 };

/* Get single bit in k-d tree */
int KdTree::getTree(int index) {
    return (tree[(index + 1) / 32] >> ((index + 1) % 32)) & 0x1;
}

/* Set single bit in k-d tree */
/* require: value = 0 or 1 */
void KdTree::setTree(int index, int value) {
    tree[(index + 1) / 32] += value << ((index + 1) % 32);
}

/* Build layer according to x, y, z axis length */
void KdTree::buildLayer() {
    layer = (int *)malloc(sizeof(int) * sumMax);
    int currIndex = 0;
    int xLeft = maxX;
    int yLeft = maxY;
    int zLeft = maxZ;
    while (xLeft + yLeft + zLeft > 0) {
        if (xLeft != 0) {
            layer[currIndex] = 0;
            currIndex++;
            xLeft--;
        }
        if (yLeft != 0) {
            layer[currIndex] = 1;
            currIndex++;
            yLeft--;
        }
        if (zLeft != 0) {
            layer[currIndex] = 2;
            currIndex++;
            zLeft--;
        }
    }

}

/* Turn index in k-d tree into (x, y, z) coordinate */
int* KdTree::positionToCoordinate(int position) {
    int* coordinate = (int *)calloc(sizeof(int), 3);
    int currIndex = 0;
    while (currIndex < sumMax) {
        coordinate[layer[currIndex]] <<= 1;
        coordinate[layer[currIndex]] += (position >> (sumMax - currIndex - 1)) & 0x1;
        currIndex++;
    }
    return coordinate;
}   

/* Turn (x, y, z) coordinate into position in k-d tree */
int KdTree::positionInTree(int x, int y, int z) {
    int position = 0;
    int xLeft = maxX;
    int yLeft = maxY;
    int zLeft = maxZ;
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

/* Calculate the min distance from one point to a boundary box */
double KdTree::minDistancePointToBox(double* point, int* box) {
    int dx = (point[0] < box[0]) ? box[0] - point[0] : (box[1] < point[0] ? box[1] - point[0] : 0);
    int dy = (point[1] < box[2]) ? box[2] - point[1] : (box[3] < point[1] ? box[3] - point[1] : 0);
    int dz = (point[2] < box[4]) ? box[4] - point[2] : (box[5] < point[2] ? box[5] - point[2] : 0);
    return sqrt(pow(dx, 2.0) + pow(dy, 2.0) + pow(dz, 2.0));
}

/* K-d tree constructor */
KdTree::KdTree(int* grid, int sizeX, int sizeY, int sizeZ) {
    maxX = (int)ceil(log(sizeX) / log(2));
    maxY = (int)ceil(log(sizeY) / log(2));
    maxZ = (int)ceil(log(sizeZ) / log(2));
    sumMax = maxX + maxY + maxZ;
    tree = (int *)malloc(sizeof(int) * ((pow(2, maxX) * pow(2, maxY) * pow(2, maxZ) * 2) / 32 + 1));
    branches = pow(2, maxX) * pow(2, maxY) * pow(2, maxZ);
    buildLayer();
    for (int x = 0; x < sizeX; x++) {
        for (int y = 0; y < sizeY; y++) {
            for (int z = 0; z < sizeZ; z++) {
                if (grid[x * sizeY * sizeZ + y * sizeZ + z] == 1) {
                    setTree(positionInTree(x, y, z) + branches - 1, 1);
                }
            }
        }
    }
    for (int i = branches - 1; i > 0; i--) {
        setTree(i - 1, getTree(i * 2 - 1) | getTree(i * 2));
    }
}

/* Find the nearest node from (x, y, z) in k-d tree */
double KdTree::findNearestNode(double x, double y, double z, int* returnValue) {
    int left[3];
    left[0] = maxX;
    left[1] = maxY;
    left[2] = maxZ;
    int hyperPlan[3];
    hyperPlan[0] = 0;
    hyperPlan[1] = 0;
    hyperPlan[2] = 0;
    double target[3];
    target[0] = x;
    target[1] = y;
    target[2] = z;
    int boundaryBox[6];
    boundaryBox[0] = 0; boundaryBox[1] = (int)pow(2.0, maxX);
    boundaryBox[2] = 0; boundaryBox[3] = (int)pow(2.0, maxY);
    boundaryBox[4] = 0; boundaryBox[5] = (int)pow(2.0, maxZ);
    return kdTreeSearch(target, left, boundaryBox, hyperPlan, -1, 0, DBL_MAX, returnValue);
}

/* Recursively find the nearest point */
double KdTree::kdTreeSearch(double* target, int* left, int* boundaryBox, int* hyperPlan, int currLayer, int node, double minDistance, int* nearestNode) {
    /* Reach a leaf node (base case) */
    if (node >= branches - 1) {
        int* coordinate;
        coordinate = positionToCoordinate(node - branches + 1);
        double distance = sqrt(pow(coordinate[0] - target[0], 2.0) + 
                               pow(coordinate[1] - target[1], 2.0) +
                               pow(coordinate[2] - target[2], 2.0));    
        if (distance < minDistance) {
            minDistance = distance;
            nearestNode[0] = coordinate[0];
            nearestNode[1] = coordinate[1];
            nearestNode[2] = coordinate[2];
        }
        return minDistance;
    /* Reach a common node */
    } else {
        /* Move current layer to next level */
        currLayer = (currLayer + 1) % 3;
        while (left[currLayer % 3] == 0) currLayer = (currLayer + 1) % 3;
        left[currLayer]--;

        /* Left child is 0, go to right branch */
        if (getTree(node * 2 + 1) == 0) {
            int prevBoundaryMin = boundaryBox[currLayer * 2];
            boundaryBox[currLayer * 2] = hyperPlan[currLayer] + (1 << left[currLayer]);
            /* Check if there are candidates in right branch */
            if (minDistancePointToBox(target, boundaryBox) <= minDistance) {
                hyperPlan[currLayer] += 1 << left[currLayer];
                minDistance = kdTreeSearch(target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 2, minDistance, nearestNode);
                hyperPlan[currLayer] -= 1 << left[currLayer];
            }
            boundaryBox[currLayer * 2] = prevBoundaryMin;
            left[currLayer]++;
            return minDistance;
        /* Right child is 0, go to left branch */    
        } else if (getTree(node * 2 + 2) == 0) {
            int prevBoundaryMax = boundaryBox[currLayer * 2 + 1];
            boundaryBox[currLayer * 2 + 1] = hyperPlan[currLayer] + (1 << left[currLayer]);
            /* Check if there are candidates in left branch */
            if (minDistancePointToBox(target, boundaryBox) <= minDistance) {
                minDistance = kdTreeSearch(target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 1, minDistance, nearestNode);
            }
            boundaryBox[currLayer * 2 + 1] = prevBoundaryMax;
            left[currLayer]++;
            return minDistance;
        }
        int prevBoundaryMin = boundaryBox[currLayer * 2];
        int prevBoundaryMax = boundaryBox[currLayer * 2 + 1];
        /* Target is more likely in left branch */
        if (target[currLayer] < hyperPlan[currLayer] + (1 << left[currLayer])) {
            boundaryBox[currLayer * 2 + 1] = hyperPlan[currLayer] + (1 << left[currLayer]);
            /* Check left branch first */
            if (minDistancePointToBox(target, boundaryBox) <= minDistance) {
                minDistance = kdTreeSearch(target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 1, minDistance, nearestNode);
            }
            boundaryBox[currLayer * 2 + 1] = prevBoundaryMax;
            boundaryBox[currLayer * 2] = hyperPlan[currLayer] + (1 << left[currLayer]);
            /* Then check right branch */
            if (minDistancePointToBox(target, boundaryBox) <= minDistance) {
                hyperPlan[currLayer] += 1 << left[currLayer];
                minDistance = kdTreeSearch(target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 2, minDistance, nearestNode);
                hyperPlan[currLayer] -= 1 << left[currLayer];
            }
            boundaryBox[currLayer * 2] = prevBoundaryMin;
            left[currLayer]++;
            return minDistance;
        /* Target is more likely in right branch */
        } else {
            boundaryBox[currLayer * 2] = hyperPlan[currLayer] + (1 << left[currLayer]);
            /* Check right branch first */
            if (minDistancePointToBox(target, boundaryBox) <= minDistance) {
                hyperPlan[currLayer] += 1 << left[currLayer];
                minDistance = kdTreeSearch(target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 2, minDistance, nearestNode);
                hyperPlan[currLayer] -= 1 << left[currLayer];
            }
            boundaryBox[currLayer * 2] = prevBoundaryMin;
            boundaryBox[currLayer * 2 + 1] = hyperPlan[currLayer] + (1 << left[currLayer]);
            /* Then check right branch */
            if (minDistancePointToBox(target, boundaryBox) <= minDistance) {
                minDistance = kdTreeSearch(target, left, boundaryBox, hyperPlan, currLayer, node * 2 + 1, minDistance, nearestNode);
            }
            boundaryBox[currLayer * 2 + 1] = prevBoundaryMax;
            left[currLayer]++;
            return minDistance;
        }
    }
}