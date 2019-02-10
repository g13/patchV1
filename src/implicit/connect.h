#ifndef CONNECT_H
#define CONNECT_H

#include "MACRO.h"

struct initialize_package {
    _float radius[NTYPE][2];
    _float neuron_type_acc_count[NTYPE+1];
	unsigned int den_axn[NTYPE];
    __host__ __device__ initialize_package() {
        // E
        radius[0][0] = 80;
        radius[0][1] = 150;
        // I
        radius[1][0] = 50;
        radius[1][1] = 100;

        // NTYPE
        neuron_type_acc_count[0] = 0;
        neuron_type_acc_count[1] = 768;
        neuron_type_acc_count[2] = 1024;
        assert(neuron_type_acc_count[NTYPE] == blockSize);

        den_axn[0] = 1;
        den_axn[1] = 2;
	}
};

__global__ void init(unsigned int* __restricted__ preType,
                     _float* __restricted__ rden,
                     _float* __restricted__ raxn,
                     _float* __restricted__ preTypeDaxn,
                     unsigned int* __restricted__ preTypeN,
                     initialize_package init_pack, unsigned long long seed);

__global__ void cal_blockPos(_float* __restricted__ pos,
                             _float* __restricted__ block_x,
                             _float* __restricted__ block_y);

__global__ void get_neighbor_blockId(_float* __restricted__ block_x,
                                     _float* __restricted__ block_y,
                                     _float* __restricted__ neighborBlockId,
                                     unsigned int* __restricted__ nNeighborBlock,
                                     _float max_radius,
                                     unsigned int nPotentialNeigbor);

__global__ void generate_connections(_float* __restricted__ pos,
                                     _float* __restricted__ neighborBlockId,
                                     _float* __restricted__ nNeighborBlock,
                                     _float* __restricted__ rden,
                                     _float* __restricted__ raxn,
                                     _float* __restricted__ conMat, //within block connections
                                     _float* __restricted__ delayMat,
                                     _float* __restricted__ conVec, //for neighbor block connections
                                     _float* __restricted__ delayVec, //for neighbor block connections
                                     unsigned int* __restricted__ preTypeConnected,
                                     unsigned int* __restricted__ preTypeAvail,
                                     unsigned int* __restricted__ preTypeN,
                                     _float* __restricted__ preTypeStrSum,
                                     _float* __restricted__ preTypeStr,
                                     unsigned int* __restricted__ preType,
                                     unsigned int* __restricted__ preTypeDaxn,
                                     curandStateMRG32k3a* __restricted__ state,
                                     unsigned int networkSize,
                                     unsigned int nNeighborMax,
                                     _float speedOfThought);

#endif
