#ifndef CONNECT_H
#define CONNECT_H

#include <curand_kernel.h>
#include "../MACRO.h"
#include "../types.h"

struct initialize_package {
    Float* radius; //[NTYPE][2];
    Float* neuron_type_acc_count; //[NTYPE+1];
	Float* den_axn; //[NTYPE];
	Float* den_den; //[NTYPE];
    __host__ 
    __device__ 
    initialize_package() {};
    __host__ 
    __device__ 
    initialize_package(Float _radius[][2], Float _neuron_type_acc_count[], Float _den_axn[], Float _den_den[]) {
        for (Size i=0; i<NTYPE; i++) {
            radius[i][0] = _radius[i][0];
            radius[i][1] = _radius[i][1];
            den_axn[i] = _den_axn[i];
            den_den[i] = _den_den[i];
        }
        // NTYPE
        for (Size i=0; i<NTYPE+1; i++) {
            neuron_type_acc_count[i] = _neuron_type_acc_count[i];
        }
	}
};

__global__ 
__launch_bounds__(blockSize, 1)
void initialize(curandStateMRG32k3a* __restrict__ state,
                           Size* __restrict__ preType,
                           Float* __restrict__ rden,
                           Float* __restrict__ raxn,
                           Float* __restrict__ dden,
                           Float* __restrict__ daxn,
                           Float*  __restrict__ sTypeMat,
                           Float*  __restrict__ pTypeMat,
                           Size* __restrict__ nTypeMat,
                           Float*  __restrict__ preTypeS,
                           Float*  __restrict__ preTypeP,
                           Size* __restrict__ preTypeN,
                           initialize_package init_pack, unsigned long long seed, Size networkSize);

__global__ 
__launch_bounds__(blockSize, 1)
void cal_blockPos(Float* __restrict__ pos,
                             Float* __restrict__ block_x,
                             Float* __restrict__ block_y,
                             Size networkSize);

__global__ 
__launch_bounds__(blockSize, 1)
void get_neighbor_blockId(Float* __restrict__ block_x,
                                     Float* __restrict__ block_y,
                                     Size* __restrict__ neighborBlockId,
                                     Size* __restrict__ nNeighborBlock,
                                     Float max_radius, Size nPotentialNeighbor);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_connections(Float* __restrict__ pos,
						  Float* __restrict__ preTypeS,
						  Float* __restrict__ preTypeP,
						  Size* __restrict__ preTypeN,
                          Size* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
                          Float* __restrict__ rden,
                          Float* __restrict__ raxn,
                          Float* __restrict__ conMat, //within block connections
                          Float* __restrict__ delayMat,
                          Float* __restrict__ conVec, //for neighbor block connections
                          Float* __restrict__ delayVec, //for neighbor block connections
                          Size* __restrict__ vecID,
                          Size* __restrict__ nVec,
                          Size* __restrict__ preTypeConnected,
                          Size* __restrict__ preTypeAvail,
                          Float* __restrict__ preTypeStrSum,
                          Size* __restrict__ preType,
                          Float* __restrict__ dden,
                          Float* __restrict__ daxn,
                          curandStateMRG32k3a* __restrict__ state,
                          Size networkSize, Size neighborSize, Size nPotentialNeighbor, Float speedOfThought);

#endif
