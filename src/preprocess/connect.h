#ifndef CONNECT_H
#define CONNECT_H

#include <curand_kernel.h>
#include "../MACRO.h"

struct initialize_package {
    _float radius[NTYPE][2];
    _float neuron_type_acc_count[NTYPE+1];
	_float den_axn[NTYPE];
	_float den_den[NTYPE];
    __host__ __device__ initialize_package() {};
    __host__ __device__ initialize_package(_float _radius[][2], _float _neuron_type_acc_count[], _float _den_axn[], _float _den_den[]) {
        for (unsigned int i=0; i<NTYPE; i++) {
            radius[i][0] = _radius[i][0];
            radius[i][1] = _radius[i][1];
            den_axn[i] = _den_axn[i];
            den_den[i] = _den_den[i];
        }
        // NTYPE
        for (unsigned int i=0; i<NTYPE+1; i++) {
            neuron_type_acc_count[i] = _neuron_type_acc_count[i];
        }
	}
};

__global__ 
__launch_bounds__(blockSize, 1)
void initialize(curandStateMRG32k3a* __restrict__ state,
                           unsigned int* __restrict__ preType,
                           _float* __restrict__ rden,
                           _float* __restrict__ raxn,
                           _float* __restrict__ dden,
                           _float* __restrict__ daxn,
                           _float*  __restrict__ sTypeMat,
                           _float*  __restrict__ pTypeMat,
                           unsigned int* __restrict__ nTypeMat,
                           _float*  __restrict__ preTypeS,
                           _float*  __restrict__ preTypeP,
                           unsigned int* __restrict__ preTypeN,
                           initialize_package init_pack, unsigned long long seed, unsigned int networkSize);

__global__ 
__launch_bounds__(blockSize, 1)
void cal_blockPos(_float* __restrict__ pos,
                             _float* __restrict__ block_x,
                             _float* __restrict__ block_y,
                             unsigned int networkSize);

__global__ 
__launch_bounds__(blockSize, 1)
void get_neighbor_blockId(_float* __restrict__ block_x,
                                     _float* __restrict__ block_y,
                                     unsigned int* __restrict__ neighborBlockId,
                                     unsigned int* __restrict__ nNeighborBlock,
                                     _float max_radius, unsigned int nPotentialNeighbor);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_connections(_float* __restrict__ pos,
						  _float* __restrict__ preTypeS,
						  _float* __restrict__ preTypeP,
						  unsigned int* __restrict__ preTypeN,
                          unsigned int* __restrict__ neighborBlockId,
                          unsigned int* __restrict__ nNeighborBlock,
                          _float* __restrict__ rden,
                          _float* __restrict__ raxn,
                          _float* __restrict__ conMat, //within block connections
                          _float* __restrict__ delayMat,
                          _float* __restrict__ conVec, //for neighbor block connections
                          _float* __restrict__ delayVec, //for neighbor block connections
                          unsigned int* __restrict__ vecID,
                          unsigned int* __restrict__ nVec,
                          unsigned int* __restrict__ preTypeConnected,
                          unsigned int* __restrict__ preTypeAvail,
                          _float* __restrict__ preTypeStrSum,
                          unsigned int* __restrict__ preType,
                          _float* __restrict__ dden,
                          _float* __restrict__ daxn,
                          curandStateMRG32k3a* __restrict__ state,
                          unsigned int networkSize, unsigned int neighborSize, unsigned int nPotentialNeighbor, _float speedOfThought);

#endif
