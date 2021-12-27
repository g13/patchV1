#ifndef CONNECT_H
#define CONNECT_H

#include <vector>
#include <cuda_runtime.h>
#include <helper_functions.h> // include cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <cassert>
#include <iostream>
using std::cout;
#include "../MACRO.h"
#include "../types.h"
#include "../util/cuda_util.cuh"

struct hInitialize_package {
	char* mem_block;
	Size* nTypeHierarchy; // [nArchtype]
    Size* typeAccCount; //[nType];
    Size* iArchType; // [nArchtype]
	Float* daxn; //[nType];
	Float* dden; //[nType];
	Float* raxn; //[nType];
	Float* rden; //[nType];
	Float* typeFeatureMat; //[nFeature, nType, nType]
	Float* sTypeMat; //[nType, nType]
	Size* nTypeMat; //[nType, nType]

    hInitialize_package() {};
    hInitialize_package(Size nArchtype, Size nType, Size nFeature,
						std::vector<Size>  &_nTypeHierarchy,
						std::vector<Size>  &_typeAccCount,
						std::vector<Float> &_raxn,
						std::vector<Float> &_rden,
						std::vector<Float> &_daxn,
						std::vector<Float> &_dden,
						std::vector<Float> &_typeFeatureMat,
						std::vector<Float> &_sTypeMat,
						std::vector<Size> &_nTypeMat)
	{
		size_t memSize = (4*nType + (1+nFeature)*nType*nType)*sizeof(Float) + (2*nArchtype + nType + nType*nType) * sizeof(Size);
		mem_block = new char[memSize];
		nTypeHierarchy = (Size*) mem_block;
		typeAccCount = nTypeHierarchy + nArchtype;
        iArchType = typeAccCount + nType;
		daxn = (Float *) (iArchType + nArchtype);
		dden = daxn + nType;
		raxn = dden + nType;
		rden = raxn + nType;
		typeFeatureMat = rden + nType;
		sTypeMat = typeFeatureMat + nFeature*nType*nType;
		nTypeMat = (Size *) (sTypeMat + nType*nType);

        for (PosInt i=0; i<nType; i++) {
            daxn[i] = _daxn[i];
            dden[i] = _dden[i];
            raxn[i] = _raxn[i];
            rden[i] = _rden[i];
        }
        for (PosInt i=0; i<nType; i++) {
			for (PosInt j=0; j<nType; j++) {
				sTypeMat[i*nType + j] = _sTypeMat[i*nType + j];
				nTypeMat[i*nType + j] = _nTypeMat[i*nType + j];
                for (PosInt k=0; k<nFeature; k++) {
                    typeFeatureMat[k*nType*nType + i*nType + j] = _typeFeatureMat[k*nType*nType + i*nType + j];
                }
			}
		}
        printf("\n");
        // nType
        for (PosInt i=0; i<nType; i++) {
            typeAccCount[i] = _typeAccCount[i];
        }
        // nHierarchy
        Size acc = 0;
        for (PosInt i=0; i<nArchtype; i++) {
            nTypeHierarchy[i] = _nTypeHierarchy[i];
            acc += nTypeHierarchy[i];
            iArchType[i] = acc;
        }
	}
	void freeMem() {
		delete []mem_block;
	}
};

struct initialize_package {
	char* mem_block;
	Size* nTypeHierarchy; // [nArchtype]
    Size* typeAccCount; //[nType];
    Size* iArchType; // [nArchtype]
	Float* daxn; //[nType];
	Float* dden; //[nType];
	Float* raxn; //[nType];
	Float* rden; //[nType];
	Float* typeFeatureMat; //[nFeature, nType, nType]
	Float* sTypeMat; //[nType, nType]
	Size* nTypeMat; //[nType, nType]

    initialize_package() {};
    initialize_package(Size nArchtype, Size nType, Size nFeature, hInitialize_package &host) {
		size_t memSize = (4*nType + (1+nFeature)*nType*nType)*sizeof(Float) + (2*nArchtype + nType + nType*nType) * sizeof(Size);
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
		nTypeHierarchy = (Size*) mem_block;
		typeAccCount = nTypeHierarchy + nArchtype;
        iArchType = typeAccCount + nType;
		daxn = (Float *) (iArchType + nArchtype);
		dden = daxn + nType;
		raxn = dden + nType;
		rden = raxn + nType;
		typeFeatureMat = rden + nType;
		sTypeMat = typeFeatureMat + nFeature*nType*nType;
		nTypeMat = (Size *) (sTypeMat + nType*nType);

        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
	}
	void freeMem() {
		checkCudaErrors(cudaFree(mem_block));
	}
};

__global__ 
__launch_bounds__(blockSize, 1)
void initialize(curandStateMRG32k3a* __restrict__ state,
                           Size*  __restrict__ preType,
                           Float* __restrict__ rden,
                           Float* __restrict__ raxn,
                           Float* __restrict__ dden,
                           Float* __restrict__ daxn,
                           Float* __restrict__ preF_type,
                           Float* __restrict__ preS_type,
                           Size* __restrict__ preN_type,
                           //Float* __restrict__ LGN_sSum,
                           Size* __restrict__ d_LGN_V1,
                           Float* __restrict__ ExcRatio,
                           Float* __restrict__ extExcRatio,
                           Float* __restrict__ synPerCon,
                           Float* __restrict__ synPerConFF,
						   Float min_FB_ratio, Float C_InhRatio, initialize_package init_pack, unsigned long long seed, Size networkSize, Size nType, Size nArchtype, Size nFeature, bool CmoreN, bool ClessI, Float p_n_LGNeff);

__global__ 
void cal_blockPos(double* __restrict__ pos,
                  Float* __restrict__ block_x,
                  Float* __restrict__ block_y,
                  Size networkSize);

__global__ 
void get_neighbor_blockId(Float* __restrict__ block_x,
                          Float* __restrict__ block_y,
                          PosInt* __restrict__ blockAcc,
                          PosInt* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
                          Size* __restrict__ nNearNeighborBlock,
						  Size nblock, Float radius, Float max_radius, Size maxNeighborBlock, PosInt in, PosInt out);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_connections(double* __restrict__ pos,
                          Float* __restrict__ preF_type,
                          Float* __restrict__ gap_preF_type, // nTypeI x nFeature x nTypeI
                          Float* __restrict__ preS_type, // nType x networkSize
                          Float* __restrict__ gap_preS_type,
                          Size* __restrict__ preN_type,
                          Size* __restrict__ gap_preN_type,
                          PosInt* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
					      Size* __restrict__ nNearNeighborBlock,
                          Float* __restrict__ rden,
                          Float* __restrict__ raxn,
                          Float* __restrict__ conMat, //within block connections
                          Float* __restrict__ delayMat,
                          Float* __restrict__ gapMat,
                          Float* __restrict__ conVec, //for neighbor block connections
                          Float* __restrict__ delayVec, //for neighbor block connections
                          Float* __restrict__ gapVec,
                          Float* __restrict__ gapDelayVec,
                          Size* __restrict__ max_N,
                          PosInt* __restrict__ _vecID,
                          Float* __restrict__ disNeighborP,
                          Float* __restrict__ gap_disNeighborP,
                          Size* __restrict__ vecID,
                          Size* __restrict__ nVec,
                          Size* __restrict__ gapVecID,
                          Size* __restrict__ nGapVec,
                          Size* __restrict__ preTypeConnected,
                          Size* __restrict__ preTypeAvail,
                          Float* __restrict__ preTypeStrSum,
                          Size* __restrict__ preTypeGapped,
                          Float* __restrict__ preTypeStrGapped,
                          Size* __restrict__ preType,
                          Float* __restrict__ feature,
                          Float* __restrict__ dden,
                          Float* __restrict__ daxn,
                          Float* __restrict__ synloc,
                          Size* __restrict__ typeAcc0,
                          curandStateMRG32k3a* __restrict__ state,
                          Size sum_max_N, Size gap_sum_max_N, PosInt block_offset, Size networkSize, Size mI, Size maxDistantNeighbor, Size gap_maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Size nType, Size nTypeE, Size nTypeI, Size nE, Size nI, Size nFeature, Float disGauss, bool strictStrength, Float tol);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_symmetry(PosInt* __restrict__ clusterID,
					   PosInt* __restrict__ neighborBlockId,
					   int* __restrict__ neighborMat,
					   Float* __restrict__ clusterGapMat,
					   Size* __restrict__ preTypeGapped,
					   Float* __restrict__ preTypeStrGapped,
					   PosInt* __restrict__ preType,
					   curandStateMRG32k3a* __restrict__ state,
					   PosInt* __restrict__ i_outstanding,
					   Float* __restrict__ v_outstanding,
					   PosInt iblock, Size nblock, Size nearNeighborBlock, Size maxNeighborBlock, Size mI, Size nE, Size nI, Size nTypeE, Size nTypeI);

#endif
