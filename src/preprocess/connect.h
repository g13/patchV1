#ifndef CONNECT_H
#define CONNECT_H

#include <vector>
#include <cuda_runtime.h>
#include <helper_functions.h> // include cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <cassert>
#include "../MACRO.h"
#include "../types.h"
#include "../util/cuda_util.cuh"

struct hInitialize_package {
	Size* mem_block;
	Size* nTypeHierarchy; // [nHierarchy]
    Size* archtypeAccCount; //[nArchtype];
	Float* daxn; //[nArchtype];
	Float* dden; //[nArchtype];
	Float* raxn; //[nArchtype];
	Float* rden; //[nArchtype];
	Float* sTypeMat; //[nType, nType]
	Float* pTypeMat; //[nType, nType]
	Size* preTypeN; //[nArchtype]

    hInitialize_package() {};
    hInitialize_package(Size nArchtype, Size nType, Size nHierarchy,
						std::vector<Size>  &_nTypeHierarchy,
						std::vector<Size>  &_archtypeAccCount,
						std::vector<Float> &_raxn,
						std::vector<Float> &_rden,
						std::vector<Float> &_daxn,
						std::vector<Float> &_dden,
						std::vector<Float> &_sTypeMat,
						std::vector<Float> &_pTypeMat,
						std::vector<Size>  &_preTypeN) 
	{
		size_t memSize = (4*nArchtype + 2*nType*nType)*sizeof(Float) + (2*nArchtype + nHierarchy) * sizeof(Size);
		mem_block = new Size[memSize];
		nTypeHierarchy = mem_block;
		archtypeAccCount = nTypeHierarchy + nHierarchy;
		daxn = (Float *) (archtypeAccCount + nArchtype);
		dden = daxn + nArchtype;
		raxn = dden + nArchtype;
		rden = raxn + nArchtype;
		sTypeMat = rden + nArchtype;
		pTypeMat = sTypeMat + nType*nType;
		preTypeN = (Size*) (pTypeMat + nType*nType);

        for (Size i=0; i<nArchtype; i++) {
            daxn[i] = _daxn[i];
            dden[i] = _dden[i];
            raxn[i] = _raxn[i];
            rden[i] = _rden[i];
			preTypeN[i] = _preTypeN[i];
        }
        for (Size i=0; i<nType; i++) {
			for (Size j=0; j<nType; j++) {
				sTypeMat[i*nType + j] = _sTypeMat[i*nType + j];
				pTypeMat[i*nType + j] = _pTypeMat[i*nType + j];
			}
		}
        // nType
        for (Size i=0; i<nArchtype; i++) {
            archtypeAccCount[i] = _archtypeAccCount[i];
        }
        // nHierarchy
        for (Size i=0; i<nHierarchy; i++) {
            nTypeHierarchy[i] = _nTypeHierarchy[i];
        }
	}
	void freeMem() {
		delete []mem_block;
	}
};

struct initialize_package {
	char* mem_block;
	Size* nTypeHierarchy; // [nHierarchy]
    Size* archtypeAccCount; //[nArchtype];
	Float* daxn; //[nArchtype];
	Float* dden; //[nArchtype];
	Float* raxn; //[nArchtype];
	Float* rden; //[nArchtype];
	Float* sTypeMat; //[nType, nType]
	Float* pTypeMat; //[nType, nType]
	Size* preTypeN; //[nArchtype]

    initialize_package() {};
    initialize_package(Size nArchtype, Size nType, Size nHierarchy, hInitialize_package &host) {
		size_t memSize = (4*nArchtype + 2*nType*nType)*sizeof(Float) + (2*nArchtype + nHierarchy) * sizeof(Size);
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
		nTypeHierarchy = (Size*) mem_block;
		archtypeAccCount = nTypeHierarchy + nHierarchy;
		daxn = (Float *) (archtypeAccCount + nArchtype);
		dden = daxn + nArchtype;
		raxn = dden + nArchtype;
		rden = raxn + nArchtype;
		sTypeMat = rden + nArchtype;
		pTypeMat = sTypeMat + nType*nType;
		preTypeN = (Size*) (pTypeMat + nType*nType);

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
                           Float* __restrict__ preS_type,
                           Float* __restrict__ preP_type,
                           Size*  __restrict__ preN,
						   Size* __restrict__ preFixType, // nSubHierarchy * networkSize
                           initialize_package init_pack, unsigned long long seed, Size networkSize, Size nType, Size nArchtype, Size nHierarchy); 

__global__ 
__launch_bounds__(blockSize, 1)
void cal_blockPos(double* __restrict__ pos,
                  Float* __restrict__ block_x,
                  Float* __restrict__ block_y,
                  Size networkSize);

__global__ 
__launch_bounds__(blockSize, 1)
void get_neighbor_blockId(Float* __restrict__ block_x,
                          Float* __restrict__ block_y,
                          PosInt* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
                          Size nblock, Float max_radius, Size maxNeighborBlock);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_connections(double* __restrict__ pos,
                          Float* __restrict__ preS_type,
                          Float* __restrict__ preP_type,
                          Size* __restrict__ preN,
                          PosInt* __restrict__ neighborBlockId,
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
                          Float* __restrict__ feature,
                          Float* __restrict__ dden,
                          Float* __restrict__ daxn,
                          curandStateMRG32k3a* __restrict__ state,
                          PosInt block_offset, Size networkSize, Size maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Size nType, Size nFeature, bool gaussian_profile);

#endif
