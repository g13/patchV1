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

/*struct hInitialize_package {
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
};*/

/*struct initialize_package {
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
}; */

__global__ 
__launch_bounds__(blockSize, 1)
void initialize(
			Float* __restrict__ nLGN_eff,
			Float* __restrict__ ffRatio,
			Size* __restrict__ typeAcc0,
            Float* __restrict__ maxCortExc,
			Size nType);

__global__ 
void cal_blockPos(
              Float* __restrict__ pos,
              Float* __restrict__ block_x,
              Float* __restrict__ block_y,
              Float* __restrict__ block_r,
              Size networkSize);

__global__ 
void get_neighbor_blockId(
                        Float* __restrict__ block_x,
                        Float* __restrict__ block_y,
                        Float* __restrict__ block_r,
                        PosInt* __restrict__ blockAcc,
                        PosInt* __restrict__ neighborBlockId,
                        Size* __restrict__ nNeighborBlock,
                        Size* __restrict__ nNearNeighborBlock,
                        Size* __restrict__ nGapNeighborBlockE,
                        Size* __restrict__ nGapNeighborBlockI,
						Size nblock, Float rden, Float raxn, Size maxNeighborBlock, Float postGapDisE, Float preGapDisE, Float postGapDisI, Float preGapDisI, PosInt ipost, PosInt ipre);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_connections(
                        curandStateMRG32k3a* __restrict__ state,
                        Float* __restrict__ post_pos,
                        Float* __restrict__ pre_pos,
                        PosInt* __restrict__ neighborBlockId,
                        Size*   __restrict__ nNeighborBlock,
                        Size*   __restrict__ nNearNeighborBlock,
                        Size*   __restrict__ nGapNeighborBlockE,
                        Size*   __restrict__ nGapNeighborBlockI,
                        Float*  __restrict__ feature,
                        Float*  __restrict__ rden,
                        Float*  __restrict__ raxn,
                        Float*  __restrict__ gapDis,
                        Float*  __restrict__ ffRatio,
                        Float*  __restrict__ inhRatio,
                        Size*   __restrict__ nTypeMat,
                        Size*   __restrict__ gap_nTypeMatE,
                        Size*   __restrict__ gap_nTypeMatI,
                        Float*  __restrict__ fTypeMat,
                        Float*  __restrict__ gap_fTypeMatE,
                        Float*  __restrict__ gap_fTypeMatI,
                        Float*  __restrict__ conMat, //within block connections
                        Float*  __restrict__ delayMat,
                        Float*  __restrict__ gapMatE,
                        Float*  __restrict__ gapMatI,
                        Float*  __restrict__ conVec, //for neighbor block connections
                        Float*  __restrict__ delayVec, //for neighbor block connections
                        Size*   __restrict__ max_N,
                        PosInt* __restrict__ _vecID,
                        Float*  __restrict__ disNeighborP,
                        Size*   __restrict__ vecID,
                        Size*   __restrict__ nVec,
                        Size*   __restrict__ preTypeConnected,
                        Size*   __restrict__ preTypeAvail,
                        Float*  __restrict__ preTypeStrSum,
                        Size*   __restrict__ preTypeGapE,
                        Size*   __restrict__ preTypeAvailGapE,
                        Float*  __restrict__ preTypeStrGapE,
                        Size*   __restrict__ preTypeGapI,
                        Size*   __restrict__ preTypeAvailGapI,
                        Float*  __restrict__ preTypeStrGapI,
                        Size**  __restrict__ typeAcc0,
                        PosInt post, PosInt pre, Size accSize, Size postAccSize, Size preAccSize, Size totalType, Size totalTypeE, Size totalTypeI, PosInt postTypeID, PosInt preTypeID, PosInt postTypeEID, PosInt preTypeEID, PosInt postTypeIID, PosInt preTypeIID, Size nF, Size ppF, Size prePerBlock, Size sum_max_N, PosInt block_offset, Size postSize, Size preSize, Size post_nType, Size pre_nType, Size post_nTypeE, Size pre_nTypeE, Size post_nTypeI, Size pre_nTypeI, Size mE, Size mI, Size maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Size maxTempBlock, Size gapNeighborBlockE, Size gapNeighborBlockI, Size post_nE, Size post_nI, Size pre_nE, Size pre_nI, Float disGauss, bool strictStrength, bool CmoreN, BigSize seed);

__global__ 
__launch_bounds__(blockSize, 1)
void generate_symmetry(
					curandStateMRG32k3a* __restrict__ state,
                    PosInt* __restrict__ clusterID,
                    Size*   __restrict__ gap_nTypeMat,
				    PosInt* __restrict__ neighborBlockId,
					Int*    __restrict__ neighborMat,
                    PosInt* __restrict__ blockAcc,
					Float*  __restrict__ postGapMat,
					Float*  __restrict__ clusterGapMat,
					Size*   __restrict__ preTypeGap,
					Float*  __restrict__ preTypeStrGap,
					Size*   __restrict__ postTypeGap,
					Float*  __restrict__ postTypeStrGap,
					PosInt* __restrict__ i_outstanding,
					Float*  __restrict__ v_outstanding,
                    Size**  __restrict__ typeAcc0,
					PosInt iblock, PosInt iLayer, PosInt jLayer, Size nLayer, Size gapNeighborBlock, Size postN, Size preN, Size postPerBlock0, Size prePerBlock0, Size prePerBlock, PosInt preTypeID, PosInt postTypeID, Size totalGapType, Size pre_nType, Size pre_nType0, Size post_nType, Size post_nType0, bool strictStrength, BigSize seed);

#endif
