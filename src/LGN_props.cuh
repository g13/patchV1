#ifndef LGN_PROPS_CUH
#define LGN_PROPS_CUH
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "types.h"
#include "LGN_props.h"
#include "DIRECTIVE.h"

// Array structure: (type, nLGN), different from spatial and temporal weight storage, see "discrete_input_convol.cu->store_weight" which are (nLGN, type).
// This is to optimize read and write in CUDA

struct Spatial_component {
    Float* mem_block;
    Float* __restrict__ x; // polar angle in rad
    Float* __restrict__ rx;
    Float* __restrict__ y; // ecc in rad 
    Float* __restrict__ ry;
    Float* __restrict__ orient;
    Float* __restrict__ k; // its sign determine On-Off

    void allocAndMemcpy(Size arraySize, hSpatial_component &host) {
		size_t memSize = 6*arraySize*sizeof(Float) ;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
        x = mem_block;
        rx = x + arraySize;
        y = rx + arraySize;
        ry = y + arraySize;
        orient = ry + arraySize;
        k = orient + arraySize;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		checkCudaErrors(cudaFree(mem_block));
	}
};

struct Temporal_component {
    Float* mem_block;
    Float* __restrict__ tauR;
    Float* __restrict__ tauD;
    Float* __restrict__ delay;
    Float* __restrict__ ratio; // 2 for parvo 1 for magno
    Float* __restrict__ nR; // factorials also defined in floating points as gamma function
    Float* __restrict__ nD;

    void allocAndMemcpy(Size arraySize, hTemporal_component &host) {
		size_t memSize = 6*arraySize*sizeof(Float);
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
        tauR = mem_block;
        tauD = tauR + arraySize;
		delay = tauD + arraySize;
        ratio = delay + arraySize;
        nR = ratio + arraySize;
        nD = nR + arraySize;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		checkCudaErrors(cudaFree(mem_block));
	}
};

struct Static_nonlinear {
    Float* mem_block;

    Float* __restrict__ c50;
    Float* __restrict__ sharpness;
    Float* __restrict__ a;
    Float* __restrict__ b;

    void allocAndMemcpy(Size arraySize, hStatic_nonlinear &host) {
        size_t memSize = 4*arraySize*sizeof(Float);
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
        c50 = mem_block;
        sharpness = c50 + arraySize;
        a = sharpness + arraySize;
        b = a + arraySize;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		checkCudaErrors(cudaFree(mem_block));
	}
    // transform convolution result (input) with logistic function and return as firing rate
    __device__
    __forceinline__
	 void load_first(unsigned int id, Float &C50, Float &K, Float &A, Float &B) {
        C50 = c50[id];
        K = sharpness[id];
        A = a[id];
        B = b[id];
    }
};
__device__
__forceinline__
Float transform(Float C50, Float K, Float A, Float B, Float input) {
    // calculation
    Float X = 1.0/(1+exponential(K*(C50-input)));
    return A*X + B;
}

// collect all the components and send to device
struct LGN_parameter {
    // dev pointer of the host structs that contain dev pointer to data
    Spatial_component* spatial;
    Temporal_component* temporal;
    Static_nonlinear* logistic;
    // host struct that contain dev pointers to data, needed for cudaFree(*);
    Spatial_component hSpat;
    Temporal_component hTemp;
    Static_nonlinear hLogi;

    SmallSize* mem_block;
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    SmallSize* __restrict__ coneType;
    Float* __restrict__ covariant; // color in the surround and center ay co-vary (
    // ASSUME: mutiple surround types vs single center type
    
    LGN_parameter(hLGN_parameter &host) {
        Size nLGN = host.nLGN;

        SmallSize nType = host.nType;
        Size arraySize = nLGN*nType;

        hTemp.allocAndMemcpy(arraySize, host.temporal);
        checkCudaErrors(cudaMalloc((void**) &temporal, sizeof(hTemp)));
        checkCudaErrors(cudaMemcpy(temporal, &hTemp, sizeof(hTemp),cudaMemcpyHostToDevice));

        hSpat.allocAndMemcpy(arraySize, host.spatial);
        checkCudaErrors(cudaMalloc((void**) &spatial, sizeof(hSpat)));
        checkCudaErrors(cudaMemcpy(spatial, &hSpat, sizeof(hSpat),cudaMemcpyHostToDevice));

        hLogi.allocAndMemcpy(nLGN, host.logistic);
        checkCudaErrors(cudaMalloc((void**) &logistic, sizeof(hLogi)));
        checkCudaErrors(cudaMemcpy(logistic, &hLogi, sizeof(hLogi),cudaMemcpyHostToDevice));

        size_t memSize = arraySize*sizeof(SmallSize)+sizeof(Float)*(nType-1)*nLGN;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));

        coneType = mem_block;
        covariant = (Float*) (coneType + arraySize);
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		hSpat.freeMem();
		hTemp.freeMem();
		hLogi.freeMem();
		checkCudaErrors(cudaFree(temporal));
		checkCudaErrors(cudaFree(spatial));
		checkCudaErrors(cudaFree(logistic));
		checkCudaErrors(cudaFree(mem_block));
	}
};

struct Zip_temporal {
    Float tauR;
    Float tauD;
	Float delay;
    Float ratio;
    Float nR;
    Float nD;
	__device__ 
	__forceinline__
    void load(Temporal_component &t, Size id) {
        // loading order corresponds to calculation order
        // think before change order of load
        // DEBUG
        __syncthreads();
        //
        nR = t.nR[id]; 
        nD = t.nD[id];
        delay = t.delay[id];
        tauR = t.tauR[id];
        tauD = t.tauD[id];
        ratio = t.ratio[id];
    }
};

struct Zip_spatial {
    Float x;
    Float rx;
    Float y;
    Float ry;
    Float orient;
    Float k;
	
	__device__
	__forceinline__
    void load(Spatial_component &s, Size id) {
        x = s.x[id];
        y = s.y[id];
        rx = s.rx[id];
        ry = s.ry[id];
        orient = s.orient[id];
        k = s.k[id];
    }
};

#endif
