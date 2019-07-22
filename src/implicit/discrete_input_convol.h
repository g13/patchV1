//#include <cufft.h>
#ifndef DISCRETE_INPUT_H
#define DISCRETE_INPUT_H

#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <math_functions.h>         //
#include "DIRECTIVE.h"
#include "cuda_util.h"

extern texture<float, cudaTextureType2DLayered> L_retinaConSig;
extern texture<float, cudaTextureType2DLayered> M_retinaConSig;
extern texture<float, cudaTextureType2DLayered> S_retinaConSig;

#ifdef SINGLE_PRECISION
	#define square_root sqrtf
	#define atan atan2f
	#define uniform curand_uniform
	#define expp expf
	#define power powf
	#define abs fabsf 
    #define copy copysignf
#else
	#define square_root sqrt
	#define atan atan2
	#define uniform curand_uniform_double
	#define expp exp 
	#define power pow
	#define abs fabs 
    #define copy copysign
#endif

struct hcone_specific {
    _float* mem_block;
    _float* __restrict__ x; // normalize to (0,1)
    _float* __restrict__ rx;
    _float* __restrict__ y; // normalize to (0,1)
    _float* __restrict__ ry;
    _float* __restrict__ k; // its sign determine On-Off
    _float* __restrict__ tauR;
    _float* __restrict__ tauD;
    _float* __restrict__ nD;
    _float* __restrict__ nR;
    _float* __restrict__ ratio; // 2 for parvo 1 for magno
    //_float* __restrict__ delay;
    //_float* __restrict__ facRatio; mem-calc balance
    void alloc(unsigned int nLGN) {
        mem_block = new _float[10*nLGN];
        x = mem_block;
        rx = x + nLGN;
        y = rx + nLGN;
        ry = y + nLGN;
        k = ry + nLGN;
        tauR = k + nLGN;
        tauD = tauR + nLGN;
        nD = tauD + nLGN;
        nR = nD + nLGN;
        ratio = nR + nLGN;
    }
	void freeMem() {
		delete []mem_block;
	}
};

struct hstatic_nonlinear {
    _float* mem_block;

    _float* __restrict__ spont;
    _float* __restrict__ c50;
    _float* __restrict__ sharpness;

    void alloc(unsigned int nLGN) {
        mem_block = new _float[3*nLGN];
        spont = mem_block;
        c50 = spont + nLGN;
        sharpness = c50 + nLGN;
    }
	void freeMem() {
		delete []mem_block;
	}
};

struct hLGN_parameter {
    // block allocation
    unsigned int nLGN;
    hcone_specific center, surround;
    hstatic_nonlinear logistic;

    unsigned int* mem_block;
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    unsigned int* __restrict__ centerType;
    unsigned int* __restrict__ surroundType;
    _float* __restrict__ covariant; // color in the surround and center ay covary
    
    hLGN_parameter(unsigned int _nLGN) {
        nLGN = _nLGN;
        center.alloc(nLGN);
        surround.alloc(nLGN);
        logistic.alloc(nLGN);


        mem_block = new unsigned int[2*nLGN];
        centerType = mem_block;
        surroundType = centerType + nLGN;
        covariant = new _float[nLGN];
    }
	void freeMem() {
		center.freeMem();
		surround.freeMem();
		logistic.freeMem();
		delete []mem_block;
		delete []covariant;
	}
};


/*struct LGN_stats {
    // min, mean, max
    // array of 19 * 3 = 57 
    const int n = 19;
    _float stats[n*3];
    // idist/stats:     0,    1,    2
    //  0: uniform:     lmin, rmax
    //  1: gauss:       mode, lstd, rstd
    //  2: log-normal:  mean, std,  cutoff
    //  3: triangle:    a,    b,    c
    unsigned int idist[n];
    //spatial parameters
    _float* __restrict__ kc, // center
    _float* __restrict__ rc;
    _float* __restrict__ ks; // surround
    _float* __restrict__ rs;
    //temporal parameters
    _float* __restrict__ cHs; // center
    _float* __restrict__ ctauS;
    _float* __restrict__ cNL;
    _float* __restrict__ ctauL;
    _float* __restrict__ cD;
    _float* __restrict__ cA;
    _float* __restrict__ sHs; // surround 
    _float* __restrict__ stauS;
    _float* __restrict__ sNL;
    _float* __restrict__ stauL;
    _float* __restrict__ sD;
    _float* __restrict__ sA;
    //static nonlineariry parameters
    _float* __restrict__ spont;
    _float* __restrict__ c50;
    _float* __restrict__ sharpness;
    LGN_stats () {

        kc = stats;
        rc = kc + 3;
        ks = rc + 3;
        rs = ks + 3;

        cHs = rs + 3;
        ctauS = cHs + 3;
        cNL = ctauS + 3;
        ctauL = cNL + 3;
        cD = ctauL + 3;
        cA = cD + 3;

        sHs = cA + 3;
        stauS = sHs + 3;
        sNL = stauS + 3;
        stauL = sNL + 3;
        sD = stauL + 3;
        sA = sD + 3;

        spont = sA + 3;
        c50 = spont + 3;
        sharpness = c50 + 3;
    }
}*/

struct cone_specific {
    _float* mem_block;
    _float* __restrict__ x; // normalize to (0,1)
    _float* __restrict__ rx;
    _float* __restrict__ y; // normalize to (0,1)
    _float* __restrict__ ry;
    _float* __restrict__ k; // its sign determine On-Off
    _float* __restrict__ tauR;
    _float* __restrict__ tauD;
    _float* __restrict__ nD;
    _float* __restrict__ nR;
    _float* __restrict__ ratio; // 2 for parvo 1 for magno
    //_float* __restrict__ delay;
    //_float* __restrict__ facRatio; mem-calc balance
    void allocAndMemcpy(unsigned int nLGN, hcone_specific &host) {
        checkCudaErrors(cudaMalloc((void**)&mem_block, 10*nLGN*sizeof(_float)));
        x = mem_block;
        rx = x + nLGN;
        y = rx + nLGN;
        ry = y + nLGN;
        k = ry + nLGN;
        tauR = k + nLGN;
        tauD = tauR + nLGN;
        nD = tauD + nLGN;
        nR = nD + nLGN;
        ratio = nR + nLGN;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, nLGN*10*sizeof(_float), cudaMemcpyHostToDevice));
    }
	void freeMem() {
		cudaFree(mem_block);
	}
};

struct static_nonlinear {
    _float* mem_block;

    _float* __restrict__ spont;
    _float* __restrict__ c50;
    _float* __restrict__ sharpness;

    void allocAndMemcpy(unsigned int nLGN, hstatic_nonlinear &host) {
        checkCudaErrors(cudaMalloc((void**)&mem_block, 3*nLGN*sizeof(_float)));
        spont = mem_block;
        c50 = spont + nLGN;
        sharpness = c50 + nLGN;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, 3*nLGN*sizeof(_float), cudaMemcpyHostToDevice));
    }
	void freeMem() {
		cudaFree(mem_block);
	}
    __device__
    _float transform(unsigned int id, _float input) {
        _float local_k = sharpness[id];
        _float local_c50 = c50[id];
        _float local_spont = spont[id];

        _float expk = expp(local_k);
        _float expkr = expp(local_k*local_c50);
        _float expkr_1 = expkr/expk;

        _float c = (expkr_1+1)/(expkr_1*(1-expk)) * (1-local_spont);
        _float a = -c*(1+expkr);

        return a/(1+expp(-local_k*input)*expkr) + c + local_spont;
    }
};

struct LGN_parameter {
    // block allocation
    unsigned int nLGN;
    cone_specific center, surround;
    static_nonlinear logistic;

    unsigned int* mem_block;
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    unsigned int* __restrict__ centerType;
    unsigned int* __restrict__ surroundType;
    _float* __restrict__ covariant; // color in the surround and center ay covary
    
    LGN_parameter(unsigned int _nLGN, hLGN_parameter &host) {
        nLGN = _nLGN;
        center.allocAndMemcpy(nLGN, host.center);
        surround.allocAndMemcpy(nLGN, host.surround);
        logistic.allocAndMemcpy(nLGN, host.logistic);

        checkCudaErrors(cudaMalloc((void**)&mem_block, 2*nLGN*sizeof(unsigned int)+nLGN*sizeof(_float)));

        centerType = mem_block;
        surroundType = centerType + nLGN;
        covariant = (_float*) (surroundType + nLGN);
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, nLGN*2*sizeof(unsigned int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(covariant, host.covariant, nLGN*sizeof(_float), cudaMemcpyHostToDevice));
    }
	void freeMem() {
		center.freeMem();
		surround.freeMem();
		logistic.freeMem();
		cudaFree(mem_block);
	}
};

struct LGN_subregion {
    _float x;
    _float y;
    _float rx;
    _float ry;
    _float k;
    _float tauR;
    _float tauD;
    _float nD;
    _float nR;
    _float ratio;
	
	__device__ LGN_subregion(cone_specific p, unsigned int id) {
        x = p.x[id];
        y = p.y[id];
        rx = p.rx[id];
        ry = p.ry[id];
        k = p.k[id];
        tauR = p.tauR[id];
        tauD = p.tauD[id];
        nD = p.nD[id];
        nR = p.nR[id];
        ratio = p.ratio[id];
    }
};

/*
 * 2D-block for spatial convolution and sum
 * loop to sum time convol
 * 1D-grid for different LGN
*/

__global__
__global__ void LGN_convol(_float* __restrict__ LGNfr,
                           LGN_parameter pLGN, // consider pointer
                           unsigned int iSample0,
                           _float samplePhase, unsigned int nKernelSample, _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D);

__global__ 
void LGN_nonlinear(_float* __restrict__ LGN_fr, static_nonlinear logistic, _float* __restrict__ max_convol);

__global__
__global__ void LGN_maxResponse(_float* __restrict__ max_convol,
                                LGN_parameter pLGN, // consider pointer
                                unsigned int nKernelSample, _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D);

#endif
