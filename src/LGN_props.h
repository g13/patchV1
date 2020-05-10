#ifndef LGN_PROPS_H
#define LGN_PROPS_H
#include "types.h"
#include <vector>

// Array structure: (type, nLGN), different from spatial and temporal weight storage which are (nLGN, type).
// This is to optimize read and write in CUDA

struct hSpatial_component {
    Float* mem_block;
    Float* x; // normalize to (0,1)
    Float* rx;
    Float* y; // normalize to (0,1)
    Float* ry;
    Float* orient;
    Float* k; // its sign determine On-Off
    Size arraySize;
    // TODO: construct from file directly, save memory
	hSpatial_component() {};
    hSpatial_component(
            Size nLGN,
            Size nType, // center and surround, 2
            const std::vector<Float> &_x,
            const std::vector<Float> &_rx,
            const std::vector<Float> &_y,
            const std::vector<Float> &_ry,
            const std::vector<Float> &_orient,
            const std::vector<Float> &_k
    ) {
        Size arraySize = nLGN*nType;
        this->arraySize = arraySize;
        mem_block = new Float[6*arraySize];
        x = mem_block;
        rx = x + arraySize;
        y = rx + arraySize;
        ry = y + arraySize;
        orient = ry + arraySize;
        k = orient + arraySize;
        for (Size i=0; i<arraySize; i++) {
            x[i] = _x[i];
            assert(!isnan(x[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            rx[i] = _rx[i];
            assert(!isnan(rx[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            y[i] = _y[i];
            assert(!isnan(y[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            ry[i] = _ry[i];
            assert(!isnan(ry[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            orient[i] = _orient[i];
            assert(!isnan(orient[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            k[i] = _k[i];
            assert(!isnan(k[i]));
        }
    }
	size_t freeMem() {
		delete []mem_block;
        size_t usingGMem = arraySize*6*sizeof(Float);
        return usingGMem;
	}
};

struct hTemporal_component {
	// check temporalKernel in discrete_input_convol.cu for the formula
	Float* mem_block;
	Float* tauR;
	Float* tauD;
	Float* delay;
	Float* ratio;
	Float* nR; // factorials are also defined for floats as gamma function
	Float* nD;
    Size arraySize;

	hTemporal_component() {};
    hTemporal_component(
            Size nLGN,
            Size nType,
            const std::vector<Float> &_tauR,
            const std::vector<Float> &_tauD,
            const std::vector<Float> &_delay,
            const std::vector<Float> &_ratio,
            const std::vector<Float> &_nR,
            const std::vector<Float> &_nD
    ) {
        Size arraySize = nLGN*nType;
        this->arraySize = arraySize;
        mem_block = new Float[6*arraySize];
        tauR = mem_block;
        tauD = tauR + arraySize;
		delay = tauD + arraySize;
        ratio = delay + arraySize;
        nR = ratio + arraySize;
        nD = nR + arraySize;
        for (Size i=0; i<arraySize; i++) {
            tauR[i] = _tauR[i];
            assert(!isnan(tauR[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            tauD[i] = _tauD[i];
            assert(!isnan(tauD[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            delay[i] = _delay[i];
            assert(!isnan(delay[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            ratio[i] = _ratio[i];
            assert(!isnan(ratio[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            nR[i] = _nR[i];
            assert(!isnan(nR[i]));
        }
        for (Size i=0; i<arraySize; i++) {
            nD[i] = _nD[i];
            assert(!isnan(nD[i]));
        }
    }
	size_t freeMem() {
		delete []mem_block;
        size_t usingGMem = arraySize*6*sizeof(Float);
        return usingGMem;
	}
};

struct hStatic_nonlinear {
    Float* mem_block;
    Float* c50;
    Float* sharpness;
    Float* a;
    Float* b;
    Size nLGN;

	hStatic_nonlinear() {};
    hStatic_nonlinear(
            Size nLGN,
            const std::vector<Float> &_spont,
            const std::vector<Float> &_c50,
            const std::vector<Float> &_sharpness
    ) {
        this->nLGN = nLGN;
        mem_block = new Float[4*nLGN];
        c50 = mem_block;
        sharpness = c50 + nLGN;
        a = sharpness + nLGN;
        b = a + nLGN;
        for (Size i=0; i<nLGN; i++) {
            c50[i] = _c50[i]; 
            assert(!isnan(c50[i]));
        }
        for (Size i=0; i<nLGN; i++) {
			sharpness[i] = _sharpness[i];
            assert(!isnan(sharpness[i]));
        }
        for (Size i=0; i<nLGN; i++) {
            // pre cache a and b
            const Float e_kc50 = exp(sharpness[i]*c50[i]);
            const Float e_kc50_k = exp(sharpness[i]*c50[i]-sharpness[i]);
            a[i] = (1-_spont[i])*(1+e_kc50)*(1+e_kc50_k)/(e_kc50-e_kc50_k);
            b[i] = 1-a[i]/(1+e_kc50_k);
        }
    }
	size_t freeMem() {
		delete []mem_block;
        size_t usingGMem = 4*nLGN*sizeof(Float);
        return usingGMem;
	}
    // to do: recalculate individual parameters when learning
};

struct hLGN_parameter {
    // block allocation
    Size nLGN, nType;
    hSpatial_component spatial;
    hTemporal_component temporal;
    hStatic_nonlinear logistic;

    char* mem_block;
    // TODO: enum this
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    SmallSize* coneType;
    Float* covariant; // color in the surround and center usually covary, for calculating the max convol
    
    hLGN_parameter(
            Size _nLGN,
            Size _nType,
            hSpatial_component &_spatial,
            hTemporal_component &_temporal,
            hStatic_nonlinear &_logistic,
            const std::vector<SmallSize> &_coneType,
            const std::vector<Float> &_covariant
    ) {
        nLGN = _nLGN;
        nType = _nType;
        Size arraySize = nLGN*nType;
        spatial = _spatial;
        temporal = _temporal;
        logistic = _logistic;

        mem_block = new char[arraySize*sizeof(SmallSize)+sizeof(Float)*(nType-1)*nType/2*nLGN];
        coneType = (SmallSize*) mem_block;
        covariant = (Float*) (coneType + arraySize);
        for (Size i=0; i<arraySize; i++) {
            coneType[i] = _coneType[i];
            assert(coneType[i] < 7);
        }
        for (Size i=0; i<(nType-1)*nType/2*nLGN; i++) {
            covariant[i] = _covariant[i];
            assert(!isnan(covariant[i]));
        }
    }
	size_t freeMem() {
        size_t usingGMem = nLGN*nType*sizeof(SmallSize)+sizeof(Float)*(nType-1)*nType/2*nLGN;
		usingGMem += spatial.freeMem();
		usingGMem += temporal.freeMem();
		usingGMem += logistic.freeMem();
		delete []mem_block;
        return usingGMem;
	}
};

#endif
