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
    Float* k; // its sign determine On-Off
    // TODO: construct from file directly, save memory
	hSpatial_component() {};
    hSpatial_component(
            Size nLGN,
            Size nType, // center and surround, 2
            const std::vector<Float> &_x,
            const std::vector<Float> &_rx,
            const std::vector<Float> &_y,
            const std::vector<Float> &_ry,
            const std::vector<Float> &_k
    ) {
        Size arraySize = nLGN*nType;
        mem_block = new Float[5*arraySize];
        x = mem_block;
        rx = x + arraySize;
        y = rx + arraySize;
        ry = y + arraySize;
        k = ry + arraySize;
        for (Size i=0; i<arraySize; i++) {
            x[i] = _x[i];
        }
        for (Size i=0; i<arraySize; i++) {
            rx[i] = _rx[i];
        }
        for (Size i=0; i<arraySize; i++) {
            y[i] = _y[i];
        }
        for (Size i=0; i<arraySize; i++) {
            ry[i] = _ry[i];
        }
        for (Size i=0; i<arraySize; i++) {
            k[i] = _k[i];
        }
    }
	void freeMem() {
		delete []mem_block;
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
        mem_block = new Float[6*arraySize];
        tauR = mem_block;
        tauD = tauR + arraySize;
		delay = tauD + arraySize;
        ratio = delay + arraySize;
        nR = ratio + arraySize;
        nD = nR + arraySize;
        for (Size i=0; i<arraySize; i++) {
            tauR[i] = _tauR[i];
        }
        for (Size i=0; i<arraySize; i++) {
            tauD[i] = _tauD[i];
        }
        for (Size i=0; i<arraySize; i++) {
            delay[i] = _delay[i];
        }
        for (Size i=0; i<arraySize; i++) {
            ratio[i] = _ratio[i];
        }
        for (Size i=0; i<arraySize; i++) {
            nR[i] = _nR[i];
        }
        for (Size i=0; i<arraySize; i++) {
            nD[i] = _nD[i];
        }
    }
	void freeMem() {
		delete []mem_block;
	}
};

struct hStatic_nonlinear {
    Float* mem_block;
    Float* c50;
    Float* sharpness;
    Float* a;
    Float* b;

	hStatic_nonlinear() {};
    hStatic_nonlinear(
            Size nLGN,
            const std::vector<Float> &_spont,
            const std::vector<Float> &_c50,
            const std::vector<Float> &_sharpness
    ) {
        mem_block = new Float[4*nLGN];
        c50 = mem_block;
        sharpness = c50 + nLGN;
        a = sharpness + nLGN;
        b = a + nLGN;
        for (Size i=0; i<nLGN; i++) {
            c50[i] = _c50[i]; 
        }
        for (Size i=0; i<nLGN; i++) {
			sharpness[i] = _sharpness[i];
        }
        for (Size i=0; i<nLGN; i++) {
            // pre cache a and b
            const Float e_kc50 = exp(sharpness[i]*c50[i]);
            const Float e_kc50_k = exp(sharpness[i]*c50[i]-sharpness[i]);
            a[i] = (1-_spont[i])*(1+e_kc50)*(1+e_kc50_k)/(e_kc50-e_kc50_k);
            b[i] = 1-a[i]/(1+e_kc50_k);
        }
    }
	void freeMem() {
		delete []mem_block;
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
        }
        for (Size i=0; i<(nType-1)*nType/2*nLGN; i++) {
            covariant[i] = _covariant[i];
        }
    }
	void freeMem() {
		spatial.freeMem();
		temporal.freeMem();
		logistic.freeMem();
		delete []mem_block;
	}
};

#endif
