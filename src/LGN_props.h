#ifndef LGN_PROPS_H
#define LGN_PROPS_H

// Array structure: (type, nLGN), different from spatial and temporal weight storage which are (nLGN, type).
// This is to optimize read and write in CUDA

struct hSpatial_component {
    Float* mem_block;
    Float* x; // normalize to (0,1)
    Float* rx;
    Float* y; // normalize to (0,1)
    Float* ry;
    Float* k; // its sign determine On-Off
    hSpatial_component(
            Size nLGN,
            Size nType,
            const vector<Float> &_x,
            const vector<Float> &_rx,
            const vector<Float> &_y,
            const vector<Float> &_rx,
            const vector<Float> &_k
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
    Float* mem_block;
    Float* tauR;
    Float* tauD;
    Float* delay;
    Float* ratio; // 2 for parvo 1 for magno
    Float* nR; // factorials are also defined for floats as gamma function
    Float* nD;

    hspatial_component(
            Size nLGN,
            Size nType,
            const vector<Float> &_tauR,
            const vector<Float> &_tauD,
            const vector<Float> &_delay,
            const vector<Float> &_ratio,
            const vector<Float> &_nR,
            const vector<Float> &_nD
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

    hStatic_nonlinear(
            Size nLGN,
            const vector<Float> &_spont,
            const vector<Float> &_c50,
            const vector<Float> &_sharpness
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
            sharpness[i] = _sharpness[i]
        }
        for (Size i=0; i<nLGN; i++) {
            // pre cache a and c
            const Float expk = exp(sharpness[i]);
            const Float expkr = exp(sharpness[i]*c50[i]);
            const Float expkr_1 = expkr/expk;
            const Float c = (expkr+expk)/(expkr*(1-expk)) * (1-spont[i]);
            a[i] = -c*(1+expkr);
            b[i] = c + spont[i];
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
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    SmallSize* coneType;
    Float* covariant; // color in the surround and center ay covary
    
    hLGN_parameter(
            Size _nLGN,
            Size _nType,
            hSpatial_component &_spatial,
            hTemporal_component &_temporal,
            hStatic_nonlinear &_logistic,
            const vector<SmallSize> &_coneType,
            const vector<Float> &_covariant
    ) {
        nLGN = _nLGN;
        nType = _nType;
        Size arraySize = nLGN*nType
        spatial = _spatial;
        temporal = _temporal;
        logistic = _logistic;

        mem_block = new char[arraySize*sizeof(SmallSize)+sizeof(Float)*(nType-1)*nLGN];
        coneType = (SmallSize*) mem_block;
        covariant = (Float*) (coneType + nLGN);
        for (Size i=0; i<arraySize; i++) {
            coneType[i] = _coneType[i];
        }
        for (Size i=0; i<(nType-1)*nLGN; i++) {
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
