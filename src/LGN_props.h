#ifndef LGN_PROPS_H
#define LGN_PROPS_H

struct hspatial_component {
    Float* mem_block;
    Float* x; // normalize to (0,1)
    Float* rx;
    Float* y; // normalize to (0,1)
    Float* ry;
    Float* k; // its sign determine On-Off
    hspatial_component(Size nLGN,
                        const vector<Float> &_x,
                        const vector<Float> &_rx,
                        const vector<Float> &_y,
                        const vector<Float> &_rx,
                        const vector<Float> &_k,
                        ) {
        mem_block = new Float[5*nLGN];
        x = mem_block;
        rx = x + nLGN;
        y = rx + nLGN;
        ry = y + nLGN;
        k = ry + nLGN;
        for (Size i=0; i<nLGN; i++) {
            x[i] = _x[i];
        }
        for (Size i=0; i<nLGN; i++) {
            rx[i] = _rx[i];
        }
        for (Size i=0; i<nLGN; i++) {
            y[i] = _y[i];
        }
        for (Size i=0; i<nLGN; i++) {
            ry[i] = _ry[i];
        }
        for (Size i=0; i<nLGN; i++) {
            k[i] = _k[i];
        }
    }
	void freeMem() {
		delete []mem_block;
	}
};

struct htemporal_component {
    Float* mem_block;
    Float* tauR;
    Float* tauD;
    Float* delay;
    Float* ratio; // 2 for parvo 1 for magno
    Float* nR; // factorials are also defined for floats as gamma function
    Float* nD;

    hspatial_component(Size nLGN,
                       const vector<Float> &_tauR,
                       const vector<Float> &_tauD,
                       const vector<Float> &_delay,
                       const vector<Float> &_ratio,
                       const vector<Float> &_nR,
                       const vector<Float> &_nD) {
        mem_block = new Float[6*nLGN];
        tauR = mem_block;
        tauD = tauR + nLGN;
		delay = tauD + nLGN;
        ratio = delay + nLGN;
        nR = ratio + nLGN;
        nD = nR + nLGN;
        for (Size i=0; i<nLGN; i++) {
            tauR[i] = _tauR[i];
        }
        for (Size i=0; i<nLGN; i++) {
            tauD[i] = _tauD[i];
        }
        for (Size i=0; i<nLGN; i++) {
            delay[i] = _delay[i];
        }
        for (Size i=0; i<nLGN; i++) {
            ratio[i] = _ratio[i];
        }
        for (Size i=0; i<nLGN; i++) {
            nR[i] = _nR[i];
        }
        for (Size i=0; i<nLGN; i++) {
            nD[i] = _nD[i];
        }
    }
	void freeMem() {
		delete []mem_block;
	}
};

struct hcone_specific {
    hspatial_component spatial;
    htemporal_component temporal;
    hcone_specific(Size nLGN,
                   hspatial_component &_spatial,
                   htemporal_component &_temporal) {
        spatial = _spatial;
        temporal = _temporal;
    }
	void freeMem() {
		spatial.freeMem();
		temporal.freeMem();
	}
};

struct hstatic_nonlinear {
    Float* mem_block;
    Float* c50;
    Float* sharpness;
    Float* a;
    Float* b;

    hstatic_nonlinear(Size nLGN,
                      const vector<Float> &_spont,
                      const vector<Float> &_c50,
                      const vector<Float> &_sharpness) {
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
    Size nLGN;
    hcone_specific center, surround;
    hstatic_nonlinear logistic;

    char* mem_block;
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    SmallSize* centerType;
    SmallSize* surroundType;
    Float* covariant; // color in the surround and center ay covary
    
    hLGN_parameter(Size _nLGN,
                   hcone_specific &_center,
                   hcone_specific &_surround,
                   hstatic_nonlinear &_logistic,
                   const vector<SmallSize> &_centerType,
                   const vector<SmallSize> &_surroundType,
                   const vector<Float> &_covariant) {
        nLGN = _nLGN;
        center = _center;
        surround = _surround;
        logistic = _logistic;

        mem_block = new char[(2*sizeof(SmallSize)+sizeof(Float))*nLGN];
        centerType = (SmallSize*) mem_block;
        surroundType = centerType + nLGN;
        covariant = (Float*) (surroundType + nLGN);
        for (Size i=0; i<nLGN; i++) {
            centerType[i] = _centerType[i];
        }
        for (Size i=0; i<nLGN; i++) {
            surroundType[i] = _surroundType[i];
        }
        for (Size i=0; i<nLGN; i++) {
            covariant[i] = _covariant[i];
        }
    }
	void freeMem() {
		center.freeMem();
		surround.freeMem();
		logistic.freeMem();
		delete []mem_block;
	}
};

#endif
