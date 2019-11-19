#ifndef LGN_PROPS_CUH
#define LGN_PROPS_CUH

struct Spatial_component {
    Float* mem_block;
    Float* __restrict__ x; // normalize to (0,1)
    Float* __restrict__ rx;
    Float* __restrict__ y; // normalize to (0,1)
    Float* __restrict__ ry;
    Float* __restrict__ k; // its sign determine On-Off

    void allocAndMemcpy(unsigned int nLGN, hspatial_component &host) {
		size_t memSize = 5*sizeof(Float)*nLGN;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
        x = mem_block;
        rx = x + nLGN;
        y = rx + nLGN;
        ry = y + nLGN;
        k = ry + nLGN;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		cudaFree(mem_block);
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

    void allocAndMemcpy(unsigned int nLGN, htemporal_component &host) {
		size_t memSize = 6*sizeof(Float)*nLGN;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
        tauR = mem_block;
        tauD = tauR + nLGN;
		delay = tauD + nLGN;
        ratio = delay + nLGN;
        nR = ratio + nLGN;
        nD = nR + nLGN;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		cudaFree(mem_block);
	}
};

struct Cone_specific {
    Spatial_component spatial;
    Temporal_component temporal;

    void allocAndMemcpy(unsigned int nLGN, hcone_specific &host) {
        spatial.allocAndMemcpy(nLGN, host.spatial);
        temporal.allocAndMemcpy(nLGN, host.temporal);
    }
};

struct Static_nonlinear {
    Float* mem_block;

    Float* __restrict__ c50;
    Float* __restrict__ sharpness;
    Float* __restrict__ a;
    Float* __restrict__ b;

    void allocAndMemcpy(unsigned int nLGN, hstatic_nonlinear &host) {
        checkCudaErrors(cudaMalloc((void**)&mem_block, 4*nLGN*sizeof(Float)));
        c50 = memblock;
        sharpness = c50 + nLGN;
        a = sharpness + nLGN;
        b = a + nLGN;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, 4*nLGN*sizeof(Float), cudaMemcpyHostToDevice));
    }
	void freeMem() {
		cudaFree(mem_block);
	}
    // transform convolution result (input) with logistic function and return as firing rate
    __device__
    Float transform(unsigned int id, Float input) {
		// loads
        Float K = sharpness[id];
        Float C50 = c50[id];
        Float A = a[id];
        Float B = b[id];
        // calculation
        Float X = 1/(1+expp(-K*(input-C50)));
        return A*X + B;
    }
};

struct LGN_parameter {
    // block allocation
    Size nLGN;
    cone_specific center, surround;
    static_nonlinear logistic;

    SmallSize* mem_block;
    // 0: L
    // 1: M
    // 2: S
    // 3: L+M+S
    // 4: L+M
    // 5: M+S
    // 6: S+L
    SmallSize* __restrict__ centerType;
    SmallSize* __restrict__ surroundType;
    Float* __restrict__ covariant; // color in the surround and center ay covary
    
    LGN_parameter(hLGN_parameter &host) {
        nLGN = host.nLGN;
        center.allocAndMemcpy(nLGN, host.center);
        surround.allocAndMemcpy(nLGN, host.surround);
        logistic.allocAndMemcpy(nLGN, host.logistic);

        size_t memSize = (2*sizeof(SmallSize)+sizeof(Float))*nLGN;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));

        centerType = mem_block;
        surroundType = centerType + nLGN;
        covariant = (Float*) (surroundType + nLGN);
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		center.freeMem();
		surround.freeMem();
		logistic.freeMem();
		cudaFree(mem_block);
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
    void load(temporal_component &t, Size id) {
        // loading order corresponds to calculation order
        // think before change order of load
        nR = t.nR[id]; 
        nD = t.nD[id];
        delay = t.delay[id];
        tauR = t.tauR[id];
        tauD = t.tauD[id];
        ratio = t.ratio[id];
    }

}

struct Zip_spatial {
    Float x;
    Float y;
    Float rx;
    Float ry;
    Float k;
	
	__device__
    load(spatial_component &s, unsigned int id) {
        x = s.x[id];
        y = s.y[id];
        rx = s.rx[id];
        ry = s.ry[id];
        k = s.k[id];
    }
};

#endif
