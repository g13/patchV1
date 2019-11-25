#ifndef LGN_PROPS_CUH
#define LGN_PROPS_CUH

// Array structure: (type, nLGN), different from spatial and temporal weight storage, see "discrete_input_convol.cu->store_weight" which are (nLGN, type).
// This is to optimize read and write in CUDA

struct Spatial_component {
    Float* mem_block;
    Float* __restrict__ x; // normalize to (0,1)
    Float* __restrict__ rx;
    Float* __restrict__ y; // normalize to (0,1)
    Float* __restrict__ ry;
    Float* __restrict__ k; // its sign determine On-Off

    void allocAndMemcpy(Size arraySize, hSpatial_component &host) {
		size_t memSize = 5*arraySize*sizeof(Float) ;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));
        x = mem_block;
        rx = x + arraySize;
        y = rx + arraySize;
        ry = y + arraySize;
        k = ry + arraySize;
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
		cudaFree(mem_block);
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
        c50 = memblock;
        sharpness = c50 + arraySize;
        a = sharpness + arraySize;
        b = a + arraySize;
        checkCudaErrors(cudaMemcpy(mem_block, host.mem_block, memSize, cudaMemcpyHostToDevice));
    }
	void freeMem() {
		cudaFree(mem_block);
	}
    // transform convolution result (input) with logistic function and return as firing rate
    __device__
    Float transform(unsigned int id, Float input) {
		// load from global memory
        Float C50 = c50[id];
        Float K = sharpness[id];
        Float A = a[id];
        Float B = b[id];
        // calculation
        Float X = 1/(1+expp(K*(C50-input)));
        return A*X + B;
    }
};

// collect all the components and send to device
struct LGN_parameter {
    Spatial_component spatial;
    Temporal_component temporal;
    Static_nonlinear logistic;

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
        temporal.allocAndMemcpy(arraySize, host.temporal);
        spatial.allocAndMemcpy(arraySize, host.spatial);
        logistic.allocAndMemcpy(nLGN, host.logistic);

        size_t memSize = arraySize*sizeof(SmallSize)+sizeof(Float)*(nType-1)*nLGN;
        checkCudaErrors(cudaMalloc((void**)&mem_block, memSize));

        coneType = mem_block;
        covariant = (Float*) (coneType + arraySize);
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
