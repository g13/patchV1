#include "connect.h"
#include <cassert>
#include "cuda_util.h"

#ifdef SINGLE_PRECISION
	//using func = _float(*)(_float);
	//__device__ func expp = &expf;
	//using func0 = _float(*)(curandStateMRG32k3a_t*);
	//__device__ func0 uniform = &curand_uniform;
	//__device__ func0 normal = &curand_normal;
	//using func1 = _float(*)(curandStateMRG32k3a_t*, _float, _float);
	//__device__ func1 log_normal = &curand_log_normal;
	//using func2 = _float(*)(_float);
    //__device__ func2 arccos = &acosf;
    //__device__ func2 square_root = &sqrtf;
    //__device__ func2 abs_value = &fabsf;

    #define expp exp 
    #define log_normal curand_log_normal_double
    #define normal curand_normal_double
    #define uniform curand_uniform_double
    #define arccos acos 
    #define square_root sqrt
    #define abs_value fabs 
#else
	//using func = _float(*)(_float);
	//__device__ func expp = &exp;
	//using func0 = _float(*)(curandStateMRG32k3a_t*);
	//__device__ func0 uniform = &curand_uniform_double;
	//__device__ func0 normal = &curand_normal_double;
	//using func1 = _float(*)(curandStateMRG32k3a_t*, _float, _float);
	//__device__ func1 log_normal = &curand_log_normal_double;
	//using func2 = _float(*)(_float);
    //__device__ func2 arccos = &acos;
    //__device__ func2 square_root = &sqrt;
    //__device__ func2 abs_value = &fabs;

    #define expp expf
    #define log_normal curand_log_normal
    #define normal curand_normal
    #define uniform curand_uniform
    #define arccos acosf 
    #define square_root sqrtf
    #define abs_value fabsf 
#endif

__global__ void initialize(curandStateMRG32k3a* __restrict__ state,
                           unsigned int* __restrict__ preType,
                           _float* __restrict__ rden,
                           _float* __restrict__ raxn,
                           _float* __restrict__ preTypeDaxn,
                           _float* __restrict__ preTypeDend,
                           _float* __restrict__ sTypeMat,
                           _float* __restrict__ pTypeMat,
                           unsigned int* __restrict__ cTypeMat,
                           _float* __restrict__ preTypeS,
                           _float* __restrict__ preTypeP,
                           unsigned int* __restrict__ preTypeN,
                           initialize_package init_pack, unsigned long long seed, unsigned int networkSize) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
    
    unsigned int itype;
    _float rd;
    _float ra;
    _float da;
    _float dd;
	#pragma unroll
    for (itype=0; itype<NTYPE; itype++) {
        if (id < init_pack.neuron_type_acc_count[itype+1]) {
            rd = init_pack.radius[itype][0];
            ra = init_pack.radius[itype][1];
            da = init_pack.den_axn[itype];
            dd = init_pack.den_den[itype];
            break;
        }
    }
    __syncwarp();
    preType[id] = itype;
    rden[id] = rd;
    raxn[id] = ra;
    preTypeDaxn[id] = da;
    preTypeDend[id] = dd;

    for (unsigned int jtype=0; jtype<NTYPE; jtype++) {
        preTypeS[jtype*networkSize+id] = sTypeMat[itype*NTYPE+jtype];
        preTypeP[jtype*networkSize+id] = pTypeMat[itype*NTYPE+jtype];
        preTypeN[jtype*networkSize+id] = cTypeMat[itype*NTYPE+jtype];
    }
}

__device__ _float tri_cos(_float a, _float b, _float c) {
    return (a*a + b*b - c*c)/(2*a*b);
}

__device__ _float seg(_float cosine, _float radius) {
    return arccos(cosine)/(radius*radius);
}

__device__ _float chord(_float radius, _float cosine) {
    _float r2 = radius*radius;
    _float cos2 = cosine*cosine;
    return square_root(r2- cos2*r2) * radius*cosine;
}

__device__ _float area(_float raxn, _float rden, _float d) {
    _float cos_theta_axn = tri_cos(raxn, d, rden);
	_float cos_theta_den = tri_cos(rden, d, raxn);
	_float seg_axn = seg(cos_theta_axn, raxn);
	_float seg_den = seg(cos_theta_den, rden);
	_float chord_axn = chord(cos_theta_axn, raxn);
	_float chord_den = chord(cos_theta_den, rden);
    return seg_axn+seg_den-chord_axn-chord_den;
}

__device__ _float connect(_float distance, _float raxn, _float rden) {
    _float weight = 0;
    if (raxn +  rden > distance && distance > abs_value(raxn - rden)) {
        weight = area(raxn, rden, distance)/(CUDA_PI*rden*rden);
    }
    __syncwarp();
    return weight;
}

__global__ void cal_blockPos(_float* __restrict__ pos,
                             _float* __restrict__ block_x,
                             _float* __restrict__ block_y,
                             unsigned int networkSize) {
    __shared__ _float x[warpSize];
    __shared__ _float y[warpSize];
    unsigned int id = blockDim.x*blockIdx.x + threadIdx.x;
    block_reduce<_float>(x, pos[id]);
    if (threadIdx.x == 0) {
        block_x[blockIdx.x] = x[0]/blockSize;
    }
    block_reduce<_float>(y, pos[networkSize + id]);
    if (threadIdx.x == 0) {
        block_y[blockIdx.x] = y[0]/blockSize;
    }
}

__device__ void compare_distance_with_neighbor_block(unsigned int* __restrict__ bid, 
													 _float bx,
													 _float by,
													 _float* __restrict__ block_x, 
													 _float* __restrict__ block_y, 
													 unsigned int* __restrict__ neighborBlockId, 
													 unsigned int offset, unsigned int nPotentialNeighbor, _float radius) {
    unsigned int blockId = offset + threadIdx.x;
    _float x = block_x[blockId] - bx;
    _float y = block_y[blockId] - by;
    _float distance = square_root(x*x + y*y);
    if (distance < radius) {
        unsigned int current_index = atomicAdd(bid, 1);
		if (current_index >= nPotentialNeighbor/2) {
			printf("%u: current index = %u, max = %u\n", blockId, current_index, nPotentialNeighbor);
			assert(current_index < nPotentialNeighbor);
		}
        
        neighborBlockId[nPotentialNeighbor*blockIdx.x + current_index] = blockId;
    }
}

__global__ void get_neighbor_blockId(_float* __restrict__ block_x,
                                     _float* __restrict__ block_y,
                                     unsigned int* __restrict__ neighborBlockId,
                                     unsigned int* __restrict__ nNeighborBlock,
                                     _float max_radius, unsigned int nPotentialNeighbor) {
    __shared__ unsigned int bid[1];
    bid[0] = 0;
    _float bx = block_x[blockIdx.x];
    _float by = block_y[blockIdx.x];
    unsigned int lefted = gridDim.x;
    unsigned int offset = 0;
    while (lefted >= blockSize) {
        lefted -= blockSize;
        compare_distance_with_neighbor_block(bid, bx, by, block_x, block_y, neighborBlockId, offset, nPotentialNeighbor, max_radius);
        offset += blockSize;
    }
    if (threadIdx.x < lefted) {
        compare_distance_with_neighbor_block(bid, bx, by, block_x, block_y, neighborBlockId, offset, nPotentialNeighbor, max_radius);
    }
    if (threadIdx.x == 0) {
        nNeighborBlock[blockIdx.x] = bid[0];
    }
}

__global__ void generate_connections(_float* __restrict__ pos,
                                     _float* __restrict__ preTypeS,
                                     _float* __restrict__ preTypeP,
                                     unsigned int* __restrict__ preTypeN,
                                     unsigned int* __restrict__ neighborBlockId,
                                     unsigned int* __restrict__ nNeighborBlock,
                                     _float* __restrict__ rden,
                                     _float* __restrict__ raxn,
                                     _float* __restrict__ conMat, //within block connections
                                     _float* __restrict__ delayMat,
                                     _float* __restrict__ conVec, //for neighbor block connections
                                     _float* __restrict__ delayVec, //for neighbor block connections
                                     unsigned int* __restrict__ vecID,
                                     unsigned int* __restrict__ preTypeConnected,
                                     unsigned int* __restrict__ preTypeAvail,
                                     _float* __restrict__ preTypeStrSum,
                                     unsigned int* __restrict__ preType,
                                     _float* __restrict__ preTypeDaxn,
                                     _float* __restrict__ preTypeDend,
                                     curandStateMRG32k3a* __restrict__ state,
                                     unsigned int networkSize, unsigned int neighborSize, unsigned int nPotentialNeighbor, _float speedOfThought) {
    __shared__ _float x1[blockSize];
    __shared__ _float y1[blockSize];
    __shared__ unsigned int ipreType[blockSize];
    _float* tempNeighbor = new _float[blockSize*nPotentialNeighbor];
    unsigned int sumConType[NTYPE];
    unsigned int sumType[NTYPE];
    unsigned int sumP[NTYPE];
    _float sumStrType[NTYPE];
    unsigned int offset = blockIdx.x*blockSize;
    unsigned int id = offset + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    unsigned int itype = preType[id];
    _float x0 = pos[id];
    _float y0 = pos[networkSize + id];
    ipreType[threadIdx.x] = itype;
    _float rd = rden[id];
    #pragma unroll
    for (unsigned int i=0; i<NTYPE; i++) {
        sumConType[i] = 0;
        sumStrType[i] = 0;
        sumType[i] = 0;
        sumP[i] = 0;
    }
    x1[threadIdx.x] = x0;
    y1[threadIdx.x] = y0;
    __syncthreads();
    //============= collect p of all ==========
    // withhin block
    #pragma unroll
    for (unsigned int i=0; i<blockSize; i++) {
        //matrix, indexed within one block
        unsigned int ipre = blockIdx.x*blockSize + i;
        //type vector, indexed across the network
        unsigned long bid = ipre*blockSize + threadIdx.x;
        unsigned int ip = ipreType[i];
        sumType[ip] += 1;
        _float x = x1[i] - x0;
        _float y = y1[i] - y0;
        _float ra = raxn[ipre];
        _float distance = square_root(x*x + y*y);
        _float p = connect(distance, ra, rd) * preTypeDaxn[id] * preTypeDend[id] * preTypeP[ip*networkSize + id];
        sumP[ip] += p;
        conMat[bid] = p;
        delayMat[bid] = distance/speedOfThought;
    }
    for (unsigned int i=0; i<nNeighborBlock[blockIdx.x]; i++) {
        #pragma unroll
        for (unsigned int j=0; j<blockSize; j++) {
            // index in the network
            unsigned int ipre = neighborBlockId[nPotentialNeighbor*blockIdx.x + i]*blockSize + j;
            // index in conVec
            unsigned long bid = i*blockSize + j;
            unsigned int ip = preType[ipre];
            sumType[ip] += 1;
            _float x = pos[ipre] - x0;
            _float y = pos[networkSize+ipre] - y0;
            _float ra = raxn[ipre];
            _float distance = square_root(x*x + y*y);
            _float p = connect(distance, ra, rd) * preTypeDaxn[id] * preTypeDend[id] * preTypeP[ip*networkSize+id];
            sumP[ip] += p;
            printf("bid = %u", bid);
            tempNeighbor[bid] = p;
        }
    }
    //============= redistribute p of all ==========
    #pragma unroll
    for (unsigned int i=0; i<blockSize; i++) {
        unsigned int ipre = blockIdx.x*blockSize + i;
        unsigned long bid = ipre*blockSize + threadIdx.x;
        unsigned int ip = ipreType[ipre];
        _float str = preTypeS[ip*networkSize+id];
        _float p = conMat[bid]/sumP[ip]*preTypeN[ip*networkSize+id];
        if (uniform(&localState) < p) {
            if (p > 1) {
                str = str*p;
            }
            p = str;
            sumConType[ip] += 1;
            sumStrType[ip] += str;
        } else {
            p = 0;
        }
        __syncwarp();
        conMat[bid] = p;
    }
    
    unsigned int nid = 0;
    for (unsigned int i=0; i<nNeighborBlock[blockIdx.x]; i++) {
        #pragma unroll
        for (unsigned int j=0; j<blockSize; j++) {
            unsigned int ipre = neighborBlockId[nPotentialNeighbor*blockIdx.x + i]*blockSize + j;
            unsigned long bid = i*blockSize + j;
            unsigned int ip = preType[ipre];
            _float str = preTypeS[ip*networkSize+id];
            _float p = tempNeighbor[bid]/sumP[ip]*preTypeN[ip*networkSize+id];
            if (uniform(&localState) < p) {
                if (p > 1) {
                    str = str*p;
                }
                sumConType[ip] += 1;
                sumStrType[ip] += str;
                conVec[id*neighborSize + nid] = str;
                vecID[nid] = ipre;
                conVec[nid] = str;
                nid += 1;
            }
        }
    }
    #pragma unroll
    for (unsigned int i=0; i<NTYPE; i++) {
        preTypeConnected[i*networkSize + id] = sumConType[i];
        preTypeAvail[i*networkSize + id] = sumType[i];
        preTypeStrSum[i*networkSize + id] = sumStrType[i];
    }
    delete []tempNeighbor;
}
