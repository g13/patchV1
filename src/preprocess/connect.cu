#include "connect.h"
#include <cassert>
#include "../util/cuda_util.h"

__global__ void initialize(curandStateMRG32k3a* __restrict__ state,
                           unsigned int* __restrict__ preType,
                           _float* __restrict__ rden,
                           _float* __restrict__ raxn,
                           _float* __restrict__ dden,
                           _float* __restrict__ daxn,
                           _float* __restrict__ sTypeMat,
                           _float* __restrict__ pTypeMat,
                           unsigned int* __restrict__ nTypeMat,
                           _float* __restrict__ preTypeS,
                           _float* __restrict__ preTypeP,
                           unsigned int* __restrict__ preTypeN,
                           initialize_package init_pack, unsigned long long seed, unsigned int networkSize) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
    
	#pragma unroll
    for (unsigned int itype=0; itype<NTYPE; itype++) {
        if (threadIdx.x < init_pack.neuron_type_acc_count[itype+1]) {
            preType[id] = itype;
            rden[id] = init_pack.radius[itype][0];
            raxn[id] = init_pack.radius[itype][1];
            dden[id] = init_pack.den_den[itype];
            daxn[id] = init_pack.den_axn[itype];
            for (unsigned int jtype=0; jtype<NTYPE; jtype++) {
                preTypeS[jtype*networkSize+id] = sTypeMat[itype*NTYPE+jtype];
                preTypeP[jtype*networkSize+id] = pTypeMat[itype*NTYPE+jtype];
                preTypeN[jtype*networkSize+id] = nTypeMat[itype*NTYPE+jtype];
            }
            return;
        }
    }
}

__device__ _float tri_cos(_float a, _float b, _float c) {
    return (a*a + b*b - c*c)/(2*a*b);
}

//__device__ _float seg(_float cosine, _float radius) {
//    return arccos(cosine)/(radius*radius);
//}

//__device__ _float chord(_float radius, _float cosine) {
//    _float r2 = radius*radius;
//    _float cos2 = cosine*cosine;
//    return square_root(r2- cos2*r2) * radius*cosine;
//}

__device__ _float area(_float raxn, _float rden, _float d) {
    _float cos_theta_axn = tri_cos(raxn, d, rden);
	_float cos_theta_den = tri_cos(rden, d, raxn);

    _float theta_axn = arccos(cos_theta_axn);
    _float theta_den = arccos(cos_theta_den);

    _float sin_theta_axn = sine(theta_axn);
    _float sin_theta_den = sine(theta_den);

    return (theta_axn-sin_theta_axn*cos_theta_axn)*raxn*raxn 
         + (theta_den-sin_theta_den*cos_theta_den)*rden*rden;
}

// co-occupied area of the presynaptic axons / dendritic area
__device__ _float connect(_float distance, _float raxn, _float rden) {
    _float weight = 0.0;
    if (raxn + rden > distance && distance > abs(raxn - rden)) {
        weight = area(raxn, rden, distance)/(CUDA_PI*rden*rden);
    } else if (distance <= abs(raxn - rden)) {
        weight = 1.0;
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
    if (distance < radius && blockId != blockIdx.x) {
        unsigned int current_index = atomicAdd(bid, 1);
		//if (current_index >= nPotentialNeighbor/2) {
		//	assert(current_index < nPotentialNeighbor);
		//}
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
    __syncthreads();
    if (threadIdx.x == 0) {
        nNeighborBlock[blockIdx.x] = bid[0];
        //printf("block #%u: %u neighbors: %u, %u, %u, %u, %u, %u, %u, %u, %u\n", blockIdx.x, bid[0], neighborBlockId[blockIdx.x*nPotentialNeighbor+0], neighborBlockId[blockIdx.x*nPotentialNeighbor+1], neighborBlockId[blockIdx.x*nPotentialNeighbor+2], neighborBlockId[blockIdx.x*nPotentialNeighbor+3],neighborBlockId[blockIdx.x*nPotentialNeighbor+4],neighborBlockId[blockIdx.x*nPotentialNeighbor+5],neighborBlockId[blockIdx.x*nPotentialNeighbor+6],neighborBlockId[blockIdx.x*nPotentialNeighbor+7],neighborBlockId[blockIdx.x*nPotentialNeighbor+8]);
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
                                     unsigned int* __restrict__ nVec,
                                     unsigned int* __restrict__ preTypeConnected,
                                     unsigned int* __restrict__ preTypeAvail,
                                     _float* __restrict__ preTypeStrSum,
                                     unsigned int* __restrict__ preType,
                                     _float* __restrict__ dden,
                                     _float* __restrict__ daxn,
                                     curandStateMRG32k3a* __restrict__ state,
                                     unsigned int networkSize, unsigned int neighborSize, unsigned int nPotentialNeighbor, _float speedOfThought) {
    __shared__ _float x1[blockSize];
    __shared__ _float y1[blockSize];
    __shared__ unsigned int ipreType[blockSize];
    //_float* tempNeighbor = new _float[];
    unsigned int offset = blockIdx.x*blockSize;
    unsigned int id = offset + threadIdx.x;
    unsigned int nb = nNeighborBlock[blockIdx.x]*blockSize;
    _float* tempNeighbor = new _float[nb];
    if (!tempNeighbor && nb > 0) {
        printf("#%u not allocated, requring %f Kb for %u units\n", id, nb*sizeof(_float)/1024.0, networkSize);
        return;
    }
    unsigned int sumConType[NTYPE];
    unsigned int sumType[NTYPE];
    _float sumP[NTYPE];
    _float sumStrType[NTYPE];
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
        _float x = x1[i] - x0;
        _float y = y1[i] - y0;
        _float ra = raxn[ipre];
        _float distance = square_root(x*x + y*y);
		// weight from area
        _float p = connect(distance, ra, rd);

        if (p > 0) {
            unsigned long bid = ipre*blockSize + threadIdx.x;
            unsigned int ip = ipreType[i];
            sumType[ip] += 1;
			// update weight with density of axon dendrites and preference over type
            p = p * daxn[ipre] * dden[id] * preTypeP[ip*networkSize + id];
            sumP[ip] += p;
            conMat[bid] = p;
            delayMat[bid] = distance/speedOfThought;
        }
    }
    for (unsigned int i=0; i<nNeighborBlock[blockIdx.x]; i++) {
        unsigned int bid = neighborBlockId[nPotentialNeighbor*blockIdx.x + i];
        #pragma unroll
        for (unsigned int j=0; j<blockSize; j++) {
            // index in the network
            unsigned int ipre = bid*blockSize + j;
            // index in conVec
            _float x = pos[ipre] - x0;
            _float y = pos[networkSize+ipre] - y0;
            _float ra = raxn[ipre];
            _float distance = square_root(x*x + y*y);
            _float p = connect(distance, ra, rd);
            //if (id == 7072 && p == 0) {
            //    printf("o:(%f,%f) <- %f, a:%f, d:%f, p=%f\n", x0, y0, distance, ra, rd, p);
            //    printf("sumType > 0\n");
            //}
            unsigned long tid = i*blockSize + j;
            tempNeighbor[tid] = 0;
            if (p > 0) {
                unsigned int ip = preType[ipre];
                sumType[ip] += 1;
                p = p * daxn[ipre] * dden[id] * preTypeP[ip*networkSize+id];
                sumP[ip] += p;
                tempNeighbor[tid] = p;
            }
        }
    }
    __syncwarp();

    //============= redistribute p of all ==========
    #pragma unroll
    for (unsigned int i=0; i<blockSize; i++) {
        unsigned int ipre = blockIdx.x*blockSize + i;
        unsigned long bid = ipre*blockSize + threadIdx.x;
        unsigned int ip = ipreType[i];
        _float str = preTypeS[ip*networkSize+id];
        _float p = conMat[bid]/sumP[ip]*preTypeN[ip*networkSize+id];
        _float xrand = uniform(&localState);
        if (xrand < p) {
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
        unsigned int bid = neighborBlockId[nPotentialNeighbor*blockIdx.x + i];
        #pragma unroll
        for (unsigned int j=0; j<blockSize; j++) {
            unsigned int ipre = bid*blockSize + j;
            unsigned long tid = i*blockSize + j;
            unsigned int ip = preType[ipre];
            _float str = preTypeS[ip*networkSize+id];
            _float p = tempNeighbor[tid]/sumP[ip]*preTypeN[ip*networkSize+id];
            _float xrand = uniform(&localState);
            if (xrand < p) {
                if (p > 1) {
                    str = str*p;
                }
                sumConType[ip] += 1;
                sumStrType[ip] += str;
                vecID[neighborSize*id + nid] = ipre;
                conVec[neighborSize*id + nid] = str;
                _float x = pos[ipre] - x0;
                _float y = pos[networkSize+ipre] - y0;
                delayVec[neighborSize*id + nid] = square_root(x*x + y*y)/speedOfThought;
                nid += 1;
                if (nid > neighborSize) {
                    printf("set bigger neighborSize, currently %u\n", neighborSize);
                    assert(nid <= neighborSize);
                }
            }
        }
    }
    nVec[id] = nid;
    #pragma unroll
    for (unsigned int i=0; i<NTYPE; i++) {
        preTypeConnected[i*networkSize + id] = sumConType[i];
        preTypeAvail[i*networkSize + id] = sumType[i];
        preTypeStrSum[i*networkSize + id] = sumStrType[i];
    }
    delete []tempNeighbor;
}
