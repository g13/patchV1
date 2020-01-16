#include "connect.h"
#include <cassert>
#include "../util/cuda_util.h"
#include "../types.h"

__global__ void initialize(curandStateMRG32k3a* __restrict__ state,
                           Size* __restrict__ preType,
                           Float* __restrict__ rden,
                           Float* __restrict__ raxn,
                           Float* __restrict__ dden,
                           Float* __restrict__ daxn,
                           Float* __restrict__ sTypeMat,
                           Float* __restrict__ pTypeMat,
                           Size* __restrict__ nTypeMat,
                           Float* __restrict__ preTypeS,
                           Float* __restrict__ preTypeP,
                           Size* __restrict__ preTypeN,
                           initialize_package init_pack, unsigned long long seed, Size networkSize) {
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
    
	#pragma unroll
    for (Size itype=0; itype<NTYPE; itype++) {
        if (threadIdx.x < init_pack.neuron_type_acc_count[itype+1]) {
            preType[id] = itype;
            rden[id] = init_pack.radius[itype][0];
            raxn[id] = init_pack.radius[itype][1];
            dden[id] = init_pack.den_den[itype];
            daxn[id] = init_pack.den_axn[itype];
            for (Size jtype=0; jtype<NTYPE; jtype++) {
                preTypeS[jtype*networkSize+id] = sTypeMat[itype*NTYPE+jtype];
                preTypeP[jtype*networkSize+id] = pTypeMat[itype*NTYPE+jtype];
                preTypeN[jtype*networkSize+id] = nTypeMat[itype*NTYPE+jtype];
            }
            return;
        }
    }
}

__device__ Float tri_cos(Float a, Float b, Float c) {
    return (a*a + b*b - c*c)/(2*a*b);
}

//__device__ Float seg(Float cosine, Float radius) {
//    return arccos(cosine)/(radius*radius);
//}

//__device__ Float chord(Float radius, Float cosine) {
//    Float r2 = radius*radius;
//    Float cos2 = cosine*cosine;
//    return square_root(r2- cos2*r2) * radius*cosine;
//}

__device__ Float area(Float raxn, Float rden, Float d) {
    Float cos_theta_axn = tri_cos(raxn, d, rden);
	Float cos_theta_den = tri_cos(rden, d, raxn);

    Float theta_axn = arccos(cos_theta_axn);
    Float theta_den = arccos(cos_theta_den);

    Float sin_theta_axn = sine(theta_axn);
    Float sin_theta_den = sine(theta_den);

    return (theta_axn-sin_theta_axn*cos_theta_axn)*raxn*raxn 
         + (theta_den-sin_theta_den*cos_theta_den)*rden*rden;
}

// co-occupied area of the presynaptic axons / dendritic area
__device__ Float connect(Float distance, Float raxn, Float rden) {
    Float weight = 0.0;
    if (raxn + rden > distance && distance > abs(raxn - rden)) {
        weight = area(raxn, rden, distance)/(CUDA_PI*rden*rden);
    } else if (distance <= abs(raxn - rden)) {
        weight = 1.0;
    }
    __syncwarp();
    return weight;
}

__global__ void cal_blockPos(Float* __restrict__ pos,
                             Float* __restrict__ block_x,
                             Float* __restrict__ block_y,
                             Size networkSize) {
    __shared__ Float x[warpSize];
    __shared__ Float y[warpSize];
    Size id = blockDim.x*blockIdx.x + threadIdx.x;
    block_reduce<Float>(x, pos[id]);
    if (threadIdx.x == 0) {
        block_x[blockIdx.x] = x[0]/blockSize;
    }
    block_reduce<Float>(y, pos[networkSize + id]);
    if (threadIdx.x == 0) {
        block_y[blockIdx.x] = y[0]/blockSize;
    }
}

__device__ void compare_distance_with_neighbor_block(Size* __restrict__ iNeighbor, 
													 Float bx,
													 Float by,
													 Float* __restrict__ block_x, 
													 Float* __restrict__ block_y, 
													 Size* __restrict__ neighborBlockId, 
													 Size offset, Size nPotentialNeighbor, Float radius) {
    Size blockId = offset + threadIdx.x;
    Float x = block_x[blockId] - bx;
    Float y = block_y[blockId] - by;
    Float distance = square_root(x*x + y*y);
    if (distance < radius && blockId != blockIdx.x) {
        Size current_index = atomicAdd(iNeighbor, 1);
		if (current_index >= nPotentialNeighbor) {
		}
        neighborBlockId[nPotentialNeighbor*blockIdx.x + current_index] = blockId;
    }
}

__global__ void get_neighbor_blockId(Float* __restrict__ block_x,
                                     Float* __restrict__ block_y,
                                     Size* __restrict__ neighborBlockId,
                                     Size* __restrict__ nNeighborBlock,
                                     Float max_radius, Size nPotentialNeighbor) {
    __shared__ Size iNeighbor[1];
    iNeighbor[0] = 0;
    Float bx = block_x[blockIdx.x]; // center of the target block
    Float by = block_y[blockIdx.x];
    tid = threadIdx.y*blockDim.x + threadIdx.x;
    Size nblock = gridDim.x;
    Size nPatch = nblock/nblockSize;
    Size remain = nblock%blockSize;

    Size offset = 0;
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        if (iPatch < nPatch || tid < remain) {
            compare_distance_with_neighbor_block(iNeighbor, bx, by, block_x, block_y, neighborBlockId, offset, nPotentialNeighbor, max_radius);
        }
        if (iPatch < nPatch) {
            offset += blockSize;
        }
    }
    __syncthreads();
    if (tid == 0) {
        nNeighbor = iNeighbor[0]+1;
        if (nNeighbor > nPotentialNeighbor) {
            printf("actual nNeighbor = %d > %d (preserved)\n", nNeighbor, nPotentialNeighbor);
		    assert(nNeighbor <= nPotentialNeighbor);
        }
        nNeighborBlock[blockIdx.x] = nNeighbor;
    }
}

__global__ void generate_connections(Float* __restrict__ pos,
                                     Float* __restrict__ preTypeS,
                                     Float* __restrict__ preTypeP,
                                     Size* __restrict__ preTypeN,
                                     Size* __restrict__ neighborBlockId,
                                     Size* __restrict__ nNeighborBlock,
                                     Float* __restrict__ rden,
                                     Float* __restrict__ raxn,
                                     Float* __restrict__ conMat, //within block connections
                                     Float* __restrict__ delayMat,
                                     Float* __restrict__ conVec, //for neighbor block connections
                                     Float* __restrict__ delayVec, //for neighbor block connections
                                     Size* __restrict__ vecID,
                                     Size* __restrict__ nVec,
                                     Size* __restrict__ preTypeConnected,
                                     Size* __restrict__ preTypeAvail,
                                     Float* __restrict__ preTypeStrSum,
                                     Size* __restrict__ preType,
                                     Float* __restrict__ dden,
                                     Float* __restrict__ daxn,
                                     curandStateMRG32k3a* __restrict__ state,
                                     Size networkSize, Size neighborSize, Size nPotentialNeighbor, Float speedOfThought) {
    __shared__ Float x1[blockSize];
    __shared__ Float y1[blockSize];
    __shared__ Size ipreType[blockSize];
    //Float* tempNeighbor = new Float[];
    Size offset = blockIdx.x*blockSize;
    Size id = offset + threadIdx.x;
    Size nb = nNeighborBlock[blockIdx.x]*blockSize;
    Float* tempNeighbor = new Float[nb];
    if (!tempNeighbor && nb > 0) {
        printf("#%u not allocated, requring %f Kb for %u units\n", id, nb*sizeof(Float)/1024.0, networkSize);
        return;
    }
    Size sumConType[NTYPE];
    Size sumType[NTYPE];
    Float sumP[NTYPE];
    Float sumStrType[NTYPE];
    curandStateMRG32k3a localState = state[id];
    Size itype = preType[id];
    Float x0 = pos[id];
    Float y0 = pos[networkSize + id];
    ipreType[threadIdx.x] = itype;
    Float rd = rden[id];
    #pragma unroll
    for (Size i=0; i<NTYPE; i++) {
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
    for (Size i=0; i<blockSize; i++) {
        //matrix, indexed within one block
        Size ipre = blockIdx.x*blockSize + i;
        //type vector, indexed across the network
        Float x = x1[i] - x0;
        Float y = y1[i] - y0;
        Float ra = raxn[ipre];
        Float distance = square_root(x*x + y*y);
		// weight from area
        Float p = connect(distance, ra, rd);

        if (p > 0) {
            unsigned long bid = ipre*blockSize + threadIdx.x;
            Size ip = ipreType[i];
            sumType[ip] += 1;
			// update weight with density of axon dendrites and preference over type
            p = p * daxn[ipre] * dden[id] * preTypeP[ip*networkSize + id];
            sumP[ip] += p;
            conMat[bid] = p;
            delayMat[bid] = distance/speedOfThought;
        }
    }
    for (Size i=0; i<nNeighborBlock[blockIdx.x]; i++) {
        Size bid = neighborBlockId[nPotentialNeighbor*blockIdx.x + i];
        #pragma unroll
        for (Size j=0; j<blockSize; j++) {
            // index in the network
            Size ipre = bid*blockSize + j;
            // index in conVec
            Float x = pos[ipre] - x0;
            Float y = pos[networkSize+ipre] - y0;
            Float ra = raxn[ipre];
            Float distance = square_root(x*x + y*y);
            Float p = connect(distance, ra, rd);
            //if (id == 7072 && p == 0) {
            //    printf("o:(%f,%f) <- %f, a:%f, d:%f, p=%f\n", x0, y0, distance, ra, rd, p);
            //    printf("sumType > 0\n");
            //}
            unsigned long tid = i*blockSize + j;
            tempNeighbor[tid] = 0;
            if (p > 0) {
                Size ip = preType[ipre];
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
    for (Size i=0; i<blockSize; i++) {
        Size ipre = blockIdx.x*blockSize + i;
        unsigned long bid = ipre*blockSize + threadIdx.x;
        Size ip = ipreType[i];
        Float str = preTypeS[ip*networkSize+id];
        Float p = conMat[bid]/sumP[ip]*preTypeN[ip*networkSize+id];
        Float xrand = uniform(&localState);
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
    
    Size nid = 0;
    for (Size i=0; i<nNeighborBlock[blockIdx.x]; i++) {
        Size bid = neighborBlockId[nPotentialNeighbor*blockIdx.x + i];
        #pragma unroll
        for (Size j=0; j<blockSize; j++) {
            Size ipre = bid*blockSize + j;
            unsigned long tid = i*blockSize + j;
            Size ip = preType[ipre];
            Float str = preTypeS[ip*networkSize+id];
            Float p = tempNeighbor[tid]/sumP[ip]*preTypeN[ip*networkSize+id];
            Float xrand = uniform(&localState);
            if (xrand < p) {
                if (p > 1) {
                    str = str*p;
                }
                sumConType[ip] += 1;
                sumStrType[ip] += str;
                vecID[neighborSize*id + nid] = ipre;
                conVec[neighborSize*id + nid] = str;
                Float x = pos[ipre] - x0;
                Float y = pos[networkSize+ipre] - y0;
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
    for (Size i=0; i<NTYPE; i++) {
        preTypeConnected[i*networkSize + id] = sumConType[i];
        preTypeAvail[i*networkSize + id] = sumType[i];
        preTypeStrSum[i*networkSize + id] = sumStrType[i];
    }
    delete []tempNeighbor;
}
