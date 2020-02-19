#include "connect.h"
#include <cassert>
#include "../util/cuda_util.cuh"
#include "../types.h"

extern __device__ __constant__ pFeature pref[];

// TODO: randomize neuronal attributes by using distribution, strength x number of con. should be controlled
__global__ 
void initialize(curandStateMRG32k3a* __restrict__ state,
                Size*  __restrict__ preType, // 
                Float* __restrict__ rden,
                Float* __restrict__ raxn,
                Float* __restrict__ dden,
                Float* __restrict__ daxn,
                Float* __restrict__ preS_type,
                Float* __restrict__ preP_type,
                Size*  __restrict__ preN,
				Size* __restrict__ preFixType, // nSubHierarchy * networkSize
                initialize_package init_pack, unsigned long long seed, Size networkSize, Size nType, Size nArchtype, Size nSubHierarchy) 
{
    //__shared__ reduced[warpSize];
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
   	Size type;
	// determine the arch neuronal type and its properties
	#pragma unroll
    for (Size i=0; i<nArchtype; i++) {
        if (threadIdx.x < init_pack.archtypeAccCount[i]) {
			type = i;
            rden[id] = init_pack.rden[i];
            raxn[id] = init_pack.raxn[i];
            dden[id] = init_pack.dden[i];
            daxn[id] = init_pack.daxn[i];
            preN[id] = init_pack.preTypeN[i];
            break;
        }
	}
	// determine subTypes
    Size nType_remain = nType/nArchtype;
	type = nType_remain * type;
	for (Size i=0; i<nSubHierarchy; i++) {
		Size subType = preFixType[i*networkSize + id];
	    nType_remain /= init_pack.nTypeHierarchy[i+1];
	    type += nType_remain * subType;
    }
    preType[id] = type;
	for (Size i=0; i<nType; i++) {
		Size tid = i*networkSize+id;
		Size ttid = i*nType + type;
    	preS_type[tid] = init_pack.sTypeMat[ttid];
    	preP_type[tid] = init_pack.pTypeMat[ttid];
	}
}

__device__ 
__forceinline__
Float tri_cos(Float a, Float b, Float c) {
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

__device__ 
__forceinline__
Float area(Float raxn, Float rden, Float d) {
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
__device__ 
__forceinline__
Float connect(Float distance, Float raxn, Float rden, bool gaussian_profile) {
	Float weight;
	if (gaussian_profile) {
		Float spread = raxn*raxn + rden*rden;
        if (distance < 3*square_root(spread/2)) {
		    weight = exponential(-distance*distance/spread)/(M_PI*spread);
        } else {
            weight = 0.0;
        }
	} else {
    	weight = 0.0;
    	if (raxn + rden > distance && distance > abs(raxn - rden)) {
    	    weight = area(raxn, rden, distance)/(M_PI*rden*rden); // conn. prob. is defined by the presynaptic point of view
    	} else if (distance <= abs(raxn - rden)) {
    	    weight = 1.0;
    	}
	}
    return weight;
}

__global__ 
void cal_blockPos(double* __restrict__ pos,
                  Float* __restrict__ block_x,
                  Float* __restrict__ block_y,
                  Size networkSize) 
{
    __shared__ double reduced[warpSize];
    Size id = (2*blockDim.x)*blockIdx.x + threadIdx.x;
    double x = pos[id];
    double y = pos[id + blockDim.x];
    block_reduce<double>(reduced, x);
    if (threadIdx.x == 0) {
        block_x[blockIdx.x] = static_cast<Float>(reduced[0]/blockDim.x);
    }
    block_reduce<double>(reduced, y);
    if (threadIdx.x == 0) {
        block_y[blockIdx.x] = static_cast<Float>(reduced[0]/blockDim.x);
    }
}

__global__ 
void get_neighbor_blockId(Float* __restrict__ block_x,
                          Float* __restrict__ block_y,
                          Size* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
                          Float max_radius, Size maxNeighborBlock) 
{
    __shared__ PosInt id[warpSize];
    __shared__ Float min[warpSize];
    __shared__ PosInt bid[nblock];
    __shared__ Float distance[nblock];


    Float bx = block_x[blockIdx.x]; // center of the target block
    Float by = block_y[blockIdx.x];
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;

    Size nblock = gridDim.x;
    Size nPatch = (nblock + blockDim.x-1)/blockDim.x - 1;
    Size remain = nblock%blockDim.x;

    Size offset = 0;
    Size nid = 0;
    Float dis;
    Size blockId;
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        if (iPatch < nPatch || tid < remain) {
            blockId = offset + threadIdx.x;
            Float x = block_x[blockId] - bx;
            Float y = block_y[blockId] - by;
            dis = square_root(x*x + y*y);
            if (dis <= max_radius) { // get blocks inside radius
                if (tid == 0) {
                    if (nid >= nblock) {
                        printf("nearby block > %u\n", blockSize);
                        assert(nid <= nblock);
                    }
                    distance[nid] = dis;
                    bid[nid] = blockId;
                }
                nid++;
            }
        }
        if (iPatch < nPatch) {
            offset += blockDim.x;
        }
    }
    if (nid > maxNeighborBlock) {
        printf("actual nNeighbor = %d > %d (preserved)\n", nid, maxNeighborBlock);
        assert(nid <= maxNeighborBlock)
    }
    __syncthreads();
    // reassign bid and distance to threads
    if (tid < nid) {
        dis = distance[tid];
        blockId =  bid[tid];
    } else {
        dis = max_radius + 1; // to be excluded
    }
    // sorting
    PosInt final_bid;
    for (Size i=0; i<nid;  i++) {
        find_min(min, dis, id);
        PosInt min_id = id[0];
        if (tid == min_id) {
            dis = max_radius + 1; // to be excluded for next min
        }
        if (tid == i) { // get read for global store
            final_bid = min_id;
        }
        __syncwarp();
    }
    if (tid < nid) {
        neighborBlockId[maxNeighborBlock*blockIdx.x + i] = final_bid;
    }
    if (tid == 0) {
        nNeighborBlock[blockIdx.x] = nid;
    }
}

__global__ 
void generate_connections(double* __restrict__ pos,
                          Float* __restrict__ preS_type,
                          Float* __restrict__ preP_type,
                          Size* __restrict__ preN,
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
                          Float* __restrict__ feature,
                          Float* __restrict__ dden,
                          Float* __restrict__ daxn,
                          curandStateMRG32k3a* __restrict__ state,
                          PosInt block_offset, Size networkSize, Size maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Float speedOfThought, Size nType, Size nFeature, bool gaussian_profile) 
{
    Size blockId = blockIdx.x + block_offset;
    Size nn = nNeighborBlock[blockId];
    Size offset = blockId*blockDim.x;
    double x0 = pos[offset*2 + threadIdx.x];
    double y0 = pos[offset*2 + threadIdx.x + blockDim.x];
    Size id = offset + threadIdx.x;
    // number of potential presynaptic connections outsied nearNeighbors, to be stored in vector.
    Size nb = 0; 
    if (nn > nearNeighbors) {
        nb = nn - nearNeighbors;
        nn = nearNeighbors;// nearNeighbors 
    } 
    Float* tempNeighbor;
    if (nb > 0) {
        nb *= blockDim.x;
        tempNeighbor = new Float[nb];
    }
    Float rd = rden[id];
    Float dd = dden[id];

    Float sumP = 0.0;
    //============= collect p of all ==========
    // withhin block and nearNeighbor
    for (Size in=0; in<nn; in++) {
        Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
        #pragma unroll
        for (Size i=0; i<blockDim.x; i++) {
            // blockwise load from gmem the (potential) presynaptic neurons' properties
            Size ii = (i + threadIdx.x) % blockDim.x; // pre-id in block
            Size ipre = bid + ii; // pre-id in network
            double x1 = pos[bid*2 + ii];
            double y1 = pos[bid*2 + blockDim.x + ii];
            Float ra = raxn[ipre];
            double x = x1 - x0;
            double y = y1 - y0;
            //type vector, indexed across the network
            Float distance = static_cast<Float>(square_root(x*x + y*y));
	    	// weight from area
            Float p = connect(distance, ra, rd, gaussian_profile);

            if (p > 0) {
                Size ip = preType[ipre];
                // id in the conMat [nblock,nearNeighborBlock,blockDim.x,blockDim.x] loop in the second axis, (last dim is the post-id)
                PosIntL mid = (offset*nearNeighborBlock + in*blockDim.x + ii)*blockDim.x + threadIdx.x;
                sumType[ip] += 1;
	    		// update weight with density of axon dendrites and preference over type
                p *= daxn[ipre] * dd * preP_type[ip*networkSize + id];
                for (Size iFeature = 0; iFeature < nFeature; iFeature++) {
                    p *= pref[iFeature](feature[id], feature[ipre]);
                }
                sumP += p;
                conMat[mid] = p;
                delayMat[mid] = distance/speedOfThought;
            }
        }
    }
    // the remaining neighbors
    if (nb > 0) {
        for (Size in=nn; in<nNeighborBlock[blockIdx.x]; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
            #pragma unroll
            for (Size i=0; i<blockDim.x; i++) {
                // blockwise load from gmem the (potential) presynaptic neurons' properties
                Size j = (i+threadIdx.x) % blockDim.x;
                Size ipre = bid + j; // pre-id in the network
                double x1 = pos[bid*2 + j];
                double y1 = pos[bid*2 + blockDim.x + j];
                Float ra = raxn[ipre];
                double x = x1 - x0;
                double y = y1 - y0;
                Float distance = static_cast<Float>(square_root(x*x + y*y));
                Float p = connect(distance, ra, rd, gaussian_profile);
                if (p > 0) {
                    Size ip = preType[ipre];
                    Size tid = (nn-in)*blockDim.x + i; // only ofr tempNeighbor, which is local, no need to coalease memory
                    sumType[ip] += 1;
                    p *= daxn[ipre] * dden[id] * preP_type[ip*networkSize+id];
                    for (Size iFeature = 0; iFeature < nFeature; iFeature ++) {
                        p *= pref[iFeature](feature[id], feature[ipre]);
                    }
                    sumP += p;
                    tempNeighbor[tid] = p;
                } else {
                	tempNeighbor[tid] = 0;
	    		}
            }
        }
    }
    __syncwarp();
    // initialize stats
    Size* sumConType = new Size[nType];
    Size* sumType = new Size[nType]; // avail
    Float* sumStrType = new Float[nType];
    // initialize  for stats collection
    #pragma unroll
    for (Size i=0; i<nType; i++) {
        sumConType[i] = 0;
        sumStrType[i] = 0;
        sumType[i] = 0;
    }
    curandStateMRG32k3a localState = state[id];
    Size pN = preN[id];
    //============= redistribute p of all ==========
    for (Size in=0; in<nn; in++) {
        Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
        #pragma unroll
        for (Size i=0; i<blockDim.x; i++) {
            Size ii = (i + threadIdx.x) % blockDim.x; // pre-id in block
            PosIntL mid = (offset*nearNeighborBlock + in*blockDim.x + ii)*blockDim.x + threadIdx.x;
            Float p = conMat[mid]/sumP*pN;
            Float xrand = uniform(&localState);
            Float str = preS_type[ip*networkSize+id];
            Size ipre = bid + i;
            Size ip = preType[ipre];
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
            conMat[mid] = p;
        }
    }
    Size nid = 0;
    if (nb > 0) {
        for (Size in=nn; in<nNeighborBlock[blockIdx.x]; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in];
            #pragma unroll
            for (Size i=0; i<blockDim.x; i++) {
                Size j = (i+threadIdx.x) % blockDim.x;
                Size ipre = bid + j;
                Size tid = (nn-in)*blockDim.x + i;
                Float p = tempNeighbor[tid]/sumP*preN[id];
                Float xrand = uniform(&localState);
                if (xrand < p) {
                    Size ip = preType[ipre];
                    Float str = preS_type[ip*networkSize+id];
                    if (p > 1) {
                        str = str*p;
                    }
                    sumConType[ip] += 1;
                    sumStrType[ip] += str;
                    vecID[maxDistantNeighbor*id + nid] = ipre;
                    conVec[maxDistantNeighbor*id + nid] = str;
                    double x = pos[bid*2 + j] - x0;
                    double y = pos[bid*2 + blockDim.x + j] - y0;
	    			Float distance = static_cast<Float>(square_root(x*x + y*y));
                    delayVec[maxDistantNeighbor*id + nid] = distance/speedOfThought;
                    nid += 1;
                    if (nid > maxDistantNeighbor) {
                        printf("set bigger maxDistantNeighbor, currently %u\n", maxDistantNeighbor);
                        assert(nid <= maxDistantNeighbor);
                    }
                }
            }
        }
    }
    nVec[id] = nid;
    #pragma unroll
    for (Size i=0; i<nType; i++) {
        preTypeConnected[i*networkSize + id] = sumConType[i];
        preTypeAvail[i*networkSize + id] = sumType[i];
        preTypeStrSum[i*networkSize + id] = sumStrType[i];
    }
    delete []sumConType;
    delete []sumType;
    delete []sumStrType;
    if (nb > 0) {
        delete []tempNeighbor;
    }
}
