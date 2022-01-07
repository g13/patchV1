#include "connect.h"

extern __device__ __constant__ pFeature pref[];

// TODO: randomize neuronal attributes by using distribution, strength x number of con. should be controlled
__global__ 
void initialize(
			Float* __restrict__ nLGN_eff,
			Float* __restrict__ ffRatio,
			Size* __restrict__ typeAcc0,
            Float* __restrict__ maxCortExc,
			Size nType)
{
	// determine the arch neuronal type and its properties
   	Size type;
    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
	#pragma unroll
    for (PosInt i=0; i<nType; i++) {
        if (threadIdx.x < typeAcc0[i+1]) {
            type = i;
            break;
        }
    }
    ffRatio[id] = nLGN_eff[id]/maxCortExc[type];
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
/*__device__ 
__forceinline__
Float connect(Float distance, Float raxn, Float rden, Float disGauss) {
	Float weight;
	if (disGauss > 0) {
		// HWHM = sqrt(raxn*raxn + rden*rden)
		// sigma = HWHM/sqrt(2*ln(2))
		Float variance = (raxn*raxn + rden*rden)/(2*logarithm(2))*disGauss*disGauss;
        if (distance < 3*square_root(variance)) {
		    //weight = exponential(-distance*distance/spread)/(2*M_PI*spread);
		    weight = exponential(-distance*distance/variance);
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
}*/
__device__ 
__forceinline__
Float connect(Float distance, Float raxn, Float rden, Float disGauss) {
    Float weight;
    if (raxn + rden < distance) {
	    if (disGauss > 0) {
	    	// HWHM = sqrt(raxn*raxn + rden*rden)
	    	// sigma = HWHM/sqrt(2*ln(2))
	    	Float variance = rden*rden*disGauss*disGauss;
	    	weight = exponential(-distance*distance/variance);
        } else {
            weight = 1.0;
            //**** revisit
    	    //weight = area(raxn, rden, distance)/(M_PI*rden*rden); // conn. prob. is defined by the presynaptic point of view
        }
    } else {
        weight = 0;
    }
    return weight;
}

__global__ 
void cal_blockPos(
              Float* __restrict__ pos,
              Float* __restrict__ block_x,
              Float* __restrict__ block_y,
              Float* __restrict__ block_r,
              Size networkSize) 
{
    __shared__ Float reduced[warpSize];
    __shared__ PosInt _id[warpSize];
    Size id = (2*blockDim.x)*blockIdx.x + threadIdx.x;
    Float x = pos[id];
    Float y = pos[id + blockDim.x];
    block_reduce<Float>(reduced, x);
    if (threadIdx.x == 0) {
        block_x[blockIdx.x] = reduced[0]/blockDim.x;
    }
    block_reduce<Float>(reduced, y);
    if (threadIdx.x == 0) {
        block_y[blockIdx.x] = reduced[0]/blockDim.x;
    }
    Float x_min, xd, y_min, yd;
    find_min(reduced, x, _id, blockDim.x);
    if (threadIdx.x == 0) {
        x_min = reduced[0];
    }
    find_min(reduced, -x, _id, blockDim.x);
    if (threadIdx.x == 0) {
        xd = -reduced[0] - x_min;
    }
    find_min(reduced, y, _id, blockDim.x);
    if (threadIdx.x == 0) {
        y_min = reduced[0];
    }
    find_min(reduced, -y, _id, blockDim.x);
    if (threadIdx.x == 0) {
        yd = -reduced[0] - y_min;
    }
    block_r[blockIdx.x] = square_root(yd*yd + xd*xd);
}

/*__global__ 
void get_neighbor_blockId(
                        Float* __restrict__ block_x,
                        Float* __restrict__ block_y,
                        Float* __restrict__ block_r,
                        PosInt* __restrict__ blockAcc,
                        PosInt* __restrict__ neighborBlockId,
                        Size* __restrict__ nNeighborBlock,
                        Size* __restrict__ nNearNeighborBlock,
                        Size* __restrict__ nGapNeighborBlockE,
                        Size* __restrict__ nGapNeighborBlockI,
                        Float* __restrict__ gapDis,
						Size nblock, Float rden, Float raxn, Size maxNeighborBlock, Float postGapDisE, Float preGapDisE, Float postGapDisI, Float preGapDisI, PosInt ipost, PosInt ipre) 
{
    __shared__ PosInt id[warpSize];
    __shared__ Float min[warpSize];
    __shared__ Int bid[blockDim.x];
	__shared__ Float distance[blockDim.x];

	extern __shared__ Float final_distance[];
	PosInt* final_bid = (PosInt*) (final_distance + maxNeighborBlock);

    Float max_radius = rden+raxn;
    Float radius = rden;
    Float gapDiusE = preGapDisE + postGapDisE;
    Float gapDiusI = preGapDisI + postGapDisI;
    PosInt center_bid = blockAcc[ipost] + blockIdx.x;
    PosInt layeredBid = blockAcc[ipre];
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    Size nPatch = (nblock + blockDim.x-1)/blockDim.x - 1;
    Size remain = nblock%blockDim.x;
	if (remain == 0) {
		remain = blockDim.x;
	}

    if (gapDiusE > radius) {
        gapDiusE = radius;
    }
    if (gapDiusI > radius) {
        gapDiusI = radius;
    }

    if (tid == 0) {
    	    //#pragma unroll
        id[0] = 0;
        if (ipost == ipre) {
            id[1] = 1; // 1 is correct, first self will be assigned
        } else {
            id[1] = 0;
        }
    }
    __syncthreads();
    bid[tid] = -1;

    Float bx = block_x[center_bid]; // center of the target block
    Float by = block_y[center_bid];

    Size offset = 0;
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        Float r;
        if (iPatch < nPatch || tid < remain) {
            PosInt blockId = layeredBid + offset + threadIdx.x;
            Float x = block_x[blockId] - bx;
            Float y = block_y[blockId] - by;
            r = block_r[blockId];
            Float dis = square_root(x*x + y*y);
            if (dis < max_radius+r) {
                distance[tid] = dis;
                bid[tid] = offset+threadIdx.x;
            }
        }
        __syncthreads();
        if (tid == 0) { // rearrange
            if (ipost == ipre) { // assign self first if same layer
			    neighborBlockId[maxNeighborBlock*blockIdx.x] = blockIdx.x;
            }
			PosInt outside_id = id[0];
            PosInt current_id = id[1];
            for (PosInt i=0; i<blockDim.x; i++) {
                if (bid[i] != -1 && (ipost != ipre || bid[i] != blockIdx.x)) {
					if (distance[i] < radius+r) {
						neighborBlockId[maxNeighborBlock*blockIdx.x + current_id] = bid[i];
                    	current_id++;
					} else {
                    	final_distance[outside_id] = distance[i]; 
                    	final_bid[outside_id] = bid[i]; 
						outside_id++;
					}
                    bid[i] = -1; 
                    if (current_id + outside_id > maxNeighborBlock) {
                        printf("actual nNeighbor = %d > %d (preserved)\n", current_id, maxNeighborBlock);
                        assert(current_id + outside_id <= maxNeighborBlock);
                    }
                }
            }
            id[0] = outside_id;
            id[1] = current_id;
        }
        __syncthreads();
        if (iPatch < nPatch) {
            offset += blockDim.x;
        }
    }
    Size nb = id[0];
    Size nn = id[1];
    if (tid == 0) {
        nNeighborBlock[blockIdx.x] = nb + id[1];
        nNearNeighborBlock[blockIdx.x] = nn;
	    //printf("%u: %u blocks in total\n", blockIdx.x, nb);
    }
    Float dis;
    PosInt local_bid;
    if (tid < nb) {
        dis = final_distance[tid];
        //DEBUG
        //  if (blockIdx.x == 2) {
        //      printf("preSort#%u:%u, %e\n", tid, final_bid[tid], dis);
        //  }
        //
    }
	// get everything closer than radius
	
    // sorting
    for (Size i=0; i<nb;  i++) {
        find_min(min, dis, id, nb);
        PosInt min_id = id[0];
        if (tid == min_id) {
            dis = max_radius + 1; // to be excluded for next min
        }
        if (tid == i) { // get read for global store
            local_bid = final_bid[min_id];
        }
		// DEBUG
        //  if (blockIdx.x == 2 && tid < nb) {
        //      printf("%u#%u:%u, %e\n",i, tid, final_bid[tid], dis);
        //  }
        //  __syncthreads();
        //
    }
    //TODO: if nb > blockSize
    if (tid < nb) {
        neighborBlockId[maxNeighborBlock*blockIdx.x + nn + tid] = local_bid;
    }
	// DEBUG
    //  if (blockIdx.x == 2) {
	//      for (PosInt j = 0; j<nb; j++) {
	//      	if (tid == j) {
	//      		if (j == 0) {
	//      			printf("block %u, %u neighbors in total\n", blockIdx.x, nb);
	//      		}
	//      		printf("#%u, %u: %f -> %f,", j, local_bid, final_distance[j], dis);
	//      		if (j == nb-1) {
	//      			printf("\n");
	//      		}
	//      	}
	//      	__syncwarp();
	//      }
    //  }
	//
}*/


__global__ 
void get_neighbor_blockId(
                        Float* __restrict__ block_x,
                        Float* __restrict__ block_y,
                        Float* __restrict__ block_r,
                        PosInt* __restrict__ blockAcc,
                        PosInt* __restrict__ neighborBlockId,
                        Size* __restrict__ nNeighborBlock,
                        Size* __restrict__ nNearNeighborBlock,
                        Size* __restrict__ nGapNeighborBlockE,
                        Size* __restrict__ nGapNeighborBlockI,
						Size nblock, Float rden, Float raxn, Size maxNeighborBlock, Float postGapDisE, Float preGapDisE, Float postGapDisI, Float preGapDisI, PosInt ipost, PosInt ipre) 
{
    __shared__ PosInt id[2];
    extern __shared__ Int bid[];
	Float* distance = (Float*) (bid + nblock);
	Float* r = distance + nblock;
	PosInt* outer_bid = (PosInt*) (r + nblock);

    Float max_radius = rden+raxn;
    Float radius = rden;
    Float gapDiusE = preGapDisE + postGapDisE;
    Float gapDiusI = preGapDisI + postGapDisI;
    PosInt center_bid = blockAcc[ipost] + blockIdx.x;
    PosInt layeredBid = blockAcc[ipre];
    Size nPatch = (nblock + blockDim.x-1)/blockDim.x - 1;
    Size remain = nblock%blockDim.x;
	if (remain == 0) {
		remain = blockDim.x;
	}
    Float ring[3]{};
    ring[2] = radius;
    Size* nNab[3];
    nNab[2] = nNearNeighborBlock;
    if (gapDiusE > radius) {
        gapDiusE = radius;
    }
    if (gapDiusI > radius) {
        gapDiusI = radius;
    }
    if (gapDiusE > gapDiusI) {
        ring[0] = gapDiusI;
        ring[1] = gapDiusE;
        nNab[0] = nGapNeighborBlockI;
        nNab[1] = nGapNeighborBlockE;
    } else {
        ring[0] = gapDiusE;
        ring[1] = gapDiusI;
        nNab[0] = nGapNeighborBlockE;
        nNab[1] = nGapNeighborBlockI;
    }

    Float bx = block_x[center_bid]; // center of the target block
    Float by = block_y[center_bid];

    Size offset = 0;
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        PosInt tid = offset+threadIdx.x;
        if (tid < nblock) {
            bid[tid] = -1;
            PosInt blockId = layeredBid + offset + threadIdx.x;
            Float x = block_x[blockId] - bx;
            Float y = block_y[blockId] - by;
            r[tid] = block_r[blockId];
            distance[tid] = square_root(x*x + y*y);
            if (distance[tid] < max_radius+r[tid]) {
                bid[tid] = tid;
            }
            offset += blockDim.x;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        id[0] = 0;
        if (ipost == ipre) {
            id[1] = 1;
		    neighborBlockId[maxNeighborBlock*blockIdx.x] = blockIdx.x;
            bid[blockIdx.x] = -1;
        } else {
            id[1] = 0;
        }
    }
    __syncthreads();
    // rearrange
    for (PosInt iR = 0; iR<3; iR++) { 
        offset = 0;
        for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
            Int local_bid;
            Int current_id = -1;
            Int outside_id = -1;
            PosInt tid = offset + threadIdx.x;
            if (tid < nblock) {
                local_bid = bid[tid];
                if (local_bid != -1) {
		    		if (distance[tid] < ring[iR]+r[tid]) {
                        current_id = atomicAdd(id + 1, 1);
		    		} else {
                        if (iR==2) {
                            outside_id = atomicAdd(id, 1);
                        }
		    		}
                }
            }
            __syncwarp();
            if (current_id >= 0) {
		        neighborBlockId[maxNeighborBlock*blockIdx.x + current_id] = local_bid;
                bid[tid] = -1;
            }
            if (outside_id >= 0) {
                outer_bid[outside_id] = local_bid; 
            }
            offset += blockDim.x;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            nNab[iR][blockIdx.x] = id[1];
        }
        nPatch = (nblock-id[1] + blockDim.x-1)/blockDim.x - 1;
        remain = (nblock-id[1])%blockDim.x;
        __syncthreads();
    }
    Size nb = id[0];
    Size nn = id[1];
    if (threadIdx.x == 0) {
        if (nb + nn > maxNeighborBlock) {
            printf("actual nNeighbor = %d > %d (preserved)\n", nb + nn, maxNeighborBlock);
            assert(nb + nn <= maxNeighborBlock);
        }
        nNeighborBlock[blockIdx.x] = nb + nn;
    }
    __syncwarp();
    nPatch = (nb + blockDim.x-1)/blockDim.x - 1;
    remain = (nb)%blockDim.x;
    offset = 0;
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        PosInt tid = offset + threadIdx.x;
        if (tid < nb) {
            neighborBlockId[maxNeighborBlock*blockIdx.x + nn + tid] = outer_bid[tid];
            offset += blockDim.x;
        }
    }
}

//__launch_bounds__(1024,1)
__global__ 
void generate_connections(
                        curandStateMRG32k3a* __restrict__ state,
                        Float* __restrict__ post_pos,
                        Float* __restrict__ pre_pos,
                        PosInt* __restrict__ neighborBlockId,
                        Size*   __restrict__ nNeighborBlock,
                        Size*   __restrict__ nNearNeighborBlock,
                        Size*   __restrict__ nGapNeighborBlockE,
                        Size*   __restrict__ nGapNeighborBlockI,
                        Float*  __restrict__ feature,
                        Float*  __restrict__ rden,
                        Float*  __restrict__ raxn,
                        Float*  __restrict__ gapDis,
                        Float*  __restrict__ ffRatio,
                        Float*  __restrict__ inhRatio,
                        Size*   __restrict__ nTypeMat,
                        Size*   __restrict__ gap_nTypeMatE,
                        Size*   __restrict__ gap_nTypeMatI,
                        Float*  __restrict__ fTypeMat,
                        Float*  __restrict__ gap_fTypeMatE,
                        Float*  __restrict__ gap_fTypeMatI,
                        Float*  __restrict__ conMat, //within block connections
                        Float*  __restrict__ delayMat,
                        Float*  __restrict__ gapMatE,
                        Float*  __restrict__ gapMatI,
                        Float*  __restrict__ conVec, //for neighbor block connections
                        Float*  __restrict__ delayVec, //for neighbor block connections
                        Size*   __restrict__ max_N,
                        PosInt* __restrict__ _vecID,
                        Float*  __restrict__ disNeighborP,
                        Size*   __restrict__ vecID,
                        Size*   __restrict__ nVec,
                        Size*   __restrict__ preTypeConnected,
                        Size*   __restrict__ preTypeAvail,
                        Float*  __restrict__ preTypeStrSum,
                        Size*   __restrict__ preTypeGapE,
                        Size*   __restrict__ preTypeAvailGapE,
                        Float*  __restrict__ preTypeStrGapE,
                        Size*   __restrict__ preTypeGapI,
                        Size*   __restrict__ preTypeAvailGapI,
                        Float*  __restrict__ preTypeStrGapI,
                        Size**  __restrict__ typeAcc0,
                        PosInt post, PosInt pre, Size accSize, Size postAccSize, Size preAccSize, Size totalType, Size totalTypeE, Size totalTypeI, PosInt postTypeID, PosInt preTypeID, PosInt postTypeEID, PosInt preTypeEID, PosInt postTypeIID, PosInt preTypeIID, Size nF, Size ppF, Size prePerBlock, Size sum_max_N, PosInt block_offset, Size postSize, Size preSize, Size post_nType, Size pre_nType, Size post_nTypeE, Size pre_nTypeE, Size post_nTypeI, Size pre_nTypeI, Size mE, Size mI, Size maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Size maxTempBlock, Size gapNeighborBlockE, Size gapNeighborBlockI, Size post_nE, Size post_nI, Size pre_nE, Size pre_nI, Float disGauss, bool strictStrength, bool CmoreN, BigSize seed) 
{
    // TODO: load with warps but more, e.g., raxn, daxn, preType
    extern __shared__ Size nMat[];
    Size* gap_nMatE = nMat + post_nType*pre_nType;
    Size* gap_nMatI = gap_nMatE + post_nTypeE*pre_nTypeE;
    Float* fMat = (Float*) (gap_nMatI + post_nTypeI*pre_nTypeI);
    Float* gap_fMatE = fMat + nF*ppF*post_nType*pre_nType;
    Float* gap_fMatI = gap_fMatE + nF*ppF*post_nTypeE*pre_nTypeE;
    Float* pre_feat = gap_fMatI + nF*ppF*post_nTypeI*pre_nTypeI;
    Float* x1 = pre_feat + nF*prePerBlock;
    Float* y1 = x1 + prePerBlock;
    Float* ra = y1 + prePerBlock;
    PosInt* pre_typeAcc = (PosInt*) (ra + pre_nType*post_nType);
    Float* pre_gapDis = (Float*) (pre_typeAcc + pre_nType);

    Size blockId = blockIdx.x + block_offset;
    Size nNab = nNeighborBlock[blockId];
    Size nNear = nNearNeighborBlock[blockId];
    Size offset = blockId*blockDim.x;
    Size id = offset + threadIdx.x;
    assert(id < postSize);
    Float x0 = post_pos[offset + id];
    Float y0 = post_pos[postSize + offset + id];
    
    // fill shared mats
    if (threadIdx.x < post_nType * pre_nType) {
        PosInt mat_id = (preTypeID + threadIdx.x/post_nType)*totalType + postTypeID + threadIdx.x%post_nType;
        nMat[threadIdx.x] = nTypeMat[mat_id];
        for (PosInt i=0; i<nF; i++) {
            for (PosInt j=0; j<ppF; j++) {
                fMat[i*j*post_nType*pre_nType + threadIdx.x] = fTypeMat[i*j*totalType*totalType + mat_id];
            }
        }
        PosInt ra_id = (preTypeID + threadIdx.x/post_nType)*totalType + postTypeID + threadIdx.x%post_nType;
        ra[threadIdx.x] = raxn[ra_id];
    }
    if (threadIdx.x < post_nTypeE * pre_nTypeE) {
        PosInt mat_id = (preTypeEID + threadIdx.x/post_nTypeE)*totalTypeE + postTypeEID + threadIdx.x%post_nTypeE;
        gap_nMatE[threadIdx.x] = gap_nTypeMatE[mat_id];
        for (PosInt i=0; i<nF; i++) {
            for (PosInt j=0; j<ppF; j++) {
                gap_fMatE[i*j*post_nTypeE*pre_nTypeE + threadIdx.x] = gap_fTypeMatE[i*j*totalTypeE*totalTypeE + mat_id];
            }
        }
    }
    if (threadIdx.x < post_nTypeI * pre_nTypeI) {
        PosInt mat_id = (preTypeIID + threadIdx.x/post_nTypeI)*totalTypeI + postTypeIID + threadIdx.x%post_nTypeI;
        gap_nMatI[threadIdx.x] = gap_nTypeMatI[mat_id];
        for (PosInt i=0; i<nF; i++) {
            for (PosInt j=0; j<ppF; j++) {
                gap_fMatI[i*j*post_nTypeI*pre_nTypeI + threadIdx.x] = gap_fTypeMatI[i*j*totalTypeI*totalTypeI + mat_id];
            }
        }
    }

    if (threadIdx.x < pre_nType) {
        pre_typeAcc[threadIdx.x] = typeAcc0[pre][threadIdx.x+1]; // without 0 in the front
        pre_gapDis[threadIdx.x] = gapDis[preTypeID + threadIdx.x];
    }
    __syncthreads();
    // determine local and global typeID
	PosInt typeID = postTypeID; // global_typeID
    PosInt ip; // local_typeID
	for (PosInt i=0; i<post_nType; i++) {
		if (threadIdx.x < typeAcc0[post][i+1]) {
			typeID += i;
            ip = i;
			break;
		}
	}
    Float* gapMat;
    Size* gap_nMat;
    Float* gap_fMat;
    Size gapType, gapType0, post_gapType, post_gapType0;
    Size pre_n, pre_n0, post_n, post_n0, nGapNab, gapNab;
    if (ip < post_nTypeE) {
        gapMat = gapMatE; 
        gap_nMat = gap_nMatE; 
        gap_fMat = gap_fMatE; 
        gapType = pre_nTypeE;
        gapType0 = 0;
        post_gapType = post_nTypeE;
        post_gapType0 = 0;
        pre_n = pre_nE;
        pre_n0 = 0;
        post_n = post_nE;
        post_n0 = 0;
        nGapNab = nGapNeighborBlockE[blockId];
        gapNab = gapNeighborBlockE;
    } else {
        gapMat = gapMatI; 
        gap_nMat = gap_nMatI;
        gap_fMat = gap_fMatI; 
        gapType = pre_nTypeI;
        gapType0 = pre_nTypeE;
        post_gapType = post_nTypeI;
        post_gapType0 = post_nTypeE;
        pre_n = pre_nI;
        pre_n0 = pre_nE;
        post_n = post_nI;
        post_n0 = post_nE;
        nGapNab = nGapNeighborBlockI[blockId];
        gapNab = gapNeighborBlockI;
    }
    PosInt gap_ip = ip-post_gapType0;
    Float rd = rden[ip];
	Float local_gapDis = gapDis[typeID];

    // number of potential presynaptic connections outside nearNeighbors, to be stored in vector.

    Size* availType = new Size[pre_nType]; // avail
    Float* sumP = new Float[pre_nType];
    #pragma unroll
    for (PosInt i=0; i<pre_nType; i++) {
        availType[i] = 0;
        sumP[i] = 0.0;
	}
    // gap
	Size* availGapType = new Size[gapType];
	Float* sumGapP = new Float[gapType];
    #pragma unroll
    for (PosInt i=0; i<gapType; i++) {
        sumGapP[i] = 0.0;
	    availGapType[i] = 0;
    }
    Float* post_feat = new Float[nF];
    for (PosInt i=0; i<nF; i++) {
        post_feat[i] = feature[i*accSize + postAccSize + id];
    }
    //============= collect p of all ==========
    // withhin block and nearNeighbor
    for (PosInt iNear=0; iNear<nNear; iNear++) {
        PosInt bid = neighborBlockId[maxNeighborBlock*blockId + iNear] * prePerBlock; // # neurons in all past blocks 
        #pragma unroll
        for (PosInt i=0; i<(prePerBlock + blockDim.x-1)/blockDim.x; i++) {
            PosInt tid = i*blockDim.x + threadIdx.x;
            if (tid < prePerBlock) {
                x1[tid] = pre_pos[bid + tid];
                y1[tid] = pre_pos[prePerBlock + bid + tid];
                for (PosInt j=0; j<nF; j++) {
                    pre_feat[j*prePerBlock + tid] = feature[j*accSize + preAccSize + tid];
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (Size i=0; i<prePerBlock; i++) {
            // blockwise load from gmem the (potential) presynaptic neurons' properties
            PosInt ipre = bid + i; // pre-id in the pre layer
            __syncwarp();
            if (id == ipre && post == pre) continue;
            PosInt jp;
            #pragma unroll
	        for (PosInt j=0; j<pre_nType; j++) {
	        	if (i < pre_typeAcc[j]) {
	        		jp = j;
	        		break;
	        	}
	        }
            Float x = static_cast<Float>(x1[ipre] - x0);
            Float y = static_cast<Float>(y1[ipre] - y0);
            //type vector, indexed across the network
            Float distance = square_root(x*x + y*y);
	    	// weight from area
            Float p = connect(distance, ra[jp*post_nType + ip], rd, disGauss);
            size_t mid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + iNear)*prePerBlock + i)*blockDim.x + threadIdx.x; // defined outside, so delayMat has access to it
            if (p > 0) { // not self-connected
                Float fp = 1.0;
                for (Size iF = 0; iF < nF; iF++) {
                    PosInt mat_id = iF*ppF*post_nType*pre_nType + jp*post_nType + ip;

					fp *= pref[iF](post_feat[iF], pre_feat[iF*prePerBlock + i], fMat[mat_id], fMat[post_nType*pre_nType + mat_id]);
                }
                if (nF > 0) {
                    p += fp;
                }
                assert(p >= 0);
				if (p > 0) {
                	availType[jp]++;
                	sumP[jp] += p;
                	conMat[mid] = p;
				}
                Int gap_jp = jp-gapType0;
                if (iNear < nGapNab && gap_jp >= 0 && gap_jp < gapType && gap_nMat[post_gapType*gap_jp + gap_ip] > 0) {
                    Float gapP = connect(distance, pre_gapDis[jp], local_gapDis, disGauss);
			        if (gapP > 0) {
                    	size_t gid = (static_cast<size_t>(blockIdx.x*gapNab + iNear)*pre_n + i-pre_n0)*post_n + threadIdx.x-post_n0;
                        Float fp = 1.0;
                        for (Size iF = 0; iF < nF; iF++) {
                            PosInt mat_id = iF*ppF*post_gapType*gapType + gap_jp*post_gapType + gap_ip;
			        		fp *= pref[iF](post_feat[iF], pre_feat[iF*prePerBlock + ipre], gap_fMat[mat_id], gap_fMat[post_gapType*gapType + mat_id]);
                        }
                        if (nF > 0) {
                            gapP += fp;
                        }
			    	    availGapType[gap_jp]++;
			    	    sumGapP[gap_jp] += gapP;
			    	    gapMat[gid] = gapP;
                    }
                }
            }
            delayMat[mid] = distance; // record even if not connected, for LFP
        }
		__syncthreads();
    }
    // the remaining neighbors
    Size nFar = nNab - nNear; 
    Float* tempNeighbor = disNeighborP + static_cast<size_t>((blockIdx.x*blockDim.x + threadIdx.x)*(maxTempBlock))*prePerBlock;
    if (nFar > 0) {
        for (Size iNab=nNear; iNab<nNab; iNab++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + iNab] * prePerBlock;
            #pragma unroll
            for (PosInt i=0; i<(prePerBlock + blockDim.x-1)/blockDim.x; i++) {
                PosInt tid = i*blockDim.x + threadIdx.x;
                if (tid < prePerBlock) {
                    x1[tid] = pre_pos[bid + tid];
                    y1[tid] = pre_pos[prePerBlock + bid + tid];
                    for (PosInt j=0; j<nF; j++) {
                        pre_feat[j*prePerBlock + tid] = feature[j*accSize + preAccSize + tid];
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (Size i=0; i<prePerBlock; i++) {
                // blockwise load from gmem the (potential) presynaptic neurons' properties
                PosInt ipre = bid + i; // pre-id in the network
                PosInt jp;
                #pragma unroll
	            for (PosInt j=0; j<pre_nType; j++) {
	            	if (ipre < pre_typeAcc[j]) {
	            		jp = j;
	            		break;
	            	}
	            }
                Float x = static_cast<Float>(x1[ipre] - x0);
                Float y = static_cast<Float>(y1[ipre] - y0);
                Float distance = static_cast<Float>(square_root(x*x + y*y));
                Float p = connect(distance, ra[jp*post_nType + ip], rd, disGauss);
                Size tid = (iNab-nNear)*prePerBlock + i; // only for tempNeighbor, which is local, no need to coalease memory
                tempNeighbor[tid] = 0;
                if (p > 0) {
                    for (Size iF = 0; iF < nF; iF++) {
                        PosInt mat_id = iF*ppF*post_nType*pre_nType + jp*post_nType + ip;
				    	p *= pref[iF](post_feat[iF], pre_feat[iF*prePerBlock + ipre], fMat[mat_id], fMat[post_nType*pre_nType + mat_id]);
                    }
                    assert(p>=0);
					if (p > 0) {
                    	availType[jp]++;
                    	sumP[jp] += p;
                    	tempNeighbor[tid] = p;
					}
                }
            }
            __syncthreads();
        }
    }
    Size* pN = new Size[pre_nType];
	Size* gap_pN = new Size[gapType];;
    PosInt gap_id = blockId*post_n + threadIdx.x-post_n0;
    #pragma unroll
    for (Size i=0; i<pre_nType; i++) {
        pN[i] = nMat[i*post_nType + ip];
        if (CmoreN) {
            if (i < pre_nTypeE && CmoreN) { // complex
                pN[i] = static_cast<Size>(ceiling(pN[i] * (1-ffRatio[id])));
            }
            if (i >= pre_nTypeE) {
                pN[i] = static_cast<Size>(ceiling(pN[i] * (1-ffRatio[id]) * inhRatio[ip]));
            }
        }
        preTypeAvail[i*postSize + id] = availType[i];
		if (availType[i] < pN[i]) {
			printf("neuron %u-%u dont have enough type %u neurons to connect to (%u/%u)\n", blockIdx.x, threadIdx.x, i, availType[i], pN[i]);
			assert(availType[i] >= pN[i]);
		}
        Int j=i-gapType0;
		if (gap_ip<post_gapType && j<gapType && j>=0) {
			gap_pN[j] = gap_nMat[j*post_gapType + gap_ip];
            if (gap_ip < post_nTypeE) preTypeAvailGapE[i*mE + gap_id] = availGapType[j];
            else preTypeAvailGapI[i*mI + gap_id] = availGapType[i];
			if (availGapType[j] < gap_pN[j]) {
				printf("neuron %u-%u dont have enough type %u inh neurons to make gap junction to (%u/%u)\n", blockIdx.x, threadIdx.x, i, availGapType[j], gap_pN[j]);
				assert(availGapType[j] >= gap_pN[j]);
			}
		}
    }
    //============= redistribute p of all ==========
    bool* typeConnected = new bool[pre_nType];
	Size _sum_max_N = 0;
	PosInt** __vecID = new PosInt*[pre_nType];
    #pragma unroll
    for (Size i=0; i<pre_nType; i++) {
     	__vecID[i] = _vecID + (blockIdx.x*blockDim.x + threadIdx.x)*sum_max_N + _sum_max_N;
		_sum_max_N += max_N[i];
		typeConnected[i] = false;
		sumP[i] = pN[i]/sumP[i]; // now is a ratio
    }

    // init rand states
    curandStateMRG32k3a localState = state[id];
    if (pre == 0) {
        curand_init(seed, id, 0, &localState);
    }
	Size count = 0;
	Size connected = false;
    Size* sumType = new Size[pre_nType];
    Float* sumStrType = new Float[pre_nType];
    Size* nid = new Size[pre_nType];
	while (!connected) {
    	for (Size i=0; i<pre_nType; i++) {
			if (!typeConnected[i]) {
    	    	sumType[i] = 0;
    	    	sumStrType[i] = 0;
				nid[i] = 0;
			}
		}
    	for (Size iNab=0; iNab<nNear; iNab++) {
    	    for (Size i=0; i<prePerBlock; i++) {
    	        size_t mid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + iNab)*prePerBlock + i)*blockDim.x + threadIdx.x;
    	        Size jp;
                #pragma unroll
	            for (PosInt j=0; j<pre_nType; j++) {
	            	if (i < pre_typeAcc[j]) {
	            		jp = j;
	            		break;
	            	}
	            }
				if (!typeConnected[jp]) {
    	        	Float p = abs(conMat[mid]);
    	        	if (p > 0) {
						if (count == 0) {
							p *= sumP[jp];
						} 
    	        	    Float xrand = uniform(&localState);
    	        	    if (xrand < p) {
    	        	        Float str = p>1?p:1;
    	        	        sumType[jp] ++;
    	        	        sumStrType[jp] += str;
							conMat[mid] = p;
    	        	    } else {
							conMat[mid] = -p;
						}
    	        	} 
				}
    	    }
    	}
    	if (nFar > 0) {
    	    for (Size iNab=nNear; iNab<nNab; iNab++) {
    	        #pragma unroll
    	        for (Size i=0; i<prePerBlock; i++) {
    	            Size tid = (iNab-nNear)*prePerBlock + i;
    	            Size jp;
                    #pragma unroll
	                for (PosInt j=0; j<pre_nType; j++) {
	                	if (i < pre_typeAcc[j]) {
	                		jp = j;
	                		break;
	                	}
	                }
					if (!typeConnected[jp]) {
    	            	Float p = tempNeighbor[tid];
    	            	if (p > 0) {
							if (count == 0) {
    	            	    	p *= sumP[jp];
								tempNeighbor[tid] = p;
							}
							if (sumType[ip] + nid[jp] < max_N[ip]) {
    	            	    	Float xrand = uniform(&localState);
    	            	    	if (xrand < p)  {
    	        	                Float str = p>1?p:1;
    	            	    	    sumType[jp] ++;
    	            	    	    sumStrType[jp] += str;
    	            	    	    __vecID[jp][nid[jp]] = tid;
    	            	    	    nid[jp]++;
    	            	    	}
							}
    	            	}
					}
    	        }
    	    }
    	}
		connected = true;
		for (PosInt i=0;i<pre_nType;i++) {
			typeConnected[i] = (sumType[i] <= ceiling(pN[i] + square_root(pN[i]))) && (sumType[i] >= flooring(pN[i] - square_root(pN[i])));
			if (!typeConnected[i]) {
				connected = false;
			}
		}
		count++;
        if (count > 0) {
            delete []sumP;
        }
		if (count > 100) {
		    for (PosInt i=0;i<pre_nType;i++) {
                if (!typeConnected[i]) {
			        printf("neuron %u-%u need to make another round(%u) of presynaptic connection of type %u, connected %u/%u (%u)-(%.1f%%)\n", blockIdx.x, threadIdx.x, count, i, sumType[i],pN[i],availType[i], pN[i]/static_cast<float>(availType[i])*100);
                }
            }

			//assert(count <= 100);
		}
	}
	__syncthreads();
	delete []typeConnected;
	delete []availType;

	Size total_nid = 0;
	for (PosInt i=0; i<pre_nType; i++) {
		total_nid += nid[i];
	}
    nVec[id] = total_nid;
    
    Float *ratio = new Float[pre_nType];
    if (strictStrength) {
        for (Size i=0; i<pre_nType; i++) {
            if (sumStrType[i] > 0) {
                ratio[i] = pN[i]/sumStrType[i];
            } else {
                ratio[i] = 0;
            }
        }
    }
    #pragma unroll
    for (Size i=0; i<pre_nType; i++) {
        preTypeConnected[i*postSize + id] = sumType[i];
        sumStrType[i] = 0;
    }
    delete []sumType;

    // ======== strictly normalize the strengths ==========
    for (Size iNab=0; iNab<nNear; iNab++) {
        for (Size i=0; i<prePerBlock; i++) {
            size_t mid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + iNab)*prePerBlock + i)*blockDim.x + threadIdx.x;
            Size jp;
            #pragma unroll
	        for (PosInt j=0; j<pre_nType; j++) {
	        	if (i < pre_typeAcc[j]) {
	        		jp = j;
	        		break;
	        	}
	        }
			Float p = conMat[mid]>1;
    	    Float str = (p>0?p:0)>1?p:1;
    		if (strictStrength) {
            	 str *= ratio[jp];
			}
			conMat[mid] = str;
			if (str > 0) {
            	sumStrType[jp] += str;
			}
        }
    }
	PosInt* qid = new PosInt[pre_nType];
	for (PosInt i=0; i<pre_nType; i++) {
		qid[i] = 0;
	}
    if (nFar > 0) {
        Size iid = 0;
        for (Size iNab=nNear; iNab<nNab; iNab++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + iNab] * prePerBlock;
            #pragma unroll
            for (PosInt i=0; i<(prePerBlock + blockDim.x-1)/blockDim.x; i++) {
                PosInt tid = i*blockDim.x + threadIdx.x;
                if (tid < prePerBlock) {
                    x1[tid] = pre_pos[bid + tid];
                    y1[tid] = pre_pos[prePerBlock + bid + tid];
                }
            }
            __syncthreads();
	    	if (iid < total_nid) {
                //#pragma unroll
                for (Size i=0; i<prePerBlock; i++) {
                    Size tid = (iNab-nNear)*prePerBlock + i;
	    	    	PosInt ipre = bid + i;
                    PosInt jp;
                    #pragma unroll
	                for (PosInt j=0; j<pre_nType; j++) {
	                	if (ipre < pre_typeAcc[j]) {
	                		jp = j;
	                		break;
	                	}
                    }
	    	    	if (qid[jp] >= nid[jp]) { // skip the rest of the type
	    	    		i = pre_typeAcc[jp]-1;
	    	    		continue;
	    	    	}
	    	    	if (__vecID[jp][qid[jp]] == tid) {
	    	    		Float p = tempNeighbor[tid];
    	                Float str = (p>0?p:0)>1?p:1;
        	    		if (strictStrength) {
	    	    			str *= ratio[jp];
	    	    		}
	    	    		vecID[maxDistantNeighbor*id + iid] = ipre;
	    	    		conVec[maxDistantNeighbor*id + iid] = str;
                    	Float x = static_cast<Float>(x1[i] - x0);
                    	Float y = static_cast<Float>(y1[i] - y0);
	    	    		Float distance = square_root(x*x + y*y);
                    	delayVec[maxDistantNeighbor*id + iid] = distance;
                    	sumStrType[jp] += str;
                    	iid ++;
	    	    		qid[jp]++;
                    	if (iid > maxDistantNeighbor) {
                    	    printf("set bigger maxDistantNeighbor, currently %u\n", maxDistantNeighbor);
                    	    assert(iid <= maxDistantNeighbor);
                    	}
	    	    	}
	    	    	if (iid >= total_nid) {
	    	    		break;
	    	    	}
                }
            }
            __syncthreads();
        }
    }
	delete []__vecID;
	delete []qid;
    delete []ratio;
    #pragma unroll
    for (Size i=0; i<pre_nType; i++) {
        preTypeStrSum[i*preSize + id] = sumStrType[i];
    }
    //if (threadIdx.x == 0) {
        //printf("done#%u\n", blockIdx.x);
    //}
	delete []nid;
    delete []pN;

	// Gap Junction
	typeConnected = new bool[gapType];
    for (Size i=0; i<gapType; i++) {
		typeConnected[i] = false;
    	sumGapP[i] = gap_pN[i]/sumGapP[i];
	}
	count = 0;
	connected = false;
 	Size* sumGapType = new Size[gapType];
 	Float* sumGapStrType = new Float[gapType];
	while (!connected) {
    	for (Size i=0; i<gapType; i++) {
			if (!typeConnected[i]) {
    	    	sumGapType[i] = 0;
    	    	sumGapStrType[i] = 0;
			}
		}
    	for (Size iNab=0; iNab<nGapNab; iNab++) {
    	    for (Size i=0; i<pre_n; i++) {
        		size_t gid = (static_cast<size_t>(blockIdx.x*gapNab + iNab)*pre_n + i-pre_n0)*post_n + threadIdx.x-post_n0; // defined outside, so delayMat has access to it
                PosInt jp;
                #pragma unroll
	            for (PosInt j=0; j<gapType; j++) {
	            	if (i + pre_n0 < pre_typeAcc[j + gapType0]) {
	            		jp = j;
	            		break;
	            	}
	            }
				if (!typeConnected[jp]) {
    	        	Float p = abs(gapMat[gid]);
    	        	if (p > 0) {
						if (count == 0) {
							p *= sumGapP[jp];
						} 
    	        	    Float xrand = uniform(&localState);
    	        	    if (xrand < p) {
    	        	        Float str = p>1?p:1;
    	        	        sumGapType[jp] ++;
    	        	        sumGapStrType[jp] += str;
							gapMat[gid] = p;
    	        	    } else {
							gapMat[gid] = -p;
						}
    	        	} 
				}
    	    }
    	}
		connected = true;
		for (PosInt i=0;i<gapType;i++) {
			typeConnected[i] = (sumGapType[i] <= ceiling(gap_pN[i]+square_root(gap_pN[i]))) && (sumGapType[i] >= flooring(gap_pN[i] - square_root(gap_pN[i])));
			if (!typeConnected[i]) {
				connected = false;
			}
		}
		count++;
		if (count > 100 && !connected) {
		    for (PosInt i=0;i<gapType;i++) {
                if (!typeConnected[i]) {
			        printf("neuron %u-%u need to make another round(%u) of gap junctions of type %u, because of %u/%u (%u)-(%.1f%%)\n", blockIdx.x, threadIdx.x, count, i, sumGapType[i], gap_pN[i], availGapType[i], gap_pN[i]/static_cast<float>(availGapType[i])*100);
                }
            }
		}
		if (count == 1) {
    		delete []sumGapP;
		}
	}
	state[id] = localState;
	__syncthreads();
	delete []typeConnected;
	delete []availGapType;
    	
    ratio = new Float[gapType];
    if (strictStrength) {
        for (Size i=0; i<gapType; i++) {
            if (sumGapStrType[i] > 0) {
                ratio[i] = gap_pN[i]/sumGapStrType[i];
            } else {
                ratio[i] = 0;
            }
        }
    }

	id = blockId*post_n + threadIdx.x-post_n0;
    #pragma unroll
    for (Size i=0; i<gapType; i++) {
        if (ip < post_nTypeE) preTypeGapE[i*mE + id] = sumGapType[i];
        else preTypeGapI[i*mI + id] = sumGapType[i];
		assert(sumGapType[i] <= ceiling(gap_pN[i] + square_root(gap_pN[i])) && sumGapType[i] >= flooring(gap_pN[i] - square_root(gap_pN[i])));
        sumGapStrType[i] = 0;
    }
    delete []sumGapType;

    // ======== strictly normalize the strengths ==========
	for (Size iNab=0; iNab<nGapNab; iNab++) {
        for (Size i=0; i<pre_n; i++) {
        	size_t gid = (static_cast<size_t>(blockIdx.x*gapNab + iNab)*pre_n + i-pre_n0)*post_n + threadIdx.x-post_n0; // defined outside, so delayMat has access to it
            PosInt jp;
            #pragma unroll
	        for (PosInt j=0; j<gapType; j++) {
	        	if (i + pre_n0 < pre_typeAcc[j + gapType0]) {
	        		jp = j;
	        		break;
	        	}
	        }
			Float p = gapMat[gid];
    	    Float str = (p>0?p:0)>1?p:1;
    		if (strictStrength) {
            	 str *= ratio[jp];
			}
			gapMat[gid] = str;
			if (str > 0) {
            	sumGapStrType[jp] += str;
			}
        }
    }
    delete []ratio;
    #pragma unroll
    for (Size i=0; i<gapType; i++) {
        if (ip < post_nTypeE) preTypeStrGapE[i*mE + id] = sumGapStrType[i];
        else preTypeStrGapI[i*mI + id] = sumGapStrType[i];
    }
    delete []sumGapStrType;
    delete []gap_pN;
	state[id] = localState;
}

__global__ // <<< nCluster, nI>>>
void generate_symmetry(
					curandStateMRG32k3a* __restrict__ state,
                    PosInt* __restrict__ clusterID,
                    Size*   __restrict__ gap_nTypeMat,
				    PosInt* __restrict__ neighborBlockId,
					Int*    __restrict__ neighborMat,
                    PosInt* __restrict__ blockAcc,
					Float*  __restrict__ postGapMat,
					Float*  __restrict__ clusterGapMat,
					Size*   __restrict__ preTypeGap,
					Float*  __restrict__ preTypeStrGap,
					Size*   __restrict__ postTypeGap,
					Float*  __restrict__ postTypeStrGap,
					PosInt* __restrict__ i_outstanding,
					Float*  __restrict__ v_outstanding,
                    Size**  __restrict__ typeAcc0,
					PosInt iblock, PosInt iLayer, PosInt jLayer, Size nLayer, Size gapNeighborBlock, Size postN, Size preN, Size postPerBlock0, Size prePerBlock0, Size prePerBlock, PosInt preTypeID, PosInt postTypeID, Size totalGapType, Size pre_nType, Size pre_nType0, Size post_nType, Size post_nType0, bool strictStrength, BigSize seed)
{
    extern __shared__ Size preTypeAcc[];
    Size* nMat = preTypeAcc + pre_nType;

    if (threadIdx.x < pre_nType) {
        preTypeAcc[threadIdx.x] = typeAcc0[jLayer][pre_nType0 + threadIdx.x+1]; // without 0 in the front
    }
    if (threadIdx.x < post_nType * pre_nType) {
        PosInt mat_id = (preTypeID + threadIdx.x/post_nType)*totalGapType + postTypeID + threadIdx.x%post_nType;
        nMat[threadIdx.x] = gap_nTypeMat[mat_id];
    }
    __syncthreads();

	PosInt id = iblock*blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[threadIdx.x];
    if (jLayer == 0) {
        curand_init(seed, threadIdx.x, 0, &localState);
    }
	// sum up reciprocal connections
	PosInt guest_bid = neighborBlockId[clusterID[blockIdx.x]]; 

	size_t home_id = static_cast<size_t>(clusterID[blockIdx.x]*blockDim.x*prePerBlock);
	int nid = neighborMat[(blockAcc[iLayer] + iblock)*blockAcc[nLayer-1] + (guest_bid + blockAcc[jLayer])]; // from post to pre (for guest)
	size_t guest_id = static_cast<size_t>(blockIdx.x*gapNeighborBlock + nid)*blockDim.x*prePerBlock; // gapNeighborBlock must be symmetric
	Size n_dir = 0;
	Size n_reciprocal = 0;
	Size n_outstanding = 0;
	PosInt* i_os = i_outstanding + static_cast<size_t>(blockIdx.x*blockDim.x + threadIdx.x)*prePerBlock;
	Float* v_os = v_outstanding + static_cast<size_t>(blockIdx.x*blockDim.x + threadIdx.x)*prePerBlock;
	// debuging parameters
	PosInt pbid = 2;
	PosInt ptid = 0;
	PosInt pib = 0;
	bool debug = false;

	for (PosInt i=0; i<prePerBlock; i++) {
		PosInt local_id = home_id + i*blockDim.x + threadIdx.x;
		assert(local_id < blockDim.x*prePerBlock*gapNeighborBlock);
		Float home_v = postGapMat[local_id];

		local_id = guest_id + threadIdx.x*prePerBlock + i;
		assert(local_id >= blockIdx.x *  prePerBlock * blockDim.x * gapNeighborBlock);
		assert(local_id < (blockIdx.x+1)*prePerBlock * blockDim.x * gapNeighborBlock);

		Float guest_v = clusterGapMat[local_id];
		// mark home<-guest as positive, guest<-home as negative
		if (clusterID[blockIdx.x] == pbid && threadIdx.x == ptid && iblock == pib && debug) {
			if ((home_v > 0 && guest_v <= 0) || (home_v <= 0 && guest_v > 0)) {
				if (guest_bid != iblock || i > threadIdx.x) { 
					printf("before: %u<-%u(%.3e), %u<-%u(%.3e)\n", threadIdx.x, i, home_v, i, threadIdx.x, guest_v);
				}
			}
		}
		if (home_v > 0) {
			n_dir++;
			if (guest_v > 0) {
				n_reciprocal++;
			} else {
				if (!(iLayer == jLayer && iblock == guest_bid) || i > threadIdx.x) { // if the home and guest block are the same, need to avoid conflict
					i_os[n_outstanding] = i;
					v_os[n_outstanding] = home_v;
					n_outstanding++; // outstands from guest to home 
				}
			}
		} else {
			if (guest_v > 0) {
				if (!(iLayer == jLayer && iblock == guest_bid) || i > threadIdx.x) { 
					i_os[n_outstanding] = i;
					v_os[n_outstanding] = -guest_v;
					n_outstanding++; // outstands from home to guest
				}
			}
		}
	}

	PosInt home_type = post_nType0;
    for (PosInt i=0; i<post_nType; i++) {
        if (threadIdx.x + postPerBlock0 < typeAcc0[iLayer][post_nType0 + i+1]) {
            home_type += i;
            break;
        }
    }
	if (n_outstanding > 0) {
		// decide remaining con prob
		Float prob = static_cast<Float>(n_dir - n_reciprocal)/n_outstanding; // n_dir comes from the  preset gap_nTypeMat, aim is to make n_dir reciprocal gap junctions
		if (iblock != guest_bid) {
			assert(prob <= 1);
		}
		// make reamining connections
		Float* deltaStr = new Float[pre_nType];
		int* nDelta = new int[pre_nType];
		for (PosInt i=0; i<pre_nType; i++) {
			deltaStr[i] = 0;
			nDelta[i] = 0;
		}
		for (PosInt i=0; i<n_outstanding; i++) {
			PosInt gid = guest_bid*prePerBlock + i_os[i];
			PosInt guest_type = post_nType0;
            for (PosInt j=0; j<pre_nType; j++) {
                if (i_os[i] + prePerBlock0 < preTypeAcc[j]) {
                    guest_type += j;
                    break;
                }
            }
            Float v = copyms(abs(v_os[i])>1?abs(v_os[i]):1, v_os[i]);
            PosInt cluster_id = guest_id + threadIdx.x*prePerBlock + i_os[i];
            PosInt post_id = home_id + i_os[i]*blockDim.x + threadIdx.x;
			Float xrand = uniform(&localState);
			if (clusterID[blockIdx.x] == pbid && threadIdx.x == ptid && iblock == pib && debug) {
				printf("%u-%u: %f\n",i, i_os[i], v);
			}

			if (xrand < prob) {
				if (v > 0) { // assign home->guest
					clusterGapMat[cluster_id] = v;	
					atomicAdd(preTypeGap + home_type * preN + gid, 1);
					atomicAdd(preTypeStrGap + home_type * preN + gid, v);
				} else { // assign guest->home
					postGapMat[post_id] = -v;	
					nDelta[guest_type]++;
					deltaStr[guest_type] -= v;// v is negative
				}
			} else {
				if (v > 0) { // remove guest->home
					postGapMat[post_id] = 0;	
					nDelta[guest_type]--;
					deltaStr[guest_type] -= v;
				} else {
					clusterGapMat[cluster_id] = 0; // remove home->guest
					atomicSub(preTypeGap + home_type * preN + gid, 1);
					atomicAdd(preTypeStrGap + home_type * preN + gid, v);// v_os[i] is negative
				}
			}
			Float home_v = postGapMat[post_id];	
			Float guest_v = clusterGapMat[cluster_id];	
			//
			if (home_v > 0) {
				if (guest_v <= 0) {
					printf("home_v = %f, guest_v = %f, v_os = %f\n", home_v, guest_v, v);
					assert(guest_v > 0);
				}
			} else {
				if (guest_v > 0) {
					printf("home_v = %f, guest_v = %f, v_os = %f\n", home_v, guest_v, v);
					assert(guest_v <= 0);
				}
			}
			//
		}
        if (iLayer == jLayer && iblock == guest_bid) {
		    for (PosInt i=0; i<pre_nType; i++) {
		    	atomicAdd(postTypeGap + i * postN + id, nDelta[i]);
		    	atomicAdd(postTypeStrGap + i * postN + id, deltaStr[i]);
		    }
        } else {
		    for (PosInt i=0; i<pre_nType; i++) {
		    	postTypeGap[i * postN + id] += nDelta[i];
		    	postTypeStrGap[i * postN + id] += deltaStr[i];
		    }
        }
		delete []deltaStr;
		delete []nDelta;
	}
	
	if (clusterID[blockIdx.x] == pbid && threadIdx.x == ptid && iblock == pib && debug) {
		printf("home block %u, guest block %u\n", iblock, guest_bid);
	}
	//certify symmetry and make strict strength
    Float *ratio = new Float[pre_nType];
	for (PosInt i=0; i<pre_nType; i++) {
		ratio[i] = nMat[i*post_nType + home_type]/postTypeStrGap[i * postN + id];
    }
	for (PosInt i=0; i<prePerBlock; i++) {
		Float home_v = postGapMat[home_id + i*blockDim.x + threadIdx.x];
        if (strictStrength) {
			PosInt guest_type = post_nType0;
            for (PosInt j=0; j<pre_nType; j++) {
                if (i_os[i] + prePerBlock0 < preTypeAcc[j]) {
                    guest_type += j;
                    break;
                }
            }
            postGapMat[home_id + i*blockDim.x + threadIdx.x] = home_v*ratio[guest_type];
        }
		Float guest_v = clusterGapMat[guest_id + threadIdx.x*prePerBlock + i];
		if (home_v > 0) {
			if (guest_v <= 0) {
				printf("%u:%u(%u)-%u, guest_v = %f, home_v =%f\n", iblock, clusterID[blockIdx.x], blockIdx.x, threadIdx.x, guest_v, home_v);
				assert(guest_v > 0);
			}
		} else {
			if (guest_v > 0) {
				printf("%u:%u(%u)-%u, guest_v = %f, home_v =%f\n", iblock, clusterID[blockIdx.x], blockIdx.x, threadIdx.x, guest_v, home_v);
				assert(guest_v <= 0);
			}
		}
	}
    delete []ratio;
}
