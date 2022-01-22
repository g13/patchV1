#include "connect.h"

extern __device__ __constant__ pFeature pref[];

// TODO: randomize neuronal attributes by using distribution, strength x number of con. should be controlled
__global__ 
void initialize(curandStateMRG32k3a* __restrict__ state,
			Size*  __restrict__ preType, // 
			Float* __restrict__ rden,
			Float* __restrict__ raxn,
			Float* __restrict__ dden,
			Float* __restrict__ daxn,
			Float* __restrict__ preF_type,
			Float* __restrict__ preS_type,
			Size*  __restrict__ preN_type,
			Size* __restrict__ nLGN_V1,
			Float* __restrict__ ExcRatio,
			Float* __restrict__ extExcRatio,
            Float* __restrict__ synPerCon,
            Float* __restrict__ synPerConFF,
			Float min_FB_ratio, Float C_InhRatio, initialize_package init_pack, unsigned long long seed, Size networkSize, Size nType, Size nArchtype, Size nFeature, bool CmoreN, bool ClessI, Float preset_nLGN)
{
    //__shared__ reduced[warpSize];
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
	curandStateMRG32k3a localState = state[id];
    curand_init(seed, id, 0, &localState);
	state[id] = localState;
   	Size type;
	// determine the arch neuronal type and its properties
	#pragma unroll
    for (Size i=0; i<nType; i++) {
        if (threadIdx.x < init_pack.typeAccCount[i]) {
            rden[id] = init_pack.rden[i];
            raxn[id] = init_pack.raxn[i];
            dden[id] = init_pack.dden[i];
            daxn[id] = init_pack.daxn[i];
            preType[id] = i;
            type = i;
            break;
        }
	}


	Float preset_LGN_syn = preset_nLGN*synPerConFF[type];
	Float preset_Cortical_syn = init_pack.nTypeMat[type]*synPerCon[type];
	//Float totalExcSyn = preset_LGN_syn + preset_Cortical_syn;

	Float LGN_syn = nLGN_V1[id]*synPerConFF[type];
    Float inhRatio = 1.0;
    Float ratio;
	if (init_pack.nTypeMat[type] == 0 || synPerCon[type] == 0) {
		ratio = 1.0;
	} else {
		ratio = (preset_Cortical_syn - (LGN_syn - preset_LGN_syn))/preset_Cortical_syn;
		if (ClessI) {
			inhRatio = LGN_syn/preset_LGN_syn;
		}
	}
    if (ratio < min_FB_ratio) ratio = min_FB_ratio;
    if (inhRatio < C_InhRatio) inhRatio = C_InhRatio;

	ExcRatio[id] = ratio;
	for (PosInt i=0; i<nType; i++) {
		PosInt tid = i*networkSize+id;
		PosInt ttid = i*nType + type;
        if (CmoreN) {
			Float excessRatio = 1.0;
            if (i < init_pack.iArchType[0]) {
				Float fpreN = ratio*init_pack.nTypeMat[ttid]*(1-extExcRatio[type]);
				Size preN = static_cast<Size>(rounding(fpreN));
                preN_type[tid] = preN;
				excessRatio *= fpreN/preN;
            } else {
				Float fpreN = init_pack.nTypeMat[ttid]*inhRatio;
				Size preN = static_cast<Size>(rounding(fpreN));
    	        preN_type[tid] = preN;
				excessRatio *= fpreN/preN;
				//if (blockIdx.x == 0) {
				//	printf("%u inhratio = %.2f, preN = %u\n", threadIdx.x, inhRatio, preN);
				//}
            }
    	    preS_type[tid] = init_pack.sTypeMat[ttid]*excessRatio;
        } else {
    	    preN_type[tid] = init_pack.nTypeMat[ttid];
            if (i < init_pack.iArchType[0]) {
                preS_type[tid] = ratio*init_pack.sTypeMat[ttid]*(1-extExcRatio[type]);
            } else {
    	        //preS_type[tid] = (ratio+ LGN_sSum/init_pack.sumType[i*nType+0])*init_pack.sTypeMat[ttid];
    	        preS_type[tid] = init_pack.sTypeMat[ttid]*inhRatio;
            }
        }
		//printf("%u-%u-%u: LGN_sSum = %f,  %u*%f = %f\n", blockIdx.x, threadIdx.x, i, LGN_sSum, preN_type[tid], preS_type[tid], preN_type[tid] * preS_type[tid]);
        for (PosInt j=0; j<nFeature; j++) {
            PosInt fid = (j*nType + i)*networkSize + id;
            preF_type[fid] = init_pack.typeFeatureMat[j*nType*nType+ttid];
        }
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
}

__global__ 
void cal_blockPos(double* __restrict__ pos,
                  Float* __restrict__ block_x,
                  Float* __restrict__ block_y,
                  Size networkSize) 
{
    __shared__ double reduced[warpSize];
    Size id = blockDim.x*blockIdx.x + threadIdx.x;
    double x = pos[id];
    double y = pos[id + gridDim.x*blockDim.x];
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
                          PosInt* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
                          Size* __restrict__ nNearNeighborBlock,
						  Size nblock, Float radius, Float max_radius, Size maxNeighborBlock) 
{
    __shared__ PosInt id[warpSize];
    __shared__ Float min[warpSize];
    __shared__ Int bid[blockSize];
	__shared__ Float distance[blockSize];

	extern __shared__ Float final_distance[];
	PosInt* final_bid = (PosInt*) (final_distance + maxNeighborBlock);

    Float bx = block_x[blockIdx.x]; // center of the target block
    Float by = block_y[blockIdx.x];
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
	//if (blockIdx.x == 0 && threadIdx.x ==0) {
	//	printf("center block %i, (%f,%f)\n", blockIdx.x, bx, by);
	//}

    Size nPatch = (nblock + blockDim.x-1)/blockDim.x - 1;
    Size remain = nblock%blockDim.x;
	if (remain == 0) {
		remain = blockDim.x;
	}

    Size offset = 0;
    if (tid == 0) {
        id[0] = 0;
        id[1] = 1; // 1 is correct, first self will be assigned
    }
    bid[tid] = -1;
    __syncthreads();
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        if (iPatch < nPatch || tid < remain) {
            PosInt blockId = offset + threadIdx.x;
            Float x = block_x[blockId] - bx;
            Float y = block_y[blockId] - by;
            Float dis = square_root(x*x + y*y);
            distance[tid] = dis;
            if (dis < max_radius) {
                bid[tid] = blockId;
            }
        }
        __syncthreads();
        if (tid == 0) { // rearrange
			// assign self first
			neighborBlockId[maxNeighborBlock*blockIdx.x] = blockIdx.x;
			PosInt outside_id = id[0];
            PosInt current_id = id[1];
            for (PosInt i=0; i<blockDim.x; i++) {
				//if (blockIdx.x == 0) {
				//	printf("patch %u, block %i(%i, (%f,%f)), distace = %f< %f(%f)\n", iPatch, i+offset, bid[i], block_x[i+offset], block_y[i+offset], distance[i], radius, max_radius);
				//}
                if (bid[i] != -1 && bid[i] != blockIdx.x) {
					if (distance[i] < radius) {
						neighborBlockId[maxNeighborBlock*blockIdx.x + current_id] = bid[i];
                    	current_id++;
					} else {
                    	final_distance[outside_id] = distance[i]; 
                    	final_bid[outside_id] = bid[i]; 
						outside_id++;
					}
                    bid[i] = -1; 
                    if (current_id + outside_id > maxNeighborBlock) {
                        printf("actual nNeighbor = %d + %d > %d (preserved)\n", current_id, outside_id, maxNeighborBlock);
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
        /*DEBUG
            if (blockIdx.x == 2) {
                printf("preSort#%u:%u, %e\n", tid, final_bid[tid], dis);
            }
        */
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
		/* DEBUG
            if (blockIdx.x == 2 && tid < nb) {
                printf("%u#%u:%u, %e\n",i, tid, final_bid[tid], dis);
            }
            __syncthreads();
        */
    }
    //TODO: if nb > blockSize
    if (tid < nb) {
        neighborBlockId[maxNeighborBlock*blockIdx.x + nn + tid] = local_bid;
    }
	/* DEBUG
        if (blockIdx.x == 2) {
	        for (PosInt j = 0; j<nb; j++) {
	        	if (tid == j) {
	        		if (j == 0) {
	        			printf("block %u, %u neighbors in total\n", blockIdx.x, nb);
	        		}
	        		printf("#%u, %u: %f -> %f,", j, local_bid, final_distance[j], dis);
	        		if (j == nb-1) {
	        			printf("\n");
	        		}
	        	}
	        	__syncwarp();
	        }
        }
	*/
}

//__launch_bounds__(1024,1)
__global__ 
void generate_connections(double* __restrict__ pos,
                          Float* __restrict__ preF_type,
                          Float* __restrict__ gap_preF_type,
                          Float* __restrict__ preS_type,
                          Float* __restrict__ gap_preS_type,
                          Size* __restrict__ preN_type,
                          Size* __restrict__ gap_preN_type,
                          PosInt* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
                          Size* __restrict__ nNearNeighborBlock,
                          Float* __restrict__ rden,
                          Float* __restrict__ raxn,
                          Float* __restrict__ conMat, //within block connections
                          Float* __restrict__ delayMat,
                          Float* __restrict__ gapMat,
                          Float* __restrict__ conVec, //for neighbor block connections
                          Float* __restrict__ delayVec, //for neighbor block connections
                          Float* __restrict__ gapVec,
                          Float* __restrict__ gapDelayVec,
                          Size* __restrict__ max_N,
                          PosInt* __restrict__ _vecID,
                          Float* __restrict__ disNeighborP,
                          Float* __restrict__ gap_disNeighborP,
                          Size* __restrict__ vecID,
                          Size* __restrict__ nVec,
                          Size* __restrict__ gapVecID,
                          Size* __restrict__ nGapVec,
                          Size* __restrict__ preTypeConnected,
                          Size* __restrict__ preTypeAvail,
                          Float* __restrict__ preTypeStrSum,
                          Size* __restrict__ preTypeGapped,
                          Float* __restrict__ preTypeStrGapped,
                          Size* __restrict__ preType,
                          Float* __restrict__ feature,
                          Float* __restrict__ dden,
                          Float* __restrict__ daxn,
                          Float* __restrict__ synloc,
                          Size* __restrict__ typeAcc0,
                          curandStateMRG32k3a* __restrict__ state,
                          Size sum_max_N, Size gap_sum_max_N, PosInt block_offset, Size networkSize, Size mI, Size maxDistantNeighbor, Size gap_maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Size nType, Size nTypeE, Size nTypeI, Size nE, Size nI, Size nFeature, Float disGauss, bool strictStrength, Float tol) 
{
    // TODO: load with warps but more, e.g., raxn, daxn, preType
    __shared__ double x1[blockSize];
    __shared__ double y1[blockSize];
    //__shared__ Float ra[blockDim.x];
    Size blockId = blockIdx.x + block_offset;
    Size nn = nNeighborBlock[blockId];
    Size ni = nNearNeighborBlock[blockId];
    Size offset = blockId*blockDim.x;
    double x0 = pos[offset + threadIdx.x];
    double y0 = pos[offset + threadIdx.x + networkSize];
    Size id = offset + threadIdx.x;
    assert(id < networkSize);
    // number of potential presynaptic connections outsied nearNeighbors, to be stored in vector.
    Size nb = nn - ni; 
    Float* tempNeighbor = disNeighborP + static_cast<size_t>((blockIdx.x*blockDim.x + threadIdx.x)*(maxNeighborBlock-nearNeighborBlock))*blockDim.x;
    Float* gap_tempNeighbor = gap_disNeighborP;
    Float rd = rden[id];
    Float dd = dden[id];
	PosInt itype;
	for (PosInt i=0; i<nType; i++) {
		if (threadIdx.x < typeAcc0[i+1]) {
			itype = i;
			break;
		}
	}
	if (itype >= nTypeE) {
 		gap_tempNeighbor = gap_disNeighborP + static_cast<size_t>((blockIdx.x*nI + threadIdx.x-nE)*(maxNeighborBlock-nearNeighborBlock))*nI;
	}
	Float* postSynLoc = new Float[nType];

    Size* availType = new Size[nType]; // avail
    Float* sumP = new Float[nType];
	Size* availInhType;
	Float* sumInhP;
	Float* gap_pF;
	if (itype >= nTypeE) {
    	sumInhP = new Float[nTypeI];
    	availInhType = new Size[nTypeI];
    	gap_pF = new Float[nFeature*nTypeI];
	}
    Float* pF = new Float[nFeature*nType];
    Float* fV = new Float[nFeature];
    #pragma unroll
    for (PosInt i=0; i<nType; i++) {
        availType[i] = 0;
        sumP[i] = 0.0;
		if (itype >= nTypeE && i >= nTypeE) {
    		sumInhP[i-nTypeE] = 0.0;
			availInhType[i-nTypeE] = 0;
		}
		postSynLoc[i] = synloc[nType*i + itype];
	}
    for (PosInt i=0; i<nFeature; i++) {
        fV[i] = feature[i*networkSize + id];
        for (PosInt j=0; j<nType; j++) {
            pF[i*nType + j] = preF_type[(i*nType+j)*networkSize + id];
			if (itype >= nTypeE && j >= nTypeE) {
    			gap_pF[i*nTypeI + j-nTypeE] = gap_preF_type[(i*nTypeI+j-nTypeE)*nTypeI + itype-nTypeE];
			}
        }
    }
    //============= collect p of all ==========
    // withhin block and nearNeighbor
    for (PosInt in=0; in<ni; in++) {
        PosInt bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x; // # neurons in all past blocks 
        x1[threadIdx.x] = pos[bid + threadIdx.x];
        y1[threadIdx.x] = pos[bid + threadIdx.x + networkSize];
        __syncthreads();
        #pragma unroll
        for (Size i=0; i<blockDim.x; i++) {
            // blockwise load from gmem the (potential) presynaptic neurons' properties
            Size ipre = bid + i; // pre-id in network
            Float ra = raxn[ipre];
            Float x = static_cast<Float>(x1[i] - x0);
            Float y = static_cast<Float>(y1[i] - y0);
            //type vector, indexed across the network
            Float distance = square_root(x*x + y*y);
	    	// weight from area
            Size ip = preType[ipre];
            Float p = connect(distance, ra, rd*postSynLoc[ip], disGauss);
            size_t mid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + in)*blockDim.x + i)*blockDim.x + threadIdx.x; // defined outside, so delayMat has access to it
            if (p > 0 && id != ipre) { // not self-connected
                /*
                size_t matSize = static_cast<size_t>(blockDim.x*blockDim.x)*nearNeighborBlock*gridDim.x;
                if (mid >= matSize) {
                    printf("(%ux%ux%ux%llu) = %u\n", gridDim.x, nearNeighborBlock, blockDim.x, blockDim.x, matSize);
                    assert(mid < matSize);
                }
                */
	    		// update weight with density of axon dendrites and preference over type
                p *= daxn[ipre] * dd;// * preP_type[ip*networkSize + id];
                //
				if (itype >= nTypeE && ip >= nTypeE) {
					Float gapP = p;
            		size_t gid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + in)*nI + i-nE)*nI + threadIdx.x-nE; // defined outside, so delayMat has access to it
                	for (Size iFeature = 0; iFeature < nFeature; iFeature++) {
						gapP *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], gap_pF[iFeature*nTypeI + ip-nTypeE]);
                	}
					availInhType[ip-nTypeE]++;
					sumInhP[ip-nTypeE] += gapP;
					gapMat[gid] = gapP;
				}
                for (Size iFeature = 0; iFeature < nFeature; iFeature++) {
					p *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], pF[iFeature*nType +ip]);
                }
                assert(p >= 0);
				if (p > 0) {
                	availType[ip]++;
                	sumP[ip] += p;
                	conMat[mid] = p;
				}
            }
            delayMat[mid] = distance; // record even if not connected, for LFP
        }
		__syncthreads();
    }
    // the remaining neighbors
    if (nb > 0) {
        for (Size in=ni; in<nn; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
			x1[threadIdx.x] = pos[bid + threadIdx.x];
        	y1[threadIdx.x] = pos[bid + threadIdx.x + networkSize];
            __syncthreads();
            #pragma unroll
            for (Size i=0; i<blockDim.x; i++) {
                // blockwise load from gmem the (potential) presynaptic neurons' properties
                Size ipre = bid + i; // pre-id in the network
                Float ra = raxn[ipre];
                double x = x1[i] - x0;
                double y = y1[i] - y0;
                Float distance = static_cast<Float>(square_root(x*x + y*y));
                Size ip = preType[ipre];
                Float p = connect(distance, ra, rd*postSynLoc[ip], disGauss);
                Size tid = (in-ni)*blockDim.x + i; // only ofr tempNeighbor, which is local, no need to coalease memory
                tempNeighbor[tid] = 0;
            	size_t gid; 
				if (itype >= nTypeE && ip >= nTypeE) {
					gid = (in-ni)*nI + i-nE;
					gap_tempNeighbor[gid] = 0;
				}
                if (p > 0) {
                    p *= daxn[ipre] * dd; // * preP_type[ip*networkSize+id];
					if (itype >= nTypeE && ip >= nTypeE) {
						Float gapP = p;
                		for (Size iFeature = 0; iFeature < nFeature; iFeature++) {
							gapP *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], gap_pF[iFeature*nTypeI + ip-nTypeE]);
                		}
						availInhType[ip-nTypeE]++;
						sumInhP[ip-nTypeE] += gapP;
						gap_tempNeighbor[gid] = gapP;
					}
                    for (Size iFeature = 0; iFeature < nFeature; iFeature ++) {
                        p *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], pF[iFeature*nType + ip]);
                    }
                    assert(p>=0);
					if (p > 0) {
                    	availType[ip]++;
                    	sumP[ip] += p;
                    	tempNeighbor[tid] = p;
					}
                }
            }
            __syncthreads();
        }
    }
	delete []postSynLoc;
    delete []pF;
    delete []fV;
    __syncwarp();
    Size* pN = new Size[nType];
	Size* gap_pN;
	if (itype >= nTypeE) {
		delete []gap_pF;
		gap_pN = new Size[nTypeI];
	}
    #pragma unroll
    for (Size i=0; i<nType; i++) {
        pN[i] = preN_type[i*networkSize + id];
        preTypeAvail[i*networkSize + id] = availType[i];
		if (availType[i] <pN[i]) {
			printf("neuron %u-%u dont have enough type %u neurons to connect to (%u/%u)\n", blockIdx.x, threadIdx.x, i, availType[i], pN[i]);
			assert(availType[i] >= pN[i]);
		}

		if (itype >= nTypeE && i >= nTypeE) {
			gap_pN[i-nTypeE] = gap_preN_type[(i-nTypeE)*nTypeI + itype-nTypeE];
			if (availInhType[i-nTypeE] < gap_pN[i-nTypeE]) {
				printf("neuron %u-%u dont have enough type %u inh neurons to make gap junction to (%u/%u)\n", blockIdx.x, threadIdx.x, i-nTypeE, availInhType[i-nTypeE], gap_pN[i-nTypeE]);
				assert(availInhType[i-nTypeE] >= gap_pN[i-nTypeE]);
			}
		}
    }
	__syncthreads();
    //============= redistribute p of all ==========
    bool* typeConnected = new bool[nType];
    Float* pS = new Float[nType];
	Size _sum_max_N = 0;
	PosInt** __vecID = new PosInt*[nType];
    #pragma unroll
    for (Size i=0; i<nType; i++) {
     	__vecID[i] = _vecID + (blockIdx.x*blockDim.x + threadIdx.x)*sum_max_N + _sum_max_N;
		_sum_max_N += max_N[i];
        pS[i] = preS_type[i*networkSize + id];
		typeConnected[i] = false;
		sumP[i] = pN[i]/sumP[i]; // now is a ratio
    }

    curandStateMRG32k3a localState = state[id];
	Size count = 0;
	Size connected = false;
    Size* sumType = new Size[nType];
    Float* sumStrType = new Float[nType];
    Size* nid = new Size[nType];
	do {
    	for (Size i=0; i<nType; i++) {
			if (!typeConnected[i]) {
    	    	sumType[i] = 0;
    	    	sumStrType[i] = 0;
				nid[i] = 0;
			}
		}
    	for (Size in=0; in<ni; in++) {
    	    PosInt bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
    	    //#pragma unroll
    	    for (Size i=0; i<blockDim.x; i++) {
    	        size_t mid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + in)*blockDim.x + i)*blockDim.x + threadIdx.x;
    	        Size ipre = bid + i;
    	        Size ip = preType[ipre];
				if (!typeConnected[ip]) {
    	        	Float p = abs(conMat[mid]);
    	        	if (p > 0) {
						if (count == 0) {
							p *= sumP[ip];
						} 
    	        	    Float xrand = uniform(&localState);
    	        	    if (xrand < p) {
    	        	        Float str = pS[ip];
    	        	        if (p > 1) {
    	        	            str = str*p;
    	        	        }
    	        	        sumType[ip] ++;
    	        	        sumStrType[ip] += str;
							conMat[mid] = p;
    	        	    } else {
							conMat[mid] = -p;
						}
    	        	} 
				}
    	    }
    	}
    	if (nb > 0) {
    	    for (Size in=ni; in<nn; in++) {
    	        Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
    	        #pragma unroll
    	        for (Size i=0; i<blockDim.x; i++) {
    	            Size tid = (in-ni)*blockDim.x + i;
    	            Size ipre = bid + i;
    	            Size ip = preType[ipre];
					if (!typeConnected[ip]) {
    	            	Float p = tempNeighbor[tid];
    	            	if (p > 0) {
							if (count == 0) {
    	            	    	p *= sumP[ip];
								tempNeighbor[tid] = p;
							}
							if (sumType[ip] + nid[ip] < max_N[ip]) {
    	            	    	Float xrand = uniform(&localState);
    	            	    	if (xrand < p)  {
    	            	    	    Float str = pS[ip];
    	            	    	    if (p > 1) {
    	            	    	        str = str*p;
    	            	    	    }
    	            	    	    sumType[ip] ++;
    	            	    	    sumStrType[ip] += str;
    	            	    	    __vecID[ip][nid[ip]] = tid;
    	            	    	    nid[ip]++;
                					//if (nid[ip] >= max_N[ip]) {
                					//    printf("set bigger max_N[\%u], currently %u\n", ip, max_N[ip]);
                					//    assert(nid[ip] <= max_N[ip]);
                					//}
    	            	    	}
							}
    	            	}
					}
    	        }
    	    }
    	}
		connected = true;
		for (PosInt i=0;i<nType;i++) {
			typeConnected[i] = (sumType[i] <= ceiling(pN[i]*(1+tol))) && (sumType[i] >= flooring(pN[i]*(1-tol)));
			if (!typeConnected[i]) {
				connected = false;
			}
		}
		count++;
		if (count > 100 && !connected) {
			printf("neuron %u-%u need to make another round(%u) of connection, because of %u/%u (%u)-(%.1f%%), %u/%u (%u)-(%.1f%%)\n", blockIdx.x, threadIdx.x, count, sumType[0],pN[0],availType[0], pN[0]/static_cast<float>(availType[0])*100, sumType[1],pN[1],availType[1], pN[1]/static_cast<float>(availType[1])*100);

			//assert(count <= 100);
		}
		/*if (count >= 100) {
			printf("neuron %u-%u don't have one (or any) of the types of neurons to connect to\n", blockIdx.x, threadIdx.x);
			assert(count < 100);
			//connected = true;
		}*/
	//} while (!connected)
	} while (false);
	__syncthreads();
    delete []sumP;
	delete []typeConnected;
	delete []availType;
	Size total_nid = 0;
	for (PosInt i=0; i<nType; i++) {
		total_nid += nid[i];
	}
    nVec[id] = total_nid;
    
    Float *ratio = new Float[nType];
    if (strictStrength) {
        for (Size i=0; i<nType; i++) {
            if (sumStrType[i] > 0) {
                ratio[i] = pS[i]*pN[i]/sumStrType[i];
            } else {
                ratio[i] = 0;
            }
        }
    }
    #pragma unroll
    for (Size i=0; i<nType; i++) {
        preTypeConnected[i*networkSize + id] = sumType[i];
		//assert(sumType[i] < round(pN[i]*1.5) && sumType[i] > round(pN[i]*2.f/3.f));
		//assert(sumType[i] <= ceiling(pN[i]*(1+tol)) && sumType[i] >= flooring(pN[i]*(1-tol)));
        sumStrType[i] = 0;
    }
    delete []sumType;

    // ======== strictly normalize the strengths ==========
    for (Size in=0; in<ni; in++) {
        Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
        //#pragma unroll
        for (Size i=0; i<blockDim.x; i++) {
            size_t mid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + in)*blockDim.x + i)*blockDim.x + threadIdx.x;
            Size ip = preType[bid + i];
			Float str = conMat[mid];
			if (str <= 0) str = 0;
			else {
				if (str > 1) str *= pS[ip];
				else str = pS[ip];
			}
    		if (strictStrength) {
            	 str *= ratio[ip];
			}
			conMat[mid] = str;
			if (str > 0) {
            	sumStrType[ip] += str;
			}
        }
    }
	PosInt* qid = new PosInt[nType];
	for (PosInt i=0; i<nType; i++) {
		qid[i] = 0;
	}
    if (nb > 0) {
        Size iid = 0;
        for (Size in=ni; in<nn; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
			x1[threadIdx.x] = pos[bid + threadIdx.x];
        	y1[threadIdx.x] = pos[bid + threadIdx.x + networkSize];
            __syncthreads();
	    	if (iid < total_nid) {
                //#pragma unroll
                for (Size i=0; i<blockDim.x; i++) {
                    Size tid = (in-ni)*blockDim.x + i;
	    	    	PosInt ipre = bid + i;
                    Size ip = preType[ipre];
	    	    	if (qid[ip] >= nid[ip]) {
	    	    		i = typeAcc0[ip+1]-1;
	    	    		continue;
	    	    	}
	    	    	if (__vecID[ip][qid[ip]] == tid) {
	    	    		Float p = tempNeighbor[tid];
	    	    		Float str = pS[ip];
	    	    		if (p > 1) str *= p;
        	    		if (strictStrength) {
	    	    			str *= ratio[ip];
	    	    		}
	    	    		vecID[maxDistantNeighbor*id + iid] = ipre;
	    	    		conVec[maxDistantNeighbor*id + iid] = str;
                    	Float x = static_cast<Float>(x1[i] - x0);
                    	Float y = static_cast<Float>(y1[i] - y0);
	    	    		Float distance = square_root(x*x + y*y);
                    	delayVec[maxDistantNeighbor*id + iid] = distance;
                    	sumStrType[ip] += str;
                    	iid ++;
	    	    		qid[ip]++;
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
    for (Size i=0; i<nType; i++) {
        preTypeStrSum[i*networkSize + id] = sumStrType[i];
		//if (strictStrength) {
		//	if (abs(sumStrType[i] - pN[i]*pS[i])/(pN[i]*pS[i]) > 1e-3) {
		//		printf("%u-%u-%u: sumStrType[i] = %f,  pN[i]*pS[i] = %f, count = %u, nid = %u\n", blockIdx.x, threadIdx.x, i, sumStrType[i], pN[i]*pS[i], count, nVec[id]);
		//		//assert(abs(sumStrType[i] - pN[i]*pS[i])/(pN[i]*pS[i]) <= 1e-3);
		//	}
		//}
    }
    //if (threadIdx.x == 0) {
        //printf("done#%u\n", blockIdx.x);
    //}
	delete []nid;
    delete []sumStrType;
    delete []pN;
    delete []pS;

	// Gap Junction
    Float* sumInhStrType;
    Size* sumInhType;
	Float* gap_pS;
	Size* gap_nid;
	PosInt** __gapVecID;
	_sum_max_N = 0;
	if (itype >= nTypeE) { // only utilizing a small number of threads
		typeConnected = new bool[nTypeI];
		gap_pS = new Float[nTypeI];
 		__gapVecID = new PosInt*[nTypeI];

    	for (Size i=0; i<nTypeI; i++) {
     		__gapVecID[i] = _vecID + (blockIdx.x*nI + threadIdx.x-nE)*gap_sum_max_N + _sum_max_N; // there's empty space between __vecID[i] and __vecID[i+1]
			_sum_max_N += gap_pN[i];
        	gap_pS[i] = gap_preS_type[i*nTypeI + itype-nTypeE];
			typeConnected[i] = false;
    		sumInhP[i] = gap_pN[i]/sumInhP[i];
		}
		Size count = 0;
		Size connected = false;
 		sumInhType = new Size[nTypeI];
 		sumInhStrType = new Float[nTypeI];
    	gap_nid = new Size[nTypeI];
		//while (!connected) {
		do {
    		for (Size i=0; i<nTypeI; i++) {
				if (!typeConnected[i]) {
    		    	sumInhType[i] = 0;
    		    	sumInhStrType[i] = 0;
					gap_nid[i] = 0;
				}
			}
    		for (Size in=0; in<ni; in++) {
    		    PosInt bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
    		    //#pragma unroll
    		    for (Size i=0; i<nI; i++) {
            		size_t gid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + in)*nI + i)*nI + threadIdx.x-nE; // defined outside, so delayMat has access to it
    		        Size ipre = bid + nE + i;
    		        Size ip = preType[ipre]-nTypeE;
					if (!typeConnected[ip]) {
    		        	Float p = abs(gapMat[gid]);
    		        	if (p > 0) {
							if (count == 0) {
								p *= sumInhP[ip];
							} 
    		        	    Float xrand = uniform(&localState);
    		        	    if (xrand < p) {
    		        	        Float str = gap_pS[ip];
    		        	        if (p > 1) {
    		        	            str = str*p;
    		        	        }
    		        	        sumInhType[ip] ++;
    		        	        sumInhStrType[ip] += str;
								gapMat[gid] = p;
    		        	    } else {
								gapMat[gid] = -p;
							}
    		        	} 
					}
    		    }
    		}
    		if (nb > 0) {
    		    for (Size in=ni; in<nNeighborBlock[blockId]; in++) {
    		        Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
    		        #pragma unroll
    		        for (Size i=0; i<nI; i++) {
    		            Size tid = (in-ni)*nI + i;
    		            Size ipre = bid + nE + i;
    		            Size ip = preType[ipre]-nTypeE;
						if (!typeConnected[ip]) {
    		            	Float p = gap_tempNeighbor[tid];
    		            	if (p > 0) {
								if (count == 0) {
    		            	    	p *= sumInhP[ip];
									gap_tempNeighbor[tid] = p;
								}
								if (sumInhType[ip] + gap_nid[ip] < gap_pN[ip]) {
    		            	    	Float xrand = uniform(&localState);
    		            	    	if (xrand < p) {
    		            	    	    Float str = gap_pS[ip];
    		            	    	    if (p > 1) {
    		            	    	        str = str*p;
    		            	    	    }
    		            	    	    sumInhType[ip] ++;
    		            	    	    sumInhStrType[ip] += str;
    		            	    	    __gapVecID[ip][gap_nid[ip]] = tid;
    		            	    	    gap_nid[ip]++;
    		            	    	}
								}
    		            	}
						}
    		        }
    		    }
    		}
			connected = true;
			for (PosInt i=0;i<nTypeI;i++) {
				typeConnected[i] = (sumInhType[i] <= ceiling(gap_pN[i]*(1+tol))) && (sumInhType[i] >= flooring(gap_pN[i]*(1-tol)));
				if (!typeConnected[i]) {
					connected = false;
				}
			}
			count++;
			if (count > 100 && !connected) {
				printf("neuron %u-%u need to make another round(%u) of gap junctions, because of %u/%u (%u)\n", blockIdx.x, threadIdx.x, count, sumInhType[0], gap_pN[0], availInhType[0]);
				//assert(count <= 100);
			}
			/*if (count >= 100) {
				printf("neuron %u-%u don't have one (or any) of the types of neurons to connect to\n", blockIdx.x, threadIdx.x);
				assert(count < 100);
				//connected = true;
			}*/
			if (count == 1) {
    			delete []sumInhP;
			}
		//} while (!connected)
		} while (false);
	}
	state[id] = localState;
	__syncthreads();
	if (itype >= nTypeE) { // only utilizing a small number of threads
		delete []typeConnected;
		delete []availInhType;
		id = blockId*nI + threadIdx.x-nE;
		total_nid = 0;
		for (PosInt i=0; i<nTypeI; i++) {
			total_nid += gap_nid[i];
		}
    	nGapVec[id] = total_nid;
    	
    	ratio = new Float[nTypeI];
    	if (strictStrength) {
    	    for (Size i=0; i<nTypeI; i++) {
    	        if (sumInhStrType[i] > 0) {
    	            ratio[i] = gap_pS[i]*gap_pN[i]/sumInhStrType[i];
    	        } else {
    	            ratio[i] = 0;
    	        }
    	    }
    	}
    	#pragma unroll
    	for (Size i=0; i<nTypeI; i++) {
    	    preTypeGapped[i*mI + id] = sumInhType[i];
			//assert(sumType[i] < round(pN[i]*1.5) && sumType[i] > round(pN[i]*2.f/3.f));
			//assert(sumInhType[i] <= ceiling(gap_pN[i]*(1+tol)) && sumInhType[i] >= flooring(gap_pN[i]*(1-tol)));
    	    sumInhStrType[i] = 0;
    	}
    	delete []sumInhType;

    	// ======== strictly normalize the strengths ==========
		for (Size in=0; in<ni; in++) {
    	    Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
    	    //#pragma unroll
    	    for (Size i=0; i<nI; i++) {
    	        size_t gid = (static_cast<size_t>(blockIdx.x*nearNeighborBlock + in)*nI + i)*nI + threadIdx.x - nE;
    	        Size ip = preType[bid + i+nE] - nTypeE;
				Float str = gapMat[gid];
				if (str <= 0) str = 0;
				else {
					if (str > 1) str *= gap_pS[ip];
					else str = gap_pS[ip];
				}
    			if (strictStrength) {
    	        	 str *= ratio[ip];
				}
				gapMat[gid] = str;
				if (str > 0) {
    	        	sumInhStrType[ip] += str;
				}
    	    }
    	}
		qid = new PosInt[nTypeI];
		for (PosInt i=0; i<nTypeI; i++) {
			qid[i] = 0;
		}
	}
    if (nb > 0) {
        Size iid = 0;
        for (Size in=ni; in<nn; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
			if (threadIdx.x >= nE) {
				x1[threadIdx.x] = pos[bid + threadIdx.x];
        		y1[threadIdx.x] = pos[bid + threadIdx.x + networkSize];
			}
            __syncthreads();
	    	if (iid < total_nid && itype >= nTypeE) {
                //#pragma unroll
                for (Size i=0; i<nI; i++) {
                    Size tid = (in-ni)*nI + i;
	    	    	PosInt ipre = bid + nE + i;
                    Size ip = preType[ipre] - nTypeE;
	    	    	if (qid[ip] >= gap_nid[ip]) {
	    	    		i = typeAcc0[nTypeE+ip+1]-nE-1;
	    	    		continue;
	    	    	}
	    	    	if (__gapVecID[ip][qid[ip]] == tid) {
	    	    		Float p = gap_tempNeighbor[tid];
	    	    		Float str = gap_pS[ip];
	    	    		if (p > 1) str *= p;
        	    		if (strictStrength) {
	    	    			str *= ratio[ip];
	    	    		}
	    	    		gapVecID[gap_maxDistantNeighbor*id + iid] = ipre;
	    	    		gapVec[gap_maxDistantNeighbor*id + iid] = str;
                    	Float x = static_cast<Float>(x1[i] - x0);
                    	Float y = static_cast<Float>(y1[i] - y0);
	    	    		Float distance = square_root(x*x + y*y);
                    	gapDelayVec[gap_maxDistantNeighbor*id + iid] = distance;
                    	sumInhStrType[ip] += str;
                    	iid ++;
	    	    		qid[ip]++;
                    	if (iid > gap_maxDistantNeighbor) {
                    	    printf("set bigger gap_maxDistantNeighbor, currently %u\n", gap_maxDistantNeighbor);
                    	    assert(iid <= gap_maxDistantNeighbor);
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
	if (itype >= nTypeE) { 
		delete []__gapVecID;
		delete []qid;
    	delete []ratio;
    	#pragma unroll
    	for (Size i=0; i<nTypeI; i++) {
    	    preTypeStrGapped[i*mI + id] = sumInhStrType[i];
			//if (strictStrength) {
			//	if (abs(sumStrType[i] - pN[i]*pS[i])/(pN[i]*pS[i]) > 1e-3) {
			//		printf("%u-%u-%u: sumStrType[i] = %f,  pN[i]*pS[i] = %f, count = %u, nid = %u\n", blockIdx.x, threadIdx.x, i, sumStrType[i], pN[i]*pS[i], count, nVec[id]);
			//		//assert(abs(sumStrType[i] - pN[i]*pS[i])/(pN[i]*pS[i]) <= 1e-3);
			//	}
			//}
    	}
    	//if (threadIdx.x == 0) {
    	    //printf("done#%u\n", blockIdx.x);
    	//}
		delete []gap_nid;
    	delete []sumInhStrType;
    	delete []gap_pN;
    	delete []gap_pS;
	}
}

__global__ // <<< nCluster, nI>>>
void generate_symmetry(PosInt* __restrict__ clusterID,
				       PosInt* __restrict__ neighborBlockId,
					   int* __restrict__ neighborMat,
					   Float* __restrict__ clusterGapMat,
					   Size* __restrict__ preTypeGapped,
					   Float* __restrict__ preTypeStrGapped,
					   PosInt* __restrict__ preType,
					   curandStateMRG32k3a* __restrict__ state,
					   PosInt* __restrict__ i_outstanding,
					   Float* __restrict__ v_outstanding,
					   PosInt iblock, Size nblock, Size nearNeighborBlock, Size maxNeighborBlock, Size mI, Size nE, Size nI, Size nTypeE, Size nTypeI)
{
	// sum up reciprocal connections
	PosInt id = iblock*nI + threadIdx.x;
	size_t home_id = static_cast<size_t>(clusterID[blockIdx.x]*nI*nI);
	PosInt bid = neighborBlockId[iblock*maxNeighborBlock + clusterID[blockIdx.x]]; 
	int nid = neighborMat[iblock*nblock + bid];
	size_t guest_id = static_cast<size_t>(blockIdx.x*nearNeighborBlock + nid)*nI*nI;
	Size n_dir = 0;
	Size n_reciprocal = 0;
	Size n_outstanding = 0;
	PosInt* i_os = i_outstanding + static_cast<size_t>(blockIdx.x*blockDim.x + threadIdx.x)*blockDim.x;
	Float* v_os = v_outstanding + static_cast<size_t>(blockIdx.x*blockDim.x + threadIdx.x)*blockDim.x;
	// debuging parameters
	PosInt pbid = 2;
	PosInt ptid = 0;
	PosInt pib = 0;
	bool debug = false;

	for (PosInt i=0; i<nI; i++) {
		//if (i==threadIdx.x && bid == iblock) continue;
		PosInt local_id = home_id + i*nI + threadIdx.x;
		assert(local_id < blockDim.x*blockDim.x*nearNeighborBlock);

		local_id = guest_id + threadIdx.x*nI + i;
		assert(local_id >= blockIdx.x *  blockDim.x * blockDim.x * nearNeighborBlock);
		assert(local_id < (blockIdx.x+1)*blockDim.x * blockDim.x * nearNeighborBlock);

		Float home_v = clusterGapMat[home_id + i*nI + threadIdx.x];
		Float guest_v = clusterGapMat[guest_id + threadIdx.x*nI + i];
		// mark home<-guest as positive, guest<-home as negative
		if (clusterID[blockIdx.x] == pbid && threadIdx.x == ptid && iblock == pib && debug) {
			if ((home_v > 0 && guest_v <= 0) || (home_v <= 0 && guest_v > 0)) {
				if (bid != iblock || i > threadIdx.x) { 
					printf("before: %u<-%u(%.3e), %u<-%u(%.3e)\n", threadIdx.x, i, home_v, i, threadIdx.x, guest_v);
				}
			}
		}
		if (home_v > 0) {
			n_dir++;
			if (guest_v > 0) {
				n_reciprocal++;
			} else {
				if (iblock != bid || i > threadIdx.x) { 
					i_os[n_outstanding] = i;
					v_os[n_outstanding] = home_v;
					n_outstanding++;
				}
			}
		} else {
			if (guest_v > 0) {
				if (iblock != bid || i > threadIdx.x) { 
					i_os[n_outstanding] = i;
					v_os[n_outstanding] = -guest_v;
					n_outstanding++;
				}
			}
		}
	}
	if (n_outstanding > 0) {
		// decide remaining con prob
		Float prob = static_cast<Float>(n_dir - n_reciprocal)/n_outstanding;
		if (iblock != bid) {
			assert(prob <= 1);
		}
    	curandStateMRG32k3a localState = state[iblock*blockSize + nE+threadIdx.x];
		// make reamining connections
		Float* deltaStr = new Float[nTypeI];
		int* nDelta = new int[nTypeI];
		for (PosInt i=0; i<nTypeI; i++) {
			deltaStr[i] = 0;
			nDelta[i] = 0;
		}
		Size home_type = preType[iblock*blockSize + threadIdx.x + nE]-nTypeE;
		for (PosInt i=0; i<n_outstanding; i++) {
			PosInt gid = bid*nI + i_os[i];
			PosInt guest_type = preType[bid*blockSize + i_os[i] + nE]-nTypeE;
			Float xrand = uniform(&localState);
			if (clusterID[blockIdx.x] == pbid && threadIdx.x == ptid && iblock == pib && debug) {
				printf("%u-%u: %f\n",i, i_os[i], v_os[i]);
			}

			Size h_id = iblock*nearNeighborBlock*nI*nI + clusterID[blockIdx.x]*nI*nI + i_os[i]*nI + threadIdx.x;
			Size g_id = bid*nearNeighborBlock*nI*nI + nid*nI*nI + threadIdx.x*nI + i_os[i];

			if (xrand < prob) {
				if (v_os[i] > 0) { // home->guest
					clusterGapMat[guest_id + threadIdx.x*nI + i_os[i]] = v_os[i];	

					atomicAdd(preTypeGapped + home_type * mI + gid, 1);
					atomicAdd(preTypeStrGapped + home_type * mI + gid, v_os[i]);

				} else { // guest->home
					clusterGapMat[home_id + i_os[i]*nI + threadIdx.x] = -v_os[i];	

					nDelta[guest_type]++;
					deltaStr[guest_type] -= v_os[i];
				}
			} else {
				if (v_os[i] > 0) {
					clusterGapMat[home_id + i_os[i]*nI + threadIdx.x] = 0;	

					nDelta[guest_type]--;
					deltaStr[guest_type] -= v_os[i];
				} else {
					clusterGapMat[guest_id + threadIdx.x*nI + i_os[i]] = 0;	

					atomicSub(preTypeGapped + home_type * mI + gid, 1);
					atomicAdd(preTypeStrGapped + home_type * mI + gid, v_os[i]);// v_os[i] is negative
				}
			}
			Float home_v = clusterGapMat[home_id + i_os[i]*nI + threadIdx.x];	
			Float guest_v = clusterGapMat[guest_id + threadIdx.x*nI + i_os[i]];	
			//
			if (home_v > 0) {
				if (guest_v <= 0) {
					printf("home_v = %f, guest_v = %f, v_os = %f\n", home_v, guest_v, v_os[i]);
					assert(guest_v > 0);
				}
			} else {
				if (guest_v > 0) {
					printf("home_v = %f, guest_v = %f, v_os = %f\n", home_v, guest_v, v_os[i]);
					assert(guest_v <= 0);
				}
			}
			//
		}
		for (PosInt i=0; i<nTypeI; i++) {
			preTypeGapped[i * mI + id] += nDelta[i];
			preTypeStrGapped[i * mI + id] += deltaStr[i];
		}
		delete []deltaStr;
		delete []nDelta;
	}
	
	if (clusterID[blockIdx.x] == pbid && threadIdx.x == ptid && iblock == pib && debug) {
		printf("home block %u, guest block %u\n", iblock, bid);
	}
	//certify symmetry
	__syncthreads();
	for (PosInt i=0; i<nI; i++) {
		Float home_v = clusterGapMat[home_id + i*nI + threadIdx.x];
		Float guest_v = clusterGapMat[guest_id + threadIdx.x*nI + i];
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
}
