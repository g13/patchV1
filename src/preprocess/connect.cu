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
                Float* __restrict__ LGN_V1_sSum,
                Float* __restrict__ ExcRatio,
                Float extExcRatio, Float min_FB_ratio, initialize_package init_pack, unsigned long long seed, Size networkSize, Size nType, Size nArchtype, Size nFeature, bool CmoreN, Float p_n_LGNeff) 
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

	Float LGN_sSum = LGN_V1_sSum[id];
	Float presetConstExc = p_n_LGNeff + init_pack.sumType[type]*(1+extExcRatio);
    Float ratio;
	if (init_pack.sumType[type] == 0) {
		ratio = 1.0;
	} else {
		ratio = (presetConstExc - LGN_sSum)/(init_pack.sumType[type]*(1+extExcRatio));
	}
    if (ratio < min_FB_ratio) ratio = min_FB_ratio;
	ExcRatio[id] = ratio;
	for (PosInt i=0; i<nType; i++) {
		PosInt tid = i*networkSize+id;
		PosInt ttid = i*nType + type;
        if (CmoreN) {
    	    preS_type[tid] = init_pack.sTypeMat[ttid];
            if (i < init_pack.iArchType[0]) {
                preN_type[tid] = static_cast<Size>(rounding(ratio*init_pack.nTypeMat[ttid]));
            } else {
    	        //preN_type[tid] = static_cast<Size>(rounding((ratio + LGN_sSum/init_pack.sumType[i*nType+0])*init_pack.nTypeMat[ttid]));
    	        preN_type[tid] = init_pack.nTypeMat[ttid];
            }
        } else {
    	    preN_type[tid] = init_pack.nTypeMat[ttid];
            if (i < init_pack.iArchType[0]) {
                preS_type[tid] = ratio*init_pack.sTypeMat[ttid];
            } else {
    	        //preS_type[tid] = (ratio+ LGN_sSum/init_pack.sumType[i*nType+0])*init_pack.sTypeMat[ttid];
    	        preS_type[tid] = init_pack.sTypeMat[ttid];
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
                          PosInt* __restrict__ neighborBlockId,
                          Size* __restrict__ nNeighborBlock,
						  Size nblock, Float max_radius, Size maxNeighborBlock) 
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

    Size nPatch = (nblock + blockDim.x-1)/blockDim.x - 1;
    Size remain = nblock%blockDim.x;
	if (remain == 0) {
		remain = blockDim.x;
	}

    Size offset = 0;
    if (tid == 0) {
        id[0] = 0;
    }
    bid[tid] = -1;
    __syncthreads();
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        if (iPatch < nPatch || tid < remain) {
            PosInt blockId = offset + threadIdx.x;
            Float x = block_x[blockId] - bx;
            Float y = block_y[blockId] - by;
            Float dis = square_root(x*x + y*y);
            if (dis < max_radius) {
                distance[tid] = dis;
                bid[tid] = blockId;
            }
        }
        __syncthreads();
        if (tid == 0) { // rearrange
            PosInt current_id = id[0];
            for (PosInt i=0; i<blockDim.x; i++) {
                if (bid[i] != -1) {
                    final_distance[current_id] = distance[i]; 
                    final_bid[current_id] = bid[i]; 
                    bid[i] = -1; 
                    current_id++;
                    if (current_id > maxNeighborBlock) {
                        printf("actual nNeighbor = %d > %d (preserved)\n", current_id, maxNeighborBlock);
                        assert(current_id <= maxNeighborBlock);
                    }
                }
            }
            id[0] = current_id;
        }
        __syncthreads();
        if (iPatch < nPatch) {
            offset += blockDim.x;
        }
    }
    Size nb = id[0];
    if (tid == 0) {
        nNeighborBlock[blockIdx.x] = nb;
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
        neighborBlockId[maxNeighborBlock*blockIdx.x + tid] = local_bid;
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

__launch_bounds__(1024,1)
__global__ 
void generate_connections(double* __restrict__ pos,
                          Float* __restrict__ preF_type,
                          Float* __restrict__ preS_type,
                          Size* __restrict__ preN_type,
                          PosInt* __restrict__ neighborBlockId,
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
                          Size* __restrict__ typeAcc0,
                          curandStateMRG32k3a* __restrict__ state,
                          PosInt block_offset, Size networkSize, Size maxDistantNeighbor, Size nearNeighborBlock, Size maxNeighborBlock, Size nType, Size nFeature, bool gaussian_profile, bool strictStrength, Float tol) 
{
    // TODO: load with warps but more, e.g., raxn, daxn, preType
    __shared__ double x1[blockSize];
    __shared__ double y1[blockSize];
    //__shared__ Float ra[blockDim.x];
    Size blockId = blockIdx.x + block_offset;
    Size nn = nNeighborBlock[blockId];
    Size offset = blockId*blockDim.x;
    double x0 = pos[offset*2 + threadIdx.x];
    double y0 = pos[offset*2 + threadIdx.x + blockDim.x];
    Size id = offset + threadIdx.x;
    // number of potential presynaptic connections outsied nearNeighbors, to be stored in vector.
    Size nb = 0; 
    if (nn > nearNeighborBlock) {
        nb = nn - nearNeighborBlock;
        nn = nearNeighborBlock;// nearNeighbors 
    } 
    Float* tempNeighbor;
    if (nb > 0) {
        nb *= blockDim.x;
        tempNeighbor = new Float[nb];
    }
    Float rd = rden[id];
    Float dd = dden[id];

    Size* sumType = new Size[nType]; // avail
    Float* sumP = new Float[nType];
    Float* pF = new Float[nFeature*nType];
    Float* fV = new Float[nFeature];
    #pragma unroll
    for (PosInt i=0; i<nType; i++) {
        sumType[i] = 0;
        sumP[i] = 0.0;
	}
    for (PosInt i=0; i<nFeature; i++) {
        fV[i] = feature[i*networkSize + id];
        for (PosInt j=0; j<nType; j++) {
            pF[i*nType + j] = preF_type[(i*nType+j)*networkSize + id];
        }
    }
    //============= collect p of all ==========
    // withhin block and nearNeighbor
    for (PosInt in=0; in<nn; in++) {
        PosInt bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x; // # neurons in all past blocks 
        x1[threadIdx.x] = pos[bid*2 + threadIdx.x];
        y1[threadIdx.x] = pos[bid*2 + blockDim.x + threadIdx.x];
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
            Float p = connect(distance, ra, rd, gaussian_profile);
            PosIntL mid = (static_cast<PosIntL>(blockIdx.x*nearNeighborBlock + in)*blockDim.x + i)*blockDim.x + threadIdx.x; // defined outside, so delayMat has access to it
                Size ip = preType[ipre];
            if (p > 0 && id != ipre) { // not self-connected
                //Size ip = preType[ipre];
                // id in the conMat [nblock,nearNeighborBlock,blockDim.x,blockDim.x] loop in the second axis, (last dim is the post-id: threadIdx.x, pre-id in the chunk: i)
                //DEBUG
                BigSize matSize = static_cast<BigSize>(blockDim.x*blockDim.x)*nearNeighborBlock*gridDim.x;
                if (mid >= matSize) {
                    printf("(%ux%ux%ux%u) = %u\n", gridDim.x, nearNeighborBlock, blockDim.x, blockDim.x, matSize);
                    assert(mid < matSize);
                }
                //
	    		// update weight with density of axon dendrites and preference over type
                //p *= daxn[ipre] * dd;// * preP_type[ip*networkSize + id];
				p *= daxn[ipre] * dd;
                for (Size iFeature = 0; iFeature < nFeature; iFeature++) {
                    //p *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], pF[iFeature*nType +ip]);
					p *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], pF[iFeature*nType +ip]);
                }
                assert(p >= 0);
				if (p > 0) {
                	sumType[ip]++;
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
        for (Size in=nn; in<nNeighborBlock[blockIdx.x]; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
            x1[threadIdx.x] = pos[bid*2 + threadIdx.x];
            y1[threadIdx.x] = pos[bid*2 + blockDim.x + threadIdx.x];
            __syncthreads();
            #pragma unroll
            for (Size i=0; i<blockDim.x; i++) {
                // blockwise load from gmem the (potential) presynaptic neurons' properties
                Size ipre = bid + i; // pre-id in the network
                Float ra = raxn[ipre];
                double x = x1[i] - x0;
                double y = y1[i] - y0;
                Float distance = static_cast<Float>(square_root(x*x + y*y));
                Float p = connect(distance, ra, rd, gaussian_profile);
                Size tid = (nn-in)*blockDim.x + i; // only ofr tempNeighbor, which is local, no need to coalease memory
                tempNeighbor[tid] = 0;
                if (p > 0) {
                    Size ip = preType[ipre];
                    p *= daxn[ipre] * dd; //* preP_type[ip*networkSize+id];
                    for (Size iFeature = 0; iFeature < nFeature; iFeature ++) {
                        p *= pref[iFeature](fV[iFeature], feature[iFeature*networkSize + ipre], pF[iFeature*nType + ip]);
                    }
                    assert(p>=0);
					if (p > 0) {
                    	sumType[ip]++;
                    	sumP[ip] += p;
                    	tempNeighbor[tid] = p;
					}
                }
            }
            __syncthreads();
        }
    }
    __syncwarp();
    Size* pN = new Size[nType];
    #pragma unroll
    for (Size i=0; i<nType; i++) {
        pN[i] = preN_type[i*networkSize + id];
        preTypeAvail[i*networkSize + id] = sumType[i];
		if (sumType[i] < ceiling(pN[i]*tol)) {
			printf("neuron %u-%u dont have enough type %u neurons to connect to\n", blockIdx.x, threadIdx.x, i);
			assert(sumType[i] >= ceiling(pN[i]*tol));
		}
    }
	__syncthreads();
    delete []pF;
    delete []fV;
    //============= redistribute p of all ==========
    Float* sumStrType = new Float[nType];
    Float* sumType0 = new Float[nType];
    Float* pS = new Float[nType];
    Size* nid = new Size[nType];
	Size max_N = 0;
    #pragma unroll
    for (Size i=0; i<nType; i++) {
		if (pN[i] > max_N) max_N = pN[i];
        pS[i] = preS_type[i*networkSize + id];
    	sumType[i] = 0;
		nid[i] = 0;
    }
    PosInt* _vecID = new PosInt[nType*max_N];

    curandStateMRG32k3a localState = state[id];
	Size count = 0;
	Size connected = false;
	while (!connected) {
    	for (Size i=0; i<nType; i++) {
			sumType0[i] = sumType[i];
			if (sumType0[i] < ceiling(tol*pN[i])) {
    	    	sumType[i] = 0;
    	    	sumStrType[i] = 0;
			}
		}
    	for (Size in=0; in<nn; in++) {
    	    PosInt bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
    	    #pragma unroll
    	    for (Size i=0; i<blockDim.x; i++) {
    	        PosIntL mid = (static_cast<PosIntL>(blockIdx.x*nearNeighborBlock + in)*blockDim.x + i)*blockDim.x + threadIdx.x;
    	        Size ipre = bid + i;
    	        Size ip = preType[ipre];
				if (sumType0[ip] < ceiling(tol*pN[ip])) {
    	        	Float p = abs(conMat[mid]);
    	        	if (p > 0) {
						if (count == 0) p *= pN[ip]/sumP[ip];
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
    	    for (Size in=nn; in<nNeighborBlock[blockIdx.x]; in++) {
    	        Size bid = neighborBlockId[maxNeighborBlock*blockId + in];
    	        #pragma unroll
    	        for (Size i=0; i<blockDim.x; i++) {
    	            Size tid = (nn-in)*blockDim.x + i;
    	            Size ipre = bid + i;
    	            Size ip = preType[ipre];
					if (sumType0[ip] < ceiling(tol*pN[ip])) {
    	            	Float p = tempNeighbor[tid];
    	            	if (p > 0) {
							if (count == 0) {
    	            	    	p *= pN[ip]/sumP[ip];
								tempNeighbor[tid] = p;
							}
    	            	    Float xrand = uniform(&localState);
    	            	    if (xrand < p) {
    	            	        Float str = pS[ip];
    	            	        if (p > 1) {
    	            	            str = str*p;
    	            	        }
    	            	        sumType[ip] ++;
    	            	        sumStrType[ip] += str;
    	            	        _vecID[ip*max_N + nid[ip]] = tid;
    	            	        nid[ip]++;
                				if (nid[ip] > max_N) {
                				    printf("set bigger max_N, currently %u\n", max_N);
                				    assert(nid[ip] <= max_N);
                				}
    	            	    }
    	            	}
					}
    	        }
				__syncthreads();
    	    }
    	}
		connected = true;
		for (PosInt i=0;i<nType;i++) {
			if (sumType[i] < ceiling(pN[i]*tol)) {
				connected = false;
			}
		}
		count++;
		if (count > 1 && !connected) {
			printf("neuron %u-%u need to make another round(%u) of connection, because of %u/%u, %u/%u\n", blockIdx.x, threadIdx.x, count, sumType[0],pN[0], sumType[1],pN[1]);
		}
		if (count >= 20) {
			printf("neuron %u-%u don't have one (or any) of the types of neurons to connect to\n", blockIdx.x, threadIdx.x);
			assert(count < 20);
			//connected = true;
		}
		if (count == 1) {
    		delete []sumP;
		}
	}
	delete []sumType0;
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
        sumStrType[i] = 0;
    }
    delete []sumType;

    // ======== strictly normalize the strengths ==========
    for (Size in=0; in<nn; in++) {
        Size bid = neighborBlockId[maxNeighborBlock*blockId + in] * blockDim.x;
        #pragma unroll
        for (Size i=0; i<blockDim.x; i++) {
            PosIntL mid = (static_cast<PosIntL>(blockIdx.x*nearNeighborBlock + in)*blockDim.x + i)*blockDim.x + threadIdx.x;
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
    if (total_nid > 0) {
		PosInt* qid = new PosInt[nType];
		for (PosInt i=0; i<nType; i++) {
			qid[i] = 0;
		}
    	Size iid = 0;
        for (Size in=nn; in<nNeighborBlock[blockIdx.x]; in++) {
            Size bid = neighborBlockId[maxNeighborBlock*blockId + in];
            x1[threadIdx.x] = pos[bid*2 + threadIdx.x];
            y1[threadIdx.x] = pos[bid*2 + blockDim.x + threadIdx.x];
            __syncthreads();
			if (iid >= total_nid) {
				break;
			}
            #pragma unroll
            for (Size i=0; i<blockDim.x; i++) {
                Size tid = (nn-in)*blockDim.x + i;
				PosInt ipre = bid + i;
                Size ip = preType[ipre];
				if (qid[ip] >= nid[ip]) {
					i = typeAcc0[ip+1]-1;
					continue;
				}
				if (_vecID[ip*max_N + qid[ip]] == tid) {
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
            __syncthreads();
        }
    }
    if (nb > 0) {
        delete []tempNeighbor;
    }

    delete []ratio;
    #pragma unroll
    for (Size i=0; i<nType; i++) {
        preTypeStrSum[i*networkSize + id] = sumStrType[i];
		if (strictStrength) {
			if (abs(sumStrType[i] - pN[i]*pS[i])/(pN[i]*pS[i]) > 1e-3) {
				printf("%u-%u-%u: sumStrType[i] = %f,  pN[i]*pS[i] = %f, count = %u, nid = %u\n", blockIdx.x, threadIdx.x, i, sumStrType[i], pN[i]*pS[i], count, nVec[id]);
				//assert(abs(sumStrType[i] - pN[i]*pS[i])/(pN[i]*pS[i]) <= 1e-3);
			}
		}
    }
    delete []sumStrType;
    delete []pN;
    delete []pS;
}
