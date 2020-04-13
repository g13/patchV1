#include "discrete_input_convol.cuh"
// CANNOT USE EXPRESSION IN (inline seems fine) FUNCTION ARGUMENTS, LEADS TO ZERO!
extern texture<float, cudaTextureType2DLayered> L_retinaInput;
extern texture<float, cudaTextureType2DLayered> M_retinaInput;
extern texture<float, cudaTextureType2DLayered> S_retinaInput;
extern surface<void, cudaSurfaceType2DLayered> LGNspikeSurface;

__global__
void cudaMemsetNonzero(
        Float* array,
        Size n,
        Float value) 
{
    Size id =  blockDim.x * blockDim.y * (gridDim.x*blockIdx.y + blockIdx.x) + blockDim.x*threadIdx.y + threadIdx.x;
	/*
    if (id == 0) {
        printf("array initialized to %f\n", value);
    }*/
    if (id < n) {
        array[id] = value;
    }
}

__device__ 
__forceinline__ 
Float spatialKernel(Float x, Float y, Float rx, Float ry) {
    return exponential(-x*x/(rx*rx) - y*y/(ry*ry));
}

__device__ 
__forceinline__ 
Float temporalKernel(Float tau, Zip_temporal &temp, Float lfac1, Float lfac2, Size lid, Size tid) {
    Float tau1 = tau/temp.tauR;
    Float tau2 = tau/temp.tauD;
    //Float A1 = power(tau1, temp.nR-1)/temp.tauR;
    //Float A2 = power(tau2, temp.nD-1)/temp.tauD;

    Float A1 = (temp.nR-1) * logrithm(tau1);
    Float A2 = (temp.nD-1) * logrithm(tau2);

    //if ((lid ==0 || lid == gridDim.x) && tid == 0) {
    //  printf("lid:%d, tid:%d\n A1 = %f, tau1 = %f\n A2 = %f, tau2 = %f\n", lid, tid, A1, tau1, A2, tau2);
    //}

    //Float tp = AexpTau(A1, tau1 + lfac1) - temp.ratio*AexpTau(A2, tau2 + lfac2);

    Float exponent = A1 - tau1 - lfac1;
    Float tpR = exponential(exponent)/temp.tauR;
    exponent = A2 - tau2 - lfac2;
    Float tpD = exponential(exponent)/temp.tauD;
    Float tp = tpR - temp.ratio*tpD;
    //if ((lid ==0 || lid == gridDim.x) && tid == 0) {
    //  printf("lid:%d, id:%d, tpR = %f, tpD = %f\n", lid, tid, tpR, tpD);
    //}
    return tp;
}

__device__ 
__forceinline__
Float get_intensity(SmallSize coneType, float x, float y, unsigned int iLayer) {
    Float contrast;
    switch (coneType) {
        case 0:
            contrast = static_cast<Float>(tex2DLayered(L_retinaInput, x, y, iLayer));
            break;
        case 1:
            contrast = static_cast<Float>(tex2DLayered(M_retinaInput, x, y, iLayer));
            break;
        case 2:
            contrast = static_cast<Float>(tex2DLayered(S_retinaInput, x, y, iLayer));
            break;
        case 3: // On-Off only magnocellular excluding S cone
            contrast = static_cast<Float>(tex2DLayered(L_retinaInput, x, y, iLayer) 
                                        + tex2DLayered(M_retinaInput, x, y, iLayer))/2.0; 
            break;
        case 4: 
            contrast = static_cast<Float>(tex2DLayered(L_retinaInput, x, y, iLayer) 
                                        + tex2DLayered(M_retinaInput, x, y, iLayer) 
                                        + tex2DLayered(S_retinaInput, x, y, iLayer))/3.0;
            break;
        default:
            printf("unrecognized cone type");
        /*
        case 4:
            break;
        case 5:
            break;
        case 6:
            break;
        */
    }
    return contrast;
}


__global__ 
void testTexture(Float L, Float M, Float S) {
    float x = (blockIdx.x*blockDim.x + threadIdx.x+0.5)/(gridDim.x*blockDim.x);
    float y = (blockIdx.y*blockDim.y + threadIdx.y+0.5)/(gridDim.y*blockDim.y);
    Float LMS[3] = {L, M, S};
    unsigned int iLayer = gridDim.z;
    for (SmallSize iType = 0; iType < 3; iType ++) {
        Float read = get_intensity(iType, x, y , iLayer);
        if (read != LMS[iType]) {
            printf("val = %f\n", read);
            assert(read == LMS[iType]);
        }
    }
}
// for 2 cone-types LGN only, can be generalized for more cone-types
// gridSize: (nLGN, nType) blocks for store 1-D nLGN for convol
// blockSize: spatialSample1D x spatialSample1D (npixel_1D)

/* TODO: speed comparison, all thread load from global memory vs load shared memory then to thread
    __device__
    __inline__
    void store_spatialWeight0(
            Spatial_component &spatial,
            Float* __restrict__ SW_storage
            Float* __restrict__ SC_storage
            Float xhspan, 
            Float yhspan,
            Float dx,
            Float dy,
            Float nsig, // span of spatialRF sample in units of std
            Size id,
            Size lid,
            Size iType,
            Size nType,
    ) {
        // parameter are stored as (nType, nLGN), weights are stored as (nLGN, nType, weight)
        Zip_spatial spat;
        spat.load(spatial, lid);
        SmallSize nSample = blockDim.x * blockDim.y;
        Size offset = (id*nType + iType)*nSample;
        Size tid = threadIdx.y*blockDim.x + threadIdx.x;
        Size storeID = offset + tid;
        Size xID = storeID + offset;
        Size yID = xID + nSample;
    
        // coord to center
        Float x = (threadIdx.x + 0.5)*dx - xhspan;
        Float y = (threadIdx.y + 0.5)*dy - yhspan;
        Float sample_vol = dx * dy;
    
        SW_storage[storeID] = spatialKernel(x, y, k, spat.rx, spat.ry)*sample_vol;
        float x_plane, y_plane;
        retina_to_plane(cx + x, cy + y, x_plane, y_plane);
        SC_storage[xID] = x_plane;
        SC_storage[yID] = y_plane;
    }
*/

// iType is for center surround (or multiple surroud) in a single LGN
// not to be confused with coneTypes
// weights are stored in shapes of (nLGN, nType, nKernelSample)
__device__
__forceinline__
void store_temporalWeight(
        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        Float* __restrict__ reduced, //__shared__
        Float &temporalWeight,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,
        Size id,
        Size tid,
        Size lid,
        SmallSize iType,
        SmallSize nType
) {
    Zip_temporal temp;
    // load temporal parameters
    temp.load(temporal, lid);
    SmallSize patchSize = blockDim.x*blockDim.y;
    Size nPatch = (nKernelSample + patchSize-1)/patchSize - 1;
    Size remain = nKernelSample%patchSize;
    if (remain == 0) {
        remain = patchSize;
    }
    /*DEBUG
    if ((lid ==0 || lid == gridDim.x) && tid == 0) {
        printf("%f, %f, %f, %f, %f, %f\n", temp.nR, temp.nD, temp.tauR, temp.tauD, temp.delay, temp.ratio);
        printf("patchSize = %u, nPatch = %u, remain = %u\n", patchSize, nPatch, remain);
    }
    __syncthreads();
    */
    
    Float lfac1, lfac2;
    lfac1 = log_gamma(temp.nR);
    lfac2 = log_gamma(temp.nD);

    /*DEBUG
    if ((lid ==0 || lid == gridDim.x) && tid == 0) {
        printf("lid:%d, tid:%d\n%f: %f, %f, %f\n %f: %f, %f, %f\n", lid, tid, temp.nR, lfac1, tgamma(temp.nR), exp(lfac1), temp.nD, lfac2, tgamma(temp.nD), exp(lfac2));
    }
	__syncthreads(); 
    */
    // account for the delay into T0
    kernelSampleT0 -= temp.delay;
    // initialize the sum of temporalWeights to 0
	if (tid == 0) {
    	temporalWeight = 0;
	}
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        Float tw;
        if (iPatch < nPatch || tid < remain) {
            // temporalKernel takes abs(now-t)
            // but we store weight in time-reverse (tau -> 0)
            // i.e., last in storage-> 0 -^v-- tau <-first in storage,   
            SmallSize twid = nKernelSample-1 - iPatch*patchSize - tid;
            Size storeID = (id*nType + iType)*nKernelSample + iPatch*patchSize + tid;

			Float t = twid * kernelSampleDt + kernelSampleT0;

            /*DEBUG
			if ((lid ==0 || lid == gridDim.x) && tid == 0) {
            	printf("lid = %d, tid = %d, t = %f\n", lid, tid, t);
            }
            */
            if (t < 0) {
                tw = 0.0;
            } else {
                tw = temporalKernel(t, temp, lfac1, lfac2, lid, tid);
            }
            /*DEBUG
			if ((lid ==0 || lid == gridDim.x) && tid == 0) {
            	printf("lid = %d, tid = %d, tw = %f\n", lid, tid, tw);
            }
            */
			
            TW_storage[storeID] = tw;
            // get absolute values ready for max_convol
            tw = abs(tw);
        } else {
            tw = 0.0;
        }
        /* DEBUG
        if (isnan(tw)) {
            printf("%u-%u: fac = %f, %f\n", lid, nKernelSample-1 - iPatch*patchSize - tid, lfac1, lfac2);
            assert(!isnan(tw));
        } */
        block_reduce<Float>(reduced, tw);
        if (tid == 0) {
            temporalWeight += reduced[0];
        }
    }
    if (tid == 0) {
        reduced[0] = temporalWeight;
    }
    __syncthreads();
    // normalize
    for (Size iPatch = 0; iPatch < nPatch+1; iPatch++) {
        if (iPatch < nPatch || tid < remain) {
            Size storeID = (id*nType + iType)*nKernelSample + iPatch*patchSize + tid;
            TW_storage[storeID] /= reduced[0];
        }
    }
}

// coordinates are stored as (2, nLGN, nType, nSample), 
// weights are stored as (nLGN, nType, nSample)
__device__
__forceinline__
void store_spatialWeight_parvo(
        Float* reduced,
		Float centerPolar,
		Float centerEcc,
		Float coso,
		Float sino,
		Float wSpan,
		Float hSpan,
		Float dw,
		Float dh,
		Float wSigSqrt2,
		Float hSigSqrt2,
		Float normViewDistance,
		Float LR_x0,
		Float LR_y0,
        bool LR,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Size storeID, // (id*nType + iType) * nSample + tid;
        Size nSample,
        bool uniform_retina
) {
    // rads relative to center
    Float w = (threadIdx.x + 0.5)*dw - wSpan;
    Float h = (threadIdx.y + 0.5)*dh - hSpan;

	if (blockIdx.x == 0 && blockIdx.y == 0) {
        Float spatialWeight = spatialKernel(w, h, wSigSqrt2, hSigSqrt2);
	    block_reduce<Float>(reduced, spatialWeight);
		SW_storage[threadIdx.x + threadIdx.y*blockDim.x] = spatialWeight/reduced[0];
	}
    float x, y;
    if (!uniform_retina) {
	    Float cosp, sinp; 
        Float cosEcc, sinEcc;
	    orthPhiRotate3D(centerPolar, centerEcc + h, w, cosp, sinp, cosEcc, sinEcc);

        Float tanEcc;
	    axisRotate3D(centerPolar, centerEcc, coso, sino, cosp, sinp, cosEcc, sinEcc, tanEcc);

        retina_to_plane(cosp, sinp, tanEcc, x, y, normViewDistance, LR_x0, LR_y0);
        // DEBUG visual field and stimulus field not matching
            if (LR) {
                if (x < 0 || x > 0.5) {
                    printf("x = %1.15e\n", x);
                    //assert(x>=0);
                    //assert(x<=0.5);
                }
            } else {
                if (x < 0.5 || x > 1) {
                    printf("x = %1.15e\n", x);
                    //assert(x>=0.5);
                    //assert(x<=1);
                }
            }
            if (y<0 || y>1) {
                printf("y = %1.15e\n", y);
                //assert(y>=0);
                //assert(y<=1);
            }
        //
    } else {
        x = LR_x0 + centerEcc * cosine(centerPolar) + w;
        y = LR_x0 + centerEcc * sine(centerPolar) + h;
        // DEBUG visual field and stimulus field not matching
            if (x<0 || x>1) {
                printf("x = %1.15e\n", x);
                assert(x>=0);
                assert(x<=1);
            }
            if (y<0 || y>1) {
                printf("y = %1.15e\n", y);
                assert(y>=0);
                assert(y<=1);
            }
        //
    }
    
    // store coords for retrieve data from texture
    SC_storage[storeID] = x; // x
               //nLGN * nType * nSample (all the x)
    SC_storage[gridDim.x*gridDim.y*nSample + storeID] = y; // y
}

//__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__launch_bounds__(1024, 2)
__global__
void store_parvo(
        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component &spatial,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Size nParvo_L, Size nMagno_L, Size nLGN,
	    Float L_x0,
	    Float L_y0,
	    Float R_x0,
	    Float R_y0,
	    Float normViewDistance,
        Float nsig, // span of spatialRF sample in units of std
        bool uniform_retina
) {
    __shared__ Float reduced[warpSize];
    __shared__ Float shared_spat[10]; // centerPolar, centerEcc, coso, sino, wSpan, hSpan, dw, dh, wSigSqrt2, hSigSqrt2
    Size id = blockIdx.x;
    SmallSize iType = blockIdx.y;
    Size lid = iType*nLGN + id;
    if (id >= nParvo_L) {
        lid += nMagno_L;
    }
    SmallSize nType = gridDim.y;
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    SmallSize nSample = blockDim.x * blockDim.y;
    __syncthreads();

    Float temporalWeight;
    store_temporalWeight(temporal, TW_storage, reduced, temporalWeight, nKernelSample, kernelSampleDt, kernelSampleT0, id, tid, lid, iType, nType);
	Float LR_x0, LR_y0;
    bool LR = id < nParvo_L;
	if (LR) {
		LR_x0 = L_x0;
		LR_y0 = L_y0;
	} else {
		LR_x0 = R_x0;
		LR_y0 = R_y0;
	}
    //if (tid == 0) {
    //    printf("id:%u-%u, sum_tw = %f\n", id, iType, temporalWeight); 
    //}

    bool use_shared = true;
    if (use_shared) { // TODO: *compare with global broadcast
        Size offset = id*nType + iType;
        if (tid == 0) {
            Zip_spatial spat;
            spat.load(spatial, lid);
            Float wSpan = nsig * spat.rx / SQRT2;
            Float dw = 2*wSpan/blockDim.x;

            Float hSpan = nsig * spat.ry / SQRT2;
            Float dh = 2*hSpan/blockDim.y;
			{
            	shared_spat[0] = spat.x;
				shared_spat[1] = spat.y;
				shared_spat[2] = cosine(spat.orient); 
				shared_spat[3] = sine(spat.orient); 
				shared_spat[4] = wSpan;
				shared_spat[5] = hSpan;
				shared_spat[6] = dw;
				shared_spat[7] = dh;
				shared_spat[8] = spat.rx; 
				shared_spat[9] = spat.ry;
			}
        }
        __syncthreads();
        // load from shared mem
        store_spatialWeight_parvo(reduced, shared_spat[0], shared_spat[1], shared_spat[2], shared_spat[3], shared_spat[4], shared_spat[5], shared_spat[6], shared_spat[7], shared_spat[8], shared_spat[9], normViewDistance, LR_x0, LR_y0, LR, SW_storage, SC_storage, offset*nSample+tid, nSample, uniform_retina);
    } 
}


__launch_bounds__(1024, 2) // must be launched by 1024
__global__
void parvo_maxConvol(Spatial_component &spatial,
                   Float* __restrict__ TW_storage,
                   Float* __restrict__ covariant,
                   Float* __restrict__ max_convol,
                   Size nSample1D, Size nParvo_L, Size nMagno_L, Size nLGN, SmallSize nKernelSample, Float kernelSampleDt, Float nsig)
{
    extern __shared__ Float swC[];
    Size nSample = nSample1D * nSample1D;
    Float* swS = swC + nSample;
    __shared__ Float reduced[warpSize];
    PosInt id = blockIdx.x;
    if (id >= nParvo_L) {
        id += nMagno_L;
    }
    Float rxs = spatial.rx[nLGN+id];
    Float rys = spatial.ry[nLGN+id];

    Float xs = spatial.x[nLGN+id];
    Float ys = spatial.y[nLGN+id];

    Float wSpan = nsig * rxs / SQRT2;
    Float hSpan = nsig * rys / SQRT2;

    Float rxc = spatial.rx[id];
    Float ryc = spatial.ry[id];

    Float R = wSpan > hSpan? wSpan: hSpan;

    Float xc = spatial.x[id];
    Float yc = spatial.y[id];

    wSpan = nsig * rxc / SQRT2;
    hSpan = nsig * ryc / SQRT2;

    Float dx = yc*cosine(xc)-ys*cosine(xs);
    Float dy = yc*sine(xc)-ys*sine(xs);
    Float r = square_root(dx*dx + dy*dy);

    Float cov = covariant[id];
    Float orients = spatial.orient[nLGN+id];
    Float orientc = spatial.orient[id];

    r += wSpan>hSpan? wSpan:hSpan;

	Float coss = cosine(orients); 
    R = r > R? r: R;
	Float sins = sine(orients);

	Float cosc = cosine(orientc); 
	Float sinc = sine(orientc); 
    Float ks = spatial.k[nLGN + id];
    Float kc = spatial.k[id];

    Float ds = 2*R/nSample1D;
    Size nPatch = nSample/blockDim.x;
    assert(nSample%blockDim.x == 0);

    Float sumC, sumS;
    if (threadIdx.x == 0) {
        sumC = 0.0;
        sumS = 0.0;
    }
    #pragma unroll 9
    for (PosInt iPatch = 0; iPatch < nPatch; iPatch++) {
        PosInt pid = iPatch*blockDim.x + threadIdx.x;
        PosInt w = pid % nSample1D;
        PosInt h = pid / nSample1D;
        Float lx = (w + 0.5)*ds - R; // origin at the center of the surround RF
        Float ly = (h + 0.5)*ds - R;
        Float x = cosc*(lx-dx) + sinc*(ly-dy);
        Float y = -sinc*(lx-dx) + cosc*(ly-dy);
        Float local_sw = spatialKernel(x, y, rxc, ryc); 
        swC[pid] = local_sw;
	    block_reduce<Float>(reduced, local_sw);
        if (threadIdx.x == 0) {
            sumC += reduced[0];
        }
        x = coss*lx + sins*ly;
        y = -sins*lx + coss*ly;
        local_sw = spatialKernel(x, y, rxs, rys);
        swS[pid] = local_sw;
	    block_reduce<Float>(reduced, local_sw);
        if (threadIdx.x == 0) {
            sumS += reduced[0];
        }
    }
    if (threadIdx.x == 0) {
        reduced[0] = sumC;
        reduced[1] = sumS;
    }
    __syncthreads();
    #pragma unroll 9
    for (PosInt iPatch = 0; iPatch < nPatch; iPatch++) {
        PosInt pid = iPatch*blockDim.x + threadIdx.x;
        swC[pid] /= reduced[0];
        swS[pid] /= reduced[1];
    }
    __syncthreads();

    Float convol;
    if (threadIdx.x == 0) {
        convol = 0.0;
    }
    #pragma unroll 16
    for (PosInt it = 0; it<nKernelSample; it++) {
        Float tc = TW_storage[blockIdx.x*2*nKernelSample + it];
        Float ts = TW_storage[(blockIdx.x*2 + 1)*nKernelSample + it];

        #pragma unroll 9
        for (PosInt iPatch = 0; iPatch < nPatch; iPatch++) {
            PosInt pid = iPatch*blockDim.x + threadIdx.x;
            Float local_decide;
            Float local_center = swC[pid] * tc*kc;
            Float local_surround = swS[pid] * ts*ks;
            if (abs(local_center) > abs(local_surround)) { // assumed contrast must have the same sign as local_center
                local_decide = abs(local_center) + local_surround * copysign(cov, local_center);
            } else { // assumed contrast must have the same sign as local_surround
                local_decide = abs(local_surround) + local_center * copysign(cov, local_surround);
            }
	        block_reduce<Float>(reduced, local_decide);
            if (threadIdx.x == 0) {
                convol += reduced[0];
            }
        }
    }
    if (threadIdx.x == 0) { // add center surround together, iType = 0, 1
		// max_convol should be initialized elsewhere 
        max_convol[id] = convol;
    }
}

// grid: [nLGN, 2, 1]
// block: [nSpatialSample1D, nSpatialSample1D, 1]
__launch_bounds__(1024, 2)
__global__ 
void LGN_convol_parvo(
        Float* __restrict__ luminance,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ convol,
        Float* __restrict__ contrast,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
		Size nParvo_L, Size nMagno_L, Size nLGN,
		Float normViewDistance,
        PosInt currentFrame,
        Size maxFrame,
		Size ntPerFrame,
        PosInt iFramePhase,
        Float Itau,
        Size iKernelSampleT0,
        Size kernelSampleInterval,
        Size nKernelSample,
        Float dt,
        Size denorm,
        bool saveOutputB4V1
) {
    __shared__ Float reducedS[warpSize];
    __shared__ Float reducedC[warpSize];
    extern __shared__ Float nSampleShared[];

    // weights are stored in shapes of (nLGN, nType, weight)
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    Size nSample = blockDim.x * blockDim.y;
	
    //TODO: Itau may take different value for different cone type
    // convolve center and update luminance
    Float convolS, convolC;
	if (tid == 0) {
		convolS = 0.0;
		convolC = 0.0;
	}
    /* kernel sampling diagram with frames
                      [,)
        frame:       curr            next              next+1
        framePhase|tPerFrame-framePhase                  ^        
        time:  <--|   ^   |-------> tPerFrame <------|   ^    |
                now-tau                                       now
        sample:           1       2       3       4       5     
                  |...|...:---|-------|-------|------:|---|---|
        dt:     -4   0    4   8       16      24  v   32    v 40
                  v                               Dt        T0 
         lastDecay|nextDecay
        e.g., tau = 40*dt
    */

    Size id = blockIdx.x;
    if (id >= nParvo_L) {
        id += nMagno_L;
    }

    // TODO store and read storage contiguously
	Size offsetS = blockIdx.x*2 + 1;
    Size storeIDS = offsetS*nSample + tid;
	Size offsetC = blockIdx.x*2 + 0;
    Size storeIDC = offsetC*nSample + tid;

    // coord on the stimulus plane
    float x0C = SC_storage[storeIDC];
    float y0C = SC_storage[gridDim.x*nSample + storeIDC];
    float x0S = SC_storage[storeIDS];
    float y0S = SC_storage[gridDim.x*nSample + storeIDS];
    assert(x0S <= 1.0);
    assert(y0S <= 1.0);
    assert(x0C <= 1.0);
    assert(y0C <= 1.0);
    assert(x0S >= 0.0);
    assert(y0S >= 0.0);
    assert(x0C >= 0.0);
    assert(y0C >= 0.0);

    Float kS, kC;
    if (tid == 0) {
        kS = spatial.k[nLGN + id];
        kC = spatial.k[id];
    }
    /* Light adaptation process:
        tau*dI/dt = -I + F(t);
        F(t) = piecewise F(t_{i}), for t in [t_{i}, t_{i+1}), t_{i} is the onset time of the i-th frame
        Float lastDecayIn = luminance[lid]; // = [sum_i2n(F_i[exp(-t_{i+1}/Itau) - exp(-t_{i}/Itau)]) - F_{n+1}*exp(-t_{i+1}/Itau)]*exp(-t/Itau)
        Float F_1 = lastF[lid]; // = F_{n+1}
        //F_i is the mean of all sampled pixel value of the ith frame in the LGN's RF.
    */
    
    SmallSize typeS = coneType[nLGN+id];
    SmallSize typeC = coneType[id];

    Float spatialWeight = SW_storage[tid];
    //initialize return value
    /* looping the following over (nPatch+1) patches on nKernelSample samples points:
        p - parallelized by all threads;
        n - needed by all threads;
        s - single thread 
    */
    // non-dimensionalized decay time-scale unit  for intensity
    //Float I_unit = dt/denorm/Itau; // denorm is the co-divisor to compare frame with dt
    Size nPatch = (nKernelSample + nSample-1)/nSample - 1;
    Size remain = nKernelSample % nSample;
    if (remain == 0) {
        remain = nSample;
    }
    for (Size iPatch=0; iPatch<nPatch+1; iPatch++) {
        Size nActive;
        // for p in time, active (for temporal samples) threads only,
        if (iPatch == nPatch) { // no divergent branch
            nActive = remain;
        } else {
            nActive = nSample;
        }
        //1. Load temporal weights: p in time
        // forward in time, stored reversed in TW_storage
        // i.e., last in storage-> t-0 -^v-- t-tau <-first in storage,   
        // convolution time start at t-tau; first sample point: t-tau+kernelSampleT0;
        Float temporalWeightS, temporalWeightC;
		if (tid < nActive) {
			temporalWeightS = TW_storage[offsetS*nKernelSample + iPatch*nSample + tid];
			temporalWeightC = TW_storage[offsetC*nKernelSample + iPatch*nSample + tid];
			//if (blockIdx.x == 0) {
			//	printf("tw[%u*%u + %u*%u + %u = %u] = %f\n", offsetC, nKernelSample, iPatch, nSample, tid, offsetC*nKernelSample + iPatch*nSample + tid, temporalWeightC);
			//}
		} 
		/*
		__syncthreads();
		if (blockIdx.x == 0 && tid == 0) {
			printf("tw: ");
			for (Size i = 0; i < nActive; i++) {
				printf("%f, ", TW_storage[offsetC*nKernelSample + iPatch*nSample + i]);
			}
			printf("\n");
			printf("id: ");
			for (Size i = 0; i < nActive; i++) {
				printf("%u, ", offsetC*nKernelSample + iPatch*nSample + i);
			}
			printf("\n");
		}
		__syncthreads();
		*/
        //2. Find new frames - n, usually just 1
        PosInt old_currentFrame = currentFrame;
        PosInt itFrames, T0;
        if (iPatch == 0) {
            T0 = iKernelSampleT0; // iKernelSampleT0 = 0 or kernelSampleInterval/2
        } else {
            T0 = kernelSampleInterval; // iKernelSampleT0 = 0 or kernelSampleInterval/2
		}
        // number of frames in one patch
        //Size nFrame = (itFrames*denorm + iFramePhase + (ntPerFrame-1)) / ntPerFrame - 1; // exclude the currentFrame within the framePhase, already in F_1
        itFrames = (T0 + (nActive-1)*kernelSampleInterval)*denorm + iFramePhase; // iKernelSampleT0 = 0 or kernelSampleInterval/2
        Size nFrame = itFrames / ntPerFrame + 1;
        // check if the first samplePoint bypass the currentFrame
        if (T0*denorm + iFramePhase >= ntPerFrame) {
            currentFrame = old_currentFrame+1;
            nFrame--;
        }
        /*DEBUG
            if (blockIdx.x==52583 && tid == 0) {
                if (iPatch == 0) {
                    printf("itFrames = %u, nFrame: %u, maxFrame = %u\n", itFrames, nFrame, maxFrame);
                }
                printf("patch #%u/%u, iframePhase = %u/%u\n", iPatch, nPatch, iFramePhase, ntPerFrame);
                //printf("itFrames = %u, nActive = %u, denorm = %u, iKernelSampleT0 = %u, kernelSampleInterval\n", iFramePhase, nActive, denorm, iKernelSampleT0, kernelSampleInterval);
                //assert(nFrame == maxFrame);
            }
        */
        //3. For all the new frames
        for (Size iFrame = 0; iFrame < nFrame; iFrame++) {
            //Get F_i by reduce - p: in space
            Float local_I = get_intensity(3, x0S, y0S, (currentFrame + iFrame) % maxFrame);
            block_reduce<Float>(reducedS, local_I);
            if (tid == 0) {
                // __shared__ to (register/local) to __shared__
                nSampleShared[iFrame] = reducedS[0]/nSample;  // shared memory now used for spatial mean luminance, F_i
            }
        }
        __syncthreads();
        /*
            //!!! Update light adapation variables here to hide latency: p in space 
            if (iPatch == 0 && tid == 0) { // first dt must be in the first patch
                PosInt itf0 = denorm + iFramePhase;
                Float luminance_tmp = lastDecayIn*exponential(-dt);
                if ((itf0 + ntPerFrame-1)/ntPerFrame > 1) { // check if a new frame is introduced within a dt
                    // here we ASSUME tPerframe > dt, i.e., at most one change of frame happen within a single dt
                    Float F_2 = nSampleShared[0]; // the first frame in the patch
                    lastF[lid] = F_2;
                    Float exponent = (itf0%ntPerFrame)*I_unit; // needs this extra variable otherwise exponential returns inf (no idea why)
                    luminance_tmp += (F_1 - F_2) * exponential(-exponent); // == exp(-(dt+framePhase-tPerFrame)*I_unit)
                } // else lastF is not changed
                // *** if (ift0 + denorm == ntPerFrame) lastF is not stored and the first frame of next convolution should be recalculated
                luminance[lid] = luminance_tmp;
                assert(abs(luminance_tmp) < abs(lastDecayIn));
            }
            // TODO: old_frame contrast (lastDecayIn, F1) data can be stored(310mb for tau 250ms dt 0.1ms) not to be calculated every time, then we can fully parallelize this step
            //4. Calculate mean_I: p in time 
            if (tid < nActive && nFrame > 0) {
                PosInt itf = tid * kernelSampleInterval;
                if (iPatch == 0) {
                    itf += iKernelSampleT0;
                }
                itf *= denorm;
                Float lastDecayIn_tmp = lastDecayIn;
                if (isnan(lastDecayIn_tmp)) {
                    printf("oldlastDecayIn = %f\n", lastDecayIn_tmp);
                    assert(!isnan(lastDecayIn_tmp));
                }
                Float exponent = itf*I_unit; // needs this extra variable otherwise exponential returns inf (no idea why)
                lastDecayIn *= exponential(-exponent); // = [sum_i2n(F_i[exp(-t_{i+1}/Itau) - exp(-t_{i}/Itau)]) - F_{n+1}*exp(-t_{i+1}/Itau)]*exp(-t/Itau)
                itf += iFramePhase;
                Size local_nFrame = (itf + ntPerFrame-1) / ntPerFrame - 1; // exclude the currentFrame within the framePhase, already in F_1
                // this loop is actually where the most of the branch divergence happen, local_nFrame can be different for each thread;
                itf = itf % ntPerFrame;
                for (Size iFrame = 0; iFrame < local_nFrame; iFrame++) {
                    // number of active threads decreases for each loop
                    Float F_2 = nSampleShared[iFrame];
                    // F_{n+1} decayed with exp(-((t+t0) - t_{i+2})/Itau); time to current kernelSampleT from currentFrame input time
                    Float exponent = (itf+(local_nFrame-1-iFrame)*ntPerFrame)*I_unit; // needs this extra variable otherwise exponential returns inf (no idea why)
                    lastDecayIn += (F_1 - F_2) * exponential(-exponent);
                    F_1 = F_2;
                }
            }
            __syncthreads(); // make sure shared memory reads complete before reuse for other data
            if (tid < nActive) {
                nSampleShared[tid] = lastDecayIn + F_1; //shared memory now used as spatiotemporal mean luminance
            }
            // broadcast [lastDecayIn, F_1] for the next convolution step if not the final patch: TODO consider move to the end
            if (iPatch <= nPatch) {
                // nSample == nActive;
                if (tid == nActive-1) {
                    reduced[0] = lastDecayIn;
                    reduced[1] = F_1;
                }
                __syncthreads();
                // broadcast [lastDecayIn, F_1]
                lastDecayIn = reduced[0];
                F_1 = reduced[1];
            }
            __syncthreads();
        */
        //5. For each sample point in time: 
        //  Get weighted contrast sum from local_I(ntensity) and mean_I(ntensity): p in space 
        Float tempFilteredC, tempFilteredS;
        // initialize with the first frame in the patch
        PosInt it = 0;
        if (iPatch == 0) {
            it = iKernelSampleT0;
        } else {
            it = kernelSampleInterval;
		}
		//PosInt iFrame = 0;
        Int preFrame = currentFrame-1;
        for (PosInt iSample = 0; iSample < nActive; iSample++) {
            PosInt frameNow = old_currentFrame + (it*denorm + iFramePhase)/ntPerFrame; //time starts with old_currentFrame, frame starts with currentFrame
            if (frameNow > preFrame) { // advance frame
                //Load mean luminance from shared memory first
                PosInt iFrame = frameNow - currentFrame;
                Float mean_I = nSampleShared[iFrame];
				/* DEBUG
					__syncthreads();
                	iFrame = frameNow - currentFrame;
					if (blockIdx.x == 52583 && tid == 0) {
						printf("iFrame: %u/%u, frameNow: %u, iSample: %u\n", iFrame, nFrame, frameNow, iSample);
					}
					__syncthreads();
				*/
                preFrame = frameNow;
				
                // surround 
                Float local_I = get_intensity(typeS, x0S, y0S, frameNow % maxFrame);
                Float local_contrast;
                if (mean_I > 0) {
                    local_contrast = local_I/mean_I - 1.0;
                } else {
                    local_contrast = local_I;
                }
                if (abs(local_contrast) > 1.0) {
                    local_contrast = copyms(1.0, local_contrast); // copyms is copysign(value, sign);
                }

                Float filteredS = spatialWeight*local_contrast;
                block_reduce<Float>(reducedS, filteredS);
                if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1 && tid ==0) {
                    contrast[nLGN+id] = reducedS[0];
                    luminance[id] = mean_I;
					/*DEBUG
						if (blockIdx.x == 52583) {
							printf("contrastS = %e, mean_I = %e, local_I = %e -> lc = %e\n", reducedS[0], mean_I, local_I, local_contrast);
						}
					*/
                }

                // center
                local_I = get_intensity(typeC, x0C, y0C, frameNow % maxFrame);
                if (mean_I > 0) {
                    local_contrast = local_I/mean_I - 1.0;
                } else {
                    local_contrast = local_I;
                }
                if (abs(local_contrast) > 1.0) {
                    local_contrast = copyms(1.0, local_contrast); // copyms is copysign(value, sign);
                }

                Float filteredC = spatialWeight*local_contrast;
                block_reduce<Float>(reducedC, filteredC);
                if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1 && tid == 0) {
                    contrast[id] = reducedC[0];
					/* DEBUG
						if (blockIdx.x == 52583) {
							printf("contrastC = %e, mean_I = %e, local_I = %e -> lc = %e\n", reducedC[0], mean_I, local_I, local_contrast);
						}
					*/
                }
				__syncthreads();
            }
            if (tid == iSample) {
                tempFilteredS = reducedS[0]*temporalWeightS; // spatially contrast convolve with temporalWeight 
                tempFilteredC = reducedC[0]*temporalWeightC;
				/* DEBUG
					if (blockIdx.x == 52583) {
						printf("%u#%u, wspC*tw: %e*%e = %e\n", iPatch, tid, reducedC[0], temporalWeightC, tempFilteredC);
					}
				*/
            }
			__syncthreads();
            // advance time
            it += kernelSampleInterval;
        }
		/*DEBUG
			if (iFrame != nFrame-1 && tid == 0) {
				printf("iFrame end with %u, nFrame = %u", iFrame, nFrame);
			}
		    if (blockIdx.x == 52583 && tid == 0) {
		    	printf("final Frame: %u, nFrame = %u \n", iFrame, nFrame);
            	assert(iFrame == nFrame-1);
		    }
		*/
		__syncthreads();
        if (tid >= nActive) {
            tempFilteredS = 0.0;
            tempFilteredC = 0.0;
        }
        //6. reduce sum with temporal weights: p in time
        block_reduce<Float>(reducedS, tempFilteredS);
        //7. add to convol: s
        if (tid == 0) {
            convolS += reducedS[0];
        }
		/*DEBUG
			__syncthreads();
			reducedS[0] = 0.0;
			if (blockIdx.x == 52583) {
				if (tid == 0) {
					printf("%u possible nonzeros, tempFilteredC = ", nActive);
				}
				for (Size i = 0; i < nSample; i++ ) {
					if (tid == i) {
						printf("%e, ", tempFilteredC);
						reducedS[0] += tempFilteredC;
					}
					__syncthreads();
				}
				if (tid == 0) {
					printf("\n");
				}
			}
		*/
        block_reduce<Float>(reducedC, tempFilteredC);
        if (tid == 0) {
			//Float old_convol = convolC;
            convolC += reducedC[0];
			/* DEBUG
				if (blockIdx.x == 52583) {
					printf("%e + patchSum #%u: %e = %e\n", old_convol, iPatch, reducedC[0], convolC);
					//if (reducedC[0] != reducedS[0]) {
					//	printf("reduce sum %e != loop sum %e\n", reducedC[0], reducedS[0]);
					//	assert(reducedC[0] == reducedS[0]);
					//}
				}
			*/
        }
		//__syncthreads();
        //9. advance [currentFrame, framePhase] if not the final patch: n
        if (iPatch < nPatch) {
			// itFrames =  [0... nSample-1]
			// fullLength =  [0... nSample-1...]
            iFramePhase = itFrames % ntPerFrame;
			// PosInt old_Frame = currentFrame; // use with DEBUG 
            currentFrame = old_currentFrame + itFrames/ntPerFrame;
            /* DEBUG
			    if (blockIdx.x == 52583 && tid == 0)  {
			    	printf("this patch starts with %u -> %u, next patch starts with %u", old_Frame, old_Frame + iFrame, currentFrame);
			    }
            */
        }
    }
    if (tid == 0) {
        // times amplitude and space-time volume, k is amplitude*dwdh
        convolC *= kC;
        convolS *= kS;
        convol[id] = (convolC + convolS);        
        //convol[blockIdx.x] = (convolC + convolS)*kernelSampleInterval*dt;
		/*DEBUG
			if (blockIdx.x == 52583) {
				printf("convol: %e*%e = %e\n", (convolC + convolS), kernelSampleInterval*dt, (convolC + convolS)*kernelSampleInterval*dt);
			}
		*/
    }
}

__inline__
__device__
Float get_spike(Size &nsp,
                Float &leftTimeRate,
                Float &lastNegLogRand,
                Float dt,
                Float rate,
                curandStateMRG32k3a *state) 
{
    nsp = 0;
    // ith spike, jth dt
    // t_{i+1}*r_{j+1} + (T_{j}-t_{i})*r_{j} = -log(rand);
    Float rT = dt*rate;
    Float next_rT = lastNegLogRand - leftTimeRate; // tsp = next_rt/rate
    if (next_rT > rT) { // spike time is larger than dt.
        leftTimeRate += rT;
        return 0.0;
    } else do { // at least one spike during current time step
        lastNegLogRand = -logrithm(uniform(state));
        next_rT += lastNegLogRand;
        nsp++;
    } while (next_rT <= rT);
    next_rT -= lastNegLogRand; // retract the tentative spike
    leftTimeRate = rT - next_rT;
    if (nsp > 0) return next_rT/(rate*nsp); // mean tsp not yet normalized by dt
    else return 0; 
}

//template<int ntimes>
__launch_bounds__(1024, 2)
__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ current_convol,
        Float* __restrict__ LGN_fr,
        Float* __restrict__ LGN_sInfo,
        int* __restrict__ sx,
        int* __restrict__ sy,
        Float* __restrict__ leftTimeRate,
        Float* __restrict__ lastNegLogRand,
		curandStateMRG32k3a* __restrict__ state,
        Float* __restrict__ lVar,
        int varSlot, LearnVarShapeFF_E_pre lE, LearnVarShapeFF_I_pre lI, Size nFF, Float dt, int learning, bool learnData_FF)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    bool engaging = id<nLGN;
    unsigned int MASK = __ballot_sync(FULL_MASK, static_cast<int>(engaging));
    if (engaging) {
        Float C50, K, A, B;
        // load in sequence
        Float current = current_convol[id];
		Float max = max_convol[id];
        Float lTR = leftTimeRate[id];
        Float lNL = lastNegLogRand[id];
        int x = sx[id];
        int y = sy[id];
        curandStateMRG32k3a local_state = state[id];

        // get firing rate
        logistic.load_first(id, C50, K, A, B);
        // Float convol = current; // use with DEBUG
        if (current < 0) {
            current = 0;
        }
        __syncwarp(MASK);
        Float fr = max * transform(C50, K, A, B, current/max);
		/* DEBUG
        if (fr < 0) {
            printf("convol = %f, fr = %f, K=%f, A= %f, B= %f, C50 =%f, max = %f\n", convol, fr, K, A, B, C50, max);
            assert(fr >= 0);
        }*/
        LGN_fr[id] = fr;
        Float var[MAX_NLEARNTYPE_FF];
        if (learning < 4) {
            #pragma unroll (sum_nLearnTypeFF)
            for (int i=0; i<nFF; i++) {
                surf2DLayeredread(&(var[i]), LGNspikeSurface, 4*x, y, 1+(3*i+varSlot)); // varSlot already 'mod'ed
            }
        }

        Size nsp;
        Float tsp = get_spike(nsp, lTR, lNL, dt, fr/1000.0, &local_state);
        Float sInfo = nsp + tsp/dt; // must be float, integer part = #spikes decimals: mean tsp normalized by dt
        if (sInfo > 0) {
            lastNegLogRand[id] = lNL;
            //if (id == 37054 || id == 37223 || id == 37647) {
            //    printf("LGN# %u fired\n");
            //}
            //printf("LGN fired, ");
        }
        leftTimeRate[id] = lTR;
        state[id] = local_state;
        if (learnData_FF) LGN_sInfo[id] = sInfo;
        // write to surface memory 
        surf2DLayeredwrite(sInfo, LGNspikeSurface, 4*x, y, 0);
        if (learning < 4) {
            Float delta_t; // from last_tsp (or start of the time step) to tsp (or end of time step)
            if (sInfo > 0) {
                delta_t = tsp;
                // decay from start to tsp
                #pragma unroll (max_nLearnTypeFF_E)
                for (PosInt i=0; i<lE.n; i++) {
                    var[i] += nsp; // increase learning variable after spike (not to be increased right after read to hide latency)
                    decay(var[i], lE.tauLTP[i], delta_t);
                }
                #pragma unroll (max_nLearnTypeFF_I)
                for (PosInt i=0; i<lI.n; i++) {
                    var[lE.n+i] += nsp; // increase learning variable after spike
                    decay(var[lE.n+i], lI.tauLTP[i], delta_t);
                }
                #pragma unroll (sum_nLearnTypeFF)
                for (int i=0; i<nFF; i++) {
                    surf2DLayeredwrite(var[i], LGNspikeSurface, 4*x, y, 1+(3*i+2)); // update at the third slot
                }
                delta_t = dt - delta_t; // remaining time to end of the timestep
            } else delta_t = dt;

            // decay from start or tsp to the end
            #pragma unroll (max_nLearnTypeFF_E)
            for (int i=0; i<lE.n; i++) {
                decay(var[i], lE.tauLTP[i], delta_t);
            }
            #pragma unroll (max_nLearnTypeFF_I)
            for (int i=0; i<lI.n; i++) {
                decay(var[lE.n+i], lI.tauLTP[i], delta_t);
            }
            // write to the next slot in surface
            #pragma unroll (sum_nLearnTypeFF)
            for (int i=0; i<nFF; i++) {
                surf2DLayeredwrite(var[i], LGNspikeSurface, 4*x, y, 1+(3*i+(varSlot+1)%2)); // update at next slot
            }
            if (learnData_FF) {
                #pragma unroll (sum_nLearnTypeFF)
                for (int i=0; i<nFF; i++) {
                    lVar[nLGN*i + id] = var[i];
                }
            }
        }
    }
}
