#include "discrete_input_convol.cuh"

extern texture<float, cudaTextureType2DLayered> L_retinaConSig;
extern texture<float, cudaTextureType2DLayered> M_retinaConSig;
extern texture<float, cudaTextureType2DLayered> S_retinaConSig;
extern __device__ __constant__ float sqrt2;

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

    Float tpR = exponential(A1 - tau1 - lfac1)/temp.tauR;
    Float tpD = exponential(A2 - tau2 - lfac2)/temp.tauD;
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
            contrast = static_cast<Float>(tex2DLayered(L_retinaConSig, x, y, iLayer));
            break;
        case 1:
            contrast = static_cast<Float>(tex2DLayered(M_retinaConSig, x, y, iLayer));
            break;
        case 2:
            contrast = static_cast<Float>(tex2DLayered(S_retinaConSig, x, y, iLayer));
            break;
        case 3:
            contrast = static_cast<Float>(tex2DLayered(L_retinaConSig, x, y, iLayer) 
                                        + tex2DLayered(M_retinaConSig, x, y, iLayer))/2.0; 
            break;
        case 4:
            contrast = static_cast<Float>(tex2DLayered(L_retinaConSig, x, y, iLayer) 
                                        + tex2DLayered(M_retinaConSig, x, y, iLayer) 
                                        + tex2DLayered(S_retinaConSig, x, y, iLayer))/3.0;
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
    if (contrast <= 0) {
        printf("(%f,%f) = %f, L:%f, M:%f, S: %f\n", x, y, contrast
                , static_cast<Float>(tex2DLayered(L_retinaConSig, x, y, iLayer))
                , static_cast<Float>(tex2DLayered(M_retinaConSig, x, y, iLayer))
                , static_cast<Float>(tex2DLayered(S_retinaConSig, x, y, iLayer)));
        assert(contrast > 0);
    }
    return contrast;
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
    Size nPatch = nKernelSample/patchSize;
    Size remain = nKernelSample%patchSize;
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
    temporalWeight = 0;
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
        assert(!isnan(tw));
        __syncthreads();
        block_reduce<Float>(reduced, tw);
        if (tid == 0) {
            temporalWeight += reduced[0];
        }
    }
}

// coordinates are stored as (2, nLGN, nType, nSample), 
// weights are stored as (nLGN, nType, nSample)
__device__
__forceinline__
void store_spatialWeight(
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
        Size nSample
) {
    // rads relative to center
    Float w = (threadIdx.x + 0.5)*dw - wSpan;
    Float h = (threadIdx.y + 0.5)*dh - hSpan;

    Float spatialWeight = spatialKernel(w, h, wSigSqrt2, hSigSqrt2);
    
	block_reduce<Float>(reduced, spatialWeight);
    // TODO: gaussian spatialWeight with fixed sample point in the unit of sigma is the same across neuorns, can be passed from host directly
    SW_storage[storeID] = spatialWeight/reduced[0];
	Float cosp, sinp; 
    Float cosEcc, sinEcc;
	orthPhiRotate3D(centerPolar, centerEcc + h, w, cosp, sinp, cosEcc, sinEcc);

    Float tanEcc;
	axisRotate3D(centerPolar, centerEcc, coso, sino, cosp, sinp, cosEcc, sinEcc, tanEcc);

    float x, y;
    retina_to_plane(cosp, sinp, tanEcc, x, y, normViewDistance, LR_x0, LR_y0);
    { // DEBUG visual field and stimulus field not matching
        if (LR) {
            if (x < 0 || x > 0.5) {
                printf("x\n");
                assert(x>=0);
                assert(x<=0.5);
            }
        } else {
            if (x < 0.5 || x > 1) {
                printf("x\n");
                assert(x>=0.5);
                assert(x<=1);
            }
        }
        if (y<0 || y>1) {
            printf("y\n");
            assert(y>=0);
            assert(y<=1);
        }
    }
    
    // store coords for retrieve data from texture
    SC_storage[storeID] = x; // x
               //nLGN * nType * nSample (all the x)
    SC_storage[gridDim.x*gridDim.y*nSample + storeID] = y; // y
}

//__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__launch_bounds__(1024, 2)
__global__
void store(
        Float* __restrict__ max_convol,

        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component &spatial,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
		Size nLGN_L,
		Float L_x0,
		Float L_y0,
		Float R_x0,
		Float R_y0,
		Float normViewDistance,
        Float nsig // span of spatialRF sample in units of std
) {
    __shared__ Float reduced[warpSize];
    __shared__ Float shared_spat[10]; // centerPolar, centerEcc, coso, sino, wSpan, hSpan, dw, dh, wSigSqrt2, hSigSqrt2
    Size id = blockIdx.x;
    SmallSize iType = blockIdx.y;
    Size lid = iType*gridDim.x + id;
    SmallSize nType = gridDim.y;
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    SmallSize nSample = blockDim.x * blockDim.y;

    Float temporalWeight;
    store_temporalWeight(temporal, TW_storage, reduced, temporalWeight, nKernelSample, kernelSampleDt, kernelSampleT0, id, tid, lid, iType, nType);
    /* DEBUG
	if ((lid ==0 || lid == gridDim.x) && tid == 0) {
        printf("temporalWeights stored\n");
        assert(!isnan(temporalWeight));
    }*/
    __syncthreads();
    //
	Float LR_x0, LR_y0;
    bool LR = id < nLGN_L;
	if (LR) {
		LR_x0 = L_x0;
		LR_y0 = L_y0;
	} else {
		LR_x0 = R_x0;
		LR_y0 = R_y0;
	}

    bool use_shared = true;
    Float k;
    if (use_shared) { // TODO: *compare with global broadcast
        Size offset = id*nType + iType;
        if (tid == 0) {
            Zip_spatial spat;
            spat.load(spatial, lid);
            Float wSpan = nsig * spat.rx / sqrt2;
            Float dw = 2*wSpan/blockDim.x;

            Float hSpan = nsig * spat.ry / sqrt2;
            Float dh = 2*hSpan/blockDim.y;
			k = spat.k;
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
        store_spatialWeight(reduced, shared_spat[0], shared_spat[1], shared_spat[2], shared_spat[3], shared_spat[4], shared_spat[5], shared_spat[6], shared_spat[7], shared_spat[8], shared_spat[9], normViewDistance, LR_x0, LR_y0, LR, SW_storage, SC_storage, offset*nSample+tid, nSample);
    } 
    //spatialWeight = abs(spatialWeight); // get absolute values ready for max_convol, always positive, not necessary
    /* DEBUG
	if ((lid ==0 || lid == gridDim.x) && tid == 0) {
        printf("spatialWeights stored\n");
        assert(!isnan(spatialWeight));
	    if (lid ==0 && tid == 0) {
            assert(max_convol[id] == 0.0);
        }
    }
    */
    // k is now integrated amplitude over space 
    if (tid == 0) { // add center surround together, iType = 0, 1
        atomicAdd(max_convol+id, temporalWeight * abs(k) * kernelSampleDt);
    }
}

// grid: [nLGN, 2, 1]
// block: [nSpatialSample1D, nSpatialSample1D, 1]
__launch_bounds__(1024, 2)
__global__ 
void LGN_convol_c1s(
        Float* __restrict__ luminance,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ current_convol,
        Float* __restrict__ contrast,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
		Size nLGN_L,
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
        Size denorm 
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
    Float contrastS, contrastC;
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
    Size lidS = 1*gridDim.x + blockIdx.x;
	Size offsetS = blockIdx.x*2 + 1;
    Size storeIDS = offsetS*nSample + tid;
    Size lidC = 0*gridDim.x + blockIdx.x;
	Size offsetC = blockIdx.x*2 + 0;
    Size storeIDC = offsetC*nSample + tid;

    // coord on the stimulus plane
    float x0C = SC_storage[storeIDC];
    float y0C = SC_storage[gridDim.x*gridDim.y*nSample + storeIDC];
    float x0S = SC_storage[storeIDS];
    float y0S = SC_storage[gridDim.x*gridDim.y*nSample + storeIDS];
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
        kS = spatial.k[lidS];
        kC = spatial.k[lidC];
    }
    /* Light adaptation process:
        tau*dI/dt = -I + F(t);
        F(t) = piecewise F(t_{i}), for t in [t_{i}, t_{i+1}), t_{i} is the onset time of the i-th frame
        Float lastDecayIn = luminance[lid]; // = [sum_i2n(F_i[exp(-t_{i+1}/Itau) - exp(-t_{i}/Itau)]) - F_{n+1}*exp(-t_{i+1}/Itau)]*exp(-t/Itau)
        Float F_1 = lastF[lid]; // = F_{n+1}
        //F_i is the mean of all sampled pixel value of the ith frame in the LGN's RF.
    */
    
    SmallSize typeS = coneType[blockIdx.x + 1*gridDim.x];
    SmallSize typeC = coneType[blockIdx.x + 0*gridDim.x];

    Float spatialWeightS = SW_storage[storeIDS];
    Float spatialWeightC = SW_storage[storeIDC];
    //initialize return value
    /* looping the following over (nPatch+1) patches on nKernelSample samples points:
        p - parallelized by all threads;
        n - needed by all threads;
        s - single thread 
    */
    // non-dimensionalized decay time-scale unit  for intensity
    Float I_unit = dt/denorm/Itau; // denorm is the co-divisor to compare frame with dt
    Size nPatch = nKernelSample/nSample;
    for (Size iPatch=0; iPatch<nPatch+1; iPatch++) {
        Size nActive;
        // for p in time, active (for temporal samples) threads only,
        if (iPatch == nPatch) { // no divergent branch
            nActive = nKernelSample % nSample;
        } else {
            nActive = nSample;
        }
        //1. Load temporal weights: p in time
        // forward in time, stored reversed in TW_storage
        // i.e., last in storage-> t-0 -^v-- t-tau <-first in storage,   
        // convolution time start at t-tau; first sample point: t-tau+kernelSampleT0;
        Size temporalWeightS = TW_storage[offsetS*nKernelSample + iPatch*nSample + tid] * (tid<nActive);  // tid < nActive, 0 otherwise
        Size temporalWeightC = TW_storage[offsetC*nKernelSample + iPatch*nSample + tid] * (tid<nActive);  // tid < nActive, 0 otherwise
        //2. Find new frames - n, usually just 1
        PosInt itFrames = (nActive-1)*kernelSampleInterval;
        if (iPatch == 0) {
            itFrames += iKernelSampleT0; // iKernelSampleT0 = 0 or kernelSampleInterval/2
        }
        // number of frames in one patch
        //Size nFrame = (itFrames*denorm + iFramePhase + (ntPerFrame-1)) / ntPerFrame - 1; // exclude the currentFrame within the framePhase, already in F_1
        Size nFrame = (itFrames*denorm + iFramePhase) / ntPerFrame + 1;
        if (blockIdx.x==0 && tid == 0) {
            if (iPatch == 0) {
                printf("nFrame: %u, maxFrame = %u\n", nFrame, maxFrame);
            }
            printf("framePhase = %u/%u\n", iFramePhase, ntPerFrame);
            //printf("itFrames = %u, nActive = %u, denorm = %u, iKernelSampleT0 = %u, kernelSampleInterval\n", iFramePhase, nActive, denorm, iKernelSampleT0, kernelSampleInterval);
            //assert(nFrame == maxFrame);
        }
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
        //  Get contrast from local_I(ntensity) and mean_I(ntensity): p in space 
        Float filteredC, filteredS;
        // initialize with the first frame in the patch
        PosInt it = 0;
        if (iPatch == 0) {
            it = iKernelSampleT0;
        }
        PosInt iFrame = 0;
        Int preFrame = currentFrame-1;
        for (PosInt iSample = 0; iSample < nActive; iSample++) {
            PosInt frameNow = currentFrame + (it*denorm + iFramePhase)/ntPerFrame;
            if (frameNow > preFrame) { // advance frame
                //Load mean luminance from shared memory first
                Float mean_I = nSampleShared[iFrame];
                // surround 
                Float local_I = get_intensity(typeS, x0S, y0S, frameNow % maxFrame);
                assert(local_I > 0);
                if (tid == 0 && blockIdx.x == 0) {
                    printf("frameNow = %u\n", frameNow);
                }
                Float local_contrast;
                if (mean_I > 0) {
                    local_contrast = local_I/mean_I - 1.0;
                } else {
                    local_contrast = local_I;
                }
                if (abs(local_contrast) > 1.0) {
                    local_contrast = copyms(1.0, local_contrast); // copyms is copysign(value, sign);
                }
                block_reduce<Float>(reducedS, spatialWeightS*local_contrast);
                if (iPatch == nPatch && iFrame == nFrame && tid ==0) {
                    contrast[gridDim.x*1+blockIdx.x] = reducedS[0];
                    luminance[lidC] = mean_I;
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
                block_reduce<Float>(reducedC, spatialWeightC*local_contrast);
                if (iPatch == nPatch && iFrame == nFrame && tid ==0) {
                    contrast[gridDim.x*0+blockIdx.x] = reducedC[0];
                    if (blockIdx.x == 0) {
                        printf("contrast0 = \n", reducedC[0]);
                    }
                }
                preFrame = frameNow;
                iFrame++;
            }
            if (tid == iSample) {
                filteredC = reducedC[0]*temporalWeightC; // shared memory have spatially convolved values 
                filteredS = reducedS[0]*temporalWeightS; // shared memory have spatially convolved values 
            }
            // advance time
            it += kernelSampleInterval;
        }
        //__syncthreads();
        //6. reduce sum with temporal weights: p in time
        if (tid >= nActive) {
            filteredS = 0.0;
            filteredC = 0.0;
        }
        block_reduce<Float>(reducedS, filteredS);
        //7. add to convol: s
        if (tid == 0) {
            convolS += reducedS[0];
        }
        block_reduce<Float>(reducedC, filteredC);
        if (tid == 0) {
            convolC += reducedC[0];
        }
        //9. advance [currentFrame, framePhase] if not the final patch: n
        if (iPatch < nPatch) {
            Size iPhaseIncrement = (nSample-1)*kernelSampleInterval;
            if (iPatch == 0) {
                iPhaseIncrement += iKernelSampleT0;
            }
            currentFrame += nFrame-1;
            iFramePhase = (iPhaseIncrement*denorm + iFramePhase) % ntPerFrame;
        }
    }
    if (tid == 0) {
        // times amplitude and space-time volume, k is amplitude*dwdh
        convolC *= kernelSampleInterval*dt*kC;
        convolS *= kernelSampleInterval*dt*kS;
    }

    Float convol = convolC + convolS;
    // update convolution data, initialized in LGN_nonlinear
    if (threadIdx.y*blockDim.x + threadIdx.x == 0) {
        atomicAdd(current_convol+blockIdx.x, convol);
    }
}

__launch_bounds__(1024, 2)
__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ current_convol,
        Float* __restrict__ LGN_fr
) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    Float max, current;
    Float C50, K, A, B;
    bool engaging = id<nLGN;
    unsigned int MASK = __ballot_sync(FULL_MASK, static_cast<int>(engaging));
    if (engaging) {
        current = current_convol[id];
		// initialize for next time step
		current_convol[id] = 0.0;
		max = max_convol[id];
        logistic.load_first(id, C50, K, A, B);
        Float convol = current;
        if (current < 0) {
            current = 0;
        }
        __syncwarp(MASK);
        Float fr = max * transform(C50, K, A, B, current/max);
        if (fr < 0) {
            printf("convol = %f, fr = %f, K=%f, A= %f, B= %f, C50 =%f, max = %f\n", convol, fr, K, A, B, C50, max);
            assert(fr >= 0);
        }
        LGN_fr[id] = fr;
        //LGN_fr[id] = max * transform(C50, K, A, B, current/max);
    }
}
