#include "discrete_input_convol.cuh"

extern texture<float, cudaTextureType2DLayered> L_retinaConSig;
extern texture<float, cudaTextureType2DLayered> M_retinaConSig;
extern texture<float, cudaTextureType2DLayered> S_retinaConSig;
extern __device__ __constant__ float sqrt2;

// TODO-10:
// block-processing without input-output delay
// http://www.cs.ust.hk/mjg_lib/bibs/DPSu/DPSu.Files/Ga95.PDF
// block processing file:///C:/Users/gueux/Desktop/FFTConvolution.pdf

// DIRECT FORM only:

__device__ 
__forceinline__ 
Float AexpTau(Float a, Float tau) {
    return a * exponential(-tau);
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
void retina_to_plane(Float polar, Float ecc, float &x, float &y, const Float curv_ratio) {
    Float r = tangent(theta);
    x = static_cast<float>(r*curv_ratio*cosine(polar));
    y = static_cast<float>(r*curv_ratio*sine(polar));
}

__device__
__forceinline__
void get_coord_in_plane(Float wSpan, Float hSpan, Float Ori, Float centerPolar, Float centerEcc, SmallSize nWidth, SmallSize nHeight, Float &polar, Float &ecc, Float &dxdy, float &x0, float &y0, bool mainThread) {
    // make change consistent with the same part in store_spatialWeight
    Float dw = 2*wSpan/nWidth;

    Float dy = 2*hSpan/nHeight;

    x = (threadIdx.x + 0.5)*dx - wSpan;
    y = (threadIdx.y + 0.5)*dy - wSpan;

    // texture coords have to be float
    retina_to_plane(cP+x, cE+y, x0, y0);
    if (mainThread) {
        dxdy = dx*dy;
    }
}

__device__ 
__forceinline__
Float get_intensity(unsigned int coneType, float x, float y, unsigned int iLayer) {
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
    return contrast;
}

// for 2 cone-types LGN only, can be generalized for more cone-types
// gridSize: (nLGN, nType) blocks for store 1-D nLGN for convol
// blockSize: spatialSample1D x spatialSample1D (npixel_1D)

/* TODO: compare all thread load from global memory vs shared
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
    SmallSize nPatch = nKernelSample/patchSize;
    SmallSize remain = nKernelSample%patchSize;
    //DEBUG
    //if ((lid ==0 || lid == gridDim.x) && tid == 0) {
    //    printf("%f, %f, %f, %f, %f, %f\n", temp.nR, temp.nD, temp.tauR, temp.tauD, temp.delay, temp.ratio);
    //    printf("patchSize = %u, nPatch = %u, remain = %u\n", patchSize, nPatch, remain);
    //}
    //__syncthreads();
    //
    
    Float lfac1, lfac2;
    lfac1 = log_gamma(temp.nR);
    lfac2 = log_gamma(temp.nD);

    //DEBUG
    //if ((lid ==0 || lid == gridDim.x) && tid == 0) {
    //    printf("lid:%d, tid:%d\n%f: %f, %f, %f\n %f: %f, %f, %f\n", lid, tid, temp.nR, lfac1, tgamma(temp.nR), exp(lfac1), temp.nD, lfac2, tgamma(temp.nD), exp(lfac2));
    //}
	//__syncthreads(); 
    //
    // account for the delay into T0
    kernelSampleT0 -= temp.delay;
    // initialize the sum of temporalWeights to 0
    temporalWeight = 0;
    for (SmallSize iPatch = 0; iPatch < nPatch+1; iPatch++) {
        Float tw;
        if (iPatch < nPatch || tid < remain) {
            // temporalKernel takes abs(now-t)
            // but we store weight in time-reverse (tau -> 0)
            SmallSize twid = nKernelSample-1 - iPatch*patchSize - tid;
            Size storeID = (id*nType + iType)*nKernelSample + iPatch*patchSize + tid;

			Float t = twid * kernelSampleDt + kernelSampleT0;

            //DEBUG
			//if ((lid ==0 || lid == gridDim.x) && tid == 0) {
            //  printf("lid = %d, tid = %d, t = %f\n", lid, tid, t);
            //}
            //
            if (t < 0) {
                tw = 0.0;
            } else {
                tw = temporalKernel(t, temp, lfac1, lfac2, lid, tid);
            }
            //DEBUG
			//if ((lid ==0 || lid == gridDim.x) && tid == 0) {
            //  printf("lid = %d, tid = %d, tw = %f\n", lid, tid, tw);
            //}
            //
			
            TW_storage[storeID] = tw;
            // get absolute values ready for max_convol
            if (tw < 0) tw = -tw;
        } else {
            tw = 0.0;
        }
        __syncthreads();
        block_reduce<Float>(reduced, tw);
        if (tid == 0) {
            //DEBUG
            //printf("total summed tw = %f\n", temporalWeight);
            //printf("patch summed tw = %f\n", reduced[0]);
            //assert(!isnan(reduced[0]));
            temporalWeight += reduced[0];
        }
    }
}

__device__
__forceinline__
Float store_spatialWeight(
        shared_spat &ss,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float nsig, // span of spatialRF sample in units of std
        Size id,
        Size tid,
        Size offset, // (id*nType + iType) * nSample
        Size nSample,
        bool storeSpatial
) {
    // parameter are stored as (nType, nLGN), weights are stored as (nLGN, nType, weight)
    Size storeID = offset + tid;

    // coord to center
    Float x = (threadIdx.x + 0.5)*ss.dx - ss.xhspan;
    Float y = (threadIdx.y + 0.5)*ss.dy - ss.yhspan;

    Float spatialWeight = spatialKernel(x, y, ss.rx, ss.ry);
    
    if (storeSpatial) {
        SW_storage[storeID] = spatialWeight;
        float x_plane, y_plane;
        retina_to_plane(ss.cx + x, ss.cy + y, x_plane, y_plane);
        // store coords for retrieve data from texture
        assert(x_plane < 1);
        assert(x_plane > 0);
        assert(y_plane < 1);
        assert(y_plane > 0);
        SC_storage[storeID] = x_plane; // x
        SC_storage[storeID + gridDim.x*blockIdx.y*nSample] = y_plane; // y
    }
    return spatialWeight;
}

// weights are stored in shapes of (nLGN, nType, weight)
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
        Float* __restrict__ dxdy_storage,
        Float nsig, // span of spatialRF sample in units of std
        bool storeSpatial
) {
    __shared__ Float reduced[warpSize]; 
    __shared__ Float spat[8]; // xhspan, yhspan, dx, dy, cx, cy, rx, ry
    Size id = blockIdx.x;
    SmallSize iType = blockIdx.y;
    Size lid = iType*gridDim.x + id;
    SmallSize nType = gridDim.y;
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    SmallSize nSample = blockDim.x * blockDim.y;

    Float temporalWeight, spatialWeight, dxdy, k;
    store_temporalWeight(temporal, TW_storage, reduced, temporalWeight, nKernelSample, kernelSampleDt, kernelSampleT0, id, tid, lid, iType, nType);
    // DEBUG
	if ((lid ==0 || lid == gridDim.x) && tid == 0) {
        printf("temporalWeights stored\n");
        assert(!isnan(temporalWeight));
    }
    __syncthreads();
    //

    bool use_shared = true;
    if (use_shared) { // TODO: compare with global broadcast
        Size offset = gridDim.x*iType + id;
        if (tid == 0) {
            Zip_spatial spat0;
            spat0.load(spatial, lid);
            Float xhspan = nsig * spat0.rx / sqrt2;
            Float dx = 2*xhspan/blockDim.x;

            Float yhspan = nsig * spat0.ry / sqrt2;
            Float dy = 2*yhspan/blockDim.y;
            dxdy = dx*dy;
			k = spat0.k;
            // store dxdy
            dxdy_storage[offset] = dxdy;

            spat[0] = xhspan; spat[1] = yhspan; spat[2] = dx; spat[3] = dy; spat[4] = spat0.x; spat[5] = spat0.y; spat[6] = spat0.rx; spat[7] = spat0.ry;
        }
        __syncthreads();
        // load from shared mem
        shared_spat ss(spat);
        spatialWeight = store_spatialWeight(ss, SW_storage, SC_storage, nsig, id, tid, offset*nSample, nSample, storeSpatial);
    } 
	if (lid ==0 || lid == gridDim.x && tid == 0) {
        printf("spatialWeights stored\n");
        assert(!isnan(spatialWeight));
    }
	if (lid ==0 && tid == 0) {
        assert(max_convol[id] == 0.0);
    }
	block_reduce<Float>(reduced, spatialWeight);

    if (tid == 0) { // iType = 0, 1
        atomicAdd(max_convol+id, reduced[0] * temporalWeight * k * dxdy * kernelSampleDt);
    }
}

__device__
__forceinline__
void sub_convol(
        SmallSize type,
        PosInt nsig,
        SmallSize currentFrame,
        SmallSize maxFrame,
		Float tPerFrame,
        Float framePhase,
        Float Itau,
        Float kernelSampleDt,
        Size nKernelSample,
        Float &convol, // for returning the convol value
        Spatial_component &spatial,
        Float* __restrict__ decayIn,
        Float* __restrict__ lastF,
        Float* __restrict__ reduced, // shared mem ptr for block_reduce
        Float* __restrict__ nSampleShared, // shared mem ptr for block_reduce
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ dxdy_storage,
        Float* __restrict__ TW_storage,
        Size tid,
        SmallSize iType,
        SmallSize nType,
        Float dt,
        bool spatialStored
) {
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
    SmallSize nSample = blockDim.x * blockDim.y;
    Size lid = iType*gridDim.x + id;

    Float dxdy, k;
    Float spatialWeight;
    float x0, y0; // coord on the stimulus plane
    if (spatialStored) {
		Size offset0 = (gridDim.x*iType + id);
        Size offset = offset0*nSample;
        Size storeID = offset + threadIdx.y*blockDim.x + threadIdx.x;
        spatialWeight = SW_storage[storeID];
        x0 = SC_storage[storeID];
        y0 = SC_storage[storeID + gridDim.x*nType*nSample];
        if (tid == 0) {
            dxdy = dxdy_storage[offset0];
            k = spatial.k[lid];
        }
    } else {
        Zip_spatial spat;
        spat.load(spatial, lid);
        Float polar, ecc;
        // texture coords have to be float
        get_coord_in_plane(spat.rx, spat.ry, spat.x, spat.y, nsig, blockDim.x, blockDim.y, x, y, dxdy, x0, y0, tid == 0); // dxdy is only given to the tid == 0
        spatialWeight = spatialKernel(polar, ecc, spat.sPolar, spat.sEcc);
        if (tid == 0) {
            k = spat.k;
        }
    }
    /* Light adaptation process:
        tau*dI/dt = -I + F(t);
        F(t) = piecewise F(t_{i}), for t in [t_{i}, t_{i+1}), t_{i} is the onset time of the i-th frame
    */
    Float lastDecayIn = decayIn[lid]; // = sum_i2n(F_i[exp(-t_{i+1}/Itau) - exp(-t_{i}/Itau)] - F_{n+1})*exp(-(t-tau)/Itau)
    Float F_1 = lastF[lid]; // = F_{n+1}
    //F_i is the mean of all sampled pixel value of the ith frame in the LGN's RF.

    // DEBUG:
    __syncthreads();
    if (id == 0 && threadIdx.y*blockDim.x + threadIdx.x == 0) {
        printf("spatial storage loaded\n");
    }
    //

    /* looping the following over (nPatch + 1) patches on nKernelSample samples points:
        p - parallelized by all threads;
        n - needed by all threads;
        s - single thread 
     */
    SmallSize nPatch = nKernelSample/nSample;
    for (SmallSize iPatch=0; iPatch<nPatch+1; iPatch++) {
        Float temporalWeight;
        SmallSize nActive;
        // for p in time, active threads only,
        if (iPatch == nPatch) {
            nActive = nKernelSample - iPatch*nSample;
        } else {
            nActive = nSample;
        }
        //1. Load temporal weights: p in time
        if (tid < nActive) {
            //convolution time start at t-tau, forward in time, reversed in temporalKernel; first sample point: t-tau+kernelSampleT0
            temporalWeight = TW_storage[iPatch*nSample + tid]; 
        }
    // DEBUG:
        __syncthreads();
        if (id == 0 && threadIdx.y*blockDim.x + threadIdx.x == 0) {
            printf("temporal storage of patch %d loaded.\n", iPatch);
        }
    //
        //2. Find new frames - n
        Float tFrames = nActive*kernelSampleDt + framePhase;
        SmallSize nFrame = static_cast<SmallSize>(tFrames/tPerFrame);
        //3. For all the new frames
        for (SmallSize iFrame = 0; iFrame < nFrame; iFrame++) {
            //Get F_i by reduce - p: in space
            Float local_I = get_intensity(type, x0, y0, (currentFrame + iFrame + 1) % maxFrame);
            block_reduce<Float>(reduced, local_I);
            if (tid == 0) {
                // __shared__ to (register/local) to __shared__
                nSampleShared[iFrame] = reduced[0]/nSample;  // shared memory now used for spatial luminance sum
            }
            __syncthreads();
        }
    // DEBUG:
        __syncthreads();
        if (id == 0 && threadIdx.y*blockDim.x + threadIdx.x == 0) {
            printf("%d frames in the new patch.\n", nFrame);
        } 
    //
        //!!! Update light adapation variables here to hide latency: p in space 
        if (iPatch == 0 && tid == 0) { // dt < kernelSampleDt, must be in the first patch
            Float tf0 = dt + framePhase;
            if (tf0 < tPerFrame) {
                //lastF is not changed
                decayIn[lid] = lastDecayIn*exponential(-dt/Itau);
            } else {
                // here we ASSUME tPerframe > dt, i.e., at most one change of frame happen within a single dt
                Float F_2 = nSampleShared[0];
                decayIn[lid] = lastDecayIn*exponential(-dt/Itau) + (F_1 - F_2) * exponential(-(tf0- tPerFrame)/Itau);
                lastF[lid] = F_2;
            }
        }
        //4. Calculate mean_I: p in time 
        if (tid < nActive) {
            Float t = (tid+1)*kernelSampleDt;
            Float tf0 = t+framePhase;
            Size local_nFrame = static_cast<Size>(tf0/tPerFrame);
            // if nFrame == 0 then F_{n+1} (F_1) is not changed
            lastDecayIn *= exponential(-t/Itau); // sum_i2n(F_i[exp(-t_{i+1}/Itau) - exp(-t_{i}/Itau)] - F_{n+1})*exp(-(t0-tau)/Itau) decayed to t = t0 + t
            for (SmallSize iFrame = 0; iFrame < local_nFrame; iFrame++) {
                // number of active threads decreases for each loop
                Float F_2 = nSampleShared[iFrame]; // load from shared memory to register first // TODO: check register usage here 
                tf0 -= tPerFrame;
                Float decay = exponential(-tf0/Itau);
                lastDecayIn += (F_1 - F_2) * decay; // F_{n+1} decayed with exp(-((t+t0) - t_{i+2})/Itau);
                F_1 = F_2;
            }
            nSampleShared[tid] = lastDecayIn + F_1; //shared memory now used as mean luminance
        }
        __syncthreads();
    // DEBUG:
        if (id == 0 && threadIdx.y*blockDim.x + threadIdx.x == 0) {
            printf("mean Intensity ready.\n");
        } 
    //
        //5. For each sample point in time: 
        //  Get contrast from local_I(ntensity) and mean_I(ntensity): p in space 
        SmallSize iFrame = static_cast<SmallSize>((kernelSampleDt + framePhase)/tPerFrame);
        Float local_I = get_intensity(type, x0, y0, (currentFrame + iFrame) % maxFrame);
        for (SmallSize iSample = 0; iSample < nSample; iSample++) {
            //Load mean luminance from shared memory first
            Float mean_I = nSampleShared[iSample];
            SmallSize frameNow = static_cast<SmallSize>((iSample*kernelSampleDt + framePhase)/tPerFrame);
            if (frameNow > iFrame) {
                // new frame, in case frame rate > sample rate, we don't increase iFrame in single units
                iFrame = frameNow;
                local_I = get_intensity(type, x0, y0, (currentFrame + iFrame) % maxFrame);
            }
            Float local_contrast = (local_I-mean_I)/mean_I;
            if (abs(local_contrast) > 1.0) {
                local_contrast = copy(1.0, local_contrast); // copy is copysign(value, sign);
            }
            block_reduce<Float>(reduced, spatialWeight*local_contrast);
            if (tid == 0) {
                // __shared__ to (register/local) to __shared__
                nSampleShared[iSample] = reduced[0]; // shared memory now stores spatially convolved values
            }
        }
    // DEBUG:
        if (id == 0 && threadIdx.y*blockDim.x + threadIdx.x == 0) {
            printf("contrast, done.\n");
        } 
    //
        //6. reduce sum with temporal weights: p in time
        Float filtered;
        if (tid < nActive) {
            filtered = nSampleShared[tid]*temporalWeight; // shared memory have spatially convolved values 
        } else {
            filtered = 0.0f;
        }
        block_reduce<Float>(reduced, filtered);
        //7. add to convol: s
        if (tid == 0) {
            convol += reduced[0];
        }
        //9 .advance currentFrame and framePhase if not the final patch: n
        if (iPatch < nPatch) {
            currentFrame += nFrame;
            framePhase = fmod(nActive*kernelSampleDt + framePhase, tPerFrame);
        }
    }
    // DEBUG:
        if (id == 0 && threadIdx.y*blockDim.x + threadIdx.x == 0) {
            printf("contrast, done.\n");
        } 
    //
    if (tid == 0) {
        // times amplitude and space-time volume
        convol *= kernelSampleDt*dxdy*k;
    }
}

// grid: [nLGN, 1, 1]
// block: [nSpatialSample1D, nSpatialSample1D, 1]
__launch_bounds__(1024, 2)
__global__ 
void LGN_convol_c1s(
        Float* __restrict__ decayIn,
        Float* __restrict__ lastF,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ dxdy_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ LGNfr,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
        Float nsig,
        SmallSize currentFrame,
        SmallSize maxFrame,
		Float tPerFrame,
        Float framePhase,
        Float Itau,
        Float kernelSampleDt,
        Size nKernelSample,
        Float dt,
        bool spatialStored
) {
    __shared__ Float reduced[warpSize];
    extern __shared__ Float nSampleShared[];
    unsigned int id = blockIdx.x;
    SmallSize type = coneType[id];
    unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;

    // weights are stored in shapes of (nLGN, nType, weight)

    Float convol;
    if (tid == 0) {
        convol = 0.0f;
    }

    //TODO: Itau may take different value for different cone type
    // convolve center and update decayIn, lastF
    sub_convol(type, nsig, currentFrame, maxFrame, tPerFrame, framePhase, Itau, kernelSampleDt, nKernelSample, convol, spatial, decayIn, lastF, reduced, nSampleShared, SW_storage, SC_storage, dxdy_storage, TW_storage, tid, 0, 2, dt, spatialStored);

    type = coneType[id + gridDim.x];

    // convolve surround and add to convol and update decayIn, lastF
    sub_convol(type, nsig, currentFrame, maxFrame, tPerFrame, framePhase, Itau, kernelSampleDt, nKernelSample, convol, spatial, decayIn, lastF, reduced, nSampleShared, SW_storage, SC_storage, dxdy_storage, TW_storage, tid, 1, 2, dt, spatialStored);

    // update convolution data 
    if (tid == 0) {
        LGNfr[id] = convol;
    }
}

__launch_bounds__(32, 64)
__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ LGN_fr
) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    Float _max_convol, current_convol;
    if (id < nLGN) {
        _max_convol = max_convol[id];
        current_convol = LGN_fr[id];
        if (current_convol < 0) {
            current_convol = 0;
        }
    }
    __syncwarp(); // check necessity

    if (id < nLGN) {
        Float ratio = logistic.transform(id, current_convol/_max_convol);
        LGN_fr[id] = current_convol * ratio;
    }
}
