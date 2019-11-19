/*
Per frame
    max spatial sample size: 32 x 32
    number of temporal kernel evaluation <= spatial sample size
*/
// grid: nLGN blocks
// block: spatialSample1D x spatialSample1D (npixel_1D)

__device__
__inline__
void get_coord(Float xsig, Float ysig, Float nsig, SmallSize nSpatialSample_1D, float &x0, float &y0) {
    // make change consistent with the same part in store_spatialWeight
    Float xhspan = nsig * xsig / sqrt2;
    Float dx = 2*xhspan/nSpatialSample_1D;

    Float yhspan = nsig * ysig / sqrt2;
    Float dy = 2*yhspan/nSpatialSample_1D;

    x = (threadIdx.x + 0.5)*dx - xhspan;
    y = (threadIdx.y + 0.5)*dy - yhspan;

    // texture coords have to be float
    x0 = static_cast<float>(spatial.x + x);
    y0 = static_cast<float>(spatial.y + y);
}

__device__
__inline__
void store_spatialWeight(Zip_spatial &spatial,
                         Float nsig, // span of spatialRF sample in units of std
                         SmallSize nSpatialSample_1D, 
                         Float* __restrict__ SWstorage) {
    Size storeId = blockIdx.x*nSample + threadIdx.y*blockDim.x + threadIdx.x;
    // coord to center
    Float xhspan = nsig * xsig / sqrt2;
    Float dx = 2*xhspan/nSpatialSample_1D;

    Float yhspan = nsig * ysig / sqrt2;
    Float dy = 2*yhspan/nSpatialSample_1D;

    Float x = (threadIdx.x + 0.5)*dx - xhspan;
    Float y = (threadIdx.y + 0.5)*dy - yhspan;
    Float sample_vol = dx * dy;

    SmallSize nSample = nSpatialSample_1D * nSpatialSample_1D;
    SW_storage[storeId] = spatialWeight(x, y, spatial.k, spatial.rx, spatial.ry)*sample_vol;
}
//TODO: 1.conform arguments, threading in __global__
//      2.store weights in size (type, nLGN)
__device__
__inline__
void store_temporalWeight(Zip_temporal &center,
                          Zip_temporal &surround,
                          Float* __restrict__ TWstorage,
                          SmallSize nKernelSample,
                          Float kernelSampleDt,
                          Float kernelSampleT0){
    unsigned int id = blockIdx.x;
    SmallSize patch_size = blockDim.x*blockDim.y;
    SmallSize npatch = nKernelSample/patchSize;
    SmallSize remain = nKernelSample%patchSize;
    
    unsigned int tid;
    zip_temporal temp;
    if (blockIdx.y == 0) {
        // load center temporal parameters
        temp.load(center);
        tid = threadIdx.x;
    } else {
        // load surround temporal parameters
        temp.load(surround);
        tid = nLGN*nKernelSample + threadIdx.x;
    }

    Float fac1, fac2;
    fac1 = tgamma(temp.nR[id]);
    fac2 = tgamma(temp.nD[id]);
    

    for (SmallSize iPatch = 0; ipatch < npatch; ipatch++) {
        // temporalKernel takes abs(t-now)
        // store t in reverse
        SmallSize twid = nKernelSample-1 - iPatch*patchSize - threadIdx.x;
        Size storeId = id*2*nKernelSample + iPatch*patchSize + tid;

        Float t = twid*kernelSampleDt + kernelSampleT0
        TWstorage[storeId] = temporalKernel(t, temp, fac1, fac2);
    }
    // store the remaining ones
    if (threadIdx.x < remain) {
        SmallSize twid = nKernelSample-1 - iPatch*patchSize - threadIdx.x;
        Size storeId = id*2*nKernelSample + iPatch*patchSize + tid;

        Float t = twid*kernelSampleDt + kernelSampleT0
        TWstorage[storeId] = temporalKernel(t, temp, fac1, fac2);
    }
}

__global__
void store_weight(Size nLGN,

                  Temporal_component* __restrict__ t_center,
                  Temporal_component* __restrict__ t_surround,
                  SmallSize nKernelSample,
                  Float* __restrict__ TWstorage,
                  Float kernelSampleDt,
                  Float kernelSampleT0

                  Spatial_component* __restrict__ s_center,
                  Spatial_component* __restrict__ s_surround,
                  Float nsig, // span of spatialRF sample in units of std
                  SmallSize nSpatialSample_1D, 
                  Float* __restrict__ SWstorage) {

    store_temporalWeight(t_center, t_surround, TWstorage, nLGN, nKernelSample, kernelSampleDt, kernelSampleT0);

    store_spatialWeight(s_center, nsig, nSpatialSample_1D, SWstorage);
    store_spatialWeight(s_surround, nsig, nSpatialSample_1D, SWstorage + nSample * nLGN);
}

__device__
__inline__
void sub_convol(SmallSize type,
                Float framePhase,
                Float ave_tau,
                Size nKernelSample,
                Float &convol, // for returning the convol value
                Float* __restrict__ decayIn,
                Float* __restrict__ decayI1,
                Float* __restrict__ reduced, // shared mem ptr for block_reduce
                Float* __restrict__ SWstorage,
                Float* __restrict__ TWstorage) {
    /* kernel sampling diagram with frames
          now-tau framePhase                                   now
        time->   | ^ |-------> tPerFrame <------|      ^      |
        frame:    curr         next                next+1
        sample:  1        2        3        4        5        6
                 |...:----|--------|--------|---:----|--------|
        dt:      0|  3    8        16       24       32       40
                  1 <--- update for next decayI 
        e.g., tau = 40*dt
    */
    SmallSize nSample = nSpatialSample_1D*nSpatialSample_1D;
    // texture coords have to be float
    float x0, y0;
    get_coord(spatial.rx, spatial.ry, nsig, nSpatialSample_1D, x0, y0);

    Float lastDecayIn, lastDecayI1, decay;
    if (threadIdx.x == 0) {
        lastDecayIn = decayIn[id]; // = sum_i2n(F_i[exp(-t_i+1/ave_tau) - exp(-t_i/ave_tau)] - F_{n+1})*exp(-(t-tau)/ave_tau), 
        lastDecayI1 = decayI1[id]; // = F_{n+1}
        //F_i is the mean of all sampled pixel value of the ith frame in the LGN's RF.
        decay0 = expp(-kernelSampleT0/ave_tau);
        decay = expp(-kernelSampleDt/ave_tau);
    }
    Float spatialWeight = *SWstorage;
    Float local_I, mean_I;

    for (unsigned int iSample=0; iSample<nKernelSample; iSample++) {
        Float t2now = iSample*kernelSampleDt + kernelSampleT0;
        if (framePhase + t2now > tPerFrame) {
            currentFrame++;
            Float local_I = get_intensity(type, x0, y0, currentFrame);
            block_reduce<Float>(reduced, local_I);
            if (threadIdx.x == 0) {
                // calculate new mean using the first thread
                Float newDecayI1 = reduced[0]/nSample;
                framePhase += t2now - tPerFrame;
                lastDecayIn = lastDecayIn * decay + (lastDecayI1 - newDecayI1) * expp(-framePhase/ave_tau);
                lastDecayI1 = newDecayI1;
                mean_I = lastDecayIn + lastDecayI1;
                reduced[0] = mean_I;
            }
            __syncthreads();
            mean_I = reduced[0]; // load to all threads
        } else {
            Float local_I = get_intensity(type, x0, y0, currentFrame);
            lastDecayIn *= decay;
            mean_I = lastDecayIn + lastDecayI1;
        }
        Float local_contrast = (local_I-mean_I)/mean_I;
        if (local_contrast > 1.0) {
            local_contrast = 1.0;
        } else {
            if (local_contrast < -1.0) {
                local_contrast = -1.0;
            }
        }
        block_reduce<Float>(reduced, spatialWeight*local_contrast);// dxdy is included in spatialWeight
        if (tid == 0) {
            Float tw = temporalWeight[iSample]; // first at t-tau+kernelSampleT0
            Float weigthed_sum = reduced[0];
            convol += weigthed_sum*tw*kernelSampleDt;
        }
    }
}

__global__ 
void LGN_convol_2Cone(Float* __restrict__ decayIn,
                      Float* __restrict__ decayI1,
                      Float* __restrict__ TWstorage,
                      Float* __restrict__ SWstorage,
                      Float* __restrict__ LGNfr,
                      SmallSize* __restrict__ center_type,
                      SmallSize* __restrict__ surround_type,
                      Spatial_component* center, // consider pointer
                      Spatial_component* surround, // consider pointer
                      Size nLGN,
                      Float framePhase,
                      SmallSize nKernelSample,
                      Float kernelSampleT0,
                      Float kernelSampleDt,
                      PosInt nsig,
                      SmallSize nSpatialSample_1D) {

    __shared__ Float reduced[warpSize];
    unsigned int id = blockIdx.x;
    SmallSize ctype = center_type[id];
    SmallSize stype = surround_type[id];
    // load spatial parameters
    Zip_spatial cen(*center, id);
    Zip_spatial sur(*surround, id);

    unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;

    SmallSize nSample = nSpatialSample_1D*nSpatialSample_1D;
    Size SWstoreId = id*nSample + threadIdx.y*blockDim.x + threadIdx.x;
    Size TWstoreId = id*nKernelSample;

    Float convol;
    if (tid == 0) {
        convol = 0.0f;
    }

    sub_convol(ctype, framePhase, ave_tau, Size nKernelSample, convol, decayIn, decayI1, reduced, SWstorage + SWstoreId, TWstorage + TWstoreId);

    SWstoreId += nLGN*nSample;
    TWstoreId += nLGN*nKernelSample;
    //TODO: ave_tau can take different value for different cone type
    sub_convol(stype, framePhase, ave_tau, Size nKernelSample, convol, decayIn, decayI1, reduced, SWstorage + SWstoreId , TWstorage + TWstoreId);

    if (tid == 0) {
        LGNfr[blockIdx.x] = convol;
    }
}
