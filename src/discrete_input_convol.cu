#include "discrete_input_convol.cuh"
// CANNOT USE EXPRESSION IN (inline seems fine) FUNCTION ARGUMENTS, LEADS TO ZERO!
//extern texture<float, cudaTextureType2DLayered> L_retinaInput;
//extern texture<float, cudaTextureType2DLayered> M_retinaInput;
//extern texture<float, cudaTextureType2DLayered> S_retinaInput;
//extern surface<void, cudaSurfaceType2DLayered> LGNspikeSurface;

__device__ 
__forceinline__ 
Float spatialKernel(Float x, Float y, Float rx, Float ry) {
    return exponential(-x*x/(rx*rx) - y*y/(ry*ry));
}

__device__ 
__forceinline__ 
Float temporalKernel(Float tau, Zip_temporal &temp, Float lfac1, Float lfac2) {
    Float tau1 = tau/temp.tauR;
    Float tau2 = tau/temp.tauD;
    //Float A1 = power(tau1, temp.nR-1)/temp.tauR;
    //Float A2 = power(tau2, temp.nD-1)/temp.tauD;

    Float A1 = (temp.nR-1) * logarithm(tau1);
    Float A2 = (temp.nD-1) * logarithm(tau2);

    //Float tp = AexpTau(A1, tau1 + lfac1) - temp.ratio*AexpTau(A2, tau2 + lfac2);

    Float exponent = A1 - tau1 - lfac1;
    Float tpR = exponential(exponent)/temp.tauR;
    exponent = A2 - tau2 - lfac2;
    Float tpD = exponential(exponent)/temp.tauD;
    Float tp = tpR - temp.ratio*tpD;
    return tp;
}

__device__ 
__forceinline__
Float get_intensity(SmallSize coneType, float x, float y, unsigned int iLayer, cudaTextureObject_t L_retinaInput, cudaTextureObject_t M_retinaInput, cudaTextureObject_t S_retinaInput) {
    Float contrast;
    switch (coneType) {
        case 0:
            contrast = static_cast<Float>(tex2DLayered<float>(L_retinaInput, x, y, iLayer));
            break;
        case 1:
            contrast = static_cast<Float>(tex2DLayered<float>(M_retinaInput, x, y, iLayer));
            break;
        case 2:
            contrast = static_cast<Float>(tex2DLayered<float>(S_retinaInput, x, y, iLayer));
            break;
        case 3: // On-Off only magnocellular excluding S cone
            contrast = (static_cast<Float>(tex2DLayered<float>(L_retinaInput, x, y, iLayer))
                      + static_cast<Float>(tex2DLayered<float>(M_retinaInput, x, y, iLayer)))/2.0; 
            break;
        case 4: 
			/* Hunt Lum
            contrast = static_cast<Float>(static_cast<Float>(tex2DLayered(L_retinaInput, x, y, iLayer) * 0.361222
                                        + static_cast<Float>(tex2DLayered(M_retinaInput, x, y, iLayer) * 0.638804
                                        + static_cast<Float>(tex2DLayered(S_retinaInput, x, y, iLayer) * (-7.127501e-6);
			*/
			// CAT02
            contrast = static_cast<Float>(tex2DLayered<float>(L_retinaInput, x, y, iLayer)) * 0.45436904
                     + static_cast<Float>(tex2DLayered<float>(M_retinaInput, x, y, iLayer)) * 0.47353315
                     + static_cast<Float>(tex2DLayered<float>(S_retinaInput, x, y, iLayer)) * 0.0720978;
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


/*
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
*/
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
Float get_virtual_convol(float x, float y, cudaTextureObject_t linearFrame, PosInt prev, PosInt next, Float r) {
	Float p0 = static_cast<Float>(tex2DLayered<float>(linearFrame, x, y, prev));
	Float p1 = static_cast<Float>(tex2DLayered<float>(linearFrame, x, y, next));
	Float p = p0 + (p1-p0)*r;
	return p;
}

__launch_bounds__(1024,1)
__global__
void virtual_LGN_convol(
        Float* __restrict__ lum,
        Float* __restrict__ contrast,
		cudaTextureObject_t* __restrict__ linearFrame,
        Float* __restrict__ current_convol,
		float* __restrict__ parvo_center,
		float* __restrict__ magno_center,
		InputType_t* __restrict__ LGN_type,
		Int inputType, Size nParvo_L, Size nMagno_L, Size nParvo_R, Size nLGN, PosInt prev, PosInt next, Float rt, bool saveOutputB4V1) 
{
    Size iLGN = blockIdx.x*blockDim.x + threadIdx.x;
	if (iLGN < nLGN) {
		float* center;
		Size nTypeLGN;
        Size jLGN;
		if (iLGN < nParvo_L || (iLGN > nParvo_L + nMagno_L && iLGN < nParvo_L + nMagno_L + nParvo_R)) {
			center = parvo_center;
			nTypeLGN = nParvo_L + nParvo_R;
            jLGN = iLGN;
            if (iLGN > nParvo_L + nMagno_L) {
                jLGN -= nMagno_L;
            }
		} else {
			center = magno_center;
			nTypeLGN = nLGN - (nParvo_L + nParvo_R);
            jLGN = iLGN - nParvo_L;
            if (iLGN > nParvo_L + nMagno_L) {
                jLGN -= nParvo_R;
            }
		}
		Float mx = center[jLGN];
		Float my = center[nTypeLGN + jLGN];
        /*
        for (PosInt i = 0; i<nLGN; i++) {
            if (iLGN == i) {
                printf("%i, (%f, %f)", iLGN, mx, my);
                if ((iLGN+1) % 14 == 0) {
                    printf("\n");
                }
            }
            __syncthreads();
        }*/
		// 0, 1, 2, 3, 4, 5, 6, 7
		// On, Off, L_on, L_off, M_on, M_off
		PosInt iType; 
		if (inputType == 0) { 
			// 0:
			// 0, 1
			// On(4), Off(5)
			iType = LGN_type[iLGN]-4;
		}
		if (inputType == 1) {
			//1:
			// 0, 1, 2, 3
			// L_on, L_off, M_on, M_off
			iType = LGN_type[iLGN];
		}
		if (inputType == 2) {
			// 2:
			// 0, 1, 2, 3, 4, 5
			// L_on(0), L_off(1), M_on(2), M_off(3), On(4), Off(5)
			iType = LGN_type[iLGN];
		}
		Float value = get_virtual_convol(mx, my, linearFrame[iType], prev, next, rt);
		/*if (iLGN == 200) {
			Float prev_value = static_cast<Float>(tex2DLayered<float>(linearFrame[iType], mx, my, prev));
			Float next_value = static_cast<Float>(tex2DLayered<float>(linearFrame[iType], mx, my, next));
			printf("\n %u: prev = %f, next =%f, rt = %f, value = %f\n", iLGN, prev_value, next_value, rt, value);
		}*/
		current_convol[iLGN] = value;
		if (saveOutputB4V1) {
			lum[iLGN] = value;
			contrast[iLGN] = value;
		}
	}
}
__device__
__forceinline__
void store_temporalWeight(
        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        Float* __restrict__ reduced, //__shared__
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
    
    Float lfac1, lfac2;
    lfac1 = log_gamma(temp.nR);
    lfac2 = log_gamma(temp.nD);

    // account for the delay into T0
    kernelSampleT0 -= temp.delay;
    // initialize the sum of temporalWeights to 0
    Float temporalWeight;
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

            if (t < 0) {
                tw = 0.0;
            } else {
                tw = temporalKernel(t, temp, lfac1, lfac2);
            }
			
            TW_storage[storeID] = tw;
            tw = abs(tw);
        } else {
            tw = 0.0;
        }
        //assert(!isnan(tw));
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
        Float* __restrict__ sample_x,
        Float* __restrict__ sample_y,
        Float* __restrict__ sample_w,
        Size nSample,
        Float nsig,
        bool uniform_retina,
        bool virtual_LGN 
) {
    // rads relative to center
    Float w;
    Float h;
    Float areal_weight;
    if (nsig > 0) {
        w = (threadIdx.x + 0.5)*dw - wSpan;
        h = (threadIdx.y + 0.5)*dh - hSpan;
        areal_weight = 1;
    } else {
        int idx = threadIdx.y * WARP_SIZE + threadIdx.x;
        w = sample_x[idx]*wSpan/nsig;
        h = sample_y[idx]*hSpan/nsig;
        areal_weight = sample_w[idx]*wSpan*hSpan/nsig/nsig;
    }

	if (blockIdx.x == 0 && blockIdx.y == 0) {
        Float spatialWeight = spatialKernel(w, h, wSigSqrt2, hSigSqrt2) * areal_weight;
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
        if (!virtual_LGN) {
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
        }
        //
        /* DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            Float x0 = normViewDistance * cosine(centerPolar);
            Float y0 = normViewDistance * sine(centerPolar);
            Float x1 = LR_x0 + x0 * centerEcc + w * normViewDistance;
            Float y1 = LR_y0 + y0 * centerEcc + h * normViewDistance;
            printf("x:%e->%e; dx:%e,%e\n", x1, x, x1-(LR_x0 + x0*centerEcc), x-(LR_x0 + tanEcc * x0));
        }*/
    } else {
        x = LR_x0 + (centerEcc * cosine(centerPolar) + w * coso - h * sino) * normViewDistance;
        y = LR_y0 + (centerEcc * sine(centerPolar) + w * sino + h * coso) * normViewDistance;
        // DEBUG visual field and stimulus field not matching
        if (!virtual_LGN) {
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
        }
        //
    }
    
    // store coords for retrieve data from texture
    SC_storage[storeID] = x; // x
               //nLGN(parvo) * nType * nSample (all the x)
    SC_storage[gridDim.x*gridDim.y*nSample + storeID] = y; // y
    /*DEBUG
    if (blockIdx.x == 3 && blockIdx.y == 1) {
		printf("%i-coord = (%1.6e, %1.6e)\n", blockDim.x*threadIdx.y + threadIdx.x, x, y);
    }*/
}

__launch_bounds__(1024,1)
__global__
void store_PM(
        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component &spatial,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        float* __restrict__ center,
        Float* __restrict__ max_convol,
        Float* __restrict__ sample_x,
        Float* __restrict__ sample_y,
        Float* __restrict__ sample_w,
        Size nBefore, Size nAfter, Size nL, Size nLGN,
	    Float L_x0,
	    Float L_y0,
	    Float R_x0,
	    Float R_y0,
	    Float normViewDistance,
        Float nsig, // span of spatialRF sample in units of std
        int PM, // 0: parvo, 1: magno
        bool uniform_retina,
        bool virtual_LGN
) {
    __shared__ Float reduced[WARP_SIZE];
    __shared__ Float shared_spat[10]; // centerPolar, centerEcc, coso, sino, wSpan, hSpan, dw, dh, wSigSqrt2, hSigSqrt2; TODO: use reduced is enough
    // nBefore: parvo:0,        magno: nParvo_L
    // nL:      parvo:nParvo_L, magno: nMagno_L
    // nAfter:  parvo:nMagno_L, magno: nParvo_R
    SmallSize iType = blockIdx.y;
    Size lid = iType*nLGN + nBefore + blockIdx.x;
    if (blockIdx.x >= nL) {
        lid += nAfter;
    }
    SmallSize nType = gridDim.y;
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    SmallSize nSample = blockDim.x * blockDim.y;

    store_temporalWeight(temporal, TW_storage, reduced, nKernelSample, kernelSampleDt, kernelSampleT0, blockIdx.x, tid, lid, iType, nType);
	Float LR_x0, LR_y0;
    bool LR = blockIdx.x < nL;
	if (LR) {
		LR_x0 = L_x0;
		LR_y0 = L_y0;
	} else {
		LR_x0 = R_x0;
		LR_y0 = R_y0;
	}

    if (PM == 0) {
        Size offset = blockIdx.x*nType + iType;
        if (tid == 0) {
            Zip_spatial spat;
            spat.load(spatial, lid);
            Float wSpan = nsig * spat.rx / SQRT2;
            Float dw = 2*wSpan/blockDim.x;

            Float hSpan = nsig * spat.ry / SQRT2;
            Float dh = 2*hSpan/blockDim.y;
			{
            	shared_spat[0] = spat.x; // polar
				shared_spat[1] = spat.y; // ecc
				shared_spat[2] = cosine(spat.orient); 
				shared_spat[3] = sine(spat.orient); 
				shared_spat[4] = wSpan;
				shared_spat[5] = hSpan;
				shared_spat[6] = dw;
				shared_spat[7] = dh;
				shared_spat[8] = spat.rx; 
				shared_spat[9] = spat.ry;
			}
            //spat.k /= dw*dh;
            if (virtual_LGN && blockIdx.y == 0) {
                float x, y;
                float cosp_d = cosine(shared_spat[0])*normViewDistance;
                float sinp_d = sine(shared_spat[0])*normViewDistance;
                if (!uniform_retina) {
                    Float tanEcc = tangent(shared_spat[1]);
                    x = LR_x0 + tanEcc * cosp_d;
                    y = LR_y0 + tanEcc * sinp_d;
                } else {
                    x = LR_x0 + shared_spat[1] * cosp_d;
                    y = LR_y0 + shared_spat[1] * sinp_d;
                }
                center[blockIdx.x] = x;
		    	center[gridDim.x + blockIdx.x] = y;
                if (!uniform_retina) {
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
                } else {
                    if (!virtual_LGN) {
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
                    }
                }
            }
        }
        __syncthreads();
        // load from shared mem
        store_spatialWeight(reduced, shared_spat[0], shared_spat[1], shared_spat[2], shared_spat[3], shared_spat[4], shared_spat[5], shared_spat[6], shared_spat[7], shared_spat[8], shared_spat[9], normViewDistance, LR_x0, LR_y0, LR, SW_storage, SC_storage, offset*nSample+tid, sample_x, sample_y, sample_w, nSample, nsig, uniform_retina, virtual_LGN);
    } else {
        assert(PM == 1);
        if (tid == 0) {
            Zip_spatial spat;
            spat.load(spatial, lid);
            shared_spat[0] = spat.x; // polar
            shared_spat[1] = spat.y; // ecc
            reduced[4] = spat.rx; // in ecc
            reduced[5] = spat.ry;
            Float c_orient = spat.orient;
            shared_spat[2] = cosine(c_orient);
            shared_spat[3] = sine(c_orient);
            shared_spat[8] = spat.k;

            spat.load(spatial, nLGN + lid);
            // visual-center-vector from center to surround 
            Float x = spat.y*cosine(spat.x) - shared_spat[1]*cosine(shared_spat[0]); // cx -> sx
            Float y = spat.y*sine(spat.x) - shared_spat[1]*sine(shared_spat[0]); // cy -> sy

            // orientation diff between center and surround
            Float d_orient = c_orient - spat.orient;
            if (abs(d_orient) > M_PI/2) {
                d_orient -= copysign(M_PI, d_orient);
            } 
            reduced[0] = cosine(d_orient);
            reduced[1] = sine(d_orient);
            Float rs = spat.rx > spat.ry ? spat.rx: spat.ry;
            Float span = nsig * rs + square_root(x*x + y*y);
            Float ds = 2*span/blockDim.x;

            shared_spat[4] = span; 
            shared_spat[5] = ds;   
            shared_spat[6] = spat.rx;
            shared_spat[7] = spat.ry;

            reduced[2] = x;
            reduced[3] = y;
        }
        __syncthreads();
        Float span = shared_spat[4];
        Float ds = shared_spat[5];
        Float centerPolar = shared_spat[0];
        Float centerEcc = shared_spat[1];
        Float coso = reduced[0];
        Float sino = reduced[1];
        Float cx,  cy, areal_weight;
        if (nsig > 0) {
            cx = (threadIdx.x + 0.5)*ds - span;
            cy = (threadIdx.y + 0.5)*ds - span;
            areal_weight = 1;
        } else {
            int idx = threadIdx.y * WARP_SIZE + threadIdx.x;
            cx = -sample_x[idx]*span/nsig;
            cy = -sample_y[idx]*span/nsig;
            areal_weight = sample_w[idx]*span*span/(nsig*nsig);
        }

        Float c_rx = reduced[4];
        Float c_ry = reduced[5];
        Float tx = cx - reduced[2]; 
        Float ty = cy - reduced[3];

        Float s_rx = shared_spat[6];
        Float s_ry = shared_spat[7];
        Float k = shared_spat[8];
        // align original center and surround sample points
        Float sx = tx*coso - ty*sino;
        Float sy = tx*sino + ty*coso;
        Float cs_weight = spatialKernel(cx, cy, c_rx, c_ry) * areal_weight;
	    block_reduce<Float>(reduced, cs_weight);
        cs_weight /= reduced[0];

        Float ss_weight = spatialKernel(sx, sy, s_rx, s_ry) * areal_weight;
	    block_reduce<Float>(reduced, ss_weight);
        ss_weight /= reduced[0];

        Float spatialWeight = copysign(cs_weight,k) - copysign(ss_weight, k); 

        coso = shared_spat[2];
        sino = shared_spat[3];

        tid = threadIdx.y*blockDim.x + threadIdx.x;
        block_reduce<Float>(reduced, abs(spatialWeight));
        SW_storage[nSample*blockIdx.x + tid] = spatialWeight/reduced[0];
        if (tid == 0) {
            max_convol[lid] = abs(k)*reduced[0]; 
            if (virtual_LGN) {
                float x, y;
                float cosp_d = cosine(centerPolar)*normViewDistance;
                float sinp_d = sine(centerPolar)*normViewDistance;
                if (!uniform_retina) {
                    x = LR_x0 + tangent(centerEcc) * cosp_d;
                    y = LR_y0 + tangent(centerEcc) * sinp_d;
                } else {
                    x = LR_x0 + centerEcc * cosp_d;
                    y = LR_y0 + centerEcc * sinp_d;
                }
                center[blockIdx.x] = x;
		    	center[gridDim.x + blockIdx.x] = y;
                if (!uniform_retina) {
                    if (LR) {
                        if (x < 0 || x > 0.5) {
                            printf("x = %1.15e\n", x);
                            assert(x>=0);
                            assert(x<=0.5);
                        }
                    } else {
                        if (x < 0.5 || x > 1) {
                            printf("x = %1.15e\n", x);
                            assert(x>=0.5);
                            assert(x<=1);
                        }
                    }
                    if (y<0 || y>1) {
                        printf("y = %1.15e\n", y);
                        assert(y>=0);
                        assert(y<=1);
                    }
                } else {
                    if (!virtual_LGN) {
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
                    }
                }
            }
        }
        float x, y;
        if (!uniform_retina) {
	        Float cosp, sinp; 
            Float cosEcc, sinEcc;
	        orthPhiRotate3D(centerPolar, centerEcc + cy, cx, cosp, sinp, cosEcc, sinEcc);

            Float tanEcc;
	        axisRotate3D(centerPolar, centerEcc, coso, sino, cosp, sinp, cosEcc, sinEcc, tanEcc);

            retina_to_plane(cosp, sinp, tanEcc, x, y, normViewDistance, LR_x0, LR_y0);
            // DEBUG visual field and stimulus field not matching
            if (!virtual_LGN) {
                if (LR) {
                    if (x < 0 || x > 0.5) {
                        printf("x = %1.15e\n", x);
                        assert(x>=0);
                        assert(x<=0.5);
                    }
                } else {
                    if (x < 0.5 || x > 1) {
                        printf("x = %1.15e\n", x);
                        assert(x>=0.5);
                        assert(x<=1);
                    }
                }
                if (y<0 || y>1) {
                    printf("y = %1.15e\n", y);
                    assert(y>=0);
                    assert(y<=1);
                }
            }
            //
        } else {
            x = LR_x0 + (centerEcc * cosine(centerPolar) + cx * coso - cy * sino) * normViewDistance;
            y = LR_y0 + (centerEcc * sine(centerPolar) + cx * sino + cy * coso) * normViewDistance;
            // DEBUG visual field and stimulus field not matching
            if (!virtual_LGN) {
                if (x<0 || x>1) {
                    printf("x = %.3e, %.3f + (%.3e + %.3e) * %.3e\n", x, LR_x0, centerEcc * cosine(centerPolar), cx * coso - cy * sino, normViewDistance);
                    assert(x>=0);
                    assert(x<=1);
                }
                if (y<0 || y>1) {
                    printf("y = %.3e, %.3f + (%.3e + %.3e) * %.3e\n", x, LR_y0, centerEcc * sine(centerPolar), cx * sino + cy * coso, normViewDistance);
                    assert(y>=0);
                    assert(y<=1);
                }
            }
            //
        }
        // store coords for retrieve data from texture
        PosInt storeID = blockIdx.x*nSample + tid;
        SC_storage[storeID] = x; // x
        SC_storage[gridDim.x*nSample + storeID] = y; // y
    }
}

// must be launched by 1024
__launch_bounds__(1024,1)
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
    __shared__ Float reduced[WARP_SIZE];
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

__launch_bounds__(1024,1)
__global__
void parvo_maxConvol_sep(Spatial_component &spatial,
                   Float* __restrict__ TW_storage,
                   Float* __restrict__ covariant,
                   Float* __restrict__ max_convol,
                   Float* __restrict__ sample_x,
                   Float* __restrict__ sample_y,
                   Float* __restrict__ sample_w,
                   Size nSample1D, Size nParvo_L, Size nMagno_L, Size nLGN, SmallSize nKernelSample, Float kernelSampleDt, Float nsig)
{
    Size nSample = nSample1D * nSample1D;
    __shared__ Float reduced[WARP_SIZE];
    PosInt id = blockIdx.x;
    if (id >= nParvo_L) {
        id += nMagno_L;
    }
    Float rxs = spatial.rx[nLGN+id];
    Float rys = spatial.ry[nLGN+id];

    Float xs = spatial.x[nLGN+id];
    Float ys = spatial.y[nLGN+id];

    Float wSpanS = nsig * rxs / SQRT2;
    Float hSpanS = nsig * rys / SQRT2;

    Float rxc = spatial.rx[id];
    Float ryc = spatial.ry[id];


    Float xc = spatial.x[id];
    Float yc = spatial.y[id];

    Float wSpanC = nsig * rxc / SQRT2;
    Float hSpanC = nsig * ryc / SQRT2;

    Float dx = yc*cosine(xc)-ys*cosine(xs);
    Float dy = yc*sine(xc)-ys*sine(xs);

    Float cov = covariant[id];
    Float s_orient = spatial.orient[nLGN+id];
    Float c_orient = spatial.orient[id];

    Float ks = spatial.k[nLGN + id];
    Float kc = spatial.k[id];

    Float d_orient = c_orient - s_orient;
    if (abs(d_orient) > M_PI/2) {
        d_orient -= copysign(M_PI, d_orient);
    } 

    Float dwC = 2*wSpanC/nSample1D;
    Float dhC = 2*hSpanC/nSample1D;
    Float dwS = 2*wSpanS/nSample1D;
    Float dhS = 2*hSpanS/nSample1D;

	Float cosd = cosine(d_orient); 
	Float sind = sine(d_orient);

    Size nPatch = nSample/blockDim.x;
    assert(nSample%blockDim.x == 0);

    Float convol;
    if (threadIdx.x == 0) {
        convol = 0.0;
    }
    #pragma unroll 16
    for (PosInt it = 0; it<nKernelSample; it++) {
        Float tc = TW_storage[blockIdx.x*2*nKernelSample + it];
        Float ts = TW_storage[(blockIdx.x*2 + 1)*nKernelSample + it];

        // center parts
        #pragma unroll 9
        for (PosInt iPatch = 0; iPatch < nPatch; iPatch++) {
            PosInt pid = iPatch*blockDim.x + threadIdx.x;
            Float x, y, areal_weight;
            if (nsig > 0) {
                PosInt w = pid % nSample1D;
                PosInt h = pid / nSample1D;
                x = (w + 0.5)*dwC - wSpanC; 
                y = (h + 0.5)*dhC - hSpanC;
                areal_weight = 1;
            } else {
                x = -sample_x[pid]*wSpanC/nsig; 
                y = -sample_y[pid]*hSpanC/nsig;
                areal_weight = sample_w[pid]*wSpanC*hSpanC/(nsig*nsig);
            }

            // origin at the center of the surround RF
            Float local_sw = spatialKernel(x, y, rxc, ryc) * areal_weight;
            Float swC = local_sw * kc/(M_PI*rxs*rys) * dwC*dhC * tc;
            Float sx = cosd*(dx + x) - sind*(dy + y);
            Float sy = sind*(dx + x) + cosd*(dy + y);
            local_sw = spatialKernel(sx, sy, rxs, rys) * areal_weight;
            Float swS = local_sw * ks/(M_PI*rxs*rys) * dwS*dhS * ts;

            Float local_decide;
            if (abs(swC) > abs(swS)) { // assumed contrast must have the same sign as local_center
                local_decide = abs(swC) + swS * copysign(cov, swC);
            } else { // assumed contrast must have the same sign as local_surround
                local_decide = abs(swS) + swC * copysign(cov, swS);
            }
	        block_reduce<Float>(reduced, local_decide);
            if (threadIdx.x == 0) {
                convol += reduced[0];
            }
        }
        // surround parts
        #pragma unroll 9
        for (PosInt iPatch = 0; iPatch < nPatch; iPatch++) {
            PosInt pid = iPatch*blockDim.x + threadIdx.x;
            Float x, y, areal_weight;
            if (nsig > 0) {
                PosInt w = pid % nSample1D;
                PosInt h = pid / nSample1D;
                x = (w + 0.5)*dwS - wSpanS; 
                y = (h + 0.5)*dhS - hSpanS;
                areal_weight = 1;
            } else {
                x = -sample_x[pid]*wSpanS/nsig; 
                y = -sample_y[pid]*hSpanS/nsig;
                areal_weight = sample_w[pid]*wSpanS*hSpanS/(nsig*nsig);
            }

            // origin at the center of the surround RF
            Float local_sw = spatialKernel(x, y, rxs, rys) * areal_weight;
            Float swS = local_sw * ks/(M_PI*rxs*rys) * dwS*dhS * ts;
            Float cx = cosd*(x - dx) - sind*(y - dy);
            Float cy = sind*(x - dx) + cosd*(y - dy);
            local_sw = spatialKernel(cx, cy, rxc, ryc) * areal_weight;
            Float swC = local_sw * kc/(M_PI*rxc*ryc) * dwC*dhC * tc;

            Float local_decide;
            if (abs(swC) > abs(swS)) { // assumed contrast must have the same sign as local_center
                local_decide = abs(swC) + swS * copysign(cov, swC);
            } else { // assumed contrast must have the same sign as local_surround
                local_decide = abs(swS) + swC * copysign(cov, swS);
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
__launch_bounds__(1024,1)
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
		cudaTextureObject_t L_retinaInput,
		cudaTextureObject_t M_retinaInput,
		cudaTextureObject_t S_retinaInput,
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
    __shared__ Float reducedC[WARP_SIZE];
    __shared__ Float reducedS[WARP_SIZE];
    extern __shared__ Float nSampleShared[];

    // weights are stored in shapes of (nLGN, nType, weight)
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    Size nSample = blockDim.x * blockDim.y;
	//Float *nSampleTemp = nSampleShared + nSample;
	
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
    float y0C = SC_storage[gridDim.x*2*nSample + storeIDC];
    float x0S = SC_storage[storeIDS];
    float y0S = SC_storage[gridDim.x*2*nSample + storeIDS];
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
    
    SmallSize typeS = coneType[id+nLGN];
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
		} 
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
        Size nFrame = (itFrames + ntPerFrame-1) / ntPerFrame;
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
			Float local_I[2];
            local_I[0] = get_intensity(typeC, x0S, y0S, (currentFrame + iFrame) % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
            local_I[1] = get_intensity(typeS, x0S, y0S, (currentFrame + iFrame) % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
			//printf("frame %u - (%u,%u), c = %.3e, s = %.3e\n", iFrame, blockIdx.x, threadIdx.x, local_I[0], local_I[1]);
            block_reduce2<Float>(reducedC, reducedS, local_I);
            if (tid == 0) {
                // __shared__ to (register/local) to __shared__
                nSampleShared[iFrame] = reducedC[0]/nSample;  // shared memory now used for spatial mean luminance, F_i
                nSampleShared[nFrame + iFrame] = reducedS[0]/nSample;  // shared memory now used for spatial mean luminance, F_i
            }
			if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1) {
            	Float local_I = get_intensity(3, x0S, y0S, (currentFrame + iFrame) % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
            	block_reduce<Float>(reducedS, local_I);
				if (tid == 0) {
                    luminance[id] = reducedS[0]/nSample;
				}
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
        Float tempFiltered[2];
        // initialize with the first frame in the patch
        PosInt it;
        if (iPatch == 0) {
            it = iKernelSampleT0;
        } else {
            it = kernelSampleInterval;
		}
		//PosInt iFrame = 0;
        Int preFrame = old_currentFrame-1;
        for (PosInt iSample = 0; iSample < nActive; iSample++) {
            PosInt frameNow = old_currentFrame + (static_cast<Int>(it*denorm + iFramePhase)-1)/static_cast<Int>(ntPerFrame); //time starts with old_currentFrame, frame starts with currentFrame, for sample points right on the edge of two frames, sample the previous frame
            if (frameNow > preFrame) { // advance frame
                //Load mean luminance from shared memory first
                PosInt iFrame = frameNow - currentFrame;
                Float meanC = nSampleShared[iFrame];
                Float meanS = nSampleShared[nFrame + iFrame];
                preFrame = frameNow;
                Float local_contrast;
				Float local_I;
                // center
                local_I = get_intensity(typeC, x0C, y0C, frameNow % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
				/*
				if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1) {
                	if (iPatch == 0 && iFrame == nFrame-1) {
						if (blockIdx.x == 0) {
							nSampleTemp[tid] = local_I;
							__syncthreads();
							if (tid == 0) {
								Float min = 2;
								Float max = -2;
								for (PosInt im = 0; im<nSample; im++) {
									if (min > nSampleTemp[im]) {
										min = nSampleTemp[im];
									}
									if (max < nSampleTemp[im]) {
										max = nSampleTemp[im];
									}
								}
								printf("type%d-frame%d: local_I:[%e, %e, %e]->[%e,%e]\n", typeC, iFrame, min, meanC, max, min/meanC-1.0, max/meanC-1.0);
							}
							__syncthreads();
						}
					}
				} */

                if (meanC > 0) {
                    local_contrast = local_I/meanC - 1.0;
                } else {
                    local_contrast = local_I;
                }
                if (abs(local_contrast) > 1.0) {
                    local_contrast = copyms(1.0, local_contrast); // copyms is copysign(value, sign);
                }

				Float filtered[2];
                filtered[0] = spatialWeight*local_contrast;
				
                // surround 
                local_I = get_intensity(typeS, x0S, y0S, frameNow % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
                if (meanS > 0) {
                    local_contrast = local_I/meanS - 1.0;
                } else {
                    local_contrast = local_I;
                }
                if (abs(local_contrast) > 1.0) {
                    local_contrast = copyms(1.0, local_contrast); // copyms is copysign(value, sign);
                }

                filtered[1] = spatialWeight*local_contrast;

                block_reduce2<Float>(reducedC, reducedS, filtered);
                if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1 && tid == 0) {
                    contrast[id] = reducedC[0];
                    contrast[nLGN+id] = reducedS[0];
                }

				/*
				if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1) {
                	if (iPatch == 0 && iFrame == nFrame-1) {
						if (blockIdx.x == 0) {
							nSampleTemp[tid] = local_I;
							__syncthreads();
							if (tid == 0) {
								Float min = 2;
								Float max = -2;
								for (PosInt im = 0; im<nSample; im++) {
									if (min > nSampleTemp[im]) {
										min = nSampleTemp[im];
									}
									if (max < nSampleTemp[im]) {
										max = nSampleTemp[im];
									}
								}
								printf("type%d-frame%d: local_I:[%e, %e, %e]->[%e,%e]\n", typeS, iFrame, min, meanS, max, min/meanS-1.0, max/meanS-1.0);
							}
							__syncthreads();
						}
					}
				} */
				//__syncthreads();
            }
            if (tid == iSample) {
                tempFiltered[0] = reducedC[0]*temporalWeightC;
                tempFiltered[1] = reducedS[0]*temporalWeightS; // spatially contrast convolve with temporalWeight 
				/* DEBUG
					if (blockIdx.x == 52583) {
						printf("%u#%u, wspC*tw: %e*%e = %e\n", iPatch, tid, reducedC[0], temporalWeightC, tempFiltered[0]);
					}
				*/	
				//
            }
			//__syncthreads();
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
        if (tid >= nActive) {
            tempFiltered[0] = 0.0;
            tempFiltered[1] = 0.0;
        }
        //6. reduce sum with temporal weights: p in time
        //block_reduce<Float>(reducedS, tempFiltered[1]);
        //7. add to convol: s
        //if (tid == 0) {
        //    convolS += reducedS[0];
        //}
		/*DEBUG
			__syncthreads();
			reducedS[0] = 0.0;
			if (blockIdx.x == 52583) {
				if (tid == 0) {
					printf("%u possible nonzeros, tempFiltered[0] = ", nActive);
				}
				for (Size i = 0; i < nSample; i++ ) {
					if (tid == i) {
						printf("%e, ", tempFiltered[0]);
						reducedS[0] += tempFiltered[0];
					}
					__syncthreads();
				}
				if (tid == 0) {
					printf("\n");
				}
			}
		*/
        block_reduce2<Float>(reducedC, reducedS, tempFiltered);
        if (tid == 0) {
			//Float old_convol = convolC;
            convolC += reducedC[0];
            convolS += reducedS[0];
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

__launch_bounds__(1024,1)
__global__ 
void LGN_convol_magno(
        Float* __restrict__ luminance,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ current_convol,
        Float* __restrict__ contrast,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
		cudaTextureObject_t L_retinaInput,
		cudaTextureObject_t M_retinaInput,
		cudaTextureObject_t S_retinaInput,
		Size nParvo_L, Size nMagno_L, Size nParvo_R,
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
    __shared__ Float reduced[WARP_SIZE];
    extern __shared__ Float nSampleShared[];

    // weights are stored in shapes of (nLGN, nType, weight)
    Size tid = threadIdx.y*blockDim.x + threadIdx.x;
    Size nSample = blockDim.x * blockDim.y;
	
    //TODO: Itau may take different value for different cone type
    // convolve center and update luminance
    Float convol;
	if (tid == 0) {
		convol = 0.0;
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

    Size id = nParvo_L + blockIdx.x;
    if (blockIdx.x >= nMagno_L) {
        id += nParvo_R;
    }

    // TODO store and read storage contiguously
    Size storeID = blockIdx.x*nSample + tid;

    // coord on the stimulus plane
    float x0 = SC_storage[storeID];
    float y0 = SC_storage[gridDim.x*nSample + storeID];
    assert(x0 <= 1.0);
    assert(y0 <= 1.0);
    assert(x0 >= 0.0);
    assert(y0 >= 0.0);

    Float k;
    if (tid == 0) {
        k = spatial.k[id];
    }
    /* Light adaptation process:
        tau*dI/dt = -I + F(t);
        F(t) = piecewise F(t_{i}), for t in [t_{i}, t_{i+1}), t_{i} is the onset time of the i-th frame
        Float lastDecayIn = luminance[lid]; // = [sum_i2n(F_i[exp(-t_{i+1}/Itau) - exp(-t_{i}/Itau)]) - F_{n+1}*exp(-t_{i+1}/Itau)]*exp(-t/Itau)
        Float F_1 = lastF[lid]; // = F_{n+1}
        //F_i is the mean of all sampled pixel value of the ith frame in the LGN's RF.
    */
    
    SmallSize type = coneType[id];

    Float spatialWeight = SW_storage[storeID];
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
        Float temporalWeight;
		if (tid < nActive) {
			temporalWeight = TW_storage[blockIdx.x*nKernelSample + iPatch*nSample + tid];
		} 
        //2. Find new frames - n, usually just 1
        PosInt old_currentFrame = currentFrame;
        PosInt itFrames, T0;
        if (iPatch == 0) {
            T0 = iKernelSampleT0; // iKernelSampleT0 = 0 or kernelSampleInterval/2
        } else {
            T0 = kernelSampleInterval; // iKernelSampleT0 = 0 or kernelSampleInterval/2
		}
        // number of frames in one patch
        itFrames = (T0 + (nActive-1)*kernelSampleInterval)*denorm + iFramePhase; // iKernelSampleT0 = 0 or kernelSampleInterval/2
        Size nFrame = itFrames / ntPerFrame + 1;
        // check if the first samplePoint bypass the currentFrame
        if (T0*denorm + iFramePhase >= ntPerFrame) {
            currentFrame = old_currentFrame+1;
            nFrame--;
        }
        //3. For all the new frames
        for (Size iFrame = 0; iFrame < nFrame; iFrame++) {
            //Get F_i by reduce - p: in space
            Float local_I = get_intensity(3, x0, y0, (currentFrame + iFrame) % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
            block_reduce<Float>(reduced, local_I);
            if (tid == 0) {
                // __shared__ to (register/local) to __shared__
				Float lum = reduced[0]/nSample;
                nSampleShared[iFrame] = lum;  // shared memory now used for spatial mean luminance, F_i
                luminance[id] = lum;
            }
        }
        __syncthreads();
        //5. For each sample point in time: 
        //  Get weighted contrast sum from local_I(ntensity) and mean_I(ntensity): p in space 
        Float tempFiltered;
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
            Float filtered, local_I, local_contrast, mean_I;
            if (frameNow > preFrame) { // advance frame
                //Load mean luminance from shared memory first
                PosInt iFrame = frameNow - currentFrame;
                //Float mean_I = nSampleShared[iFrame];
                mean_I = nSampleShared[iFrame];
                preFrame = frameNow;
				
                //Float local_I = get_intensity(type, x0, y0, frameNow % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
                //Float local_contrast;
                local_I = get_intensity(type, x0, y0, frameNow % maxFrame, L_retinaInput, M_retinaInput, S_retinaInput);
                if (mean_I > 0) {
                    local_contrast = local_I/mean_I - 1.0;
                } else {
                    local_contrast = local_I;
                }
                if (abs(local_contrast) > 1.0) {
                    local_contrast = copyms(1.0, local_contrast); // copyms is copysign(value, sign);
                }
                //Float filtered = spatialWeight*local_contrast;
                filtered = spatialWeight*local_contrast;
				// DEBUG
                if (iPatch == nPatch) {
                }
				//
                block_reduce<Float>(reduced, filtered);
                if (saveOutputB4V1 && iPatch == nPatch && iFrame == nFrame-1 && tid ==0) {
                    contrast[id] = reduced[0];
                    luminance[id] = mean_I;
                }
		        //__syncthreads();
            }
            if (tid == iSample) {
                tempFiltered = reduced[0]*temporalWeight; // spatially contrast convolve with temporalWeight 
            }
            /*DEBUG
			if (blockIdx.x == 138 || blockIdx.x == 394) {
                if (iSample == 103) {
				    if (tid == nSample/2 + blockDim.x/2-1) {
				    	printf("#%u: local_I: %e, mean_I: %e, contrast: %e, spatialWeight: %e, filtered: %e\n", blockIdx.x, local_I, mean_I, local_contrast, spatialWeight, filtered);
				    }
                    if (tid == iSample) {
			            printf("#%u: sw contrast: %e, tw:%e, convol: %e\n", blockIdx.x, reduced[0], temporalWeight, tempFiltered);
                    }
                }
            }*/
		    //__syncthreads();
            // advance time
            it += kernelSampleInterval;
        }
		//__syncthreads();
        if (tid >= nActive) {
            tempFiltered = 0.0;
        }
        //6. reduce sum with temporal weights: p in time
        block_reduce<Float>(reduced, tempFiltered);
        //7. add to convol: s
        if (tid == 0) {
            convol += reduced[0];
        }
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
        current_convol[id] = convol * abs(k);
        //convol[blockIdx.x] = (convolC + convolS)*kernelSampleInterval*dt;
		/*DEBUG
			if (blockIdx.x == 138 || blockIdx.x == 394) {
				printf("#%u: total convol: %e\n", blockIdx.x, convol);
                assert(!isnan(convol));
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
                curandStateMRG32k3a *state,
				PosInt id, 
				Float tRef = 0) 
{
	nsp = 0;
	if (true) {
    	// ith spike, jth dt
    	// t_{i+1}*r_{j+1} + (T_{j}-t_{i})*r_{j} = -log(rand);
    	Float rT = dt*rate;
    	Float next_rT = lastNegLogRand - leftTimeRate; // tsp = next_rt/rate
		Float old_rT;
    	if (next_rT > rT) { // spike time is larger than dt.
    	    leftTimeRate += rT;
    	    return 0;
    	} else do { // at least one spike during current time step
			Float rand = uniform(state); 
    	    //lastNegLogRand = -logarithm(rand);
    	    lastNegLogRand = fmaxf(-logarithm(rand),tRef*rate);
			old_rT = next_rT;
    	    next_rT += lastNegLogRand;
    	    nsp++;
    	} while (next_rT <= rT);
    	//next_rT -= lastNegLogRand; // retract the tentative spike
    	leftTimeRate = rT - old_rT; // just use old_rT
    	if (nsp > 0) return old_rT/(rate*nsp); // mean tsp not yet normalized by dt
    	else return 0; 
	} else {
		Float rand = uniform(state);
		Float tsp = 0;
		if (rand < rate*dt) {
			nsp = 1;
			if (leftTimeRate > dt) {
				tsp = rand*dt;
			} else {
				tsp = (1-rand)*dt;
			}
			leftTimeRate = tsp;
		} 
		leftTimeRate += dt*rate;
		return tsp;
	}	
}

//template<int ntimes>
__launch_bounds__(1024,1)
__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ current_convol,
        Float convolRatio,
        Float* __restrict__ LGN_fr,
        Float* __restrict__ LGN_sInfo,
        int* __restrict__ sx,
        int* __restrict__ sy,
        Float* __restrict__ leftTimeRate,
        Float* __restrict__ lastNegLogRand,
		curandStateMRG32k3a* __restrict__ state,
		InputType_t* __restrict__ LGN_type,
		Float* __restrict__ switch_value,
        InputActivation typeStatus,
        Float* __restrict__ lVar,
		cudaSurfaceObject_t LGNspikeSurface,
        Float frRatio, int varSlot, LearnVarShapeFF_E_pre lE, LearnVarShapeFF_I_pre lI, Size nFF, Float dt, int learning, bool learnData_FF, bool LGN_switch, bool getLGN_sp, bool virtual_LGN, int switchNow)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    bool engaging = id<nLGN;
    unsigned int MASK = __ballot_sync(FULL_MASK, static_cast<int>(engaging));
    Float var[MAX_NLEARNTYPE_FF];
	Size nsp;
	Float sInfo, tsp;
 	int x, y;
    if (engaging) {
        Float C50, K, A, B;
        // load in sequence
        Float current = current_convol[id] * convolRatio;
		Float max = max_convol[id];
        Float lTR = leftTimeRate[id];
        Float lNL = lastNegLogRand[id];
        InputType_t type;
		Float local_switch;
        if (switchNow > 0) {
            type = LGN_type[id];
		}
        if (virtual_LGN) {
            current *= max;
        }
        //int x = sx[id];
        //int y = sy[id];
        x = sx[id];
        y = sy[id];
        curandStateMRG32k3a local_state = state[id];
        //PosInt type = static_cast<PosInt>(LGN_type[id]);
		if (LGN_switch) {
			if (switchNow > 0) {
				switch (switchNow) {
					case 1:
						if (uniform(&local_state) > typeStatus.actPercent[type]) {
							local_switch = 0;
							switch_value[id] = 0;
						} else {
							local_switch = 1;
							switch_value[id] = 1.0;
						}
						break;
					case 2:
						switch_value[id] = typeStatus.actPercent[type];
						local_switch = typeStatus.actPercent[type];
						break;
				}
        	} else {
				local_switch = switch_value[id];
			}
			//if (id == 0) {
			//	printf("switchNow = %i, switch value = %f, actPercent[%u] = %f\n", switchNow, local_switch, type, typeStatus.actPercent[type]); 
			//}
		}

        // get firing rate
        logistic.load_first(id, C50, K, A, B);
        // Float convol = current; // use with DEBUG
        if (LGN_switch) {
			current *= local_switch;
		}
        if (current < 0) {
            current = 0;
        }

        __syncwarp(MASK);
        Float fr = frRatio * max * transform(C50, K, A, B, current/max);
		// DEBUG
		/* DEBUG
        if (fr < 0) {
            printf("convol = %f, fr = %f, K=%f, A= %f, B= %f, C50 =%f, max = %f\n", convol, fr, K, A, B, C50, max);
            assert(fr >= 0);
        }*/
        LGN_fr[id] = fr;
        //Float var[MAX_NLEARNTYPE_FF];
        //if (isnan(max) || max == 0 || isnan(current) || isnan(fr)) {
        //    printf("current/max: %1.3e/%1.3e = %1.3e == %1.3e\n", current, max, current/max, fr);
        //}
        if (learning < 4) {
            #pragma unroll (sum_nLearnTypeFF)
            for (int i=0; i<nFF; i++) {
				float varf;
                surf2DLayeredread(&varf, LGNspikeSurface, 4*x, y, 1+(3*i+varSlot)); // varSlot already 'mod'ed, learnVar at the start of this tstep
				var[i] = static_cast<Float>(varf);
            }
        }

        //Size nsp;
        //Float tsp = get_spike(nsp, lTR, lNL, dt, fr/1000.0, &local_state);
        //Float sInfo = nsp + tsp/dt; // must be float, integer part = #spikes decimals: mean tsp normalized by dt
        tsp = get_spike(nsp, lTR, lNL, dt, fr/1000.0, &local_state, id);
        sInfo = nsp + tsp/dt; // must be float, integer part = #spikes decimals: mean tsp normalized by dt
		/* debug for snapshot
			if (id == 0) {
				printf("LGN: fr = %f, lTR = %f, lNL = %f, rand = %f\n", fr, lTR, lNL, uniform(&local_state));
			}
		 */
        if (sInfo > 0) {
            lastNegLogRand[id] = lNL;
            //if (id == 37054 || id == 37223 || id == 37647) {
            //    printf("LGN# %u fired\n");
            //}
            //printf("LGN fired, ");
			//if (id == 24277) {
			//	printf("\nLGN (%i, %i) fired at %f\n", x, y, tsp);
			//}
        }
        leftTimeRate[id] = lTR;
        state[id] = local_state;
        if (learnData_FF || getLGN_sp) LGN_sInfo[id] = sInfo;
		//printf("%u: ltr = %lf, lnl = %lf, nsp = %f, sInfo = %lf\n", id, lTR, lNL, nsp, LGN_sInfo[id]);
        // write to surface memory 
		float sInfof = static_cast<float>(sInfo);
        surf2DLayeredwrite(sInfof, LGNspikeSurface, 4*x, y, 0);
	}
	__syncthreads();
    if (engaging) {
        if (learning && learning < 4) {
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
					float varf = static_cast<float>(var[i]);
                    surf2DLayeredwrite(varf, LGNspikeSurface, 4*x, y, 1+(3*i+2)); // store learnVar right before the LGN spike in the current time step, always update at the third slot, 
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
				float varf = static_cast<float>(var[i]);
                surf2DLayeredwrite(varf, LGNspikeSurface, 4*x, y, 1+(3*i+(varSlot+1)%2)); // update at next slot, learnVar at the end of the current step
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
