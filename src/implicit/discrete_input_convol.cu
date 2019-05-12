#include "discrete_input_convol.h"

// TO-DO
// update_kernel
// update_static_nonlinearity

// TO-DO 
// block-processing without input-output delay
// http://www.cs.ust.hk/mjg_lib/bibs/DPSu/DPSu.Files/Ga95.PDF
// block processing file:///C:/Users/gueux/Desktop/FFTConvolution.pdf
// direct form:
__inline__ __device__ _float AexpTau(_float a, _float tau) {
    return a * expp(-tau);
}

__inline__ __device__ _float spatialProduct(_float x, _float y, _float contrast, _float k, _float rx, _float ry) {
    return k * expp(-power(x/rx, 2) - power(y/ry, 2)) * contrast;
}

__inline__ __device__ _float temporalKernel(_float tau, LGN_subregion subr) {

    _float facRatio = 1.0f;
    for (unsigned int i=subr.nR; i<subr.nD; i++) facRatio*=i;
    _float A = facRatio;
    for (unsigned int i=1; i<subr.nR; i++) A*=i;

    _float tau1 = tau/subr.tauR;
    _float tau2 = tau/subr.tauD;
    _float A1 = facRatio * power(tau1, subr.nR-1)/subr.tauR;
    _float A2 = power(tau2, subr.nD-1)/subr.tauD;

    _float tp = (subr.ratio * AexpTau(A1, tau1) - AexpTau(A2, tau2))/A;
    return tp;
}

__device__ __inline__ _float get_contrast(unsigned int coneType, float x, float y, unsigned int iFrame) {
    _float contrast;
    switch (coneType) {
        case 0:
            contrast = tex2DLayered(L_retinaConSig, x, y, iFrame);
            break;
        case 1:
            contrast = tex2DLayered(M_retinaConSig, x, y, iFrame);
            break;
        case 2:
            contrast = tex2DLayered(S_retinaConSig, x, y, iFrame);
            break;
        case 3:
            contrast = tex2DLayered(L_retinaConSig, x, y, iFrame) + tex2DLayered(M_retinaConSig, x, y, iFrame) + tex2DLayered(S_retinaConSig, x, y, iFrame);
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
Per frame
    max spatial sample size: 32 x 32
    number of temporal kernel evaluation <= spatial sample size
*/
__global__ void LGN_convol(_float* __restrict__ LGNfr,
                           LGN_parameter pLGN, // consider pointer
                           unsigned int nFrame, _float framePhase, _float tPerFrame, unsigned int frame0, _float kernelSampleDt, _float tau, unsigned int nsig, unsigned int npixel_1D) {

    __shared__ _float linearResponse[warpSize];
    // consider store LGN_subregion, facRatio to __shared__

    unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
    unsigned int id = blockIdx.x;
    _float sqrt2 = square_root(2.0);
    _float convol;
    if (tid == 0) {
        convol = 0.0f; 
    }
        /*
         
       framePhase                    tPerFrame - framePhase
time     | ^ |-------> tPerFrame <------|      ^      |
frame: prev curr                   next
sample:  1        2        3        4        5        6
         |...:----|--------|--------|---:----|--------|
dt:      0   3    8        16       24       32       40
tau = 40*dt
          
        */

    // center
    LGN_subregion center(pLGN.center, id);
    unsigned int type = pLGN.centerType[id];

    float xhspan = nsig * center.rx / sqrt2;
    float dx = 2*xhspan/npixel_1D;

    float yhspan = nsig * center.ry / sqrt2;
    float dy = 2*yhspan/npixel_1D;

    float x = threadIdx.x*dx - xhspan;
    float y = threadIdx.y*dy - yhspan;
    float x0 = center.x + x;
    float y0 = center.y + y;
    
    _float t = 0;
    for (unsigned int iFrame=0; iFrame<nFrame; iFrame++) {
        // frame phase vs. dt
        _float contrast = static_cast<_float>(get_contrast(type, x0, y0, (frame0 + iFrame) % nFrame));
        _float filtered = spatialProduct(x, y, contrast, center.k, center.rx, center.ry);
        __syncwarp();
        block_reduce<_float>(linearResponse, filtered);

        _float blocked_t;
        unsigned int nSample = 1;
        // no branching
        if (iFrame==0) {
            blocked_t = framePhase;
        } else {
            if (iFrame == nFrame - 1) {
                blocked_t = tau - t;
            } else {
                blocked_t = framePhase + iFrame*tPerFrame - t;
            }
        }
        nSample += static_cast<unsigned int>(floor(blocked_t/kernelSampleDt));

        if (tid == 0) { // acquire spatial filtered input
            contrast = linearResponse[0]*dx*dy; 
        }
        __syncthreads();

        _float thread_t;

        if (tid < nSample) {
            thread_t = t + tid*kernelSampleDt;
            filtered = temporalKernel(thread_t, center);
        } else {
            filtered = 0.0f;
        }
        __syncwarp();
        block_reduce<_float>(linearResponse, filtered);

        if (tid == 0) {
            convol += linearResponse[0]*kernelSampleDt*contrast;
        }

        t += nSample*kernelSampleDt;
    }
    
    // surround
    LGN_subregion surround(pLGN.surround, id);
    type = pLGN.surroundType[id];

    xhspan = nsig * surround.rx / sqrt2;
    dx = 2*xhspan/npixel_1D;

    yhspan = nsig * surround.ry / sqrt2;
    dy = 2*yhspan/npixel_1D;

    x = threadIdx.x*dx - xhspan;
    y = threadIdx.y*dy - yhspan;
    x0 = surround.x + x;
    y0 = surround.y + y;
    
    t = 0;
    for (unsigned int iFrame=0; iFrame<nFrame; iFrame++) {
        // frame phase vs. dt
        _float contrast = static_cast<_float>(get_contrast(type, x0, y0, (frame0 + iFrame) % nFrame));
        _float filtered = spatialProduct(x, y, contrast, surround.k, surround.rx, surround.ry);
        __syncwarp();
        block_reduce<_float>(linearResponse, filtered);
         
        _float blocked_t;
        unsigned int nSample = 1;
        // no branching
        if (iFrame==0) {
            blocked_t = framePhase;
        } else {
            if (iFrame == nFrame - 1) {
                blocked_t = tau - t;
            } else {
                blocked_t = framePhase + iFrame*tPerFrame - t;
            }
        }
        nSample += static_cast<unsigned int>(floor(blocked_t/kernelSampleDt));

        if (tid == 0) { // acquire spatial filtered input
            contrast = linearResponse[0]*dx*dy; 
        }
        __syncthreads();

        if (tid < nSample) {
            filtered = temporalKernel(t, surround);
        } else {
            filtered = 0.0f;
        }
        __syncwarp();
        block_reduce<_float>(linearResponse, filtered);

        if (tid == 0) {
            convol += linearResponse[0]*kernelSampleDt*contrast;
        }

        t += nSample*kernelSampleDt;
    }

    // output
    if (tid == 0) {
        LGNfr[blockIdx.x] = convol;
    }
}

__global__ void LGN_nonlinear(_float* __restrict__ LGN_fr, static_nonlinear logistic, _float* __restrict__ max_convol) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    _float current_convol = LGN_fr[id];
	_float _max_convol = max_convol[id];
    _float ratio = logistic.transform(id, current_convol/_max_convol);
    LGN_fr[id] *= ratio;
}

__global__ void LGN_maxResponse(_float* __restrict__ max_convol,
                                LGN_parameter pLGN, // consider pointer
                                _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D, unsigned int nKernelSample) {

    __shared__ _float linearResponse[warpSize];
    extern __shared__ _float temporalWeight[];

    _float LMcovariant = 0.53753461391295254;
    // consider store LGN_subregion, facRatio to __shared__

    unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
    unsigned int id = blockIdx.x;
    _float covariant = pLGN.covariant[id];
    _float sqrt2 = square_root(2.0);
    _float convol;
    if (tid == 0) {
        convol = 0.0f;
    }

    // load
    LGN_subregion center(pLGN.center, id);
    LGN_subregion surround(pLGN.surround, id);
    
    // load temporal weights
    unsigned int block_size = blockDim.x*blockDim.y;
    unsigned int nblock = nKernelSample/block_size;
    for (unsigned int iblock = 0; iblock < nblock; iblock++) {
        unsigned int twid = iblock*block_size + tid;
        _float t = twid*kernelSampleDt;
        temporalWeight[twid] = temporalKernel(t, center);
        temporalWeight[nKernelSample + twid] = temporalKernel(t, surround);
    }
    
    if (tid < nKernelSample - nblock*block_size) {
        unsigned int twid = nblock*block_size + tid;
        _float t = twid*kernelSampleDt;
        temporalWeight[twid] = temporalKernel(t, center);
        temporalWeight[nKernelSample + twid] = temporalKernel(t, surround);
    }
    __syncthreads();

    // pair spatial kernel
        // center
    float xhspan = nsig * center.rx / sqrt2;
    float dx = 2*xhspan/npixel_1D;

    float yhspan = nsig * center.ry / sqrt2;
    float dy = 2*yhspan/npixel_1D;

    float x = threadIdx.x*dx - xhspan;
    float y = threadIdx.y*dy - yhspan;

    float x_prime = center.x + x - surround.x;
    float y_prime = center.y + y - surround.y;

    float sample_vol = dx * dy * kernelSampleDt;
    
    for (unsigned int it=0; it<nKernelSample; it++) {
        // frame phase vs. dt
        _float tmp = temporalWeight[it];
        _float filtered = spatialProduct(x, y, copy(1.0, tmp)*copy(1.0,center.k), center.k, center.rx, center.ry)*tmp;

        tmp = temporalWeight[nKernelSample + it];
        _float filtered_prime = spatialProduct(x_prime, y_prime, copy(1.0, tmp)*copy(1.0, surround.k), surround.k, surround.rx, surround.ry)*tmp;

        if (filtered < filtered_prime) {
            filtered *= (-covariant);
        }
        __syncwarp();
        block_reduce<_float>(linearResponse, filtered);

        if (tid == 0) { // acquire spatial filtered input
            convol += linearResponse[0]*sample_vol;
        }
        __syncthreads();
    }
    
        // surround
    xhspan = nsig * surround.rx / sqrt2;
    dx = 2*xhspan/npixel_1D;

    yhspan = nsig * surround.ry / sqrt2;
    dy = 2*yhspan/npixel_1D;

    x = threadIdx.x*dx - xhspan;
    y = threadIdx.y*dy - yhspan;

    x_prime = surround.x + x - center.x;
    y_prime = surround.y + y - center.y;

    sample_vol = dx * dy * kernelSampleDt;
    
    for (unsigned int it=0; it<nKernelSample; it++) {
        // frame phase vs. dt
        _float tmp = temporalWeight[nKernelSample + it];
        _float filtered = spatialProduct(x, y, copy(1.0, tmp)*copy(1.0,surround.k), surround.k, surround.rx, surround.ry)*tmp;

        tmp = temporalWeight[it];
        _float filtered_prime = spatialProduct(x_prime, y_prime, copy(1.0, tmp)*copy(1.0, center.k), center.k, center.rx, center.ry)*tmp;

        if (filtered < filtered_prime) {
            filtered *= (-covariant);
        }
        __syncwarp();
        block_reduce<_float>(linearResponse, filtered);

        if (tid == 0) { // acquire spatial filtered input
            convol += linearResponse[0]*sample_vol; 
        }
        __syncthreads();
    }

    // output
    if (tid == 0) {
        max_convol[blockIdx.x] = convol;
    }
}
/*
    _float one_unkown_second_order_eq_solver(_float a, _float b, _float c, int sb) {
        _float delta = b*b - 4*a*c;
        if (delta < 0) {
            printf("delta = %f < 0\n", delta);
        }
        if (sb > 0) {
            return (-b + sqrt(delta))/2/a;
        } else {
            return (-b - sqrt(delta))/2/a;
        }
    }
    
    _float var_triag_dist(_float a, _float b, _float c) {
        return (a*a + b*b + c*c - a*b - b*c - a*c)/18;
    }
    
    void set_triag_dist(_float &a, _float &b, _float &c, _float m, _float sd0) {
        _float constant = a + b;
        c = 3*m - constant;
        _float var = var_triag_dist(a, b, c);
        _float p = a*b - 6*(sd0*sd0-var);
        b = one_unkown_second_order_eq_solver(1.0f, -constant, p, 1);
        a = constant - b;
    }
    
    __device__ _float get_triag_dist(_float a, _float b, _float c, _float rand) {
        _float F_c = (c-a)/(b-a)
        _float x;
        if (rand < F_c) {
            x = a + sqrt(x*(b-a)*(c-a));
        } else {
            x = b - sqrt((1-x)*(b-a)*(b-c));
        }
        return x;
    }
    
    __global__ initialize_LGN(LGN_stats s, LGN_parameter p, curandStateMRG32k3a* __restrict__ rState, unsigned long seed) {
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, id, 0, &rState[id]);
        for (int i=1; i<s.n; i++) {
            unsigned int index = i*p.nLGN + id;
            _float x = uniform(&localState);
            switch (s.idist[i]) {
                //  0: uniform
                case 0:
                    p.parameter[index] = s.stats[i*3] + x*(s.stats[i*3+1] - s.stats[i*3]);
                    break;
                //  1: gauss
                case 1:
                    break;
                //  2: log-normal
                case 2:
                    break;
                //  3: triangle
                case 3: 
                    p.parameter[index] = get_triag_dist(s.stats[i*3], s.stats[i*3+1], s.stats[i*3+2], x);
                    break;
                default:
                    printf("dist %i not defined\n", stats.idist[i]);
            }
        }
    }
    
    __device__ __inline__ unsigned int get_index0(_float x, _float xmin, _float dx) {
        unsigned int index = x>xim? round((x-xmin)/dx): 0;
        return index;
    }
    
    __device__ __inline__ unsigned int get_index1(_float x, _float xmin, _float xmax, _float dx) {
        unsigned int index = x<xmax? round((x-xmin)/dx): round((xmax-xmin)/dx);
        return index;
    }
*/
