#include "discrete_input_convol.h"

// TO-DO
// update_kernel
// update_static_nonlinearity

// TO-DO 
// block-processing without input-output delay
// http://www.cs.ust.hk/mjg_lib/bibs/DPSu/DPSu.Files/Ga95.PDF
// block processing file:///C:/Users/gueux/Desktop/FFTConvolution.pdf
// direct form:
__device__ 
__inline__ 
_float AexpTau(_float a, _float tau) {
    return a * expp(-tau);
}

__device__ 
__inline__ 
_float spatialKernel(Float x, Float y, Float k, Float rx, Float ry) {
    return k * expp(-x*x/(rx*rx) - y*y/(ry*ry));
}

__device__ 
__inline__ 
_float temporalKernel(Float tau, Temporal_component &temp, Float fac1, Float fac2) {
    Float tau1 = tau/subr.tauR;
    Float tau2 = tau/subr.tauD;
    Float A1 = power(tau1, subr.nR-1)/(subr.tauR * fac1);
    Float A2 = power(tau2, subr.nD-1)/(subr.tauD * fac2);

    Float tp = subr.ratio * AexpTau(A1, tau1) - AexpTau(A2, tau2);
    return tp;
}

__device__ 
__inline__
Float get_intensity(unsigned int coneType, Float x0, Float y0, unsigned int iLayer) {
    float x, y;
    retina_to_plane(x0, y0, x, y);
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


/*
Per frame
    max spatial sample size: 32 x 32
    number of temporal kernel evaluation <= spatial sample size
*/
// grid: nLGN blocks
// block: spatialSample1D x spatialSample1D (npixel_1D)
__global__ void LGN_convol(_float* __restrict__ LGNfr,
                           LGN_parameter pLGN, // consider pointer
                           unsigned int iSample0,
                           _float samplePhase, unsigned int nKernelSample, _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D) {

    __shared__ _float linearResponse[warpSize];
    extern __shared__ _float temporalWeight[];
    Float convol;

    // consider store LGN_subregion, facRatio to __shared__

    unsigned int id = blockIdx.x;

    // load
    LGN_subregion center(pLGN.center, id);
    LGN_subregion surround(pLGN.surround, id);

    _float sqrt2 = square_root(2.0);
    unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
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

    // load temporal weights using all the threads in the block
    unsigned int block_size = blockDim.x*blockDim.y;
    unsigned int nblock = nKernelSample/block_size;

    for (unsigned int iblock = 0; iblock < nblock; iblock++) {
        unsigned int twid = iblock*block_size + tid;
        _float t = twid*kernelSampleDt + samplePhase;
        temporalWeight[twid] = temporalKernel(t, center);
        temporalWeight[nKernelSample + twid] = temporalKernel(t, surround);
    }
    
    if (tid < nKernelSample - nblock*block_size) {
        unsigned int twid = nblock*block_size + tid;
        _float t = twid*kernelSampleDt + samplePhase;
        temporalWeight[twid] = temporalKernel(t, center);
        temporalWeight[nKernelSample + twid] = temporalKernel(t, surround);
    }
    __syncthreads();

    // calculate spatial filter
        // center
    unsigned int type = pLGN.centerType[id];

    _float xhspan = nsig * center.rx / sqrt2;
    _float dx = 2*xhspan/npixel_1D;

    _float yhspan = nsig * center.ry / sqrt2;
    _float dy = 2*yhspan/npixel_1D;

    _float x = threadIdx.x*dx - xhspan;
    _float y = threadIdx.y*dy - yhspan;
    float x0 = static_cast<float>(center.x + x);
    float y0 = static_cast<float>(center.y + y);

    _float sample_vol = dx * dy;
    _float spatialWeight = spatialProduct(x, y, 1.0f, center.k, center.rx, center.ry);
    
    Float decay = expp(-samplePhase/tau_ave);
    unsigned int it = 0;
    Float old_qs = q[id]; // sum_i(F_i[exp(-t_i+1/tau) - exp(-t_i/tau)])*exp(-t/tau)
    for (unsigned int iSample=iSample0; iSample<iSample0+nKernelSample; iSample++) {
        unsigned int jSample = iSample % nKernelSample;

        Float local_I = get_intensity(type, x0, y0, jSample);
        block_reduce<Float>(linearResponse, local_I);
        if (tid == 0) {
            old_qs
            Float this_q = linearResponse[0];
            Float sum_I = linearResponse[0]*(1-decay);
            sum_I += old_qs*decay + ;
            if samplePhase
            old_I = old_I + expp(
        }
        __device__
        __inline__
        Float next_I

        Float filtered = spatialWeight * get_intensity(type, x0, y0, jSample);

        block_reduce<Float>(linearResponse, filtered);

        if (tid == 0) {
            filtered = linearResponse[0];
            if (it == 0) {
                filtered *= kernelSampleDt - samplePhase;
            } else {
                if (it == nKernelSample-1) {
                    filtered *= samplePhase;
                } else {
                    filtered *= kernelSampleDt;
                }
            }
            convol += filtered*temporalWeight[nKernelSample-1-it]*sample_vol; 
            it++;
        }
    }

        // surround
    type = pLGN.surroundType[id];

    xhspan = nsig * surround.rx / sqrt2;
    dx = 2*xhspan/npixel_1D;

    yhspan = nsig * surround.ry / sqrt2;
    dy = 2*yhspan/npixel_1D;

    x = threadIdx.x*dx - xhspan;
    y = threadIdx.y*dy - yhspan;
    x0 = static_cast<float>(surround.x + x);
    y0 = static_cast<float>(surround.y + y);

    sample_vol = dx * dy;
    spatialWeight = spatialProduct(x, y, 1.0f, surround.k, surround.rx, surround.ry);
    
    it = 0;
    for (unsigned int iSample=iSample0+nKernelSample; iSample>iSample0; iSample--) {
        unsigned int jSample = iSample % nKernelSample;
        Float filtered = spatialWeight * get_intensity(type, x0, y0, jSample);

        block_reduce<Float>(linearResponse, filtered);
         
        if (tid == 0) { // acquire spatial filtered input
            filtered = linearResponse[0];
            if (it == 0) {
                filtered *= samplePhase;
            } else {
                if (it == nKernelSample-1) {
                    filtered *= kernelSampleDt - samplePhase;
                } else {
                    filtered *= kernelSampleDt;
                }
            }
            convol += filtered*temporalWeight[nKernelSample + it]*sample_vol; 
        }
        it++;
    }

    // output
    if (tid == 0) {
        LGNfr[blockIdx.x] = convol;
    }
}

__global__ void LGN_nonlinear(_float* __restrict__ LGN_fr, static_nonlinear logistic, _float* __restrict__ max_convol) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	_float _max_convol = max_convol[id];
	// min = -max;
	_float current_convol = LGN_fr[id];
    if (current_convol < 0) {
        current_convol = 0;
    }
    __syncwarp(); // check necessity

    _float ratio = logistic.transform(id, current_convol/_max_convol);
    LGN_fr[id] = current_convol * ratio;
}

__global__ void LGN_maxResponse(_float* __restrict__ max_convol,
                                LGN_parameter pLGN, // consider pointer
                                unsigned int nKernelSample, _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D) {

    __shared__ _float linearResponse[warpSize];
    extern __shared__ _float temporalWeight[];
    Float convol;

    // consider store LGN_subregion, facRatio to __shared__

    unsigned int id = blockIdx.x;

    // load
    LGN_subregion center(pLGN.center, id);
    LGN_subregion surround(pLGN.surround, id);
    
    unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
    _float covariant = pLGN.covariant[id];
    _float sqrt2 = square_root(2.0);
    if (tid == 0) {
        convol = 0.0f;
    }

    // load temporal weights
    unsigned int block_size = blockDim.x*blockDim.y;
    unsigned int nblock = nKernelSample/block_size;

    for (unsigned int iblock = 0; iblock < nblock; iblock++) {
        unsigned int twid = iblock*block_size + tid;
        _float t = (twid + 1)*kernelSampleDt;
        temporalWeight[twid] = temporalKernel(t, center);
        temporalWeight[nKernelSample + twid] = temporalKernel(t, surround);
    }
    
    if (tid < nKernelSample - nblock*block_size) {
        unsigned int twid = nblock*block_size + tid;
        _float t = (twid + 1)*kernelSampleDt;
        temporalWeight[twid] = temporalKernel(t, center);
        temporalWeight[nKernelSample + twid] = temporalKernel(t, surround);
    }
    __syncthreads();

    _float signCS = copy(1.0, center.k * surround.k);
    // pair spatial kernel
        // center
    _float xhspan = nsig * center.rx / sqrt2;
    _float dx = 2*xhspan/npixel_1D;

    _float yhspan = nsig * center.ry / sqrt2;
    _float dy = 2*yhspan/npixel_1D;

    _float x = threadIdx.x*dx - xhspan;
    _float y = threadIdx.y*dy - yhspan;

    _float x_prime = center.x + x - surround.x;
    _float y_prime = center.y + y - surround.y;

    _float sample_vol = dx * dy * kernelSampleDt;

    _float spatialWeightC = spatialProduct(x, y, 1.0, center.k, center.rx, center.ry);
    _float spatialWeightS = spatialProduct(x_prime, y_prime, 1.0, surround.k, surround.rx, surround.ry);
    
    for (unsigned int it=0; it<nKernelSample; it++) {
        Float filter = abs(spatialWeightC * temporalWeight[it]);
        Float filter_prime = abs(spatialWeightS * temporalWeight[nKernelSample + it]);

        if (filter < filter_prime) {
            filter *= copy(covariant, signCS);
        } 

        block_reduce<Float>(linearResponse, filter);

        if (tid == 0) { // acquire spatial filtered input
            convol += linearResponse[0]*sample_vol;
        }
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
    spatialWeightS = spatialProduct(x, y, 1.0, surround.k, surround.rx, surround.ry);
    spatialWeightC = spatialProduct(x_prime, y_prime, 1.0, center.k, center.rx, center.ry);
    
    for (unsigned int it=0; it<nKernelSample; it++) {

        _float filter = abs(spatialWeightS * temporalWeight[nKernelSample + it]);
        _float filter_prime = abs(spatialWeightC * temporalWeight[it]);

        if (filter < filter_prime) {
            filter *= copy(covariant, signCS);
        }

        block_reduce<Float>(linearResponse, filter);

        if (tid == 0) { // acquire spatial filtered input
            convol += linearResponse[0]*sample_vol; 
        }
    }

    // output
    if (tid == 0) {
        assert(convol >= 0);
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
