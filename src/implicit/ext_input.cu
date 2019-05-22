#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <helper_cuda.h>         // helper functions for CUDA error check
#include "discrete_input_convol.h"
#include "global.h"

// the retinal discrete x, y as cone receptors id
texture<float, cudaTextureType2DLayered> LMS_frame;

void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
    layer.addressMode[0] = cudaAddressModeBorder;
    layer.addressMode[1] = cudaAddressModeBorder; layer.filterMode = cudaFilterModeLinear;
    layer.normalized = true;
}

void prep_sample(unsigned int width, unsigned int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, unsigned int nSample) { 
    cudaMemcpy3DParms params = {0};
    params.srcPos = make_cudaPos(0, 0, 0);
    params.dstPos = make_cudaPos(0, 0, 0);
    params.extent = make_cudaExtent(width, height, nSample);
    params.kind = cudaMemcpyDeviceToDevice;

    params.srcPtr = make_cudaPitchedPtr(L, width * sizeof(float), width, height);
    params.dstArray = dL;
    checkCudaErrors(cudaMemcpy3D(&params));

    params.srcPtr = make_cudaPitchedPtr(M, width * sizeof(float), width, height);
    params.dstArray = dM;
    checkCudaErrors(cudaMemcpy3D(&params));

    params.srcPtr = make_cudaPitchedPtr(S, width * sizeof(float), width, height);
    params.dstArray = dS;
    checkCudaErrors(cudaMemcpy3D(&params));
}

unsigned int find_denorm(unsigned int u1, unsigned int u2, bool MorN, unsigned int &norm) { 
    unsigned int m, n;
    if (MorN) { //u2 > u1
        m = u1;
        n = u2 - u1*(u2/u1);
    } else {
        m = u2;
        n = u1 - u2*(u1/u2);
    }
    printf("m = %u, n = %u\n", m, n);
    assert (m>n);
    for (int i=n; i>1; i--) {
        if (n%i==0 && m%i==0) { 
            norm = n/i;
            return m/i;
        } 
    }
    norm = 1;
    return m;
} 

__global__ void plane_to_retina(float* __restrict__ LMS,
                                unsigned int iFrame,
                                unsigned int maxFrame,
                                unsigned int nPixelPerFrame,
                                unsigned int width, unsigned int height,
                                float sup, float inf,
                                float tem, float nas,
                                float yb, float yt,
                                float xr, float xl)
{
    // transform from normalized plane coord to retinal coord (tan)
    
    // all positives
    //  retina(right-hand) FRAME(left-hand)
    //         sup              yb 
    //     tem     nas  ->  xr      xl
    //         sup              yt 
    //
    unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        float *outputChannel = LMS + blockIdx.z*maxFrame*nPixelPerFrame +iFrame*nPixelPerFrame + iy*width + ix;

        float x = static_cast<float>(ix+0.5)/width;
        float y = static_cast<float>(iy+0.5)/height;
        float xmin = 1.0f/(2.0f*width);
        float ymin = 1.0f/(2.0f*height);
        float dx = xmin*2;
        float dy = ymin*2;
        
        float xspan = (width-1.0f)/width;
        float yspan = (height-1.0f)/height;
        // xmin, ymin, dx, dy, xspan, yspan, are the same for retinal coordi because they are both normalized in texture memory
        
        float x0 = x - xmin;
        float y0 = y - ymin;
        
        float tanx = tanf(x * (tem + nas)-nas);
        float tany = tanf(y * (sup + inf)-sup);
        
        float tanx0 = tanf(x0 * (tem + nas)-nas);
        float tany0 = tanf(y0 * (sup + inf)-sup);
        
        float tanx1 = tanf((x+xmin) * (tem + nas)-nas);
        float tany1 = tanf((y+ymin) * (sup + inf)-sup);
        
        x0 = xmin + (xl + tanx0)/(xr+xl)*xspan;
        y0 = ymin + (yb + tany0)/(yt+yb)*yspan;
        
        x = xmin + (xl + tanx)/(xr+xl)*xspan;
        y = ymin + (yb + tany)/(yt+yb)*yspan;
        
        // distance in the frame between the two adjacent pixel in the retina
        float xs = (tanx1-tanx0)/(xr+xl)*xspan;
        float ys = (tany1-tany0)/(yt+yb)*yspan;
        
        unsigned int nx = static_cast<unsigned int>(ceil(xs/dx));
        unsigned int ny = static_cast<unsigned int>(ceil(ys/dy));

        float output = 0.0f;

        if (nx == 1 && ny == 1) {
            output = tex2DLayered(LMS_frame, x, y, blockIdx.z);
        } else { // sum across
            if (nx > 1 && ny == 1) {
                for (unsigned int jx = 0; jx<nx; jx++) {
                    float ax = x0 + jx*dx;
                    output += tex2DLayered(LMS_frame, ax, y, blockIdx.z);
                }
            } else {
                if (nx == 1 && ny > 1) {
                    for (unsigned int jy = 0; jy<ny; jy++) {
                        float ay = y0 + jy*dy;
                        output += tex2DLayered(LMS_frame, x, ay, blockIdx.z);
                    }
                } else {
                    for (unsigned int jx = 0; jx<nx; jx++) {
                        float ax = x0 + jx*dx;
                        for (unsigned int jy = 0; jy<ny; jy++) {
                            float ay = y0 + jy*dy;
                            output += tex2DLayered(LMS_frame, ax, ay, blockIdx.z);
                        }
                    }
                }
            }
        }
        __syncwarp();
        // average 
        *outputChannel = output/(nx*ny);
    }
}
// to-do:
// 1. use non-uniform temporal filtering to mimick cone behavior
__global__ void intensity_to_contrast(float* __restrict__ LMS,
                                      unsigned int latestFrame,
									  unsigned int maxFrame,
                                      unsigned int nPixelPerFrame,
                                      _float framePhase,
                                      _float tPerFrame,
									  _float kernelSampleDt,
                                      _float ave_tau)
{
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < nPixelPerFrame) {
        _float current;
        _float average;
        unsigned int iSample = blockIdx.y;
        unsigned int iChannel = blockIdx.z;
        // gridDim.y = nKernelSample
        // gridDim.z = 3 Channels
        // blockDim.x * gridDim.x = nPixelPerFrame

        float *inputChannel = LMS + iChannel*maxFrame*nPixelPerFrame;
        float *outputChannel = LMS + gridDim.z*maxFrame*nPixelPerFrame + iChannel*gridDim.y*nPixelPerFrame + iSample*nPixelPerFrame + id;
        // 1                       latestframe   0
        // :    |->          |->          |->     :
        //  fP1              fP0              fP
        assert(ave_tau > tPerFrame);
        _float t0 = kernelSampleDt * iSample - framePhase; //from latestFrame
        
        _float framePhase0;
        unsigned int frame0;
        if (t0 < 0) {
            framePhase0 = -t0;
            frame0 = latestFrame + maxFrame;
        } else {
            framePhase0 = tPerFrame - fmod(t0, tPerFrame);
            frame0 = latestFrame + maxFrame - (1 + static_cast<unsigned int>(floor(t0/tPerFrame)));
        }

        if (ave_tau > framePhase0) { // then there is the ending frame, frame1 to be consider

            _float add_t;
            unsigned int iFrame;

            unsigned int frame1 = latestFrame + maxFrame - (1+ static_cast<unsigned int>(floor((t0 + ave_tau)/tPerFrame)));
            iFrame = frame1 % maxFrame;
            _float last = static_cast<_float>(inputChannel[iFrame*nPixelPerFrame + id]);
            _float framePhase1 = fmod(ave_tau - framePhase0, tPerFrame);

            _float mid = 0.0;
            for (unsigned int frame = frame1 + 1; frame < frame0; frame++) {
                iFrame = frame % maxFrame;
                mid += static_cast<_float>(inputChannel[iFrame*nPixelPerFrame + id]);
            }

            iFrame = frame0 % maxFrame;
            current = static_cast<_float>(inputChannel[iFrame*nPixelPerFrame + id]);

            add_t = framePhase0 + framePhase1 + (frame0-frame1-1)*tPerFrame;
            average = (current*framePhase0 + mid*tPerFrame + last*framePhase1)/add_t;
        } else {
            // averaging windows ends within framePhase0
            unsigned int iFrame = frame0 % maxFrame;
            average = static_cast<_float>(inputChannel[iFrame*nPixelPerFrame + id]);
        }

        _float c;

        if (average > 0.0) { 
            c = (current - average)/average;
            if (c > 1.0) {
                c = 1.0;
            } else {
                if (c < -1.0) {
                    c = -1.0;
                }
            }
        } else {
            if (current == 0.0) {
                c = 0.0;
            } else { 
                c = 1.0;
            }
        }
        __syncwarp();
        *outputChannel = c;
    }
}

int main(int argc, char **argv) {
   
    cudaDeviceProp deviceProps;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

    // from the retina facing out
    std::ofstream fplane, fretina, fcontrast, fLGN_fr, fLGN_convol, fmax_convol;
    fplane.open("3xplane.bin", std::ios::out | std::ios::binary);
    fretina.open("3xretina.bin", std::ios::out | std::ios::binary);
    fcontrast.open("contrast.bin", std::ios::out | std::ios::binary);
    fLGN_fr.open("LGNfr.bin", std::ios::out | std::ios::binary);
    fLGN_convol.open("LGN_convol.bin", std::ios::out | std::ios::binary);
    fmax_convol.open("max_convol.bin", std::ios::out | std::ios::binary);

    float init_luminance = 1.0/6.0; //1.0/6.0;

    float sup = 70*M_PI/180.0f; 
    float inf = 30*M_PI/180.0f;
    float tem = 45*M_PI/180.0f;
    float nas = 85*M_PI/180.0f; // 100 see wiki, compromised for tan(pi/2)

    unsigned int width = 96;
    unsigned int height = 64;
    unsigned int local_width = width/8;
    unsigned int local_height = height/8;
	unsigned int nLGN_x;
	unsigned int nLGN_y;
    unsigned int nSpatialSample1D = 8; // spatial kernel sample size = nSpatialSample1D x nSpatialSample1D
    unsigned int nsig = 3;
    float tau = 250.0f; 
    float ave_tau = 300.0f; // in ms .. cone adaptation at 300ms https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289
    _float dt = 0.1f; // in ms, better in fractions of binary 
    unsigned int sampleRate = static_cast<unsigned int>(round(1000/dt));
    unsigned int nt;
    sscanf(argv[argc-3], "%u", &nt);
	sscanf(argv[argc-2], "%u", &nLGN_x);
	sscanf(argv[argc-1], "%u", &nLGN_y);
    printf("simulating for %u steps, t = %f ms\n", nt, nt*dt);
    unsigned int nKernelSample = 128;
    unsigned int frameRate = 60; // Hz
    unsigned int nLGN = nLGN_x * nLGN_y;
    hLGN_parameter hLGN(nLGN);

    // set test param for LGN subregion RF kernel 
    /*
        Group the same cone-specific types together for warp performance
    */
    for (unsigned int i=0; i<nLGN; i++) {
		hLGN.center.k[i] = 100.0 * (i % 2 == 0 ? 1 : -1);
		hLGN.center.x[i] = 0.5f / width + fmod(i*(1.0f - 0.5f / width)/8.0f, (1.0f - 0.5f / width));
		hLGN.center.y[i] = 0.5f / height + fmod((i/8)*(1.0f - 0.5f / height)/8.0f, (1.0f - 0.5f / height));
        hLGN.center.rx[i] = 0.015;
		hLGN.center.ry[i] = 0.015;
        hLGN.center.tauR[i] = 5.5;
        hLGN.center.tauD[i] = 14;
        hLGN.center.nR[i] = 6;
        hLGN.center.nD[i] = 7;
		assert(hLGN.center.nR[i] < hLGN.center.nD[i]);
        hLGN.center.ratio[i] = 1;

        hLGN.surround.k[i] = -copy(50.0, hLGN.center.k[i]);
		hLGN.surround.x[i] = 0.5f / width + fmod(i*(1.0f - 0.5f / width)/8.0f, (1.0f - 0.5f / width));
		hLGN.surround.y[i] = 0.5f / height + fmod((i/8)*(1.0f - 0.5f / height)/8.0f, (1.0f - 0.5f / height));
        hLGN.surround.rx[i] = 0.03;
		hLGN.surround.ry[i] = 0.03;
        hLGN.surround.tauR[i] = 8;
        hLGN.surround.tauD[i] = 14;
        hLGN.surround.nR[i] = 6;
        hLGN.surround.nD[i] = 7;
		assert(hLGN.surround.nR[i] < hLGN.surround.nD[i]);
		hLGN.surround.ratio[i] = 1;

        hLGN.covariant[i] = 0.53753461391295254; 
        hLGN.centerType[i] = i%2;
        hLGN.surroundType[i] = (i+1)%2;

        hLGN.logistic.spont[i] = 0.0;
        hLGN.logistic.c50[i] = 0.5;
        hLGN.logistic.sharpness[i] = 1.0;
    }

    LGN_parameter dLGN(nLGN, hLGN);

    assert(nSpatialSample1D <= 32);

    unsigned nPixelPerFrame = width*height;

    float yb = tan(sup);
    float yt = tan(inf);
    float xr = tan(tem);
    float xl = tan(nas);

    // set params for layerd texture
    init_layer(L_retinaConSig);
    init_layer(M_retinaConSig);
    init_layer(S_retinaConSig);

    init_layer(LMS_frame);

    unsigned int nRetrace = static_cast<unsigned int>(round(tau/dt));
    tau = nRetrace * dt;
    _float kernelSampleDt = tau/nKernelSample;
    printf("temporal kernel retraces %f ms, samples %u points, sample rate = %f Hz\n", tau, nKernelSample, 1000/kernelSampleDt);

    unsigned int maxFrame;
    _float maxFramef = (tau+ave_tau)/1000 * frameRate;
    if (round(maxFramef) == maxFramef) {
       maxFrame = static_cast<unsigned int>(round(maxFramef) + 1);
    } else {
       maxFrame = static_cast<unsigned int>(round(maxFramef));
    }
    
    printf("temporal kernel retrace (tau) + contrast computation (ave_tau) = %f ms needs at most %u frames\n", tau+ave_tau, maxFrame);
	_float tPerFrame = 1000.0f / frameRate; //ms
    printf("~ %f plusminus 1 kernel sample per frame\n", tPerFrame/kernelSampleDt);

    bool moreDt = true;
    if (sampleRate < frameRate) {
        moreDt = false;
        printf("frameRate > simulation rate!!");
    }

    unsigned int norm;
    unsigned int denorm = find_denorm(frameRate, sampleRate , moreDt, norm);
    unsigned *exact_norm = new unsigned int[denorm];
    unsigned int current_norm = 0;
    printf("%u exact phases in [0,1]: ", denorm);
    for (unsigned int i=0; i<denorm; i++) {
        if (current_norm>denorm) {
            current_norm -= denorm;
        } 
        exact_norm[i] = current_norm;
        if (i<denorm - 1) {
            printf("%u/%u, ", current_norm, denorm);
        } else {
            printf("%u/%u\n", current_norm, denorm);
        }
        current_norm += norm;
    }
    assert(current_norm == denorm);

    unsigned int *exact_it = new unsigned int[denorm];
    for (unsigned int i=0; i<denorm; i++) {
        exact_it[i] = (i*sampleRate)/frameRate;
    }
    assert((sampleRate*denorm) % frameRate == 0);
    unsigned int co_product = (sampleRate*denorm)/frameRate;


    unsigned int nChannel = 3; // L, M, S
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArr_L;
    cudaArray *cuArr_M;
    cudaArray *cuArr_S;
    checkCudaErrors(cudaMalloc3DArray(&cuArr_L, &channelDesc, make_cudaExtent(width, height, nKernelSample), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_M, &channelDesc, make_cudaExtent(width, height, nKernelSample), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_S, &channelDesc, make_cudaExtent(width, height, nKernelSample), cudaArrayLayered));

    cudaArray *cuArr_frame;
    checkCudaErrors(cudaMalloc3DArray(&cuArr_frame, &channelDesc, make_cudaExtent(width, height, nChannel), cudaArrayLayered));

    // bind texture to cudaArrays
    checkCudaErrors(cudaBindTextureToArray(L_retinaConSig,  cuArr_L, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_retinaConSig,  cuArr_M, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_retinaConSig,  cuArr_S, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(LMS_frame,  cuArr_frame, channelDesc));

    float* LMS;

    // LMS retina projection intensity array of [maxFrame] frames from t - (tau + ave_tau)
    float* __restrict__ dL;
    float* __restrict__ dM;
    float* __restrict__ dS;
    // LMS contrast signal
    float* __restrict__ cL;
    float* __restrict__ cM;
    float* __restrict__ cS;

    checkCudaErrors(cudaMalloc((void **) &LMS, nPixelPerFrame*sizeof(float)*3*maxFrame + nPixelPerFrame*sizeof(float)*3*nKernelSample));

    dL = LMS;
    dM = dL + nPixelPerFrame*maxFrame;
    dS = dM + nPixelPerFrame*maxFrame;

    cL = dS + nPixelPerFrame*maxFrame;
    cM = cL + nPixelPerFrame*nKernelSample;
    cS = cM + nPixelPerFrame*nKernelSample;

    _float* LGN_fr = new _float[nLGN];
    _float* __restrict__ d_LGN_fr;
    _float* __restrict__ max_convol;
    checkCudaErrors(cudaMalloc((void **) &d_LGN_fr, nLGN*sizeof(_float)));
    checkCudaErrors(cudaMalloc((void **) &max_convol, nLGN*sizeof(_float)));

    // initialize average to normalized mean luminnace
    cudaStream_t s0, s1, s2;
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaStreamCreate(&s2));

    dim3 initBlock(blockSize, 1, 1);
    dim3 initGrid((nPixelPerFrame*nKernelSample + blockSize-1) / blockSize, 1, 1);

    initGrid.x = (nPixelPerFrame*maxFrame + blockSize - 1)/blockSize;
    init<<<initGrid, initBlock, 0, s0>>>(dL, init_luminance, nPixelPerFrame*maxFrame);
    init<<<initGrid, initBlock, 0, s1>>>(dM, init_luminance, nPixelPerFrame*maxFrame);
    init<<<initGrid, initBlock, 0, s2>>>(dS, init_luminance, nPixelPerFrame*maxFrame);

    cudaEvent_t i0, i1, i2;

    checkCudaErrors(cudaEventCreate(&i0));
    checkCudaErrors(cudaEventCreate(&i1));
    checkCudaErrors(cudaEventCreate(&i2));

    dim3 convolBlock(nSpatialSample1D, nSpatialSample1D, 1);
    dim3 convolGrid(nLGN, 1, 1);

    float* __restrict__ hLMS_frame = new float[nChannel*nPixelPerFrame];
    float* __restrict__ L = hLMS_frame;
    float* __restrict__ M = L + nPixelPerFrame;
    float* __restrict__ S = M + nPixelPerFrame;

    dim3 localBlock(local_width, local_height, 1);
    dim3 globalGrid((width + local_width - 1)/local_width, (height + local_height - 1)/local_height, nChannel);

    dim3 spatialBlock(blockSize, 1, 1);
    dim3 sampleGrid((nPixelPerFrame + blockSize - 1)/blockSize, nKernelSample, nChannel);

    // determine the maximums of LGN kernel convolutions
    LGN_maxResponse<<<convolGrid, convolBlock, sizeof(_float)*2*nKernelSample>>>(max_convol, dLGN, nKernelSample, kernelSampleDt, nsig, nSpatialSample1D);
    // calc LGN firing rate at the end of current dt
    unsigned int currentFrame = 0; // current frame number from stimulus
    unsigned int iFrame = 0; //latest frame inserted into the dL dM dS,  initialization not necessary
    unsigned int iPhase = 0;
    unsigned int iPhase_old = 0; // initialization not necessary
    _float framePhase;
    unsigned int jt = 0;
    for (unsigned int it = 0; it < nt; it++) {

        _float t = it*dt;

        if (it+1 > (currentFrame/denorm)*co_product + exact_it[iPhase]) {
            jt = it;
            iFrame = currentFrame % maxFrame;
            // generate/read new frame
            for (unsigned int ih = 0; ih < height; ih++) {
                for (unsigned int iw = 0; iw < width; iw++) {
                    //L[ih*width + iw] =    iw*1.0f/width + ih + iFrame*height;
                    //M[ih*width + iw] = 2*(iw*1.0f/width + ih + iFrame*height);
                    //S[ih*width + iw] = 3*(iw*1.0f/width + ih + iFrame*height);
                    unsigned int id = ih*width + iw;
                    L[id] = (1+currentFrame/30) * 1.0/6.0;
                    M[id] = (1+currentFrame/30) * 1.0/6.0;
                    S[id] = (1+currentFrame/30) * 1.0/6.0;
                }
            }
            //fplane.write((char*)hLMS_frame, nChannel*nPixelPerFrame*sizeof(float));

            {
                //cp to texture mem in device
                cudaMemcpy3DParms params = {0};
                params.srcPos = make_cudaPos(0, 0, 0);
                params.dstPos = make_cudaPos(0, 0, 0);
                params.extent = make_cudaExtent(width, height, nChannel);
                params.kind = cudaMemcpyHostToDevice;
                params.srcPtr = make_cudaPitchedPtr(hLMS_frame, width * sizeof(float), width, height);
                params.dstArray = cuArr_frame;
                checkCudaErrors(cudaMemcpy3D(&params));

                checkCudaErrors(cudaEventRecord(i0));
                checkCudaErrors(cudaEventSynchronize(i0));
            }
            // transform from euclidean coord to retinal visual field in rad 1:1 and store to dLMS for later average
            plane_to_retina<<<globalGrid, localBlock>>>(LMS, iFrame, maxFrame, nPixelPerFrame, width, height, sup, inf, tem, nas, yb, yt, xr, xl);
            getLastCudaError("plane_to_retina failed");

            checkCudaErrors(cudaEventRecord(i1));
            checkCudaErrors(cudaEventSynchronize(i1));

            ///* check
                checkCudaErrors(cudaMemcpy(L, &(dL[iFrame*nPixelPerFrame]), nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(M, &(dM[iFrame*nPixelPerFrame]), nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(S, &(dS[iFrame*nPixelPerFrame]), nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaEventRecord(i2));
                checkCudaErrors(cudaEventSynchronize(i2));
                // store frame
                fretina.write((char*)hLMS_frame, nChannel*nPixelPerFrame*sizeof(float));
            //*/
            printf("frame #%i prepared at t = %f, %f%% from previous t = %f.\n", currentFrame, currentFrame*tPerFrame, static_cast<_float>(exact_norm[iPhase])/denorm, t);
            currentFrame++;
            iPhase_old = iPhase;
            iPhase = (iPhase + 1) % denorm;
        }

        framePhase = (((it - jt + 1)*denorm - exact_norm[iPhase_old])*dt)/denorm;
        // convert intensity to contrast for temporal kernel sample points, thus frame-wise info incorporated into kernel sample points
        intensity_to_contrast<<<sampleGrid, spatialBlock>>>(LMS, iFrame, maxFrame, nPixelPerFrame, framePhase, tPerFrame, kernelSampleDt, ave_tau);
		getLastCudaError("intensity_to_contrast failed");
        checkCudaErrors(cudaMemcpy(L, &(cL[2*nPixelPerFrame]), nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaEventRecord(i2));
        checkCudaErrors(cudaEventSynchronize(i2));
        fcontrast.write((char*)L, nPixelPerFrame*sizeof(float));
		// copy contrast signals to texture
        prep_sample(width, height, cL, cM, cS, cuArr_L, cuArr_M, cuArr_S, nKernelSample);
        // perform kernel convolution with built-in texture interpolation
        LGN_convol<<<convolGrid, convolBlock, sizeof(_float)*2*nKernelSample>>>(d_LGN_fr, dLGN, nKernelSample, kernelSampleDt, nsig, nSpatialSample1D);
		getLastCudaError("LGN_convol failed");
        checkCudaErrors(cudaMemcpy(LGN_fr, d_LGN_fr, nLGN*sizeof(_float), cudaMemcpyDeviceToHost));
        fLGN_convol.write((char*)LGN_fr, nLGN*sizeof(_float));

		// generate LGN fr with logistic function
        LGN_nonlinear<<<nLGN_x, nLGN_y>>>(d_LGN_fr, dLGN.logistic, max_convol);
		getLastCudaError("LGN_nonlinear failed");
        checkCudaErrors(cudaMemcpy(LGN_fr, d_LGN_fr, nLGN*sizeof(_float), cudaMemcpyDeviceToHost));
        fLGN_fr.write((char*)LGN_fr, nLGN*sizeof(_float));
    }
    checkCudaErrors(cudaMemcpy(LGN_fr, max_convol, nLGN*sizeof(_float), cudaMemcpyDeviceToHost));
    fmax_convol.write((char*)LGN_fr, nLGN*sizeof(_float));

    checkCudaErrors(cudaDeviceSynchronize());
    delete []hLMS_frame;
    delete []LGN_fr;
    delete []exact_norm;
    delete []exact_it;
    
    fretina.close();
    fplane.close();
    fLGN_fr.close();
    fLGN_convol.close();
    fcontrast.close();
    fmax_convol.close();

    dLGN.freeMem();
    hLGN.freeMem();
	checkCudaErrors(cudaStreamDestroy(s0));
    checkCudaErrors(cudaStreamDestroy(s1));
    checkCudaErrors(cudaStreamDestroy(s2));
    checkCudaErrors(cudaFree(LMS));
    checkCudaErrors(cudaFree(d_LGN_fr));
    checkCudaErrors(cudaFree(max_convol));
    checkCudaErrors(cudaFreeArray(cuArr_L));
    checkCudaErrors(cudaFreeArray(cuArr_M));
    checkCudaErrors(cudaFreeArray(cuArr_S));
    checkCudaErrors(cudaFreeArray(cuArr_frame));
    cudaDeviceReset();
    return 0;
}

/*
    __global__ void load(float *data, int nx, int ny) {
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
        surf2Dwrite(data[y * nx + x], outputSurface, x*sizeof(float), y, cudaBoundaryModeTrap);
    }
    
    void init_tex(texture<float, cudaTextureType2D, cudaReadModeElementType> &tex) {
        tex.addressMode[0] = cudaAddressModeBorder;
        tex.addressMode[1] = cudaAddressModeBorder;
        tex.filterMode = cudaFilterModeLinear;
        tex.normalized = true;
    }
*/
