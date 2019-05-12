#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <helper_cuda.h>         // helper functions for CUDA error check
#include "discrete_input_convol.h"
#include "global.h"

// the retinal discrete x, y as cone receptors id
texture<float, cudaTextureType2D> L_frame;
texture<float, cudaTextureType2D> M_frame;
texture<float, cudaTextureType2D> S_frame;

void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
    layer.addressMode[0] = cudaAddressModeBorder;
    layer.addressMode[1] = cudaAddressModeBorder; layer.filterMode = cudaFilterModeLinear;
    layer.normalized = true;
}

void init_tex(texture<float, cudaTextureType2D, cudaReadModeElementType> &tex) {
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;
}

void prep_frame(int width, int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, int nFrame) { 
    cudaMemcpy3DParms params = {0};
    params.srcPos = make_cudaPos(0, 0, 0);
    params.dstPos = make_cudaPos(0, 0, 0);
    params.extent = make_cudaExtent(width, height, nFrame);
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

__global__ void intensity_to_contrast(float* __restrict__ dL,
                                      float* __restrict__ dM,
                                      float* __restrict__ dS,
                                      float* __restrict__ L,
                                      float* __restrict__ M,
                                      float* __restrict__ S,
                                      unsigned int latestFrame,
									  unsigned int maxFrame,
                                      unsigned int nPixelPerFrame,
                                      _float framePhase,
                                      _float tPerFrame,
									  _float kernelSampleDt,
                                      _float ave_tau)
{
    _float current_L, current_M, current_S;
    _float last_L, last_M, last_S;
    _float average_L, average_M, average_S;
    _float framePhase1;
    unsigned int frame1;

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iSample = blockIdx.y;
    // 1                    0      latestframe (frame0)
    // :    |           |   :        |
    //  fP1              fP0    fP
    assert(ave_tau > tPerFrame);
    _float t0 = kernelSampleDt * iSample + framePhase; //from latestFrame
    unsigned int frame0 = latestFrame + static_cast<unsigned int>(floor(t0/tPerFrame));
    _float framePhase0 = tPerFrame -  fmod(t0, tPerFrame);
    if (ave_tau > framePhase0) {
        framePhase1 = fmod(ave_tau - framePhase0, tPerFrame);
        frame1 = latestFrame + static_cast<unsigned int>(floor((t0 + ave_tau)/tPerFrame));

        unsigned int iFrame = frame0 % maxFrame;
        current_L = L[iFrame*nPixelPerFrame + id];
        current_M = M[iFrame*nPixelPerFrame + id];
        current_S = S[iFrame*nPixelPerFrame + id];
        
        iFrame = frame1 % maxFrame;
        last_L = L[iFrame*nPixelPerFrame + id];
        last_M = M[iFrame*nPixelPerFrame + id];
        last_S = S[iFrame*nPixelPerFrame + id];

        average_L = current_L*framePhase0;
        average_M = current_M*framePhase0;
        average_S = current_S*framePhase0;

        average_L += last_L*framePhase1;
        average_M += last_M*framePhase1;
        average_S += last_S*framePhase1;

        for (unsigned int frame = frame0 + 1; frame < frame1; frame++) {
            unsigned int iFrame = frame % maxFrame; 
            _float mid_L = L[iFrame*nPixelPerFrame + id];
            _float mid_M = M[iFrame*nPixelPerFrame + id];
            _float mid_S = S[iFrame*nPixelPerFrame + id];

            average_L += mid_L*tPerFrame;
            average_M += mid_M*tPerFrame;
            average_S += mid_S*tPerFrame;
        }
    } else {
        // averaging windows ends within framePhase0
        unsigned int iFrame = frame0 % maxFrame;
        average_L = L[iFrame*nPixelPerFrame + id] * ave_tau;
        average_M = M[iFrame*nPixelPerFrame + id] * ave_tau;
        average_S = S[iFrame*nPixelPerFrame + id] * ave_tau;
    }
     
    average_L /= ave_tau;
    average_M /= ave_tau;
    average_S /= ave_tau;

    _float cL = (current_L - average_L)/average_L;
    _float cM = (current_M - average_M)/average_M;
    _float cS = (current_S - average_S)/average_S;

    if (cL > 1.0) {
        cL = 1.0;
    } else {
        if (cL < -1.0) {
            cL = -1.0;
        }
    }

    if (cM > 1.0) {
        cM = 1.0;
    } else {
        if (cM < -1.0) {
            cM = -1.0;
        }
    }

    if (cS > 1.0) {
        cS = 1.0;
    } else {
        if (cS < -1.0) {
            cS = -1.0;
        }
    }

    L[iSample*nPixelPerFrame + id] = cL;
    M[iSample*nPixelPerFrame + id] = cM;
    S[iSample*nPixelPerFrame + id] = cS;
}

__global__ void plane_to_retina(float* __restrict__ L,
                                float* __restrict__ M,
                                float* __restrict__ S,
                                unsigned int iFrame,
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

    unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix < width && iy < height) {
        unsigned int id = iFrame * height * width + iy*width + ix;
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

        float _L = 0;
        float _M = 0;
        float _S = 0;

        if (nx == 1 && ny == 1) {
            _L = tex2D(L_frame, x, y);
            _M = tex2D(M_frame, x, y);
            _S = tex2D(S_frame, x, y);
        } else { // sum across
            if (nx > 1 && ny == 1) {
                for (unsigned int jx = 0; jx<nx; jx++) {
                    float ax = x0 + jx*dx;
                    _L += tex2D(L_frame, ax, y);
                    _M += tex2D(M_frame, ax, y);
                    _S += tex2D(S_frame, ax, y);
                }
            } else {
                if (nx == 1 && ny > 1) {
                    for (unsigned int jy = 0; jy<ny; jy++) {
                        float ay = y0 + jy*dy;
                        _L += tex2D(L_frame, x, ay);
                        _M += tex2D(M_frame, x, ay);
                        _S += tex2D(S_frame, x, ay);
                    }
                } else {
                    for (unsigned int jx = 0; jx<nx; jx++) {
                        float ax = x0 + jx*dx;
                        for (unsigned int jy = 0; jy<ny; jy++) {
                            float ay = y0 + jy*dy;
                            _L += tex2D(L_frame, ax, ay);
                            _M += tex2D(M_frame, ax, ay);
                            _S += tex2D(S_frame, ax, ay);
                        }
                    }
                }
            }
        }
        __syncwarp();
        
        // set contrast
        L[id] = _L/(nx*ny);
        M[id] = _M/(nx*ny);
        S[id] = _S/(nx*ny);
    }
}

int main(int argc, char **argv) {
   
    cudaDeviceProp deviceProps;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

    // from the retina facing out
    std::ofstream fplane, fretina, fLGN_fr;
    fplane.open("3xplane.bin", std::ios::out | std::ios::binary);
    fretina.open("3xretina.bin", std::ios::out | std::ios::binary);
    fLGN_fr.open("LGNfr.bin", std::ios::out | std::ios::binary);

    float init_luminance = 0.5;

    float sup = 70*M_PI/180.0f; 
    float inf = 30*M_PI/180.0f;
    float tem = 45*M_PI/180.0f;
    float nas = 85*M_PI/180.0f; // 100 see wiki, compromised for tan(pi/2)

    unsigned int width = 64;
    unsigned int height = 64;
    unsigned int local_width = width/8;
    unsigned int local_height = height/8;
	unsigned int nLGN_x;
	unsigned int nLGN_y;
    unsigned int nSpatialSample1D = 8; // spatial kernel sample size = nSpatialSample1D x nSpatialSample1D
    unsigned int nsig = 3;
    float tau = 200.0f; 
    float ave_tau = 300.0f; // in ms .. cone adaptation at 300ms https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289
    float dt = 0.1f; // in ms
    unsigned int nt;
    sscanf(argv[argc-3], "%u", &nt);
	sscanf(argv[argc-2], "%u", &nLGN_x);
	sscanf(argv[argc-1], "%u", &nLGN_y);
    printf("simulating for %u steps, t = %f ms\n", nt, nt*dt);
    unsigned int nKernelSample = 256;
    float frameRate = 60; // Hz
    unsigned int nLGN = nLGN_x * nLGN_y;
    LGN_parameter pLGN(nLGN);

    // set test param for LGN subregion RF kernel 
    /*
        Group the same cone-specific types together for warp performance
    */
    for (unsigned int i=0; i<nLGN; i++) {
		pLGN.center.k[i] = 30.0 * (i % 2 == 0 ? 1 : -1);
		pLGN.center.x[i] = 0.5f/width + fmod(1.0f*i/8.0f, 1.0f - 0.5f/width);
		pLGN.center.y[i] = 0.5f/height + fmod(1.0f*(i/8)/8.0f, 1.0f - 0.5f/height);
        pLGN.center.rx[i] = 0.15;
		pLGN.center.ry[i] = 0.15;
        pLGN.center.tauR[i] = 8;
        pLGN.center.tauD[i] = 14;
        pLGN.center.nD[i] = 4;
        pLGN.center.nR[i] = 3;
		assert(pLGN.center.nR[i] < pLGN.center.nD[i]);
        pLGN.center.ratio[i] = 2;

        pLGN.surround.k[i] = -copy(10.0, pLGN.center.k[i]);
		pLGN.surround.x[i] = 0.5f / width + fmod(1.0f*i / 8.0f, 1.0f - 0.5f / width);
		pLGN.surround.y[i] = 0.5f / height + fmod(1.0f*(i / 8) / 8.0f, 1.0f - 0.5f / height);
        pLGN.surround.rx[i] = 0.3;
		pLGN.surround.ry[i] = 0.3;
        pLGN.surround.tauR[i] = 10;
        pLGN.surround.tauD[i] = 16;
        pLGN.surround.nD[i] = 4;
        pLGN.surround.nR[i] = 3;
		assert(pLGN.surround.nR[i] < pLGN.surround.nD[i]);
		pLGN.surround.ratio[i] = 2;

        pLGN.covariant[i] = 0.53753461391295254; 
        pLGN.centerType[i] = i%2;
        pLGN.surroundType[i] = (i+1)%2;

        pLGN.logistic.spont[i] = 0.1;
        pLGN.logistic.c50[i] = 0.1;
        pLGN.logistic.sharpness[i] = 1.5;
    }

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

    init_tex(L_frame);
    init_tex(M_frame);
    init_tex(S_frame);

    unsigned int nRetrace = static_cast<unsigned int>(round(tau/dt));
    tau = nRetrace * dt;
    _float kernelSampleDt = tau/nKernelSample;
    printf("temporal kernel retraces %f ms, sample rate %f Hz\n", tau, 1000/kernelSampleDt);

    unsigned int maxFrame;
    _float maxFramef = (tau+ave_tau)/1000 * frameRate;
    if (round(maxFramef) == maxFramef) {
       maxFrame = static_cast<unsigned int>(round(maxFramef) + 1);
    } else {
       maxFrame = static_cast<unsigned int>(round(maxFramef));
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArr_L;
    cudaArray *cuArr_M;
    cudaArray *cuArr_S;
    checkCudaErrors(cudaMalloc3DArray(&cuArr_L, &channelDesc, make_cudaExtent(width, height, nKernelSample), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_M, &channelDesc, make_cudaExtent(width, height, nKernelSample), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_S, &channelDesc, make_cudaExtent(width, height, nKernelSample), cudaArrayLayered));

    cudaArray *cuArr_Lf;
    cudaArray *cuArr_Mf;
    cudaArray *cuArr_Sf;
    checkCudaErrors(cudaMallocArray(&cuArr_Lf, &channelDesc, width, height, cudaArrayDefault));
    checkCudaErrors(cudaMallocArray(&cuArr_Mf, &channelDesc, width, height, cudaArrayDefault));
    checkCudaErrors(cudaMallocArray(&cuArr_Sf, &channelDesc, width, height, cudaArrayDefault));

    // bind texture to cudaArrays
    checkCudaErrors(cudaBindTextureToArray(L_retinaConSig,  cuArr_L, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_retinaConSig,  cuArr_M, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_retinaConSig,  cuArr_S, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(L_frame,  cuArr_Lf, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_frame,  cuArr_Mf, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_frame,  cuArr_Sf, channelDesc));

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

    float* __restrict__ LGN_fr = new float[nLGN];
    float* __restrict__ d_LGN_fr;
    float* __restrict__ max_convol;
    checkCudaErrors(cudaMalloc((void **) &d_LGN_fr, nLGN*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &max_convol, nLGN*sizeof(float)));

    // initialize average to normalized mean luminnace
    cudaStream_t s0, s1, s2;
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaStreamCreate(&s2));

    dim3 initBlock(blockSize, 1, 1);
    dim3 initGrid((nPixelPerFrame + blockSize-1) / blockSize, 1, 1);

    init<<<initGrid, initBlock, 0, s0>>>(cL, init_luminance, nPixelPerFrame*nKernelSample);
    init<<<initGrid, initBlock, 0, s1>>>(cM, init_luminance, nPixelPerFrame*nKernelSample);
    init<<<initGrid, initBlock, 0, s2>>>(cS, init_luminance, nPixelPerFrame*nKernelSample);

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

    _float framePhase = 0;
    _float framePhase_old = 0;
    _float tPerFrame = 1000.0f/frameRate; //ms
    unsigned int iFrame = 0;
    bool newFrame = true;

    float* __restrict__ L = new float[nPixelPerFrame];
    float* __restrict__ M = new float[nPixelPerFrame];
    float* __restrict__ S = new float[nPixelPerFrame];

    dim3 localBlock(local_width, local_height, 1);
    dim3 globalGrid((width + local_width - 1)/local_width, (height + local_height - 1)/local_height, 1);

    dim3 spatialBlock(nSpatialSample1D, nSpatialSample1D, 1);
    dim3 sampleGrid(nKernelSample, 1, 1);

    _float t;

    LGN_maxResponse<<<convolGrid, convolBlock, sizeof(float)*nKernelSample>>>(max_convol, pLGN, kernelSampleDt, nsig, nSpatialSample1D, nKernelSample);
    for (unsigned int it = 0; it < nt; it++) {
        t = it*dt;
        if (newFrame) {
            
            // generate/read new frame
            for (unsigned int ih = 0; ih < height; ih++) {
                for (unsigned int iw = 0; iw < width; iw++) {
                    L[ih*width + iw] =    iw*1.0f/width + ih + iFrame*height;
                    M[ih*width + iw] = 2*(iw*1.0f/width + ih + iFrame*height);
                    S[ih*width + iw] = 3*(iw*1.0f/width + ih + iFrame*height);
                }
            }
            printf("frame #%i data received.\n", iFrame);
            /* check
            fplane.write((char*)L, nPixelPerFrame*sizeof(float));
            fplane.write((char*)M, nPixelPerFrame*sizeof(float));
            fplane.write((char*)S, nPixelPerFrame*sizeof(float));
            */

            //cp to texture mem in device
            //to-do: async
            checkCudaErrors(cudaMemcpyToArray(cuArr_Lf, 0, 0, L, nPixelPerFrame*sizeof(float), cudaMemcpyHostToDevice)); 
            checkCudaErrors(cudaMemcpyToArray(cuArr_Mf, 0, 0, M, nPixelPerFrame*sizeof(float), cudaMemcpyHostToDevice)); 
            checkCudaErrors(cudaMemcpyToArray(cuArr_Sf, 0, 0, S, nPixelPerFrame*sizeof(float), cudaMemcpyHostToDevice)); 
            checkCudaErrors(cudaEventRecord(i0));
            checkCudaErrors(cudaEventSynchronize(i0));

            // transform from euclidean coord to retinal visual field in rad 1:1
            plane_to_retina<<<globalGrid, localBlock, 0>>>(dL, dM, dS, iFrame, width, height, sup, inf, tem, nas, yb, yt, xr, xl);
            getLastCudaError("frame to retina projection failed");

            checkCudaErrors(cudaEventRecord(i1));
            checkCudaErrors(cudaEventSynchronize(i1));

            printf("frame #%i transformed.\n", iFrame);

            /* check
            checkCudaErrors(cudaMemcpy(L, dL, nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(M, dM, nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(S, dS, nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
            */

            // store frame
            checkCudaErrors(cudaEventRecord(i2));
            checkCudaErrors(cudaEventSynchronize(i2));

            /*
            fretina.write((char*)L, nPixelPerFrame*sizeof(float));
            fretina.write((char*)M, nPixelPerFrame*sizeof(float));
            fretina.write((char*)S, nPixelPerFrame*sizeof(float));
            */

            newFrame = false;
        }

        framePhase_old = framePhase;
        framePhase = fmod(t, tPerFrame);

        if (framePhase < framePhase_old) {
            iFrame++;
            iFrame = iFrame % maxFrame;
            newFrame = true;
        }

        intensity_to_contrast<<<sampleGrid, spatialBlock>>>(dM, dL, dS, cL, cM, cS, iFrame, maxFrame, nPixelPerFrame, framePhase, tPerFrame, kernelSampleDt, ave_tau);
        // copy contrast signals to texture
        prep_frame(width, height, cL, cM, cS, cuArr_L, cuArr_M, cuArr_S, nKernelSample);
        // perform kernel convolution with built-in texture interpolation
        LGN_convol<<<convolGrid, convolBlock>>>(d_LGN_fr, pLGN, maxFrame, framePhase, tPerFrame, iFrame, kernelSampleDt, tau, nsig, nSpatialSample1D);
        // generate LGN fr with logistic function
        LGN_nonlinear<<<nLGN_x, nLGN_y>>>(d_LGN_fr, pLGN.logistic, max_convol);

        checkCudaErrors(cudaMemcpy(LGN_fr, d_LGN_fr, nLGN*sizeof(float), cudaMemcpyDeviceToHost));
        fLGN_fr.write((char*)LGN_fr, nLGN*sizeof(float));
    }
    checkCudaErrors(cudaDeviceSynchronize());
    delete []L;
    delete []M;
    delete []S;
    delete []LGN_fr;
    
    fretina.close();
    fplane.close();
    fLGN_fr.close();
	checkCudaErrors(cudaStreamDestroy(s0));
    checkCudaErrors(cudaStreamDestroy(s1));
    checkCudaErrors(cudaStreamDestroy(s2));
    checkCudaErrors(cudaFree(LMS));
    checkCudaErrors(cudaFree(d_LGN_fr));
    checkCudaErrors(cudaFree(max_convol));
    checkCudaErrors(cudaFreeArray(cuArr_L));
    checkCudaErrors(cudaFreeArray(cuArr_M));
    checkCudaErrors(cudaFreeArray(cuArr_S));
    checkCudaErrors(cudaFreeArray(cuArr_Lf));
    checkCudaErrors(cudaFreeArray(cuArr_Mf));
    checkCudaErrors(cudaFreeArray(cuArr_Sf));
    cudaDeviceReset();
    return 0;
}

/*
__global__ void load(float *data, int nx, int ny) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    surf2Dwrite(data[y * nx + x], outputSurface, x*sizeof(float), y, cudaBoundaryModeTrap);
}*/

