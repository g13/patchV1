#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "discrete_input_convol.h"
#include "global.h"

// the retinal discrete x, y as cone receptors id
texture<float, cudaTextureType2DLayered> LMS_frame;

void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
    layer.addressMode[0] = cudaAddressModeBorder;
    layer.addressMode[1] = cudaAddressModeBorder; 
    layer.filterMode = cudaFilterModeLinear;
    layer.normalized = true; //accessing coordinates are normalized
}

void prep_sample(unsigned int iSample, unsigned int width, unsigned int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, unsigned int nSample) {
    // copy the three channels L, M, S of the #iSample frame to the cudaArrays dL, dM and dS
    cudaMemcpy3DParms params = {0};
    params.srcPos = make_cudaPos(0, 0, 0);
    params.dstPos = make_cudaPos(0, 0, iSample);
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


// what is contrast, when stimuli is natural, in general not drifiting grating
// here defined as T-300 ms average of previous intensity
__global__ 
void intensity_to_contrast(float* __restrict__ LMS,
						   unsigned int maxFrame,
                           unsigned int nPixelPerFrame,
                           unsigned int nKernelSample,
                           Float tPerFrame,
                           Float ave_tau,
                           unsigned int frame0,
                           unsigned int frame1,
                           Float framePhase0,
                           Float framePhase1,
                           bool simpleContrast)
{
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id < nPixelPerFrame) {
        Float current;
        Float average;
        unsigned int iChannel = blockIdx.y;
        // gridDim.y = 3 Channels
        // blockDim.x * gridDim.x = nPixelPerFrame

        float *inputChannel = LMS + iChannel*maxFrame*nPixelPerFrame;
        Float *output = (Float*)(LMS + gridDim.y*maxFrame*nPixelPerFrame) + iChannel*nKernelSample*nPixelPerFrame + id;
        // 1                       latestframe   0
        // :    |->          |->          |->     :
        //  fP1              fP0              fP
        unsigned int iFrame = frame0 % maxFrame;
        current = static_cast<Float>(inputChannel[iFrame*nPixelPerFrame + id]);
        if (!simpleContrast) {
            if (ave_tau > framePhase0) { // then there is the ending frame, frame1, and frames inbetween to be considered

                _float add_t;
                iFrame = frame1 % maxFrame;
                _float last = static_cast<_float>(inputChannel[iFrame*nPixelPerFrame + id]);
                _float mid = 0.0;
                for (unsigned int frame = frame1 + 1; frame < frame0; frame++) {
                    iFrame = frame % maxFrame;
                    mid += static_cast<_float>(inputChannel[iFrame*nPixelPerFrame + id]);
                }

                add_t = framePhase0 + framePhase1 + (frame0-frame1-1)*tPerFrame;
                average = (current*framePhase0 + mid*tPerFrame + last*framePhase1)/add_t;
            } else {
                // averaging windows ends within framePhase0
                average = current;
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
            *output = c;
        } else {
            *output = 2*current - 1;
        }
    }
}
// to-do:
// 1. use non-uniform temporal filtering to mimick cone behavior, is it necessary?

int main(int argc, char **argv) {
    cudaDeviceProp deviceProps;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

#ifdef SINGLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    

    std::ofstream fplane, fretina, fcontrast, fLGN_fr, fLGN_convol, fmax_convol;
    fplane.open("3xplane.bin", std::ios::out | std::ios::binary);
    fretina.open("3xretina.bin", std::ios::out | std::ios::binary);
    fcontrast.open("contrast.bin", std::ios::out | std::ios::binary);
    fLGN_fr.open("LGNfr.bin", std::ios::out | std::ios::binary);
    fLGN_convol.open("LGN_convol.bin", std::ios::out | std::ios::binary);
    fmax_convol.open("max_convol.bin", std::ios::out | std::ios::binary);

    Float init_luminance = 2.0/6.0; //1.0/6.0;

    // from the retina facing out
    const Float toRad = M_PI/180.0f;
    Float ecc; 

    Size width;
    Size height;
	Size nLGN_x;
	Size nLGN_y;
    Size nSpatialSample1D; // spatial kernel sample size = nSpatialSample1D x nSpatialSample1D
    PosInt nsig = 3;
    Float tau = 256.0f; // required length of memory for LGN temporal kernel
    Float Itau = 300.0f; // in ms .. cone adaptation at 300ms https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289
    PosInt frameRate; // Hz
    Float dt; // in ms, better in fractions of binary 
    unsigned int nt;
    PosInt nKernelSample; // kernel sampling
    char tmpStr[101]
        Itau
        tau
        frameRate
        dt
    sscanf(argv[argc-6], "%u", &nt);
	sscanf(argv[argc-5], "%u", &nLGN_x);
	sscanf(argv[argc-4], "%u", &nLGN_y);
    sscanf(argv[argc-3], "%u", &nKernelSample);
    sscanf(argv[argc-2], "%u", &nSpatialSample1D);
    sscanf(argv[argc-1], "%100s", &tmpStr);
    bool simpleContrast = true; // no cone adaptation if set to true
    PosInt stepRate = round(1000/dt);
    if (round(1000/dt) != 1000/dt) {
        cout << "stepRate = 1000/dt has to be a integer.\n";
        return EXIT_FAILURE;
    }
    if (frameRate > stepRate) {
        cout << "stepRate cannot be smaller than frameRate, 1000/dt.\n";
        return EXIT_FAILURE;
    }

    Size nPixelPerFrame = width*height;
    Float dy = 2*tan(ecc)/height;
    Float dx = 2*tan(ecc)/width;
    
    Size local_width = width/8;
    Size local_height = height/8;

    string LGN_prop_file = tmpStr;
    printf("simulating for %u steps, t = %f ms\n", nt, nt*dt);
    Size nLGN = nLGN_x * nLGN_y;
    // setup LGN here
    LGN_parameter hLGN(nLGN);

    // set test param for LGN subregion RF kernel 
    /*
        Group the same cone-specific types together for warp performance
		Check ./patchV1/LGN_kernel.ipynb
    */
    for (unsigned int i=0; i<nLGN; i++) {
		// subject to function of eccentricity
		hLGN.center.x[i] = 0.5f/width + (i%nLGN_x) * (1.0f-1.0f/width)/(nLGN_x-1);
		hLGN.center.y[i] = 0.5f/height + (i/nLGN_y) * (1.0f-1.0f/height)/(nLGN_y-1);
        hLGN.center.rx[i] = 0.015;		
		hLGN.center.ry[i] = 0.015;

		// subject to some distribution
		hLGN.center.k[i] = 16.5 * (i % 4 < 2 ? 1 : -1); // on - off center
        hLGN.center.tauR[i] = 5.2/1.55;
        hLGN.center.tauD[i] = 17/1.55;
        hLGN.center.nR[i] = 10;
        hLGN.center.nD[i] = 6;
		hLGN.center.delay[i] = 3.5+2.95;
        hLGN.center.ratio[i] = 25.9/16.5;

		// subject to function of eccentricity
		hLGN.surround.x[i] = hLGN.center.x[i];
		hLGN.surround.y[i] = hLGN.center.y[i];
        hLGN.surround.rx[i] = 0.03;  	// function of eccentricity
		hLGN.surround.ry[i] = 0.03;

		// subject to some distribution
        hLGN.surround.k[i] = -copy(6.4, hLGN.center.k[i]);
        hLGN.surround.tauR[i] = 8;
        hLGN.surround.tauD[i] = 14;
        hLGN.surround.nR[i] = 13;
        hLGN.surround.nD[i] = 6;
		hLGN.surround.delay[i] = 3.5+1.21+15;
		hLGN.surround.ratio[i] = 17.6/6.4;

        hLGN.covariant[i] = 0.53753461391295254; //between L and M, significantly overlap in cone response curve
        hLGN.centerType[i] = i%2;
        hLGN.surroundType[i] = (i+1)%2;

        hLGN.logistic.spont[i] = 0.0;
        hLGN.logistic.c50[i] = 0.5;
        hLGN.logistic.sharpness[i] = 1.0;
    }

    LGN_parameter dLGN(nLGN, hLGN);
    // finish LGN setup
    if (nSpatialSample1D > 32) {
        cout << "nSpatialSample1D has to be smaller than 32 (1024 threads per block).\n"
        return EXIT_FAILURE;
    }

    // set params for layerd texture memory
    init_layer(L_retinaConSig);
    init_layer(M_retinaConSig);
    init_layer(S_retinaConSig);

    init_layer(LMS_frame);

    Size nRetrace = static_cast<Size>(round(tau/dt));
    if (round(tau/dt) != tau/dt) {
        cout << "tau should be divisible by dt.\n"; 
        return EXIT_FAILURE;
    }
    if (nRetrace % nKernelSample != 0) {
        cout << "tau in #dt should be divisible by nKernelSample.\n"; 
        return EXIT_FAILURE;
    }
    Float kernelSampleRate = static_cast<PosInt>(nKernelSample/tau*1000);

    PosInt kernelSampleInterval = nRetrace/nKernelSample;
    Float kernelSampleT0 = (kernelSampleInterval/2)*dt; 
    Float kernelSampleDt = sampleInterval*dt;

    printf("temporal kernel retraces %f ms, samples %u points, sample rate = %u Hz\n", tau, nKernelSample, kernelSampleRate);

    PosInt maxFrame = static_cast<PosInt>(ceil(tau/1000 * frameRate) + 1; // max frame need to be stored in texture for temporal convolution with the LGN kernel.
    //  |---|--------|--|
    
    printf("temporal kernel retrace (tau) = %f ms needs at most %u frames\n", tau, maxFrame);
	_float tPerFrame = 1000.0f / frameRate; //ms
    printf("~ %f plusminus 1 kernel sample per frame\n", tPerFrame/kernelSampleDt);

    // calculate phase difference between sampling point and next-frame point  
    bool moreDt = true;
    if (stepRate < frameRate) {
        moreDt = false;
        printf("frameRate > simulation rate!!");
    }
    // norm/denorm = the normalized advancing phase at each step, 
    // denorm is the mininum number of frames such that total frame length is a multiple of dt.
    PosInt norm;
    PosInt denorm = find_denorm(frameRate, stepRate , moreDt, norm);
    PosInt *exact_norm = new PosInt[denorm]; // to be normalized by denorm to form normalized phase
    PosInt current_norm = 0;
    printf("%u exact phases in [0,1]: ", denorm);
    for (PosInt i=0; i<denorm; i++) {
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

    PosInt *exact_it = new PosInt[denorm];
    for (PosInt i=0; i<denorm; i++) {
        exact_it[i] = (i*stepRate)/frameRate; // => i*Tframe/Tdt
        printf("i == %f", (exact_it[i] + (Float)exact_norm[i]/denorm)*dt*frameRate);
    }
    // i frames' length in steps = exact_it[i] + exact_norm[i]/denorm
    assert((stepRate*denorm) % frameRate == 0);
    unsigned int co_product = (stepRate*denorm)/frameRate; 
    // the number of minimum steps to meet frameLength * denorm with 0 phase

    unsigned int nChannel = 3; // L, M, S
    // one cudaArray per channel
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArr_L;
    cudaArray *cuArr_M;
    cudaArray *cuArr_S;
    // allocate cudaArrays on the device
    checkCudaErrors(cudaMalloc3DArray(&cuArr_L, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_M, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_S, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));

    cudaArray *cuArr_frame;
    checkCudaErrors(cudaMalloc3DArray(&cuArr_frame, &channelDesc, make_cudaExtent(width, height, nChannel), cudaArrayLayered));

    // bind texture to cudaArrays
    checkCudaErrors(cudaBindTextureToArray(L_retinaConSig,  cuArr_L, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_retinaConSig,  cuArr_M, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_retinaConSig,  cuArr_S, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(LMS_frame,  cuArr_frame, channelDesc));

    float* LMS; // memory head

    // LMS retina projection intensity array of [maxFrame] frames from t - (tau + ave_tau) on device
    float* __restrict__ dL;
    float* __restrict__ dM;
    float* __restrict__ dS;
    // LMS contrast signal on device
    Float* __restrict__ cL;
    Float* __restrict__ cM;
    Float* __restrict__ cS;

    // allocate the memory for a video of maxFrame with channels, and the memory for the contrast signal extracted from the video by sampling
    checkCudaErrors(cudaMalloc((void **) &LMS, nPixelPerFrame*sizeof(float)*3*maxFrame + nPixelPerFrame*sizeof(Float)*3*nKernelSample));

    dL = LMS;
    dM = dL + nPixelPerFrame*maxFrame;
    dS = dM + nPixelPerFrame*maxFrame;

    cL = (Float*) (dS + nPixelPerFrame*maxFrame);
    cM = cL + nPixelPerFrame*nKernelSample;
    cS = cM + nPixelPerFrame*nKernelSample;

    Float* LGN_fr = new _float[nLGN];
    Float* __restrict__ d_LGN_fr;
    Float* __restrict__ max_convol;
    checkCudaErrors(cudaMalloc((void **) &d_LGN_fr, nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &max_convol, nLGN*sizeof(Float)));

    // initialize average to normalized mean luminnace
    cudaStream_t s0, s1, s2;
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaStreamCreate(&s2));

    Float init_contrast = init_luminance;
    dim3 initBlock(blockSize, 1, 1);
    dim3 initGrid((nPixelPerFrame*nKernelSample + blockSize-1) / blockSize, 1, 1);
    init<<<initGrid, initBlock, 0, s0>>>(cL, init_contrast, nPixelPerFrame * nKernelSample);
    init<<<initGrid, initBlock, 0, s1>>>(cM, init_contrast, nPixelPerFrame * nKernelSample);
    init<<<initGrid, initBlock, 0, s2>>>(cS, init_contrast, nPixelPerFrame * nKernelSample);

    prep_sample(0, width, height, cL, cM, cS, cuArr_L, cuArr_M, cuArr_S, nKernelSample);

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
    dim3 sampleGrid((nPixelPerFrame + blockSize - 1)/blockSize, nChannel, 1);

    // determine the maximums of LGN kernel convolutions
    LGN_maxResponse<<<convolGrid, convolBlock, sizeof(_float)*2*nKernelSample>>>(max_convol, dLGN, nKernelSample-1, kernelSampleDt, nsig, nSpatialSample1D);
    // calc LGN firing rate at the end of current dt
    unsigned int currentFrame = 0; // current frame number from stimulus
    unsigned int iFrame = 0; //latest frame inserted into the dL dM dS,  initialization not necessary
    unsigned int iPhase = 0;
    unsigned int iPhase_old = 0; // initialization not necessary
    Float framePhase;
    unsigned int jt = 0;

    unsigned int currentSample = 0; // current frame number from stimulus
    unsigned int iSample = 0; //latest frame inserted into the dL dM dS,  initialization not necessary
    unsigned int kPhase = 0;
    unsigned int kPhase_old = 0; // initialization not necessary
    Float samplePhase;
    unsigned int kt = 0;

    for (unsigned int it = 0; it < nt; it++) {

        _float t = it*dt;

        // next frame comes between (t, t+dt), read and store frame to texture memory
        if (it+1 > (currentFrame/denorm)*co_product + exact_it[iPhase]) {
            jt = it; // starting it for the current frame
            iFrame = currentFrame % maxFrame;
            // TODO: read the new frame
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
            printf("frame #%i prepared at t = %f, in (%f,%f) ~ %.3f%%.\n", currentFrame, currentFrame*tPerFrame, t, t+dt, exact_norm[iPhase]*100.0f/denorm);
            currentFrame++;
            iPhase_old = iPhase;
            iPhase = (iPhase + 1) % denorm;
        }
        //->|  |<-exact_norm/denorm
        //  |--|------frame------|
        //  |----|----|
        // jt   jt+1  it
        framePhase = ((it - jt + 1)*denorm - exact_norm[iPhase_old])/denorm*dt; // current frame length shown till t+dt

        // next kernel sample, use intensity to contrast to get contrast signal
        if (it+1 > (currentSample/kernel_denorm)*co_kernel_product + exact_kernel_it[kPhase]) {
            kt = it;
            iSample = currentSample % nKernelSample;
    
            unsigned int frame0, frame1;
            _float framePhase0, framePhase1;
            assert(ave_tau > tPerFrame);
            framePhase0 = framePhase;
            frame0 = iFrame + maxFrame;
            if (ave_tau > framePhase0) { 
                // then there is the ending frame, frame1 to be consider
                frame1 = iFrame + maxFrame - static_cast<unsigned int>(ceil((ave_tau-framePhase)/tPerFrame));
                framePhase1 = fmod(ave_tau - framePhase, tPerFrame);
            }

            intensity_to_contrast<<<sampleGrid, spatialBlock>>>(LMS, maxFrame, nPixelPerFrame, nKernelSample, tPerFrame, ave_tau, frame0, frame1, framePhase0, framePhase1, simpleContrast);
		    getLastCudaError("intensity_to_contrast failed");
            checkCudaErrors(cudaMemcpy(L, cL, nPixelPerFrame*sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaEventRecord(i2));
            checkCudaErrors(cudaEventSynchronize(i2));
            fcontrast.write((char*)L, nPixelPerFrame*sizeof(float));
		    // copy contrast signals to texture
            prep_sample(iSample, width, height, cL, cM, cS, cuArr_L, cuArr_M, cuArr_S, 1);

            currentSample++;
            kPhase_old = kPhase;
            kPhase = (kPhase + 1) % kernel_denorm;
        }
        samplePhase = (((it - kt + 1)*kernel_denorm - exact_kernel_norm[kPhase_old])*dt)/kernel_denorm;

        // perform kernel convolution with built-in texture interpolation
        LGN_convol<<<convolGrid, convolBlock, sizeof(_float)*2*nKernelSample>>>(d_LGN_fr, dLGN, iSample, framePhase, nKernelSample, kernelSampleDt, nsig, nSpatialSample1D);
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
    delete []exact_kernel_norm;
    delete []exact_kernel_it;
    
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
