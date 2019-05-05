#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check


texture<float, cudaTextureType2DLayered> L_retinaProj;
texture<float, cudaTextureType2DLayered> M_retinaProj;
texture<float, cudaTextureType2DLayered> S_retinaProj;

void next_layer(int ilayer, int width, int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, int nlayer = 1) { 
    cudaMemcpy3DParms params = {0};
    params.srcPos = make_cudaPos(0,0,0);
    params.dstPos = make_cudaPos(0, 0, ilayer);
    params.extent = make_cudaExtent(width, height, nlayer);
    params.kind = cudaMemcpyHostToDevice;

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

__global__ void plane_to_retina(/*_float* __restrict__ x,
                      _float* __restrict__ y,
                      vInput v*/ int width, int height,
                      int jlayer) 
{
    unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
    float x = static_cast<float>(ix)/width;
    float y = static_cast<float>(iy)/height;

    // x, y as on the retina
    // coord(x,y) 0,0 at fovea positive value pointing towards left and bottom (as in the image would be top and left)

    // 1,-1  0,-1  -1,-1
    // 
    // 1,0   0,0   -1,0
    // 
    // 1,1   0,1   -1,1

    // id 0,0 at top right

    if (x == 0.0 || x == 1.0) {
        float testpt = tex2DLayered(L_retinaProj, x, y, jlayer);
        printf("[%i,(%i-%f,%i-%f)]: %f\n", jlayer, x, y, testpt);
    }
    //x[id] = retina_radius*atan(v.xl + (v.xr -v.xl)*id/v.nx, distance);
    //y[id-v.nx] = retina_radius*atan(v.yb + (v.yt -v.yb)*(id-v.nx)/v.ny, distance);
}


int main(int argc, char **argv) {
    
    cudaDeviceProp deviceProps;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

    if (deviceProps.major < 2)
    {
        printf("Surface requires SM >= 2.0 to support Texture Arrays.  Test will be waived... \n");
        cudaDeviceReset();
        exit(EXIT_WAIVED);
    }

// set params for layerd texture
    L_retinaProj.addressMode[0] = cudaAddressModeWrap;
    L_retinaProj.addressMode[1] = cudaAddressModeWrap;
    L_retinaProj.filterMode = cudaFilterModeLinear;
    L_retinaProj.normalized = true;  // access with normalized texture coordinates
    M_retinaProj.addressMode[0] = cudaAddressModeWrap;
    M_retinaProj.addressMode[1] = cudaAddressModeWrap;
    M_retinaProj.filterMode = cudaFilterModeLinear;
    M_retinaProj.normalized = true;  // access with normalized texture coordinates
    S_retinaProj.addressMode[0] = cudaAddressModeWrap;
    S_retinaProj.addressMode[1] = cudaAddressModeWrap;
    S_retinaProj.filterMode = cudaFilterModeLinear;
    S_retinaProj.normalized = true;  // access with normalized texture coordinates
    // readin plane data
    unsigned int width = 16;
    unsigned int height = 16;
    unsigned int num_layers = 16;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *timeLayeredPlaneL;
    cudaArray *timeLayeredPlaneM;
    cudaArray *timeLayeredPlaneS;
    checkCudaErrors(cudaMalloc3DArray(&timeLayeredPlaneL, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&timeLayeredPlaneM, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&timeLayeredPlaneS, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered));

    // fill with initial layers
    size_t size = width*height*num_layers*sizeof(float);
    float* __restrict__ L = new float[size];
    float* __restrict__ M = new float[size];
    float* __restrict__ S = new float[size];
    for (int ilayer = 0; ilayer < num_layers; ilayer++) {
        for (int ih = 0; ih < height; ih++) {
            for (int iw = 0; iw < width; iw++) {
                if (ilayer == 0) {
                    L[width*height*ilayer + ih*width + iw] = iw/width + ih/height + ilayer;
                    M[width*height*ilayer + ih*width + iw] = 2*(iw/width + ih/height + ilayer);
                    S[width*height*ilayer + ih*width + iw] = 3*(iw/width + ih/height + ilayer);
                } else {
                    L[width*height*ilayer + ih*width + iw] = 0;
                    M[width*height*ilayer + ih*width + iw] = 0;
                    S[width*height*ilayer + ih*width + iw] = 0;
                }
            }
        }
    }
    next_layer(0, width, height, L, M, S, timeLayeredPlaneL, timeLayeredPlaneM, timeLayeredPlaneS, num_layers);

    checkCudaErrors(cudaBindTextureToArray(L_retinaProj, timeLayeredPlaneL, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_retinaProj, timeLayeredPlaneM, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_retinaProj, timeLayeredPlaneS, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    
    plane_to_retina<<< dimGrid, dimBlock, 0 >>>(width, height, 0);

    getLastCudaError("#0 Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    for (unsigned int ilayer = 1; ilayer < num_layers; ilayer++) {
        for (int ih = 0; ih < height; ih++) {
            for (int iw = 0; iw < width; iw++) {
                L[ih*width + iw] = iw/width + ih/height;
                M[ih*width + iw] = 2*(iw/width + ih/height);
                S[ih*width + iw] = 3*(iw/width + ih/height);
            }
        }
        int jlayer = ilayer % num_layers;
        next_layer(jlayer, width, height, L, M, S, timeLayeredPlaneL, timeLayeredPlaneM, timeLayeredPlaneS);
        plane_to_retina<<< dimGrid, dimBlock, 0 >>>(width, height, jlayer);
        getLastCudaError("#i>0 Kernel execution failed");
    }
    checkCudaErrors(cudaDeviceSynchronize());
    delete []L;
    delete []M;
    delete []S;
    checkCudaErrors(cudaFreeArray(timeLayeredPlaneL));
    checkCudaErrors(cudaFreeArray(timeLayeredPlaneM));
    checkCudaErrors(cudaFreeArray(timeLayeredPlaneS));
    cudaDeviceReset();
    return 0;
}

/*
__global__ void load(float *data, int nx, int ny) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    surf2Dwrite(data[y * nx + x], outputSurface, x*sizeof(float), y, cudaBoundaryModeTrap);
}*/

