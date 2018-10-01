#include <iostream>
using std::cout;
using std::endl;
void printDevProp(cudaDeviceProp devProp)
{
    cout << "Major revision number:         " <<  devProp.major << endl;
    cout << "Minor revision number:         " <<  devProp.minor << endl;
    cout << "Name:                          " <<  devProp.name << endl;
    cout << "Total global memory:           " <<  devProp.totalGlobalMem << endl;
    cout << "Total shared memory per block: " <<  devProp.sharedMemPerBlock << endl;
    cout << "Total registers per block:     " <<  devProp.regsPerBlock << endl;
    cout << "Warp size:                     " <<  devProp.warpSize << endl;
    cout << "Maximum memory pitch:          " <<  devProp.memPitch << endl;
    cout << "Maximum threads per block:     " <<  devProp.maxThreadsPerBlock << endl;
    for (int i = 0; i < 3; ++i)
        cout << "Maximum dimension " << i << " of block:  " << devProp.maxThreadsDim[i] << endl;
    for (int i = 0; i < 3; ++i) 
        cout << "Maximum dimension " << i << " of grid:   " << devProp.maxGridSize[i] << endl;
    cout << "Clock rate:                    " <<  devProp.clockRate << endl;
    cout << "Total constant memory:         " <<  devProp.totalConstMem << endl;
    cout << "Texture alignment:             " <<  devProp.textureAlignment << endl;
    cout << "Concurrent copy and execution: " <<  (devProp.deviceOverlap ? "Yes" : "No") << endl;
    cout << "Number of multiprocessors:     " <<  devProp.multiProcessorCount << endl;
    cout << "Kernel execution timeout:      " <<  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << endl;
    return;
}
