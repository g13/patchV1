#include "cudaDeviceQuery.h"
int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    cout << "CUDA Device Query..." << endl;
    cout << "There are " << devCount << " CUDA devices" << endl;
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        cout << endl << "CUDA Device #" << i << endl;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
}
