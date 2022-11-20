#define _USE_MATH_DEFINES
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_functions.h> // include cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>

int main(int argc, char** argv) {
	using namespace std;
	int nDevice;
	checkCudaErrors(cudaGetDeviceCount(&nDevice));
	// cout << nDevice << " gpu on the node\n";
	//int iDevice;
	size_t maxFree = 0;
	for (int i = 0; i < nDevice; i++) {
		checkCudaErrors(cudaSetDevice(i));
		size_t free;
		size_t total;
		checkCudaErrors(cudaMemGetInfo(&free, &total));
		if (free > maxFree) {
			//iDevice = i;
			maxFree = free;
		}
	}
    int requiredMem_Mb;
    if (argc > 1) {
        requiredMem_Mb = atoi(argv[1]);
    }
    if (requiredMem_Mb == 0 || argc == 1) { // default
        requiredMem_Mb = 256;
    }
    if (maxFree < 1024*1024*requiredMem_Mb) {
        cout << "insufficient memory\n";
		cout << 1;
        return 1;
    } else {
		cout << 0;
        return 0;
    }
}
