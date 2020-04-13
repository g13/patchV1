#define _USE_MATH_DEFINES
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>
#include <chrono>
#include "boost/program_options.hpp"
#include "LGN_props.cuh"
#include "LGN_props.h"
#include "condShape.h"
#include "discrete_input_convol.cuh"
#include "coredynamics.cuh"
#include "stats.cuh"
#include "util/util.h"
#include "util/po.h"
#include "preprocess/RFtype.h"
#include "global.h"
#include "MACRO.h"

inline void read_LGN(std::string filename, Float* &array, Size &maxList, Float s_ratio, bool pinMem, bool print) {
	std::ifstream input_file;
	input_file.open(filename, std::fstream::in | std::fstream::binary);
	if (!input_file) {
		std::string errMsg{ "Cannot open or find " + filename + "\n" };
		throw errMsg;
	}
    Size nList;
    input_file.read(reinterpret_cast<char*>(&nList), sizeof(Size));
    input_file.read(reinterpret_cast<char*>(&maxList), sizeof(Size));
    size_t arraySize = nList*maxList;
    if (pinMem) {
        checkCudaErrors(cudaMallocHost((void**) &array, arraySize*sizeof(Float)));
    } else {
        array = new Float[arraySize];
    }
    for (PosInt i=0; i<nList; i++) {
        Size listSize;
        input_file.read(reinterpret_cast<char*>(&listSize), sizeof(Size));
		input_file.read(reinterpret_cast<char*>(&array[i*maxList]), listSize * sizeof(Float));
		if (s_ratio != 1) {
			for (PosInt j=0; j<listSize; j++) {
				array[i*maxList + j] *= s_ratio;
			}
		}
        if (print) {
            std::cout << i << ": ";
            for (PosInt j=0; j<listSize; j++) {
                std::cout << array[i*maxList + j];
                if (j == listSize-1) std::cout << "\n";
                else std::cout << ", ";
            }
        }
    }
	input_file.close();
}

inline int checkGMemUsage(size_t usingGMem, size_t GMemAvail) {
	if (usingGMem > GMemAvail) {
		std::cout << usingGMem / 1024.0 / 1024.0 << " > GMem available on device: " << GMemAvail << "\n";
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}
// the retinal discrete x, y as cone receptors id
inline void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
	layer.addressMode[0] = cudaAddressModeBorder;
	layer.addressMode[1] = cudaAddressModeBorder; 
	layer.filterMode = cudaFilterModeLinear;
	layer.normalized = true; //accessing coordinates are normalized
}

void prep_sample(unsigned int iSample, unsigned int width, unsigned int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, unsigned int nSample, cudaMemcpyKind cpyKind) {
	// copy the three channels L, M, S of the #iSample frame to the cudaArrays dL, dM and dS
	cudaMemcpy3DParms params = {0};
	params.srcPos = make_cudaPos(0, 0, 0);
	params.dstPos = make_cudaPos(0, 0, iSample);
	// if a cudaArray is involved width is element not byte size
	params.extent = make_cudaExtent(width, height, nSample);
	params.kind = cpyKind;

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
