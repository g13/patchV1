#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <fenv.h>
#include <boost/program_options.hpp>

#include "connect.h"
#include "../types.h"
#include "../util/po.h" // custom validator for reading vector in the configuration file
#include "../util/util.h"

#define nFunc 2
const Size pPerFeature = 2;
const Float defaultFeatureValue[4]{0.5, 0.5, 0, 0};

__device__ __host__
Float ODpref(Float post, Float pre, Float r1, Float r2) { // ocular dominance
    Float p;
    if (post*pre < 0) {
        p = r1;
    } else {
        p = r2;
    }
    return p; 
}
__device__ pFeature p_OD = ODpref;

__device__
Float OPpref(Float post, Float pre, Float k, Float b) { // orientation preference
	Float p;
	if (k > 0) {
    	Float dp = post - pre;
    	if (abs(dp) > M_PI/2) {
    	    dp += copyms(M_PI, -dp);
    	}
    	//Float sig = M_PI/r; // set spread here
    	//Float A = square_root(2*M_PI);
    	//p = exponential(-dp*dp/(2*sig*sig))/(A*sig);
		Float base = boostOri[iType*2 + 0];
		Float amplitude = 1-boostOri[iType*2 + 0];
		Float vonMisesAmp = 1-exponential(-2*boostOri[iType*2 + 1]);
        boost = base + amplitude * (exponential(boostOri[iType*2 + 1]*(cosine(dOri*M_PI*2)-1))-1+vonMisesAmp)/vonMisesAmp; // von Mises

	} else {
		p = 1;
	}
    return p; 
}
__device__ pFeature p_OP = OPpref;

__device__
Float pass(Float post, Float pre, Float r) {
    return 1.0;
}
__device__ pFeature p_pass = pass;

__device__ __constant__ pFeature pref[nFunc];

void initializePreferenceFunctions(Size nFeature) {
    pFeature h_pref[nFunc];
    if (nFunc != nFeature) {
        std::cout << "number of preference functions = " << nFunc << ", is inconsistent with the number of the features: " << nFeature << "\n";
        assert(nFunc == nFeature);
    }
    checkCudaErrors(cudaMemcpyFromSymbol(&h_pref[0], p_OD, sizeof(pFeature), 0, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_pref[1], p_OP, sizeof(pFeature), 0, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyToSymbol(pref, h_pref, nFunc*sizeof(pFeature), 0, cudaMemcpyHostToDevice));
}

void read_LGN_V1_stats(std::string filename, Float sLGN[], Size nLGN[], vector<PosInt> &inputLayer, vector<Size> &networkSizeAcc) {
	std::ifstream input_file;
	input_file.open(filename, std::fstream::in | std::fstream::binary);
	if (!input_file) {
		std::string errMsg{ "Cannot open or find " + filename + "\n" };
		throw errMsg;
	}
    Size nInputV1;
    Size max_nLGN;
    input_file.read(reinterpret_cast<char*>(&nInputV1), sizeof(Size));
    input_file.read(reinterpret_cast<char*>(&max_nLGN), sizeof(Size));
    Size _n = 0;
    for (int i=0;i<inputLayer.size(); i++) {
        _n += networkSizeAcc[inputLayer[i]+1] - networkSizeAcc[inputLayer[i]];
    }
    assert(_n == nInputV1);
    PosInt currentLayer = inputLayer[0];
    Float *sInputLGN = new Float[max_nLGN];
    Size n;
    PosInt index = networkSizeAcc[currentLayer];
    for (PosInt i=0; i<nInputV1; i++) {
        input_file.read(reinterpret_cast<char*>(&n), sizeof(Size));
		input_file.read(reinterpret_cast<char*>(sInputLGN), n * sizeof(Float));
        sLGN[index] = *accumulate(sInputLGN, sInputLGN+n, 0.0);
        nLGN[index] = n;
        if (print) {
            std::cout << i << ": ";
            for (PosInt j=0; j<listSize; j++) {
                std::cout << array[i*maxList + j];
                if (j == listSize-1) std::cout << "\n";
                else std::cout << ", ";
            }
        }
        index++;
        if (index == networkSizeAcc[currentLayer+1]) {
            currentLayer++;
            index = networkSizeAcc[currentLayer];
        }
    }
    delete []sInputLGN;
	input_file.close();
}

void read_LGN_V1(std::string filename, Size nLGN_V1[], Size nLGN_V1_Max[], Size typeAcc[], Size nType) {
    std::ifstream input_file;
    input_file.open(filename, std::fstream::in | std::fstream::binary);
    if (!input_file) {
        std::string errMsg{ "Cannot open or find " + filename + "\n" };
        throw errMsg;
    }
    Size nList, maxList;
    input_file.read(reinterpret_cast<char*>(&nList), sizeof(Size));
    input_file.read(reinterpret_cast<char*>(&maxList), sizeof(Size));

    Float *array = new Float[nList*maxList];
    for (PosInt i=0; i<nType; i++) {
        //nLGN_V1Mean[i] = 0.0;
        nLGN_V1_Max[i] = 0.0;
    }
    for (PosInt i=0; i<nList; i++) {
        Size listSize;
        input_file.read(reinterpret_cast<char*>(&listSize), sizeof(Size));
		nLGN_V1[i] = listSize;
        input_file.read(reinterpret_cast<char*>(&array[i*maxList]), listSize * sizeof(Float));
        PosInt k = i%blockSize;
        for (PosInt j = 0; j<nType; j++) {
            if (k<typeAcc[j]) {
                k = j;
                break;
            }
        }
        if (nLGN_V1[i] > nLGN_V1_Max[k]) {
            nLGN_V1_Max[k] = nLGN_V1[i];
        }
    }
    delete []array;
    input_file.close();
}
