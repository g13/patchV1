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
#include <boost/filesystem.hpp>

#include "connect.h"
#include "../types.h"
#include "../util/po.h" // custom validator for reading vector in the configuration file
#include "../util/util.h"

#define nFunc 2
const Size pPerFeature = 2;
const Float defaultFeatureParameter[4]{0.5, 0.5, 0, 0};
using std::vector;

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
		Float amp = 1-exponential(-2*k);
        p = b + (1-b) * (exponential(k*(cosine(dp*M_PI*2)-1))-1+amp)/amp; // von Mises
	} else {
		p = 0;
	}
    return p; 
}
__device__ pFeature p_OP = OPpref;

__device__
Float pass(Float post, Float pre, Float r, Float s) {
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

void read_LGN_V1(std::string sFile, std::string idFile, Float nLGN_eff[], vector<PosInt> &inputLayer, vector<Size> &networkSizeAcc, vector<Float> &synOccupyRaito, vector<Size> &mL, vector<Size> &mR, Size mLayer, bool print) {
    // typeAcc
    vector<PosInt> typeAcc;
    vector<PosInt> typeID;
    assert(mL.size() == mR.size());
    assert(synOccupyRaito.size() == mLayer);
    PosInt id = 0; 
    for (PosInt i=0; i < mLayer; i++) {
        typeID.push_back(i);
        id += mL[i];
        typeAcc.push_back(id);
    }
    for (PosInt i=0; i < mLayer; i++) {
        typeID.push_back(i);
        id += mR[i];
        typeAcc.push_back(id);
    }
    // open files
	std::ifstream fs, fid;
	fs.open(sFile, std::fstream::in | std::fstream::binary);
	if (!fs) {
		std::string errMsg{ "Cannot open or find " + sFile + "\n" };
		throw errMsg;
	}
	fid.open(idFile, std::fstream::in | std::fstream::binary);
	if (!fs) {
		std::string errMsg{ "Cannot open or find " + idFile + "\n" };
		throw errMsg;
	}

    Size nInputV1;
    Size max_nLGN;
    fs.read(reinterpret_cast<char*>(&nInputV1), sizeof(Size));
    Size _nInputV1;
    fid.read(reinterpret_cast<char*>(&_nInputV1), sizeof(Size));
    assert(_nInputV1 == nInputV1);
    fs.read(reinterpret_cast<char*>(&max_nLGN), sizeof(Size));
    Size _n = 0;
    for (int i=0;i<inputLayer.size(); i++) {
        _n += networkSizeAcc[inputLayer[i]+1] - networkSizeAcc[inputLayer[i]];
    }
    assert(_n == nInputV1);
    PosInt currentLayer = inputLayer[0];
    Float *sInputLGN = new Float[max_nLGN];
    Float *idInputLGN = new Float[max_nLGN];
    Size n;
    PosInt index = networkSizeAcc[currentLayer];
    for (PosInt i=0; i<nInputV1; i++) {
        fs.read(reinterpret_cast<char*>(&n), sizeof(Size));
        fid.read(reinterpret_cast<char*>(&_n), sizeof(Size));
        assert(_n == n);
		fs.read(reinterpret_cast<char*>(sInputLGN), n * sizeof(Float));
		fid.read(reinterpret_cast<char*>(idInputLGN), n * sizeof(PosInt));
        nLGN_eff[index] = 0;
        for (PosInt j=0; j<n; j++) {
            for (PosInt k=0; k<typeAcc.size(); k++) {
                if (idInputLGN[j] < typeAcc[k]) {
                    nLGN_eff[index] += sInputLGN[j]*synOccupyRaito[k];
                    break;
                }
            }
        }
        if (print) {
            std::cout << i << ": ";
            for (PosInt j=0; j<n; j++) {
                PosInt type;
                for (PosInt k=0; k<typeAcc.size(); k++) {
                    if (idInputLGN[j] < typeAcc[k]) {
                        type = k;
                        break;
                    }
                }
                std::cout << sInputLGN[j] << "(" << type << ")";
                if (j == n-1) std::cout << "\n";
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
    delete []idInputLGN;
	fs.close();
	fid.close();
}

/*void read_LGN_V1(std::string filename, Size nLGN_V1[], Size nLGN_V1_Max[], Size typeAcc[], Size nType) {
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
}*/
