#include "retinotopic_connections.h"
using namespace std;

/* 
    Purpose:
        Connect neurons with visual field centered at (eccentricity, polar) in retinotopic Sheet1 :pre-synaptically: to neurons in another retinotopic Sheet2, with visual field centered near the same spot.
*/

template<typename T>
void original2LR(vector<T> &original, vector<T> &seqByLR, vector<Int> LR, Size nL) {
    seqByLR.assign(LR.size(), 0);
    Size jL = 0;
    Size jR = 0;
    for (Size i = 0; i<LR.size(); i++) {
        if (LR[i] < 0) {
            seqByLR[jL] = original[i];
            jL++;
        } else{
            seqByLR[nL + jR] = original[i];
            jR++;
        }
    }
}
template<typename T>
void LR2original(vector<T> &seqByLR, vector<T> &original, vector<Int> LR, Size nL) {
    assert(seqByLR.size() == LR.size());
    Size jL = 0;
    Size jR = 0;
    for (Size i = 0; i<LR.size(); i++) {
        if (LR[i] < 0) {
            original.push_back(seqByLR[jL]);
            jL ++;
        } else {
            original.push_back(seqByLR[nL + jR]);
            jR ++;
        }
    }
    assert(seqByLR.size() == original.size());
}

Float get_acuity(Float ecc) {
    Float k = 0.20498;
    Float log_cpd0 = 3.67411;
    Float cpd = exponential(-k*ecc + log_cpd0);
    Float acuity = 1.0/cpd/4.0;
    return acuity;
}

vector<vector<Float>> retinotopic_connection(
        vector<vector<Size>> &poolList,
        RandomEngine &rGen,
        Float p_n_LGNeff,
        Size max_LGNeff,
        Float envelopeSig,
        const Size n,
		const pair<vector<Float>, vector<Float>> &cart, // V1 VF position (tentative)
        const pair<vector<Float>,vector<Float>> &cart0, // LGN VF position
		const vector<RFtype> &V1Type,
        const vector<Float> &theta,
        vector<Float> &phase,
        vector<Float> &sfreq,
        vector<Float> &modAmp_nCon,
        const vector<Float> &baRatio,
        const vector<Float> &a,
		vector<OutputType> &RefType,
        const vector<InputType> &LGNtype,
        vector<Float> &cx, // V1 VF position (final)
        vector<Float> &cy,
        const vector<Float> &ecc,
        Int SimpleComplex,
        Float conThres,
        Float ori_tol,
        Float disLGN,
		bool strictStrength,
		bool top_pick
) {
    uniform_real_distribution<Float> dist(0,1);
    vector<vector<Float>> srList;
    srList.reserve(n);
	// different V1 RF types ready for pointer RF to use
	LinearReceptiveField* RF;		
	NonOpponent_CS NO_CS;
	NonOpponent_Gabor NO_G;
	DoubleOpponent_CS DO_CS;
	DoubleOpponent_Gabor DO_G;
	SingleOpponent SO;
    // for each neuron i in sheet 2 at (e, p)
    for (Size i=0; i<n; i++) {
        Size m = poolList[i].size();
		if (m > 0) { // if the pool has LGN neuron
			// intermdiate V1 VF position (tentative)
			vector<Float> x, y;
			// LGN types of the pool
			vector<InputType> iType;
			// assert no need to reset at each end of loop
			//assert(x.size() == 0);
			//assert(y.size() == 0);
			//assert(iType.size() == 0);
			iType.reserve(m);
			x.reserve(m);
			y.reserve(m);
			assert(x.size() == 0);
			for (Size j = 0; j < m; j++) {
				x.push_back(cart0.first[poolList[i][j]]);
				y.push_back(cart0.second[poolList[i][j]]);
				iType.push_back(LGNtype[poolList[i][j]]);
			}
			// initialize the V1 neuron to the corresponding RF type
			switch (V1Type[i]) {
				case RFtype::nonOppopent_cs: 
					assert(RefType[i] == OutputType::LonMon || RefType[i] == OutputType::LoffMoff);
					RF = &NO_CS;
					break;
				case RFtype::nonOppopent_gabor: 
					assert(RefType[i] == OutputType::LonMon || RefType[i] == OutputType::LoffMoff);
					RF = &NO_G;
					break;
				case RFtype::doubleOppopent_cs: 
					assert(RefType[i] == OutputType::LonMoff || RefType[i] == OutputType::LoffMon);
					RF = &DO_CS;
					break;
				case RFtype::doubleOppopent_gabor: 
					assert(RefType[i] == OutputType::LonMoff || RefType[i] == OutputType::LoffMon);
					RF = &DO_G;
					break;
				case RFtype::singleOppopent: 
					assert(RefType[i] == OutputType::LonMoff || RefType[i] == OutputType::LoffMon);
					RF = &SO;
					break;
				default: throw "no such type of receptive field for V1 so defined (./src/preprocess/RFtype.h";
			}
			// construct pooled LGN neurons' connection to the V1 neuron
			vector<Float> strengthList;
            bool percentOrNumber = p_n_LGNeff < 0;
            if (percentOrNumber) {
                p_n_LGNeff = -p_n_LGNeff; 
            }
			Size maxN = max_LGNeff > poolList[i].size()? poolList.size(): max_LGNeff;
			if (conThres >= 0) {
				Float qfreq;
				RF->setup_param(m, sfreq[i], phase[i], 1.0, theta[i], a[i], baRatio[i], RefType[i], strictStrength, envelopeSig);
				//m = RF->construct_connection_thres(x, y, iType, poolList[i], strengthList, rGen, maxN, conThres, modAmp_nCon[i], qfreq, cart.first[i], cart.second[i], i);
				m = RF->construct_connection_opt(x, y, iType, poolList[i], strengthList, modAmp_nCon[i], qfreq, cart.first[i], cart.second[i], i, ori_tol, get_acuity(ecc[i])/a[i]*disLGN);
				sfreq[i] = qfreq;
			} else {
            	if (SimpleComplex == 0) {
				    RF->setup_param(m, sfreq[i], phase[i], modAmp_nCon[i], theta[i], a[i], baRatio[i], RefType[i], strictStrength, envelopeSig);
				    m = RF->construct_connection_N(x, y, iType, poolList[i], strengthList, rGen, p_n_LGNeff*1.0, percentOrNumber, maxN, top_pick);
            	} else {
				    RF->setup_param(m, sfreq[i], phase[i], 1.0, theta[i], a[i], baRatio[i], RefType[i], strictStrength, envelopeSig);
				    m = RF->construct_connection_N(x, y, iType, poolList[i], strengthList, rGen, p_n_LGNeff*modAmp_nCon[i], percentOrNumber, maxN, top_pick);
            	}
			}
            RefType[i] = RF->oType;
			srList.push_back(strengthList);
			if (m > 0) { 
				x.clear();
				y.clear();
				for (Size j = 0; j < m; j++) {
					x.push_back(cart0.first[poolList[i][j]]);
					y.push_back(cart0.second[poolList[i][j]]);
				}
				tie(cx[i], cy[i]) = average2D<Float>(x, y);
			} else {
				cx[i] = cart.first[i];
				cy[i] = cart.second[i];
			}
            phase[i] = RF->phase;
			// reset reusable variables
			RF->clear();
		} else {
			// keep tentative VF position
			cx[i] = cart.first[i];
			cy[i] = cart.second[i];
            phase[i] = 0;
			// empty list of connection strength, idList is already empty
			srList.push_back(vector<Float>());
		}
    }
    return srList;
}

Float norm_vec(Float x, Float y) {
    return sqrt(pow(x,2)+pow(y,2));
}
// *** Boundary cases are simply cut-off, assuming on calloscal connections not being activated under half-field stimulation. 
vector<Size> draw_from_radius(
        const Float x0,
        const Float y0, 
        const pair<vector<Float>,vector<Float>> &cart,
        const Size start,
        const Size end,
        const Float radius) {
    vector<Size> index;

    //Float mx = 0.0f;
    //Float my = 0.0f;
	//Float min_dis;
	//Size min_id;
    for (Size i=start; i<end; i++) {
		Float dis = norm_vec(cart.first[i] - x0, cart.second[i] - y0);
		/*if (i == 0) {
			min_dis = dis;
		}
		else {
			if (dis < min_dis) {
				min_id = i;
			}
		}*/
        if (dis < radius) {
            index.push_back(i);
			//mx = mx + cart.first[i];
			//my = my + cart.second[i];
        }
    }
	/*
	if (index.size() > 0) {
		cout << "(" << mx / index.size() << ", " << my / index.size() << ")\n";
	} else {
		cout << "receive no ff input\n";
	}
	*/
    return index;
}

__host__
__device__
__forceinline__
bool inside_ellipse(Float x, Float y, Float theta, Float a, Float b, Float &value) {
    Float tx = cosine(theta) * x + sine(theta) * y;
	Float ty = -sine(theta) * x + cosine(theta) * y;
	value = (tx*tx/(a*a) + ty*ty/(b*b));
	return value <= 1.0-1e-7;
}

__launch_bounds__(1024,2)
__global__
void vf_pool_CUDA( // for each eye
	Float* __restrict__ x,
	Float* __restrict__ y,
	Float* __restrict__ x0,
	Float* __restrict__ y0,
    Float* __restrict__ VFposEcc,
    Float* __restrict__ baRatio,
    Float* __restrict__ a,
    Float* __restrict__ theta,
    Size* __restrict__ poolList,
    Size* __restrict__ nPool,
    Size i0, Size n,
    Size j0, Size m,
    BigSize seed,
    Float LGN_V1_RFratio,
    Size maxLGNperV1pool // total LGN size to the eye
) {
    __shared__ Float sx[blockSize]; 
    __shared__ Float sy[blockSize]; 
    Size V1_id = i0 + blockDim.x*blockIdx.x + threadIdx.x;
    Size nPatch = (m + blockSize - 1)/blockDim.x;
    Size nRemain = m%blockDim.x;
	if (nRemain == 0) nRemain = blockDim.x;
    assert((nPatch-1)*blockSize + nRemain == m);
    curandStateMRG32k3a rGen;
    Float V1_x, V1_y;
    if (V1_id < i0+n) { // get radius
        Float ecc = VFposEcc[V1_id];
        Float baR = baRatio[V1_id];
        V1_x = x[V1_id];
        V1_y = y[V1_id];
		curand_init(seed + V1_id, 0, 0, &rGen);
        Float R = mapping_rule_CUDA(ecc*60.0, rGen, LGN_V1_RFratio)/60.0;
        // R = sqrt(area)
        Float sqr = square_root(M_PI*baR);
		Float local_a = R/sqr;
        a[V1_id] = local_a;
    }
    // scan and pooling LGN
    Size iPool = 0;
    for (Size iPatch = 0; iPatch < nPatch; iPatch++) {
        // load LGN pos to __shared__
        Size nLGN;
        if (iPatch < nPatch - 1 || threadIdx.x < nRemain) {
            Size LGN_id = j0 + iPatch*blockDim.x + threadIdx.x;
            sx[threadIdx.x] = x0[LGN_id];
            sy[threadIdx.x] = y0[LGN_id];
        }
        // put outside the loop to make sure nLGN is assigned for threads not used to assign shared sx,sy, because they still need to access those shared variable
        if (iPatch < nPatch - 1) {
            nLGN = blockDim.x;
        } else {
            nLGN = nRemain;
        }
        __syncthreads();
        // check for distance
        if (V1_id < i0+n) {
			#pragma unroll (16)
            for (Size i = 0; i < nLGN; i++) {
							// threadIdx.x roll over V1 neurons in blocks
                Size iLGN = (threadIdx.x + i)%nLGN;
				Float value;
                if (inside_ellipse(sx[iLGN]-V1_x, sy[iLGN]-V1_y, theta[V1_id], a[V1_id], a[V1_id]*baRatio[V1_id], value)) {
                    PosInt LGN_id = j0 + iPatch*blockDim.x + iLGN;
                    if (LGN_id >= j0+m) {
                        printf("%u/%u, LGN_id:%u < j0(%u) + m(%u); nRemain = %u, nLGN = %u\n", iPatch, nPatch, LGN_id, j0, m, nRemain, nLGN); 
                        assert(LGN_id < j0+m);
                    }
                    PosIntL pid = static_cast<PosIntL>(V1_id)*maxLGNperV1pool + iPool;
                    poolList[pid] = LGN_id;
					//if (V1_id == 41652) {
                    //    printf("## %u-%u(%u):, dx = %f, dy = %f, theta = %f, a =%f, b = %f\n lgnx = %f, lgny = %f\n v1x = %f, v1y = %f\n 1-value = %e\n", V1_id, iPool, LGN_id, sx[iLGN]-V1_x, sy[iLGN]-V1_y, theta[V1_id]*180/M_PI, a[V1_id], a[V1_id]*baRatio[V1_id], sx[iLGN], sy[iLGN], V1_x, V1_y, 1-value); 
					//}
                    if (iPool > maxLGNperV1pool) {
                        printf("%u > %u, a = %f, b = %f, theta = %f\n", iPool, maxLGNperV1pool, a[V1_id], a[V1_id]*baRatio[V1_id], theta[V1_id]*180/M_PI); 
                        assert(iPool <= maxLGNperV1pool);
                    }
                    iPool++;
                }
            }
        }
        __syncthreads();
    }
    if (V1_id < i0+n) {
        nPool[V1_id] = iPool;
    }
}

/* 
    Purpose:
        Pool neurons with visual field centered at (eccentricity, polar) in retinotopic Sheet1 :pre-synaptically: to neurons in another retinotopic Sheet2, with visual field centered near the same spot.
    
    Inputs:
        1. visual field positions of neurons in Sheet1 in ((e)ccentricity, (p)olar angle) of size (2,N1)
            cart0 --- ([Float], [Float]) 
        2. visual field positions of neurons in Sheet2 in in (e,p) of size (2,N2)
            cart --- ([Float], [Float]) 
        3. mapping rule: given the (e,p), return the radius of VF
            rmap --- (Float, Float) -> Float:
    Output:
        1. index list as an adjacency matrix:
            poolList --- [[Int]];
    
    Source file structure:

        "retinotopic_vf_pool.h"
            define mapping rule 
                1. analytic function
                2. interpolation from data
            declare retinotopic_vf_pool

        "retinotopic_vf_pool.cpp"
            implement pooling 
                for i-th neuron (e_i,p_i) in Sheet2
                use tentative radius of interest rmap(e_i,p_i) on Sheet 1 and make sure that it contains at least one neuron, otherwsie only include the nearest one to the poolList.
*/

vector<vector<Size>> retinotopic_vf_pool(
        pair<vector<Float>,vector<Float>> &cart,
        pair<vector<Float>,vector<Float>> &cart0,
        vector<PosInt> &layerMatch,
        vector<Float> &VFposEcc,
        bool useCuda,
        RandomEngine &rGen,
        vector<Float> &baRatio,
        vector<Float> &a,
        vector<Float> &theta,
        vector<Int> &LR,
        Size nL,
        vector<Size> &mL,
        vector<Size> &m,
        Size maxLGNperV1pool,
		BigSize seed,
		Float LGN_V1_RFratio
) {
    vector<vector<PosInt>> poolList;
    const Size n = cart.first.size();
        
    poolList.reserve(n);
    if (useCuda) {
		vector<Float> xLR, yLR, baRatioLR, VFposEccLR, thetaLR;
		original2LR(cart.first, xLR, LR, nL);
		original2LR(cart.second, yLR, LR, nL);
		original2LR(baRatio, baRatioLR, LR, nL);
		original2LR(VFposEcc, VFposEccLR, LR, nL);
		original2LR(theta, thetaLR, LR, nL);
        Float *d_memblock;
        size_t d_memSize = ((2+4)*n + 2*m)*sizeof(Float) + static_cast<size_t>(n)*maxLGNperV1pool*sizeof(PosInt) + n*sizeof(Size);
		cout << "poolList need memory of " << n << "x" << maxLGNperV1pool << "x" << sizeof(PosInt) << " = " << static_cast<PosIntL>(n)*maxLGNperV1pool*sizeof(PosInt) << " = " << static_cast<PosIntL>(n)*maxLGNperV1pool*sizeof(PosInt) / 1024.0 / 1024.0 << "mb\n";
        checkCudaErrors(cudaMalloc((void **) &d_memblock, d_memSize));
		cout << "need global memory of " << d_memSize / 1024.0 / 1024.0 << "mb in total\n";
        Float* d_x = d_memblock;
        Float* d_y = d_x + n;
        Float* d_x0 = d_y + n;
        Float* d_y0 = d_x0 + m;
        Float* d_baRatio = d_y0 + m;
        Float* d_a = d_baRatio + n; // to be filled
        Float* d_theta = d_a + n;
        Float* d_VFposEcc = d_theta + n;
		// to be filled
        PosInt* d_poolList = (PosInt*) (d_VFposEcc + n);
        Size* d_nPool = (Size*) (d_poolList + n*maxLGNperV1pool);

        checkCudaErrors(cudaMemcpy(d_x, &(xLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, &(yLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_x0, &(cart0.first[0]), m*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y0, &(cart0.second[0]), m*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_baRatio, &(baRatioLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_VFposEcc, &(VFposEccLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_theta, &(thetaLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        // TODO: stream kernels in chunks for large network
        Size nblock = (nL + blockSize-1)/blockSize;
        cout <<  "<<<" << nblock << ", " << blockSize << ">>>\n";
        vf_pool_CUDA<<<nblock, blockSize>>>(d_x, d_y, d_x0, d_y0, d_VFposEcc, d_baRatio, d_a, d_theta, d_poolList, d_nPool, 0, nL, 0, mL, seed, LGN_V1_RFratio, maxLGNperV1pool);
        getLastCudaError("vf_pool for the left eye failed");
        cudaDeviceSynchronize();
		nblock = (n-nL + blockSize-1) / blockSize;
        cout <<  "<<<" << nblock << ", " << blockSize << ">>>\n";
        vf_pool_CUDA<<<nblock, blockSize>>>(d_x, d_y, d_x0, d_y0, d_VFposEcc, d_baRatio, d_a, d_theta, d_poolList, d_nPool, nL, n-nL, mL, m-mL, seed, LGN_V1_RFratio, maxLGNperV1pool);
        getLastCudaError("vf_pool for the right eye failed");
		vector<vector<PosInt>> poolListLR;
        size_t largeSize = static_cast<BigSize>(n)*maxLGNperV1pool;
        PosInt* poolListArray = new Size[largeSize];
        Size* nPool = new Size[n];
		vector<Float> aLR(n);
        checkCudaErrors(cudaMemcpy(&aLR[0], d_a, n*sizeof(Float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(poolListArray, d_poolList, largeSize*sizeof(PosInt), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(nPool, d_nPool, n*sizeof(Size), cudaMemcpyDeviceToHost));
        for (Size i=0; i<n; i++) {
            vector<Size> iPool(nPool[i]);
            for (Size j=0; j<nPool[i]; j++) {
                size_t pid = static_cast<PosIntL>(i)*maxLGNperV1pool + j;
                iPool[j] = poolListArray[pid];
            }
            poolListLR.push_back(iPool);
        }
        Size maxPool = *max_element(nPool, nPool+n);
        delete []poolListArray;
        delete []nPool;
        assert(maxPool < maxLGNperV1pool);
		LR2original(poolListLR, poolList, LR, nL);
		LR2original(aLR, a, LR, nL);
        checkCudaErrors(cudaFree(d_memblock));
    } else {
        vector<Float> normRand;
        vector<Float> rMap;
        normRand.reserve(n);
        rMap.reserve(n);
        a.reserve(n);
        normal_distribution<Float> dist(0, 1);

        // generate random number for RF size distribution
        for (Size i=0; i<n; i++) {
            normRand.push_back(dist(rGen));
        }
        // find radius from mapping rule at (e,p)
        for (Size i=0; i<n; i++) {
			// convert input eccentricity to mins and convert output back to degrees.
            Float R = mapping_rule(VFposEcc[i]*60, normRand[i], rGen, LGN_V1_RFratio)/60.0;
            a.push_back(R/sqrt(M_PI*baRatio[i]));
            Float b = R*R/M_PI/a[i];
            if (a[i] > b) {
                rMap.push_back(a[i]);
            } else {
                rMap.push_back(b);
            }
        }
        // next nearest neuron j to (e,p) in sheet 1
        Size maxLGNperV1pool = 0;
        for (Size i=0; i<n; i++) {
            if (LR[i] > 0) {
                poolList.push_back(draw_from_radius(cart.first[i], cart.second[i], cart0, mL, m, rMap[i]));
            } else {
                poolList.push_back(draw_from_radius(cart.first[i], cart.second[i], cart0, 0, mL, rMap[i]));
            }
            if (poolList[i].size() > maxLGNperV1pool) maxLGNperV1pool = poolList[i].size();
        }
        cout << "actual maxLGNperV1pool reaches " << maxLGNperV1pool << "\n";
    }
	Float ecc_min = *min_element(VFposEcc.begin(), VFposEcc.end());
	Float ecc_mean = accumulate(VFposEcc.begin(), VFposEcc.end(), 0.0)/VFposEcc.size();
	Float ecc_max = *max_element(VFposEcc.begin(), VFposEcc.end());
	Float a_min = *min_element(a.begin(), a.end());
	Float a_mean = accumulate(a.begin(), a.end(), 0.0)/a.size();
	Float a_max = *max_element(a.begin(), a.end());
	Float baR_min = *min_element(baRatio.begin(), baRatio.end());
	Float baR_mean = accumulate(baRatio.begin(), baRatio.end(), 0.0)/baRatio.size();
	Float baR_max = *max_element(baRatio.begin(), baRatio.end());
	cout << "LGN_V1_RFratio = " << LGN_V1_RFratio << "\n";
	cout << "ecc = [" << ecc_min << ", " << ecc_mean << ", " << ecc_max << "] deg\n";
	cout << "a = [" << a_min << ", " << a_mean << ", " << a_max << "] deg\n";
	cout << "baRatio = [" << baR_min << ", " << baR_mean << ", " << baR_max << "] deg\n";
	cout << "R = [" << sqrt(M_PI*baR_min)*a_min << ", " << sqrt(M_PI*baR_mean)*a_mean << ", " << sqrt(M_PI*baR_max)*a_max << "] deg\n";
    return poolList;
}
 
int main(int argc, char *argv[]) {
	namespace po = boost::program_options;
    bool readFromFile, useCuda;
    Size nLayer, mLayer;
	Size max_LGNeff;
	vector<Size> maxLGNperV1pool;
	vector<Size> nType;
	vector<Float> p_n_LGNeff;
    Int SimpleComplex;
	Float conThres;
	Float ori_tol;
	Float disLGN;
	bool strictStrength;
	bool top_pick;
    vector<Float> pureComplexRatio;
	vector<Size> typeAccCount;
    vector<PosInt> layerMatch;
    Float LGN_V1_RFratio;
    Float envelopeSig;
    BigSize seed;
    vector<Size> nRefTypeV1_RF, V1_RefTypeID;
    vector<Float> V1_RFtypeAccDist, V1_RefTypeDist;
	string LGN_vpos_filename, V1_vpos_filename, V1_RFpreset_filename, V1_feature_filename, conLGN_suffix;
    string res_suffix;
	po::options_description generic("generic options");
	generic.add_options()
		("help,h", "print usage")
		("seed,s", po::value<BigSize>(&seed)->default_value(1885124), "random seed")
		("readFromFile,f", po::value<bool>(&readFromFile)->default_value(false), "whether to read V1 RF properties from file")
		("useCuda,u", po::value<bool>(&useCuda)->default_value(false), "whether to use cuda")
		("cfg_file,c", po::value<string>()->default_value("LGN_V1.cfg"), "filename for configuration file");

	po::options_description input_opt("input options");
	input_opt.add_options()
		("res_suffix", po::value<string>(&res_suffix)->default_value(""), "conLGN_suffix for resource files")
		("nLayer", po::value<int>(&nLayer)->default_value(1), "number of layers in V1, to cross check with the corresponding V1_allpos*.bin")
		("nLayer", po::value<vector<Size>>(&nType), "number of types per layer in V1, to cross check with the corresponding V1_allpos*.bin")
		("mLayer", po::value<int>(&mLayer)->default_value(1), "number of layers in LGN, to cross check with the corresponding LGN_vpos*.bin")
		("p_n_LGNeff", po::value<vector<Float>>(&p_n_LGNeff), "LGN conneciton probability [-1,0], or number of connections [0,n]")
		("top_pick", po::value<bool>(&top_pick)->default_value(true), "preset number of connection, n, and connect to the neurons with the top n prob")
		("max_LGNeff", po::value<Size>(&max_LGNeff)->default_value(10), "max realizable number of connections [0,n]")
		("LGN_V1_RFratio,r", po::value<Float>(&LGN_V1_RFratio)->default_value(1.0), "LGN's contribution to the total RF size")
		("maxLGNperV1pool", po::value<vector<Size>>(&maxLGNperV1pool), "maximum pooling of LGN neurons per V1 neuron, array of nLayers")
		("envelopeSig", po::value<Float>(&envelopeSig)->default_value(1.177), "LGN's pools connection probability envelope sigma on distance")
		("SimpleComplex", po::value<Int>(&SimpleComplex)->default_value(1), "determine how simple complex is implemented, through modulation modAmp_nCon(0) or number of LGN connection(1)")
		("conThres", po::value<Float>(&conThres)->default_value(-1), "connect to LGN using conThres")
		("ori_tol", po::value<Float>(&ori_tol)->default_value(15), "tolerance of preset orientation deviation in degree")
		("disLGN", po::value<Float>(&disLGN)->default_value(1.0), "average visual distance between LGN cells")
		("strictStrength", po::value<bool>(&strictStrength)->default_value(true), "make nLGN*sLGN strictly as preset")
		("pureComplexRatio", po::value<vector<Float>>(&pureComplexRatio), "determine the proportion of simple and complex in V1 of size nType")
		("typeAccCount",po::value<vector<Size>>(&typeAccCount), "neuronal types' discrete accumulative distribution size of nType")
		("V1_RFtypeAccDist", po::value<vector<Float>>(&V1_RFtypeAccDist), "determine the relative portion of each V1 RF type")
		("nRefTypeV1_RF", po::value<vector<Size>>(&nRefTypeV1_RF), "determine the number of cone/ON-OFF combinations for each V1 RF type")
		("V1_RefTypeID", po::value<vector<Size>>(&V1_RefTypeID), "determine the ID of the available cone/ON-OFF combinations in each V1 RF type")
		("V1_RefTypeDist", po::value<vector<Float>>(&V1_RefTypeDist), "determine the relative portion of the available cone/ON-OFF combinations in each V1 RF type")
        ("V1_LGN_layer_match", po::value<vector<PosInt>>(&layerMatch), "nLayer arrays of how each V1 layer recieve inputs from the LGN layers (mLayer)")
		("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file that stores V1 neurons' parameters")
		("fV1_RFpreset", po::value<string>(&V1_RFpreset_filename)->default_value("V1_RFpreset.bin"), "file that stores V1 neurons' parameters")
		("fLGN_vpos", po::value<string>(&LGN_vpos_filename)->default_value("LGN_vpos.bin"), "file that stores LGN position in visual field (and on-cell off-cell label)")
		("fV1_vpos", po::value<string>(&V1_vpos_filename)->default_value("V1_vpos.bin"), "file that stores V1 position in visual field)");

    string V1_RFprop_filename, idList_filename, sList_filename, output_cfg_filename;
	po::options_description output_opt("output options");
	output_opt.add_options()
		("conLGN_suffix", po::value<string>(&conLGN_suffix)->default_value(""), "conLGN_suffix for output file")
		("fV1_RFprop", po::value<string>(&V1_RFprop_filename)->default_value("V1_RFprop"), "file that stores V1 neurons' information")
		("fLGN_V1_ID", po::value<string>(&idList_filename)->default_value("LGN_V1_idList"), "file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&sList_filename)->default_value("LGN_V1_sList"), "file stores LGN to V1 connection strengths")
		("fLGN_V1_cfg", po::value<string>(&output_cfg_filename)->default_value("LGN_V1_cfg"), "file stores LGN_V1.cfg parameters");

	po::options_description cmdline_options;
	cmdline_options.add(generic).add(input_opt).add(output_opt);

	po::options_description config_file_options;
	config_file_options.add(generic).add(input_opt).add(output_opt);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
	if (vm.count("help")) {
		cout << cmdline_options << "\n";
		return EXIT_SUCCESS;
	}
	string cfg_filename = vm["cfg_file"].as<string>();
	ifstream cfg_file{cfg_filename.c_str()};
	if (cfg_file) {
		po::store(po::parse_config_file(cfg_file, config_file_options), vm);
		cout << "Using configuration file: " << cfg_filename << "\n";
	} else {
		cout << "No configuration file is given, default values are used for non-specified parameters\n";
	}
	po::notify(vm);
    
    if (!conLGN_suffix.empty()) {
        conLGN_suffix = '_' + conLGN_suffix;
    }
    conLGN_suffix = conLGN_suffix + ".bin";

    if (!res_suffix.empty())  {
        res_suffix = "_" + res_suffix;
    }
    res_suffix = res_suffix + ".bin";

    if (useCuda) {
        cudaDeviceProp deviceProps;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
        printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
        printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("total global memory: %f Mb.\n", deviceProps.totalGlobalMem/1024.0/1024.0);
    
    #ifdef SINGLE_PRECISION
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    #else
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    #endif
        printf("maximum threads per MP: %d.\n", deviceProps.maxThreadsPerMultiProcessor);
        printf("shared memory per block: %zu bytes.\n", deviceProps.sharedMemPerBlock);
        printf("registers per block: %d.\n", deviceProps.regsPerBlock);
        cout << "\n";
    }
    
	Size* nblock = new Size[nLayer];
    Size* neuronPerBlock = new Size[nLayer];
    Size* n = new Size[nLayer];
	vector<pair<vector<Float>, vector<Float>>> cart(nLayer), // V1 VF position (preset)
	vector<vector<Float>> VFposEcc(nLayer);
	ifstream fV1_vpos;
	fV1_vpos.open(V1_vpos_filename + res_suffix, fstream::in | fstream::binary);
	if (!fV1_vpos) {
		cout << "Cannot open or find " << V1_allpos_filename + res_suffix <<" to read V1 positions.\n";
		return EXIT_FAILURE;
	} else {
        Size _nLayer;
		fV1_vpos.read(reinterpret_cast<char*>(&_nLayer), sizeof(Size));
        if (_nLayer != nLayer) {
            cout << "nLayer = " << nLayer << " is not consistent with nLayer(" << _nLayer << ") in V1_allpos.bin"
            return EXIT_FAILURE;
        }
		fV1_vpos.read(reinterpret_cast<char*>(&nblock[0]), sizeof(Size)*nLayer);
		fV1_vpos.read(reinterpret_cast<char*>(&neuronPerBlock[0]), sizeof(Size)*nLayer);
        auto double2Float = [](double x) {
            return static_cast<Float>(x);
	    };
	    auto polar2xD2F = [] (double ecc, double polar) {
	    	return static_cast<Float>(ecc*cos(polar));
	    };
	    auto polar2yD2F = [] (double ecc, double polar) {
	    	return static_cast<Float>(ecc*sin(polar));
	    };
        for (int iLayer = 0; iLayer < nLayer; i++) {
            n[iLayer] = nblock[iLayer] * neuronPerBlock[iLayer];
            vector<double> decc(n[iLayer]);
	        vector<double> dpolar(n[iLayer]);
            VFposEcc[iLayer].reserve(n[iLayer]);
	        fV1_vpos.read(reinterpret_cast<char*>(&decc[0]), n * sizeof(double)); 
	        fV1_vpos.read(reinterpret_cast<char*>(&dpolar[0]), n * sizeof(double));
            transform(decc.begin(), decc.end(), VFposEcc[iLayer].begin(), double2Float);
	        vector<Float> x(n);
	        vector<Float> y(n);
	        transform(decc.begin(), decc.end(), dpolar.begin(), x.begin(), polar2xD2F);
	        transform(decc.begin(), decc.end(), dpolar.begin(), y.begin(), polar2yD2F);
            cout << n[iLayer] << "V1 neurons for layer " << iLayer << "\n";
            cout << "V1_x: [" << *min_element(x.begin(), x.end()) << ", " << *max_element(x.begin(), x.end()) << "]\n";
            cout << "V1_y: [" << *min_element(y.begin(), y.end()) << ", " << *max_element(y.begin(), y.end()) << "]\n";
	        cart[iLayer] = make_pair(x, y);
        }
    }
	fV1_vpos.close();

	Size* mL;
	Size* mR;
    Size* m;
	Float max_ecc;
	vector<vector<InputType>> LGNtype(mLayer);
    vector<pair<vector<Float>, vector<Float>>> cart0(mLayer), // V1 VF position (preset)
	ifstream fLGN_vpos;
	fLGN_vpos.open(LGN_vpos_filename, fstream::in | fstream::binary);
	if (!fLGN_vpos) {
		cout << "Cannot open or find " << LGN_vpos_filename << "\n";
		return EXIT_FAILURE;
	} else {
        Size _mLayer;
	    fLGN_vpos.read(reinterpret_cast<char*>(&_mLayer), sizeof(Size));
        if (_mLayer != mLayer) {
            cout << "mLayer = " << mLayer << " is inconsistent with mLayer(" << _mLayer << ") in LGN_vpos.bin"
            return EXIT_FAILURE;
        }
	    fLGN_vpos.read(reinterpret_cast<char*>(&max_ecc), sizeof(Float));
	    mL = new Size[mLayer];
	    mR = new Size[mLayer];
        m = new Size[mLayer];
	    fLGN_vpos.read(reinterpret_cast<char*>(mL), mLayer*sizeof(Size));
	    fLGN_vpos.read(reinterpret_cast<char*>(mR), mLayer*sizeof(Size));
	    {// not used
	    	Float tmp;
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // x0
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // xspan
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // y0
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // yspan
	    }
        Size m_total = 0;
        for (iLayer = 0; iLayer < mLayer; iLayer++) {
            m[iLayer] = mL[iLayer] + mR[iLayer];
	        vector<Float> x0(m[iLayer]);
	        vector<Float> y0(m[iLayer]);
	        LGNtype[iLayer].resize(m[iLayer]);
	        fLGN_vpos.read(reinterpret_cast<char*>(&x0[0]), m[iLayer]*sizeof(Float));
	        fLGN_vpos.read(reinterpret_cast<char*>(&y0[0]), m[iLayer]*sizeof(Float));
	        fLGN_vpos.read(reinterpret_cast<char*>(&LGNtype[iLayer][0]), m[iLayer] * sizeof(PosInt));
            m_total += m[iLayer];
            cout << m[iLayer] << " LGN neurons, " << mL[iLayer] << " from left eye, " << mR[iLayer] << " from right eye in layer " << iLayer << ".\n";
            cout << "LGN_x: [" << *min_element(x0.begin(), x0.end()) << ", " << *max_element(x0.begin(), x0.end()) << "]\n";
            cout << "LGN_y: [" << *min_element(y0.begin(), y0.end()) << ", " << *max_element(y0.begin(), y0.end()) << "]\n";
	        cart0[iLayer] = make_pair(x0, y0);
        }
	    cout << "need " << 3 * m_total * sizeof(Float) / 1024 / 1024 << "Mb mem\n";
    }
	fLGN_vpos.close();

	cout << "carts ready\n";
    vector<BigSize> seeds{seed,seed+13};
    seed_seq seq(seeds.begin(), seeds.end());
    RandomEngine rGen(seq);

	Size nFeature; // not used here
	vector<vector<Int>> featureLayer(nLayer);
	vector<vector<Int>> LR(nLayer);
	vector<vector<Float>> theta(nLayer); // [0, 1] control the LGN->V1 RF orientation
	vector<vector<Float>> OD(nLayer);
    vector<vector<Float>>* feature[2] = {&OD, &theta};
    ifstream fV1_feature;
    fV1_feature.open(V1_feature_filename, ios::in|ios::binary);
	if (!fV1_feature) {
		cout << "failed to open pos file:" << V1_feature_filename << "\n";
		return EXIT_FAILURE;
	} else {
        fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
        for (int i=0; i<2; i++) { // only care OD and theta
            Size nfl;
            fV1_feature.read(reinterpret_cast<char*>(&nfl), sizeof(Size));
            fV1_feature.read(reinterpret_cast<char*>(&featureLayer[i]), nfl*sizeof(Size));
            vector<Size> _n(nfl); 
            fV1_feature.read(reinterpret_cast<char*>(&_n[0]), nfl*sizeof(Size));
            for (int j=0; j<nfl; j++) {
                PosInt iLayer = featureLayer[i];
                if (n[iLayer] != _n[j]) {
                    cout << "feature size in layer " << iLayer << " is " << _n[j] << ", inconsistent with the size from V1_vpos.bin, " << n[iLayer] << "\n";
                    return EXIT_FAILURE;
                } else {
                    (*(feature[i])->at(iLayer).resize(_n[j]);
	                fV1_feature.read(reinterpret_cast<char*>(&(*feature[i])->(iLayer)[0], _n[j] * sizeof(Float));
                }
            }
        }
    }
	fV1_feature.close();

    for (iLayer = 0; iLayer < nLayer; iLayer++) {
        cout << "layer " << iLayer << "\n";  
        Size nL = 0;
        Size nR = 0;
        for (PosInt i = 0; i<n[iLayer]; i++) {
            if (OD[iLayer][i] < 0) {
                LR[iLayer][i] = -1;
                nL++;
            } else {
                LR[iLayer][i] = 1;
                nR++
            }
            theta[iLayer][i] = (theta[iLayer][i]-0.5)*M_PI;
        }
        cout << "   left eye: " << nL << " V1 neurons\n";
        cout << "   right eye: " << nR << " V1 neurons\n";
        vector<Float> a; // radius of the VF, to be filled
	    vector<Float> baRatio = generate_baRatio(n[iLayer], rGen);
        cout << "   max pool of LGN = " << maxLGNperV1pool[iLayer] << "\n";
        vector<vector<Size>> poolList = retinotopic_vf_pool(cart[iLayer], cart0, layerMatch, VFposEcc[iLayer], useCuda, rGen, baRatio, a, theta[iLayer], LR[iLayer], nL, mL, m, maxLGNperV1pool[iLayer], seed, LGN_V1_RFratio);
        Size minPool_L = maxLGNperV1pool[iLayer]; 
        Size maxPool_L = 0; 
        Float meanPool_L = 0; 
        Size zeroPool_L = 0;

        Size minPool_R = maxLGNperV1pool[iLayer]; 
        Size maxPool_R = 0; 
        Float meanPool_R = 0; 
        Size zeroPool_R = 0;

        Float rzeroL = 0;
        Float rmeanL = 0;
        Float rzeroR = 0;
        Float rmeanR = 0;
	    
	    Int jL = 0;
	    Int jR = 0;
        for (PosInt i=0; i<n; i++) {
	    	Int iLR;
	    	if (LR[i] < 0) {
	    		iLR = jL;
	    		jL++;
	    	} else {
	    		iLR = nL + jR;
	    		jR++;
	    	}
            Size iSize = poolList[i].size();
	    	for (PosInt j=0; j<iSize; j++) {
	    		Float dx = (cart0.first[poolList[i][j]] - cart[iLayer].first[i]);
	    		Float dy = (cart0.second[poolList[i][j]] - cart[iLayer].second[i]);
	    		PosInt lgn_id = poolList[i][j];
	    		Float value;
	    		if (!inside_ellipse(dx, dy, theta[i], a[i], a[i]*baRatio[i], value)) {
        			Float tx = cosine(theta[i]) * dx + sine(theta[i]) * dy;
	    			Float ty = -sine(theta[i]) * dx + cosine(theta[i]) * dy;
	    			cout << "#" << iLR << "-" << j << "(" << poolList[i][j]<< "): x = " << dx << ", y = " << dy << ", theta = " << theta[i]*180/M_PI << ", a = " << a[i] << ", b = " << a[i]*baRatio[i] << "\n";
	    			cout << "1-value = " << 1-value << "\n";
	    			cout << "lgnx = " << cart0.first[lgn_id] << ", lgny = " << cart0.second[lgn_id] << "\n";
	    			cout << "v1x = " << cart.first[i] << ", v1y = " << cart.second[i] << "\n";
	    			cout << "tx = " << tx << ", ty = " << ty << "\n";
	    			assert(inside_ellipse(dx, dy, theta[i], a[i], a[i]*baRatio[i], value));
	    		}
	    	}

            Float r = a[i]*square_root(baRatio[i]*baRatio[i] + 1);
	    	if (LR[i] > 0) {
            	if (iSize > maxPool_R) maxPool_R = iSize;
            	if (iSize < minPool_R) minPool_R = iSize;
            	if (iSize == 0) {
            	    zeroPool_R++;
            	    rzeroR += r;
            	}
            	rmeanR += r;
            	meanPool_R += iSize;
	    	} else {
            	if (iSize > maxPool_L) maxPool_L = iSize;
            	if (iSize < minPool_L) minPool_L = iSize;
            	if (iSize == 0) {
            	    zeroPool_L++;
            	    rzeroL += r;
            	}
            	rmeanL += r;
            	meanPool_L += iSize;
	    	}
        }
        meanPool_L /= nL;
        rzeroL /= zeroPool_L;
        rmeanL /= nL;
        meanPool_R /= nL;
        rzeroR /= zeroPool_R;
        rmeanR /= nR;
        cout << "right poolSizes: [" << minPool_R << ", " << meanPool_R << ", " << maxPool_R << " < " << maxLGNperV1pool[iLayer] << "]\n";
        cout << "among them " << zeroPool_R << " would have no connection from LGN, whose average radius is " << rzeroR << ", compared to population mean " <<  rmeanR << "\n";

        cout << "left poolSizes: [" << minPool_L << ", " << meanPool_L << ", " << maxPool_L << " < " << maxLGNperV1pool[iLayer] << "]\n";
        cout << "among them " << zeroPool_L << " would have no connection from LGN, whose average radius is " << rzeroL << ", compared to population mean " <<  rmeanL << "\n";

	    cout << "poolList and R ready for layer " << iLayer << "\n";
    }

	vector<vector<RFtype>> V1Type(nLayer); // defined in RFtype.h [0..4], RF shapes
	vector<vector<OutputType>> RefType(nLayer); // defined in in RFtype.h [0..3] conetype placements
	vector<vector<Float>> phase(nLayer); // [0, 2*pi], control the phase
    // theta is read from fV1_feature
	vector<vector<Float>> modAmp_nCon(nLayer); // [0,1] controls simple complex ratio, through subregion overlap ratio, or number of LGN connected.
	vector<Size> nType(nLayer);
	vector<vector<Size>> typeAcc0(nLayer);

    
    if (readFromFile) {
        fstream fV1_RFpreset(V1_RFpreset_filename, fstream::in | fstream::binary);
	    if (!fV1_RFpreset) {
	    	cout << "Cannot open or find " << V1_RFpreset_filename <<"\n";
	    	return EXIT_FAILURE;
	    } else {
            Size _nLayer;
	        fV1_RFpreset.read(reinterpret_cast<char*>(&_nLayer), sizeof(Size));
            vector<Size> _n(_nLayer);
		    fV1_RFpreset.read(reinterpret_cast<char*>(&_n[0]), sizeof(Size)*nLayer);
            if (_nLayer != nLayer) {
                cout << "nLayer = " << nLayer << " is inconsistent with nLayer(" << _nLayer << ") in V1_allpos.bin"
                return EXIT_FAILURE;
            }
            for (int iLayer = 0; iLayer<nLayer; iLayer++) {
                if (n[iLayer] != _n[iLayer]) {
                    cout << "feature size in layer " << iLayer << " is " << _n[iLayer] << ", inconsistent with the size from V1_vpos.bin, " << n[iLayer] << "\n";
                    return EXIT_FAILURE;
                }
	            fV1_RFpreset.read(reinterpret_cast<char*>(&nType), sizeof(Size));
	            fV1_RFpreset.read(reinterpret_cast<char*>(&typeAccCount[0]), nType*sizeof(Size));

	            fV1_RFpreset.read(reinterpret_cast<char*>(&V1Type[0]), n * sizeof(RFtype_t));
	            fV1_RFpreset.read(reinterpret_cast<char*>(&RefType[0]), n * sizeof(OutputType_t));
	            fV1_RFpreset.read(reinterpret_cast<char*>(&phase[0]), n * sizeof(Float));
	            fV1_RFpreset.read(reinterpret_cast<char*>(&modAmp_nCon[0]), n * sizeof(Float));
            }
            fV1_RFpreset.close();
    } else {
        nType = typeAccCount.size();
	    if (nType > max_nType) {
	    	cout << "the accumulative distribution of neuronal type <typeAccCount> has size of " << nType << " > " << max_nType << "\n";
	    	return EXIT_FAILURE;
	    }

	    typeAcc0.push_back(0);
	    for (PosInt i=0; i<nType; i++) {
	    	typeAcc0.push_back(typeAccCount[i]);
	    }

	    Size nblock = n/blockSize;
	    if (typeAccCount.back() != blockSize) {
	    	cout << "neuron per block != " << blockSize << "\n";
	    	return EXIT_FAILURE;
	    }
	    if (nblock * blockSize != n) {
	    	cout << "number of V1 neuron cannot be divided by " << blockSize << "\n";
	    	return EXIT_FAILURE;
	    }
        // discrete portions randomly distributed
        uniform_real_distribution<Float> uniform_01(0,1.0);
        normal_distribution<Float> norm_01(0,1.0);

        vector<Size> nTypes(nRFtype,0);
        assert(V1_RFtypeAccDist.size() == nRFtype  && V1_RFtypeAccDist.back() == 1.0);
        auto genV1_RFtype = [&uniform_01, &V1_RFtypeAccDist, &rGen, &nTypes] () {
            Float rand = uniform_01(rGen);
            for (PosInt i=0; i<V1_RFtypeAccDist.size(); i++) {
                if (rand < V1_RFtypeAccDist[i]) {
                    nTypes[i]++;
                    return static_cast<RFtype>(i);
                }
            }
        };
        generate(V1Type.begin(), V1Type.end(), genV1_RFtype);
        cout << "None-G,    None-CS,   Single,  Double-CS,   Double-G\n";
        for (PosInt i=0; i<nRFtype; i++) {
            cout << nTypes[i] << "    ";
        }
        cout << "\n";

        // vec to list of vec
        vector<vector<Float>> RefTypeDist;
        vector<vector<Size>> RefTypeID;
        Size iTmp = 0;
        Size checksum = 0;
        for (Size iV1type = 0; iV1type<nRFtype; iV1type++){
            vector<Float> RefTypeDistTmp;
            vector<Size> RefTypeID_Tmp;
            for (Size iRefType = 0; iRefType < nRefTypeV1_RF[iV1type]; iRefType++) {
                RefTypeDistTmp.push_back(V1_RefTypeDist[iTmp]);
                RefTypeID_Tmp.push_back(V1_RefTypeID[iTmp]);
                iTmp++;
            }
            checksum += nRefTypeV1_RF[iV1type];
            RefTypeDist.push_back(RefTypeDistTmp);
            RefTypeID.push_back(RefTypeID_Tmp);
        }
        assert(V1_RefTypeDist.size() == checksum);
        assert(V1_RefTypeID.size() == checksum);

        auto genV1_RefType = [&uniform_01, &nRefTypeV1_RF, &RefTypeDist, &RefTypeID, &rGen] (RFtype V1TypeI) {
            Float rand = uniform_01(rGen);
            Size iRFtype = static_cast<Size>(V1TypeI);
            for (Size i = 0; i < nRefTypeV1_RF[iRFtype]; i++) {
                if (rand < RefTypeDist[iRFtype][i]) {
                    return static_cast<OutputType>(RefTypeID[iRFtype][i]);
                }
            }
        };
        transform(V1Type.begin(), V1Type.end(), RefType.begin(), genV1_RefType);

        // uniform distribution
        auto genPhase = [&uniform_01, &rGen] () {
            return uniform_01(rGen) * 2 * M_PI;
			//return 0;
        };
        generate(phase.begin(), phase.end(), genPhase);

    	
		for (PosInt i = 0; i<nblock; i++) {
			for (PosInt j=0; j<nType; j++) { 
				Float ratio = pureComplexRatio[j];
				if (SimpleComplex == 1) { // ** else == 1 not implemented
					// uniformly distribute simple cell's modulation ratio
        			auto genModAmp_nCon = [&uniform_01, &rGen, &ratio] () {
        			    Float rand = uniform_01(rGen);
						Float v;
        			    if (rand >= ratio) {
        			        //v = 1.0;
        			        v = rand;
        			    } else {
        			        v = 0.0;
        			    }
						//Float v = 1.0;
        			    return v;
        			};
        			generate(modAmp_nCon.begin() + i*blockSize + typeAcc0[j], modAmp_nCon.begin() + i*blockSize + typeAcc0[j+1], genModAmp_nCon);
				} else {
        			auto genModAmp_nCon = [&uniform_01, &norm_01, &rGen, &ratio, &max_LGNeff] () {
        			    Float rand = uniform_01(rGen);
						Float v;
        			    if (rand >= ratio) {
        			        //v = 1.0;
        			        //v = rand;
							v = ratio + (1-ratio)/2 + norm_01(rGen)*square_root(0.5*0.5/max_LGNeff);
        			    } else {
        			        v = 0.0;
        			    }
						if (v>1) v=1;
						if (v<0) v=0;
						//Float v = 1.0;
        			    return v;
        			};
        			generate(modAmp_nCon.begin() + i*blockSize + typeAcc0[j], modAmp_nCon.begin() + i*blockSize + typeAcc0[j+1], genModAmp_nCon);
					// binomially distribute simple cell's modulation ratio over nLGN connections
					/*
        			auto genModAmp_nCon = [&norm_01, &rGen, &ratio, &p_n_LGNeff, &max_LGNeff] () {
						Float std = 
        			    Float rand = norm_01(rGen)*std + 0.5;
						Float v;
        			    return v;
        			};
        			generate(modAmp_nCon.begin() + i*blockSize + typeAcc0[j], modAmp_nCon.begin() + i*blockSize + typeAcc0[j+1], genModAmp_nCon);
					*/
				}
			}
		}
    }
	cout << "V1 RF properties ready.\n";

    assert(p_n_LGNeff != 0);
    assert(p_n_LGNeff >= -1);

	
	vector<Float> sfreq = generate_sfreq(n, rGen);

    
    Float mean_sfreq = accumulate(sfreq.begin(), sfreq.end(), 0.0)/n;
    Float mean_a = accumulate(a.begin(), a.end(), 0.0)/n;
    Float suggesting_SF = mean_sfreq/mean_a/2;
    cout << "mean a: " << mean_a << " degPerRF\n";
    cout << "mean sfreq: " << mean_sfreq << " cpRF\n";
    cout << "mean SF suggested based on preset: " << suggesting_SF << " cpd\n";

	vector<Float> cx(n);
	vector<Float> cy(n);
    vector<vector<Float>> srList = retinotopic_connection(poolList, rGen, p_n_LGNeff, max_LGNeff, envelopeSig, n, cart, cart0, V1Type, theta, phase, sfreq, modAmp_nCon, baRatio, a, RefType, LGNtype, cx, cy, VFposEcc, SimpleComplex, conThres, ori_tol, disLGN, strictStrength, top_pick);

    if (!readFromFile) {
        ofstream fV1_RFpreset(V1_RFpreset_filename, fstream::out | fstream::binary);
	    if (!fV1_RFpreset) {
	    	cout << "Cannot open or find " << V1_RFpreset_filename <<"\n";
	    	return EXIT_FAILURE;
	    }

		fV1_RFpreset.write((char*)&nType, sizeof(Size));
		fV1_RFpreset.write((char*)&typeAccCount[0], nType*sizeof(Size));
	    fV1_RFpreset.write((char*)&V1Type[0], n * sizeof(RFtype_t));
	    fV1_RFpreset.write((char*)&RefType[0], n * sizeof(OutputType_t));
	    fV1_RFpreset.write((char*)&phase[0], n * sizeof(Float));
	    fV1_RFpreset.write((char*)&modAmp_nCon[0], n * sizeof(Float));
        fV1_RFpreset.close();
    }

	ofstream fV1_RFprop(V1_RFprop_filename + conLGN_suffix, fstream::out | fstream::binary);
	if (!fV1_RFprop) {
		cout << "Cannot open or find V1." << V1_RFprop_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	}
    fV1_RFprop.write((char*)&n, sizeof(Size));
    fV1_RFprop.write((char*)&cx[0], n * sizeof(Float));
	fV1_RFprop.write((char*)&cy[0], n * sizeof(Float));
    fV1_RFprop.write((char*)&a[0], n * sizeof(Float));
    fV1_RFprop.write((char*)&phase[0], n * sizeof(Float));
	fV1_RFprop.write((char*)&sfreq[0], n * sizeof(Float));
    fV1_RFprop.write((char*)&baRatio[0], n * sizeof(Float));
	fV1_RFprop.close();

	Float sfreq2 = 0;
	Float min_sfreq = 100.0; // 60 as max sfreq
	Float max_sfreq = 0.0;
    Size nonZero = 0;
	for (PosInt i = 0; i<n; i++) {
		if (sfreq[i] > 0 && poolList[i].size() > 0) {
            Float sf = sfreq[i]/(mean_a*2);
            if (sf < min_sfreq) {
                min_sfreq = sf; 
            }
            if (sf > max_sfreq) {
                max_sfreq = sf; 
            }
			sfreq2 += sf*sf;
			mean_sfreq += sf;
            nonZero++;
		}
	}
	mean_sfreq /= nonZero;
    cout << "mean SF suggested after connection: " << mean_sfreq << " cpd\n";
    cout << "min non-zero SF suggested after connection: " << min_sfreq << " cpd\n";
    cout << "max SF suggested after connection: " << max_sfreq << " cpd\n";
	sfreq2 /= nonZero;
    cout << "std SF suggested after connection: " << square_root(sfreq2 - mean_sfreq*mean_sfreq) << " cpd\n";

    Size nonZeroMinPool = maxLGNperV1pool[iLayer]; 
    Float nonZeroMeanPool = 0; 
    Size nonZeroMaxPool = 0;

    Size minPool = maxLGNperV1pool[iLayer]; 
    Size maxPool = 0; 
    Float meanPool = 0; 
    Size zeroPool = 0;
	Float minSum = max_LGNeff;
	Float meanSum = 0;
	Float maxSum = 0;
	Float sum2 = 0;
	Float pool2 = 0;
	PosInt ic_max = 0;
	PosInt is_max = 0;
    for (PosInt i=0; i<n; i++) {
        Size iSize = poolList[i].size();
        if (iSize > maxPool) {
			maxPool = iSize;
			ic_max = i;
		}
        if (iSize < minPool) minPool = iSize;
        if (iSize > 0) {
            if (iSize < nonZeroMinPool) nonZeroMinPool = iSize;
            nonZeroMeanPool += iSize;
            if (iSize > nonZeroMaxPool) nonZeroMaxPool = iSize;
        }

        meanPool += iSize;
        if (iSize == 0) zeroPool++;
		assert(poolList[i].size() == srList[i].size());
		Float strength = accumulate(srList[i].begin(), srList[i].end(), 0.0);
		if (strength > maxSum) {
			maxSum = strength;
			is_max = i;
		}
		if (strength < minSum) minSum = strength;
		meanSum += strength;
		sum2 += strength * strength;
		pool2 += iSize * iSize;
    }
	if (SimpleComplex == 1) {
		Size nblock = n/blockSize;
		Float nonzeroN = 0;
		for (PosInt i=0; i<nType; i++) {
			nonzeroN += nblock*(typeAcc0[i+1]-typeAcc0[i])*(1-pureComplexRatio[i]);
		}
		cout << nonzeroN << " nonzero LGN cells\n";
		meanSum /= nonzeroN;
    	meanPool /= nonzeroN;
		sum2 /= nonzeroN;
		pool2 /= nonzeroN;
	} else {
		meanSum /= n;
    	meanPool /= n;
		sum2 /= n;
		pool2 /= n;
	}
    cout << "# connections: [" << minPool << ", " << meanPool << ", " << maxPool << "]\n";
    cout << "among them " << zeroPool << " would have no connection from LGN\n";
    cout << "# nonzero connections: [" << nonZeroMinPool << ", " << nonZeroMeanPool/(n-zeroPool) << ", " << nonZeroMaxPool << "]\n";
    cout << "# totalConnectionStd: " << square_root(pool2 - meanPool*meanPool) << "\n";
    cout << "# totalStrength: [" << minSum << ", " << meanSum << ", " << maxSum << "]\n";
    cout << "# totalStrengthStd: " << square_root(sum2 - meanSum*meanSum) << "\n";

	/*
	cout << ic_max << "has max connections:\n";
	if (srList[ic_max].size() > 0) {
		for (PosInt i=0; i<srList[ic_max].size()-1; i++) {
			cout << srList[ic_max][i] << ", ";
		}
		cout << srList[ic_max].back() << "\n";
	}
	if (srList[is_max].size() > 0) {
		cout << is_max << " has max strength:\n";
		for (PosInt i=0; i<srList[is_max].size()-1; i++) {
			cout << srList[is_max][i] << ", ";
		}
		cout << srList[is_max].back() << "\n";
	} */

    // write poolList to disk, to be used in ext_input.cu and genCon.cu
	write_listOfList<Size>(idList_filename + conLGN_suffix, poolList, false);
	write_listOfListForArray<Float>(sList_filename + conLGN_suffix, srList, false); // read with read_listOfListToArray
	ofstream fLGN_V1_cfg(output_cfg_filename + conLGN_suffix, fstream::out | fstream::binary);
	if (!fLGN_V1_cfg) {
		cout << "Cannot open or find " << output_cfg_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
		fLGN_V1_cfg.write((char*) &p_n_LGNeff, sizeof(Float));
		fLGN_V1_cfg.write((char*) &max_LGNeff, sizeof(Size));
		fLGN_V1_cfg.write((char*) &nType, sizeof(Size));
		fLGN_V1_cfg.write((char*) &typeAccCount[0], nType*sizeof(Size));
		fLGN_V1_cfg.write((char*) &meanPool, sizeof(Float));
		fLGN_V1_cfg.write((char*) &meanSum, sizeof(Float));
		fLGN_V1_cfg.close();
	}

    return 0;
}
