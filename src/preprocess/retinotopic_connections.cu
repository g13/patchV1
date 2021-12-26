#include "retinotopic_connections.h"
using namespace std;

/* 
    Purpose:
        Connect neurons with visual field centered at (eccentricity, polar) in retinotopic Sheet1 :pre-synaptically: to neurons in another retinotopic Sheet2, with visual field centered near the same spot.
*/

template<typename T>
void original2typeLR(Size n, vector<PosInt> &seqByTypeLR, vector<T> &original, vector<T> &typeLR) {
    assert(n == original.size());
    for (PosInt i=0; i<n; i++) {
        typeLR[i] = original[seqByTypeLR[i]];
    }
}

template<typename T>
void typeLR2original(Size n, vector<PosInt> &seqByTypeLR, vector<T> &typeLR, vector<T> &original) {
    assert(seqByTypeLR.size() == typeLR.size());
    assert(original.size() == typeLR.size());
    assert(n == seqByTypeLR.size());
    for (PosInt i = 0; i<n; i++) {
        original[seqByTypeLR[i]] = typeLR[i];
    }
}

void seqTypeLR(Size n, vector<PosInt> &seqByTypeLR, vector<PosInt> &type, vector<Int> &LR, vector<Size> &accTypeL, vector<Size> &accTypeR, vector<Size> &accType, Size nType, Size nL, Size nB) {
    assert(n == seqByTypeLR.size());
    assert(n == type.size());
    assert(n == LR.size());

    vector<Size> jL(nType,0);
    vector<Size> jR(nType,0);
    vector<Size> jB(nType,0);
    vector<Size> accTypeB(nType,0);
    for (PosInt i = 0; i<type.size(); i++) {
        if (LR[i] < 0) {
            accTypeL[type[i]]++;
        } else {
            if (LR[i] == 0) {
                accTypeB[type[i]]++;
            } else {
                accTypeR[type[i]]++;
            }
        }
    }
    for (PosInt i = 0; i<nType; i++) {
        for (PosInt j = i+1; j<nType; j++) {
            accTypeL[j] += accTypeL[i];
        }
    }
    accTypeL.insert(accTypeL.begin(),0);

    for (PosInt i = 0; i<nType; i++) {
        for (PosInt j = i+1; j<nType; j++) {
            accTypeB[j] += accTypeB[i];
        }
    }
    accTypeB.insert(accTypeB.begin(),0);

    Size _nL = accTypeL.back();
    assert(_nL == nL);
    for (PosInt i = 0; i<nType+1; i++) {
        accTypeB[i] += _nL;
    }
    for (PosInt i = 0; i<nType; i++) {
        for (PosInt j = i+1; j<nType; j++) {
            accTypeR[j] += accTypeR[i];
        }
    }
    accTypeR.insert(accTypeR.begin(),0);
    Size _nB = accTypeB.back();
    assert(_nB == nB + nL);
    for (PosInt i = 0; i<nType+1; i++) {
        accTypeR[i] += _nB;
    }
    for (Size i = 0; i<LR.size(); i++) {
        if (LR[i] < 0) {
            seqByTypeLR[accTypeL[type[i]] + jL[type[i]]] = i;
            jL[type[i]]++;
        } else{
            if (LR[i] == 0) {
                seqByTypeLR[accTypeB[type[i]] + jB[type[i]]] = i;
                jB[type[i]]++;
            } else {
                seqByTypeLR[accTypeR[type[i]] + jR[type[i]]] = i;
                jR[type[i]]++;
            }
        }
    }
    accTypeL.pop_back();
    accTypeL.insert(accTypeL.end(), accTypeB.begin(), accTypeB.end());
    accType.insert(accType.end(), accTypeL.begin(), accTypeL.end());
    accType.pop_back();
    accType.insert(accType.end(), accTypeR.begin(), accTypeR.end());
    accTypeB.pop_back();
    accTypeR.insert(accTypeR.begin(), accTypeB.begin(), accTypeB.end());
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
        Float p_n_LGNeff[], // size: nType
        Size max_LGNeff[], // size: nType
        Float envelopeSig,
        Size qn,
        PosInt nType,
        const Size n,
		const pair<vector<Float>, vector<Float>> &cart, // V1 VF position (tentative)
        const pair<vector<Float>,vector<Float>> &cart0, // LGN VF position
		const vector<RFtype> &V1Type,
		const vector<PosInt> &type,
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
    vector<vector<Float>> srList(n);
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
			vector<Float> x(m);
            vector<Float> y(m);
			// LGN types of the pool
			vector<InputType> iType(m);
			// assert no need to reset at each end of loop
			for (Size j = 0; j < m; j++) {
				x[j] = (cart0.first[poolList[i][j]]);
				y[j] = (cart0.second[poolList[i][j]]);
				iType[j] = (LGNtype[poolList[i][j]]);
			}
			// initialize the V1 neuron to the corresponding RF type
			switch (V1Type[qn + i]) {
				case RFtype::nonOppopent_cs: 
					assert(RefType[qn + i] == OutputType::LonMon || RefType[qn + i] == OutputType::LoffMoff);
					RF = &NO_CS;
					break;
				case RFtype::nonOppopent_gabor: 
					assert(RefType[qn + i] == OutputType::LonMon || RefType[qn + i] == OutputType::LoffMoff);
					RF = &NO_G;
					break;
				case RFtype::doubleOppopent_cs: 
					assert(RefType[qn + i] == OutputType::LonMoff || RefType[qn + i] == OutputType::LoffMon);
					RF = &DO_CS;
					break;
				case RFtype::doubleOppopent_gabor: 
					assert(RefType[qn + i] == OutputType::LonMoff || RefType[qn + i] == OutputType::LoffMon);
					RF = &DO_G;
					break;
				case RFtype::singleOppopent: 
					assert(RefType[qn + i] == OutputType::LonMoff || RefType[qn + i] == OutputType::LoffMon);
					RF = &SO;
					break;
				default: throw "no such type of receptive field for V1 so defined (./src/preprocess/RFtype.h";
			}
			// construct pooled LGN neurons' connection to the V1 neuron
            Float p = p_n_LGNeff[type[i]];
            bool percentOrNumber = p < 0;
            if (percentOrNumber) {
                p = -p;
            }
			Size maxN = max_LGNeff[type[i]] > poolList[i].size()? poolList.size(): max_LGNeff[type[i]];
			vector<Float> strengthList;
            strengthList.reserve(maxN);
			if (conThres >= 0) {
				Float qfreq;
				RF->setup_param(m, sfreq[i], phase[qn+i], 1.0, theta[i], a[i], baRatio[i], RefType[qn+i], strictStrength, envelopeSig);
                PosInt qotype0 = static_cast<PosInt>(RefType[qn + i]);
				m = RF->construct_connection_opt(x, y, iType, poolList[i], strengthList, modAmp_nCon[qn+i], qfreq, cart.first[i], cart.second[i], i, ori_tol, get_acuity(ecc[i])/a[i]*disLGN);
                PosInt qotype1 = static_cast<PosInt>(RF->oType);
                if (qn == 430528 && i == 1) {
                    cout << qotype0 << ", " << qotype1 << "\n";
                }
                assert((qotype0 - 1.5) * (qotype1-1.5) > 0);
				sfreq[i] = qfreq;
			} else {
            	if (SimpleComplex == 0) {
				    RF->setup_param(m, sfreq[i], phase[qn+i], modAmp_nCon[qn+i], theta[i], a[i], baRatio[i], RefType[qn+i], strictStrength, envelopeSig);
				    m = RF->construct_connection_N(x, y, iType, poolList[i], strengthList, rGen, p*1.0, percentOrNumber, maxN, top_pick);
            	} else {
				    RF->setup_param(m, sfreq[i], phase[qn+i], 1.0, theta[i], a[i], baRatio[i], RefType[qn+i], strictStrength, envelopeSig);
				    m = RF->construct_connection_N(x, y, iType, poolList[i], strengthList, rGen, p*modAmp_nCon[qn+i], percentOrNumber, maxN, top_pick);
            	}
			}
            RefType[qn+i] = RF->oType;
			srList[i] = strengthList;
			if (m > 0) { 
				x.resize(m);
				y.resize(m);
				for (Size j = 0; j < m; j++) {
					x[j] = cart0.first[poolList[i][j]];
					y[j] = cart0.second[poolList[i][j]];
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
			//srList[i] = vector<Float>();
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
        const vector<Size> &m,
        const vector<Size> &idOffset,
        const Float radius) {
    vector<Size> index;

    //Float mx = 0.0f;
    //Float my = 0.0f;
	//Float min_dis;
	//Size min_id;
    for (PosInt i=0; i<idOffset.size(); i++) {
        for (Size j=idOffset[i]; j<idOffset[i]+m[i]; j++) {
	    	Float dis = norm_vec(cart.first[j] - x0, cart.second[j] - y0);
	    	/*if (i == 0) {
	    		min_dis = dis;
	    	}
	    	else {
	    		if (dis < min_dis) {
	    			min_id = i;
	    		}
	    	}*/
            if (dis < radius) {
                index.push_back(j);
	    		//mx = mx + cart.first[i];
	    		//my = my + cart.second[i];
            }
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
	Float* __restrict__ x0, // only relevant layers are included
	Float* __restrict__ y0,
    Float* __restrict__ VFposEcc,
    Float* __restrict__ baRatio,
    Float* __restrict__ a,
    Float* __restrict__ theta,
    Size* __restrict__ poolList,
    Size* __restrict__ nPool,
    PosInt* __restrict__ id, // for neuronal id in layers
    PosInt* __restrict__ idOffset,
    Size* __restrict__ mLpVp, // LGN pool size per neuron (based on type)
    Size* __restrict__ accType,
    BigSize* __restrict__ accTypeN,
    Size i0, Size n, Size m, Size mLayer,
    BigSize seed,
    Float LGN_V1_RFratio,
    Size nType,
    PosInt iLayer
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
    Size iPool;
    PosIntL pool_id;
    Size max_LGN;
    if (V1_id < i0+n) { // get radius
        PosInt iType;
        for (PosInt i=1; i<2*nType+1; i++) {
            if (V1_id < accType[i]) {
                iType = i-1;
                break;
            }
        }
        Float ecc = VFposEcc[V1_id];
        Float baR = baRatio[V1_id];
        V1_x = x[V1_id];
        V1_y = y[V1_id];
		curand_init(seed + V1_id, 0, 0, &rGen);
        Float R = mapping_rule_CUDA(ecc*60.0, rGen, LGN_V1_RFratio, iLayer)/60.0;
        // R = sqrt(area)
        max_LGN = mLpVp[iType];
        Float sqr = square_root(M_PI*baR);
		Float local_a = R/sqr;
        a[V1_id] = local_a;
        iPool = nPool[V1_id]; // read from array, could be binocular
        pool_id = (V1_id-accType[iType])*max_LGN + accTypeN[iType];
    }
    // scan and pooling LGN
    for (Size iPatch = 0; iPatch < nPatch; iPatch++) {
        // load LGN pos to __shared__
        Size nLGN;
        if (iPatch < nPatch - 1 || threadIdx.x < nRemain) {
            Size LGN_id = iPatch*blockDim.x + threadIdx.x;
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
                    PosInt LGN_id = iPatch*blockDim.x + iLGN;
                    for (PosInt iLayer=0; iLayer<mLayer; iLayer++) {
                        if (LGN_id < id[iLayer+1]) {
                            LGN_id += idOffset[iLayer] + id[iLayer];
                            break;
                        }
                    }
                    if (LGN_id >= idOffset[mLayer-1] + id[mLayer]) {
                        printf("%u/%u, LGN_id:%u < j0(%u) + m(%u); nRemain = %u, nLGN = %u\n", iPatch, nPatch, LGN_id, idOffset[mLayer-1], id[mLayer], nRemain, nLGN); 
                        assert(LGN_id < idOffset[mLayer-1] + id[mLayer]);
                    }

                    PosIntL pid = pool_id + static_cast<PosIntL>(iPool);
                    poolList[pid] = LGN_id;
					//if (V1_id == 41652) {
                    //    printf("## %u-%u(%u):, dx = %f, dy = %f, theta = %f, a =%f, b = %f\n lgnx = %f, lgny = %f\n v1x = %f, v1y = %f\n 1-value = %e\n", V1_id, iPool, LGN_id, sx[iLGN]-V1_x, sy[iLGN]-V1_y, theta[V1_id]*180/M_PI, a[V1_id], a[V1_id]*baRatio[V1_id], sx[iLGN], sy[iLGN], V1_x, V1_y, 1-value); 
					//}
                    if (iPool > max_LGN) {
                        printf("V1%u:#%u > %u, a = %f, b = %f, theta = %f\n", V1_id, iPool, max_LGN, a[V1_id], a[V1_id]*baRatio[V1_id], theta[V1_id]*180/M_PI); 
                        assert(iPool <= max_LGN);
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
        PosInt layerMatch[],
        vector<Float> &VFposEcc,
        bool useCuda,
        RandomEngine &rGen,
        vector<Float> &baRatio,
        vector<Float> &a,
        vector<Float> &theta,
        vector<PosInt> &type,
        vector<Int> &LR,
        Size nL,
        Size nB,
        vector<Size> &mL,
        vector<Size> &mR,
        Size mLayer,
        Size maxLGNperV1pool[], // size of nType
		BigSize seed,
		Float LGN_V1_RFratio,
        Size nType,
        Size iLayer
) {
    const Size n = cart.first.size();
    vector<vector<PosInt>> poolList(n);
    a.resize(n);
    if (useCuda) {
		vector<Float> xLR(n);
        vector<Float> yLR(n);
        vector<Float> baRatioLR(n);
        vector<Float> VFposEccLR(n);
        vector<Float> thetaLR(n);
		vector<PosInt> seqByTypeLR(n);
		vector<PosInt> accTypeL(nType,0);
		vector<PosInt> accTypeR(nType,0);
		vector<PosInt> accType;
        seqTypeLR(n, seqByTypeLR, type, LR, accTypeL, accTypeR, accType, nType, nL, nB);
		original2typeLR(n, seqByTypeLR, cart.first, xLR);
		original2typeLR(n, seqByTypeLR, cart.second, yLR);
		original2typeLR(n, seqByTypeLR, baRatio, baRatioLR);
		original2typeLR(n, seqByTypeLR, VFposEcc, VFposEccLR);
		original2typeLR(n, seqByTypeLR, theta, thetaLR);
        Float *d_memblock;

        Size mL_total = 0;
        Size mR_total = 0;
        vector<PosInt> lid; // offset within picked layer
        vector<PosInt> lidOffset; // offset within all lgn layers
        lid.push_back(0);
        vector<PosInt> pickLayer;
        Size temp_offset = 0;
        for (PosInt i=0; i<mLayer; i++) {
            if (layerMatch[i] == 1) {
                mL_total += mL[i];          // 
                lid.push_back(mL_total);    // before which "i" apply the offset.
                lidOffset.push_back(temp_offset);
                mR_total += mR[i];
                pickLayer.push_back(i);
            } else {
                temp_offset += mL[i];
            }
        }
        Size nPickLayer = pickLayer.size();

        Size m_max_total;
        if (mL_total > mR_total) {
            m_max_total = mL_total;
        } else {
            m_max_total = mR_total;
        }

        vector<Size> mLpVp(nType*2);
        vector<BigSize> accTypeL_N(nType*2+1);
        vector<BigSize> accTypeR_N(nType*2+1);
        vector<BigSize> accType_N;
        BigSize poolSizeL = 0;
        BigSize poolSizeR = 0;
        for (PosInt i=0; i<nType*2; i++) {
            accTypeL_N[i] = poolSizeL;
            accTypeR_N[i] = poolSizeR;
            mLpVp[i] = maxLGNperV1pool[i%nType];
            poolSizeL += (accTypeL[i+1]- accTypeL[i])*mLpVp[i];
            poolSizeR += (accTypeR[i+1]- accTypeR[i])*mLpVp[i];
        }
        accTypeL_N[nType*2] = poolSizeL;
        accType_N.insert(accType_N.end(), accTypeL_N.begin(), accTypeL_N.end()-1);
        accTypeR_N[nType*2] = poolSizeR;
        for (PosInt i=0; i<nType*2+1; i++) {
            accTypeR_N[i] += poolSizeL;
            accType_N.push_back(accTypeR_N[i]);
        }
        BigSize poolSize = poolSizeL + poolSizeR;

        cout << "nL = " << nL << "\n";
        cout << "nB = " << nB << "\n";
        cout << "n = " << n << "\n";

        cout << "(accType, accType_N):\n";
        for (PosInt i=0; i<nType*3+1; i++) {
            cout << "(" << accType[i] << ", " << accType_N[i] << ")";
            if (i < nType*3) {
                cout << ", ";
            } else {
                cout << "\n";
            }
        }
        cout << "poolSize = " << poolSize << "\n";
        assert(poolSize == accType_N.back());


        size_t d_memSize = ((2+4)*n + 2*m_max_total)*sizeof(Float) + static_cast<size_t>(poolSize)*sizeof(PosInt) + n*sizeof(Size) + (nPickLayer*2+1)*sizeof(Size) + (4*nType+1)*sizeof(Size) + (2*nType+1)*sizeof(BigSize);
		cout << "poolList need "<< static_cast<size_t>(poolSize)*sizeof(PosInt)/1024.0/1024.0 << "mb memory.\n";
        checkCudaErrors(cudaMalloc((void **) &d_memblock, d_memSize));
		cout << "need global memory of " << d_memSize / 1024.0 / 1024.0 << "mb in total\n";
        Float* d_x = d_memblock;
        Float* d_y = d_x + n;
        Float* d_x0 = d_y + n;
        Float* d_y0 = d_x0 + m_max_total;
        Float* d_baRatio = d_y0 + m_max_total;
        Float* d_a = d_baRatio + n; // to be filled
        Float* d_theta = d_a + n;
        Float* d_VFposEcc = d_theta + n;
        PosInt* d_id = (PosInt*) (d_VFposEcc + n);
        PosInt* d_idOffset = d_id + nPickLayer+1;
		// to be filled
        PosInt* d_poolList = (PosInt*) (d_idOffset + nPickLayer);
        Size* d_nPool = (Size*) (d_poolList + poolSize); // count of LGN connected
        Size* d_mLpVp = d_nPool + n;
        Size* d_accType = d_mLpVp + nType*2;
        BigSize* d_accTypeN = (BigSize*) (d_accType + nType*2+1);

        checkCudaErrors(cudaMemset(d_nPool, 0, n*sizeof(Size)));
        checkCudaErrors(cudaMemcpy(d_x, &(xLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, &(yLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_baRatio, &(baRatioLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_VFposEcc, &(VFposEccLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_theta, &(thetaLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        for (PosInt i = 0; i < nPickLayer; i++) {
            checkCudaErrors(cudaMemcpy(d_x0 + lid[i], &(cart0.first[lidOffset[i]+lid[i]]), mL[pickLayer[i]]*sizeof(Float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_y0 + lid[i], &(cart0.second[lidOffset[i]+lid[i]]), mL[pickLayer[i]]*sizeof(Float), cudaMemcpyHostToDevice));
        }
        checkCudaErrors(cudaMemcpy(d_id, &(lid[0]), (nPickLayer+1)*sizeof(Size), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_idOffset, &(lidOffset[0]), nPickLayer*sizeof(Size), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_mLpVp, &(mLpVp[0]), 2*nType*sizeof(Size), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_accType, &(accTypeL[0]), (2*nType+1)*sizeof(Size), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_accTypeN, &(accTypeL_N[0]), (2*nType+1)*sizeof(BigSize), cudaMemcpyHostToDevice));
        // TODO: stream kernels in chunks for large network
        Size nblock = (nL + blockSize-1)/blockSize;
        cout <<  "left: <<<" << nblock << ", " << blockSize << ">>>\n";
        vf_pool_CUDA<<<nblock, blockSize>>>(d_x, d_y, d_x0, d_y0, d_VFposEcc, d_baRatio, d_a, d_theta, d_poolList, d_nPool, d_id, d_idOffset, d_mLpVp, d_accType, d_accTypeN, 0, nL + nB, mL_total, nPickLayer, seed, LGN_V1_RFratio, nType, iLayer);
        getLastCudaError("vf_pool for the left eye failed");
        cudaDeviceSynchronize();

        mR_total = 0;
        vector<PosInt> ridOffset;
        vector<PosInt> rid;
        rid.push_back(0);
        temp_offset = mL_total;
        for (PosInt i=0; i<mLayer; i++) {
            if (layerMatch[i] == 1) {
                mR_total += mR[i];
                rid.push_back(mR_total);
                ridOffset.push_back(temp_offset);
            } else {
                temp_offset += mR[i];
            }
        }

		if (mR_total > 0 && (n-nL)> 0) {
			for (PosInt i = 0; i < nPickLayer; i++) {
        	    checkCudaErrors(cudaMemcpy(d_x0 + rid[i], &(cart0.first[ridOffset[i] + rid[i]]), mR[pickLayer[i]]*sizeof(Float), cudaMemcpyHostToDevice));
        	    checkCudaErrors(cudaMemcpy(d_y0 + rid[i], &(cart0.second[ridOffset[i] + rid[i]]), mR[pickLayer[i]]*sizeof(Float), cudaMemcpyHostToDevice));
        	}
        	checkCudaErrors(cudaMemcpy(d_id, &(rid[0]), (nPickLayer+1)*sizeof(Size), cudaMemcpyHostToDevice));
        	checkCudaErrors(cudaMemcpy(d_idOffset, &(ridOffset[0]), nPickLayer*sizeof(Size), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_accType, &(accTypeR[0]), (2*nType+1)*sizeof(Size), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_accTypeN, &(accTypeR_N[0]), (2*nType+1)*sizeof(BigSize), cudaMemcpyHostToDevice));

			nblock = (n-nL + blockSize-1) / blockSize;
        	cout <<  "right: <<<" << nblock << ", " << blockSize << ">>>\n";
        	vf_pool_CUDA<<<nblock, blockSize>>>(d_x, d_y, d_x0, d_y0, d_VFposEcc, d_baRatio, d_a, d_theta, d_poolList, d_nPool, d_id, d_idOffset, d_mLpVp, d_accType, d_accTypeN, nL, n-nL, mR_total, nPickLayer, seed, LGN_V1_RFratio, nType, iLayer);
        	getLastCudaError("vf_pool for the right eye failed");
		}
		vector<vector<PosInt>> poolListLR(n);
        PosInt* poolListArray = new Size[poolSize];
        Size* nPool = new Size[n];
		vector<Float> aLR(n);
        checkCudaErrors(cudaMemcpy(&aLR[0], d_a, n*sizeof(Float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(poolListArray, d_poolList, poolSize*sizeof(PosInt), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(nPool, d_nPool, n*sizeof(Size), cudaMemcpyDeviceToHost));
        for (Size i=0; i<n; i++) {
            PosInt iType;
            for (Size j=1; j<nType*3+1; j++) {
                if (i < accType[j]) {
                    iType = j-1;
                    break;
                }
            }
            size_t pid = static_cast<PosIntL>(i-accType[iType])*maxLGNperV1pool[iType] + accType_N[iType];
            vector<Size> iPool(poolListArray+pid, poolListArray+pid+nPool[i]);
            poolListLR[i] = iPool;
        }
        Size maxPool = *max_element(nPool, nPool+n);
        delete []poolListArray;
        delete []nPool;
        assert(maxPool < *max_element(maxLGNperV1pool, maxLGNperV1pool+nType));
		typeLR2original(n, seqByTypeLR, poolListLR, poolList);
		typeLR2original(n, seqByTypeLR, aLR, a);
        checkCudaErrors(cudaFree(d_memblock));
    } else {
        vector<Float> normRand(n);
        vector<Float> rMap(n);
        normal_distribution<Float> dist(0, 1);
        // find radius from mapping rule at (e,p)
        for (Size i=0; i<n; i++) {
            normRand[i] = dist(rGen); // generate random number for RF size distribution
			// convert input eccentricity to mins and convert output back to degrees.
            Float R = mapping_rule(VFposEcc[i]*60, normRand[i], rGen, LGN_V1_RFratio, iLayer)/60.0;
            a[i] = R/sqrt(M_PI*baRatio[i]);
            Float b = R*R/M_PI/a[i];
            if (a[i] > b) {
                rMap[i] = a[i];
            } else {
                rMap[i] = b;
            }
        }
        // next nearest neuron j to (e,p) in sheet 1
        vector<PosInt> lidOffset;
        vector<PosInt> bidOffset;
        vector<Size> mL_pick;
        vector<Size> m_pick;
        Size offset = 0;
        for (PosInt i=0; i<mLayer; i++) {
            if (layerMatch[i] == 1) {
                lidOffset.push_back(offset);
                bidOffset.push_back(offset);
                mL_pick.push_back(mL[i]);
                m_pick.push_back(mL[i]);
            }
            offset += mL[i];
        }

        vector<Size> mR_pick;
        vector<PosInt> ridOffset;
        for (PosInt i=0; i<mLayer; i++) {
            if (layerMatch[i] == 1) {
                bidOffset.push_back(offset);
                ridOffset.push_back(offset);
                m_pick.push_back(mR[i]);
                mR_pick.push_back(mR[i]);
            }
            offset += mR[i];
        }

        Size _maxLGNperV1pool = 0;
        for (Size i=0; i<n; i++) {
            if (LR[i] > 0) {
                poolList[i] = draw_from_radius(cart.first[i], cart.second[i], cart0, mL_pick, lidOffset, rMap[i]);
            } else {
                if (LR[i] == 0) {
                    poolList[i] = draw_from_radius(cart.first[i], cart.second[i], cart0, m_pick, bidOffset, rMap[i]);
                } else {
                    poolList[i] = draw_from_radius(cart.first[i], cart.second[i], cart0, mR_pick, ridOffset, rMap[i]);
                }
            }
            if (poolList[i].size() > _maxLGNperV1pool) _maxLGNperV1pool = poolList[i].size();
        }
        cout << "max LGN per V1 pool reaches " << _maxLGNperV1pool << "\n";
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
	vector<Size> max_LGNeff;
	vector<Size> maxLGNperV1pool;
	vector<Size> nTypeEI;
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
    vector<PosInt> inputLayer;
    vector<Float> V1_RFtypeAccDist, V1_RefTypeDist;
	string LGN_vpos_filename, V1_allpos_filename, V1_RFpreset_filename, V1_feature_filename, conLGN_suffix, res_suffix, inputFolder, outputFolder;
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
		("inputFolder", po::value<string>(&inputFolder)->default_value(""), "Folder that stores the input files")
		("outputFolder", po::value<string>(&outputFolder)->default_value(""), "Folder that stores the output files")
		("top_pick", po::value<bool>(&top_pick)->default_value(true), "preset number of connection, n, and connect to the neurons with the top n prob")
		("LGN_V1_RFratio,r", po::value<Float>(&LGN_V1_RFratio)->default_value(1.0), "LGN's contribution to the total RF size")
		("maxLGNperV1pool", po::value<vector<Size>>(&maxLGNperV1pool), "maximum pooling of LGN neurons per V1 neuron, array of nInputLayer x nType")
		("envelopeSig", po::value<Float>(&envelopeSig)->default_value(1.177), "LGN's pools connection probability envelope sigma on distance")
		("SimpleComplex", po::value<Int>(&SimpleComplex)->default_value(1), "determine how simple complex is implemented, through modulation modAmp_nCon(0) or number of LGN connection(1)")
		("conThres", po::value<Float>(&conThres)->default_value(-1), "connect to LGN using conThres")
		("ori_tol", po::value<Float>(&ori_tol)->default_value(15), "tolerance of preset orientation deviation in degree")
		("disLGN", po::value<Float>(&disLGN)->default_value(1.0), "average visual distance between LGN cells")
		("strictStrength", po::value<bool>(&strictStrength)->default_value(true), "make nLGN*sLGN strictly as preset")
		("pureComplexRatio", po::value<vector<Float>>(&pureComplexRatio), "determine the proportion of simple and complex in each input layer, with size of nType")
		("V1_RFtypeAccDist", po::value<vector<Float>>(&V1_RFtypeAccDist), "determine the relative portion of each V1 RF type")
		("nRefTypeV1_RF", po::value<vector<Size>>(&nRefTypeV1_RF), "determine the number of cone/ON-OFF combinations for each V1 RF type")
		("V1_RefTypeID", po::value<vector<Size>>(&V1_RefTypeID), "determine the ID of the available cone/ON-OFF combinations in each V1 RF type")
		("V1_RefTypeDist", po::value<vector<Float>>(&V1_RefTypeDist), "determine the relative portion of the available cone/ON-OFF combinations in each V1 RF type")
        ("inputLayer", po::value<vector<PosInt>>(&inputLayer), "array of V1 layer IDs that recieve inputs from the LGN layers with size nInputLayer")
		("nTypeEI", po::value<vector<Size>>(&nTypeEI), "a vector of hierarchical types differs in non-functional properties: reversal potentials, characteristic lengths of dendrite and axons, e.g. in the form of nTypeE, nTypeI for each layer (2 x nLayer). {Exc-Pyramidal, Exc-stellate; Inh-PV, Inh-SOM, Inh-LTS} then for that input layer the element would be {3, 2}")
		("typeAccCount",po::value<vector<Size>>(&typeAccCount), "neuronal types' discrete accumulative distribution size of nType for input layers only")
		("max_LGNeff", po::value<vector<Size>>(&max_LGNeff), "max realizable number of connections [0,n] for inputLayers")
		("p_n_LGNeff", po::value<vector<Float>>(&p_n_LGNeff), "LGN conneciton probability [-1,0], or number of connections [0,n] for nInputLayer x nType")
        ("V1_LGN_layer_match", po::value<vector<PosInt>>(&layerMatch), "nInputLayer arrays of how each V1 layer recieve inputs from the LGN layers (mLayer)")
		("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature"), "file that stores V1 neurons' parameters")
		("fV1_RFpreset", po::value<string>(&V1_RFpreset_filename)->default_value("V1_RFpreset"), "file that stores V1 neurons' parameters")
		("fLGN_vpos", po::value<string>(&LGN_vpos_filename)->default_value("LGN_vpos"), "file that stores LGN position in visual field (and on-cell off-cell label)")
		("fV1_allpos", po::value<string>(&V1_allpos_filename)->default_value("V1_allpos"), "file that stores V1 position in visual field)");

    string V1_RFprop_filename, idList_filename, srList_filename, output_cfg_filename;
	po::options_description output_opt("output options");
	output_opt.add_options()
		("conLGN_suffix", po::value<string>(&conLGN_suffix)->default_value(""), "conLGN_suffix for output file")
		("fV1_RFprop", po::value<string>(&V1_RFprop_filename)->default_value("V1_RFprop"), "file that stores V1 neurons' information")
		("fLGN_V1_ID", po::value<string>(&idList_filename)->default_value("LGN_V1_idList"), "file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&srList_filename)->default_value("LGN_V1_sList"), "file stores LGN to V1 connection strengths")
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

    Size nLayer;
	vector<Size> nblock;
    vector<Size> neuronPerBlock;
    vector<Size> n;
	vector<pair<vector<Float>, vector<Float>>> cart; // V1 VF position (preset)
	vector<vector<Float>> VFposEcc;
	ifstream fV1_allpos;
	fV1_allpos.open(inputFolder + V1_allpos_filename + res_suffix, fstream::in | fstream::binary);
	if (!fV1_allpos) {
		cout << "Cannot open or find " << V1_allpos_filename + res_suffix <<" to read V1 positions.\n";
		return EXIT_FAILURE;
	} else {
		fV1_allpos.read(reinterpret_cast<char*>(&nLayer), sizeof(Size));
	    nblock.resize(nLayer);
        neuronPerBlock.resize(nLayer);
        n.resize(nLayer);
	    cart.resize(nLayer);
        VFposEcc.resize(nLayer);

		fV1_allpos.read(reinterpret_cast<char*>(&nblock[0]), sizeof(Size)*nLayer);
		fV1_allpos.read(reinterpret_cast<char*>(&neuronPerBlock[0]), sizeof(Size)*nLayer);
        auto double2Float = [](double x) {
            return static_cast<Float>(x);
	    };
	    auto xy2eccD2F = [] (double x, double y) {
	    	return static_cast<Float>(square_root(x*x+y*y));
	    };
        for (int iLayer = 0; iLayer < nLayer; iLayer++) {
            n[iLayer] = nblock[iLayer]*neuronPerBlock[iLayer];
            cout << n[iLayer] << " neurons for V1 layer " << iLayer << "\n";
	        vector<double> x0(n[iLayer]);
	        vector<double> y0(n[iLayer]);
	        fV1_allpos.seekg(2*n[iLayer]*sizeof(double), fV1_allpos.cur); // skip cortical positions
	        fV1_allpos.read(reinterpret_cast<char*>(&x0[0]), n[iLayer] * sizeof(double)); 
	        fV1_allpos.read(reinterpret_cast<char*>(&y0[0]), n[iLayer] * sizeof(double));
            VFposEcc[iLayer].resize(n[iLayer]);
            transform(x0.begin(), x0.end(), y0.begin(), VFposEcc[iLayer].begin(), xy2eccD2F);
	        vector<Float> x(n[iLayer]);
	        vector<Float> y(n[iLayer]);
	        transform(x0.begin(), x0.end(), x.begin(), double2Float);
	        transform(y0.begin(), y0.end(), y.begin(), double2Float);
            cout << "V1_x: [" << *min_element(x.begin(), x.end()) << ", " << *max_element(x.begin(), x.end()) << "]\n";
            cout << "V1_y: [" << *min_element(y.begin(), y.end()) << ", " << *max_element(y.begin(), y.end()) << "]\n";
	        cart[iLayer] = make_pair(x, y);
        }
    }
	fV1_allpos.close();

    Size mLayer;
	vector<Size> mL;
	vector<Size> mR;
    vector<Size> m;
    Size m_total;
    Size mL_total;
    Size mR_total;
	Float max_ecc;
	vector<InputType> LGNtype;
    pair<vector<Float>, vector<Float>> cart0; // V1 VF position (preset)
	ifstream fLGN_vpos;
	fLGN_vpos.open(inputFolder + LGN_vpos_filename + res_suffix, fstream::in | fstream::binary);
	if (!fLGN_vpos) {
		cout << "Cannot open or find " << LGN_vpos_filename << "\n";
		return EXIT_FAILURE;
	} else {
	    fLGN_vpos.read(reinterpret_cast<char*>(&mLayer), sizeof(Size));
	    mL.resize(mLayer);
	    mR.resize(mLayer);
        m.resize(mLayer);

	    fLGN_vpos.read(reinterpret_cast<char*>(&mL[0]), mLayer*sizeof(Size)); 
        fLGN_vpos.read(reinterpret_cast<char*>(&mR[0]), mLayer*sizeof(Size));
	    fLGN_vpos.read(reinterpret_cast<char*>(&max_ecc), sizeof(Float));
	    {// not used
	    	Float tmp;
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // x0
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // xspan
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // y0
	    	fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // yspan
	    }
        m_total = 0;
        mL_total = 0;
        mR_total = 0;
	    vector<Float> x0;
	    vector<Float> y0;
        for (PosInt iLayer = 0; iLayer < mLayer; iLayer++) {
            m[iLayer] = mL[iLayer] + mR[iLayer];
            m_total += m[iLayer];
            mL_total += mL[iLayer];
            mR_total += mR[iLayer];
	        vector<Float> x(mL[iLayer]);
	        vector<Float> y(mL[iLayer]);
	        fLGN_vpos.read(reinterpret_cast<char*>(&x[0]), mL[iLayer]*sizeof(Float));
	        fLGN_vpos.read(reinterpret_cast<char*>(&y[0]), mL[iLayer]*sizeof(Float));
            cout << m[iLayer] << " LGN neurons, " << mL[iLayer] << " from left eye in layer " << iLayer << "\n"; 
            cout << "LGN_x: [" << *min_element(x.begin(), x.end()) << ", " << *max_element(x.begin(), x.end()) << "]\n";
            cout << "LGN_y: [" << *min_element(y.begin(), y.end()) << ", " << *max_element(y.begin(), y.end()) << "]\n";
            x0.insert(x0.end(), std::make_move_iterator(x.begin()),std::make_move_iterator(x.end()));
            y0.insert(y0.end(), std::make_move_iterator(y.begin()),std::make_move_iterator(y.end()));
        }
        if (mR_total > 0) {
            for (PosInt iLayer = 0; iLayer < mLayer; iLayer++) {
	            vector<Float> x(mR[iLayer]);
	            vector<Float> y(mR[iLayer]);
	            fLGN_vpos.read(reinterpret_cast<char*>(&x[0]), mR[iLayer]*sizeof(Float));
	            fLGN_vpos.read(reinterpret_cast<char*>(&y[0]), mR[iLayer]*sizeof(Float));
                cout << m[iLayer] << " LGN neurons, "<< mR[iLayer] << " from right eye in layer " << iLayer << ".\n";
                cout << "LGN_x: [" << *min_element(x.begin(), x.end()) << ", " << *max_element(x.begin(), x.end()) << "]\n";
                cout << "LGN_y: [" << *min_element(y.begin(), y.end()) << ", " << *max_element(y.begin(), y.end()) << "]\n";
                x0.insert(x0.end(), std::make_move_iterator(x.begin()),std::make_move_iterator(x.end()));
                y0.insert(y0.end(), std::make_move_iterator(y.begin()),std::make_move_iterator(y.end()));
            }
        }
	    cart0 = make_pair(x0, y0);
	    LGNtype.resize(m_total);
	    fLGN_vpos.read(reinterpret_cast<char*>(&LGNtype[0]), m_total * sizeof(PosInt));
    }
	fLGN_vpos.close();

	cout << "carts ready\n";

    vector<BigSize> seeds{seed,seed+13};
    seed_seq seq(seeds.begin(), seeds.end());
    RandomEngine rGen(seq);

	vector<RFtype> V1Type; // defined in RFtype.h [0..4], RF shapes
	vector<OutputType> RefType; // defined in in RFtype.h [0..3] conetype placements
	vector<Float> phase; // [0, 2*pi], control the phase
    // theta is read from fV1_feature
	vector<Float> modAmp_nCon; // [0,1] controls simple complex ratio, through subregion overlap ratio, or number of LGN connected.
	vector<vector<Size>> typeAcc0(nLayer);

    Size nInputLayer;
    Size input_ntotal = 0;
    Size totalType = 0;
    vector<Size> nType;
    if (readFromFile) {
        cout << "ignoring inputLayer, nTypeEI, typeAccCount, V1Type, RefType in the config file\n";
        fstream fV1_RFpreset(inputFolder + V1_RFpreset_filename, fstream::in | fstream::binary);
	    if (!fV1_RFpreset) {
	    	cout << "Cannot open or find " << V1_RFpreset_filename <<"\n";
	    	return EXIT_FAILURE;
	    } else {
	        fV1_RFpreset.read(reinterpret_cast<char*>(&nInputLayer), sizeof(Size));
            if (nInputLayer > nLayer) {
                cout << "nInputLayer = " << nInputLayer << " is larger than the number of V1 layers available nLayer(" << nLayer << ")";
                return EXIT_FAILURE;
            }
	        nType.resize(nInputLayer);
            vector<Size> _inputLayer(nInputLayer);
		    fV1_RFpreset.read(reinterpret_cast<char*>(&_inputLayer[0]), sizeof(Size)*nInputLayer);
            vector<Size> _n(nInputLayer);
		    fV1_RFpreset.read(reinterpret_cast<char*>(&_n[0]), sizeof(Size)*nInputLayer);
	        fV1_RFpreset.read(reinterpret_cast<char*>(&nTypeEI[0]), sizeof(Size)*2*nInputLayer);
            for (int iLayer = 0; iLayer<nInputLayer; iLayer++) {
                if (n[inputLayer[iLayer]] != _n[iLayer]) {
                    cout << "feature size in layer " << inputLayer[iLayer] << " is " << _n[iLayer] << ", inconsistent with the size from V1_allpos.bin, " << n[inputLayer[iLayer]] << "\n";
                    return EXIT_FAILURE;
                }
                input_ntotal += _n[iLayer];

                nType[iLayer] = nTypeEI[2*iLayer] + nTypeEI[2*iLayer+1];
                totalType += nType[iLayer];
            }

            typeAccCount.resize(totalType);
	        fV1_RFpreset.read(reinterpret_cast<char*>(&typeAccCount[0]), totalType*sizeof(Size));
            V1Type.resize(input_ntotal);
            RefType.resize(input_ntotal);
            phase.resize(input_ntotal);
            modAmp_nCon.resize(input_ntotal);

	        fV1_RFpreset.read(reinterpret_cast<char*>(&V1Type[0]), input_ntotal * sizeof(RFtype_t));
	        fV1_RFpreset.read(reinterpret_cast<char*>(&RefType[0]), input_ntotal * sizeof(OutputType_t));
            Size qn = 0;
            for (int iLayer = 0; iLayer<nInputLayer; iLayer++) {
	            fV1_RFpreset.read(reinterpret_cast<char*>(&phase[qn]), n[inputLayer[iLayer]] * sizeof(Float));
	            fV1_RFpreset.read(reinterpret_cast<char*>(&modAmp_nCon[qn]), n[inputLayer[iLayer]] * sizeof(Float));
                qn += n[inputLayer[iLayer]];
            }
            fV1_RFpreset.close();
        }
    } else {
        nInputLayer = inputLayer.size();    
        nType.resize(nLayer);
        if (nInputLayer == 0) {
            cout << "no input layers assigned\n";
            return EXIT_FAILURE;
        }
        // discrete portions randomly distributed
        uniform_real_distribution<Float> uniform_01(0,1.0);
        normal_distribution<Float> norm_01(0,1.0);
        if (V1_RFtypeAccDist.size() != nInputLayer*nRFtype) {
            cout << "V1_RFtypeAccDist has a incorrect size should be " << nInputLayer << "(nInputLayer) x " << nRFtype << "(nRFtype)\n";
            return EXIT_FAILURE;
        }
        if (nTypeEI.size() != 2*nInputLayer) {
            cout << "size of nTypeEI is inconsistent with nInputLayer("<<nInputLayer<<")\n";
            return EXIT_FAILURE;
        }

        for (PosInt iLayer = 0; iLayer<nInputLayer; iLayer++) {
            nType[iLayer] = nTypeEI[2*iLayer] + nTypeEI[2*iLayer+1];
        }

        if (pureComplexRatio.size() == nInputLayer) {
            PosInt i=0;
            for (PosInt iLayer = 0; iLayer<nInputLayer; iLayer++) {

                pureComplexRatio.insert(pureComplexRatio.begin() + i+1, nType[iLayer]-1, pureComplexRatio[i]);
                i += nType[iLayer];
            }
            assert(pureComplexRatio.size() == i);
        } else {
            Size _totalType = 0;
            for (PosInt iLayer = 0; iLayer<nInputLayer; iLayer++) {
                _totalType += nType[iLayer]; 
            }
            if (pureComplexRatio.size() != _totalType) {
                cout << "pureComplexRatio size must be nInputLayer x nType\n"; 
                return EXIT_FAILURE;
            }
        }

        for (PosInt iLayer = 0; iLayer < nInputLayer; iLayer++) {
            PosInt jLayer = inputLayer[iLayer];
            if (accumulate(V1_RFtypeAccDist.begin() + iLayer*nRFtype,  V1_RFtypeAccDist.begin() + (iLayer+1)*nRFtype, 0.0) != 1.0) {
                cout << "V1_RFtypeAccDist does not sum to 1 for inputLayer:" << iLayer << " which is layer " << jLayer << "\n";
                return EXIT_FAILURE;
            }
        }
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

        PosInt iType = 0;
        Size qn = 0;
        for (PosInt iLayer=0; iLayer<nInputLayer; iLayer++) {
            input_ntotal += n[inputLayer[iLayer]];
        }
        V1Type.resize(input_ntotal);
        RefType.resize(input_ntotal);
        phase.resize(input_ntotal);
        modAmp_nCon.resize(input_ntotal);

        for (PosInt iLayer=0; iLayer<nInputLayer; iLayer++) {
            PosInt jLayer = inputLayer[iLayer];
            cout << "layer " << jLayer << ":\n";
	        if (nType[iLayer] > max_nType) {
	        	cout << "the accumulative distribution of neuronal type <typeAccCount> in layer" << jLayer << " has size of " << nType[iLayer] << " > " << max_nType << "\n";
	        	return EXIT_FAILURE;
	        }
	        typeAcc0[iLayer].push_back(0);
	        for (PosInt i=0; i<nType[iLayer]; i++) {
	        	typeAcc0[iLayer].push_back(typeAccCount[iType+i]);
	        }
            iType += nType[iLayer];

            vector<Size> nTypes(nRFtype,0);
            Size qType = nRFtype*iLayer;
            auto genV1_RFtype = [&uniform_01, &V1_RFtypeAccDist, &rGen, &nTypes, &qType] () {
                Float rand = uniform_01(rGen);
                for (PosInt i=qType; i<qType+nRFtype; i++) {
                    if (rand < V1_RFtypeAccDist[i]) {
                        nTypes[i-qType]++;
                        return static_cast<RFtype>(i-qType);
                    }
                }
            };
            generate(V1Type.begin() + qn, V1Type.begin()+ qn + n[jLayer], genV1_RFtype);
            cout << "None-G,    None-CS,   Single,  Double-CS,   Double-G\n";
            for (PosInt i=0; i<nRFtype; i++) {
                cout << nTypes[i] << "    ";
            }
            cout << "\n";

            auto genV1_RefType = [&uniform_01, &nRefTypeV1_RF, &RefTypeDist, &RefTypeID, &rGen] (RFtype V1TypeI) {
                Float rand = uniform_01(rGen);
                Size iRFtype = static_cast<Size>(V1TypeI);
                for (Size i = 0; i < nRefTypeV1_RF[iRFtype]; i++) {
                    if (rand < RefTypeDist[iRFtype][i]) {
                        return static_cast<OutputType>(RefTypeID[iRFtype][i]);
                    }
                }
            };
            transform(V1Type.begin() + qn, V1Type.begin() + qn + n[jLayer], RefType.begin() + qn, genV1_RefType);

            // uniform distribution
            auto genPhase = [&uniform_01, &rGen] () {
                return static_cast<Float>(uniform_01(rGen) * 2 * M_PI);
		    	//return 0;
            };
            generate(phase.begin() + qn, phase.begin() + qn + n[jLayer], genPhase);

    	    
		    for (PosInt i = 0; i<nblock[jLayer]; i++) {
		    	for (PosInt j=0; j<nType[iLayer]; j++) { 
		    		Float ratio = pureComplexRatio[totalType + j];
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
            			generate(modAmp_nCon.begin() + qn + i*neuronPerBlock[jLayer] + typeAcc0[iLayer][j], modAmp_nCon.begin() + qn + i*neuronPerBlock[jLayer] + typeAcc0[iLayer][j+1], genModAmp_nCon);
		    		} else {
            			auto genModAmp_nCon = [&uniform_01, &norm_01, &rGen, &ratio, &max_LGNeff, &iLayer] () {
            			    Float rand = uniform_01(rGen);
		    				Float v;
            			    if (rand >= ratio) {
            			        //v = 1.0;
            			        //v = rand;
		    					v = ratio + (1-ratio)/2 + norm_01(rGen)*square_root(0.5*0.5/max_LGNeff[iLayer]);
            			    } else {
            			        v = 0.0;
            			    }
		    				if (v>1) v=1;
		    				if (v<0) v=0;
		    				//Float v = 1.0;
            			    return v;
            			};
            			generate(modAmp_nCon.begin() + qn + i*neuronPerBlock[jLayer] + typeAcc0[iLayer][j], modAmp_nCon.begin() + qn + i*neuronPerBlock[jLayer] + typeAcc0[iLayer][j+1], genModAmp_nCon);
		    			// binomially distribute simple cell's modulation ratio over nLGN connections
		    			/*
            			auto genModAmp_nCon = [&norm_01, &rGen, &ratio, &p_n_LGNeff, &max_LGNeff, &iLayer] () {
		    				Float std = 
            			    Float rand = norm_01(rGen)*std + 0.5;
		    				Float v;
            			    return v;
            			};
            			generate(modAmp_nCon.begin() + i*neuronPerBlock[jLayer] + typeAcc0[iLayer][j], modAmp_nCon.begin() + i*neuronPerBlock[jLayer] + typeAcc0[iLayer][j+1], genModAmp_nCon);
		    			*/
		    		}
		    	}
            }
            qn += n[jLayer];
            totalType += nType[iLayer];
		}
    }

    vector<Size> input_n(nInputLayer);
    for (int iLayer = 0; iLayer<nInputLayer; iLayer++) {
        input_n[iLayer] = n[inputLayer[iLayer]];
    }

	Size nFeature; // not used here
	vector<vector<PosInt>> featureLayer(nLayer);
	vector<vector<Float>> theta(nLayer); // [0, 1] control the LGN->V1 RF orientation
	vector<vector<Float>> OD(nLayer);
    vector<vector<Float>>* feature[2] = {&OD, &theta};
    ifstream fV1_feature;
    fV1_feature.open(inputFolder + V1_feature_filename + res_suffix, ios::in|ios::binary);
	if (!fV1_feature) {
		cout << "failed to open pos file:" << V1_feature_filename << "\n";
		return EXIT_FAILURE;
	} else {
        fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
        for (int i=0; i<2; i++) { // only care OD and theta
            Size nfl;
            fV1_feature.read(reinterpret_cast<char*>(&nfl), sizeof(Size));
            featureLayer[i].resize(nfl);
            fV1_feature.read(reinterpret_cast<char*>(&(featureLayer[i][0])), nfl*sizeof(PosInt));
            for (int j=0; j<nInputLayer; j++) { // the first two feature are input-related
                bool contained = false;
                for (PosInt k=0; k<nfl; k++) {
                    if (inputLayer[j] == featureLayer[i][k]) {
                        contained = true;
                        break;
                    }
                }
                if (!contained) {
                    cout << "inputLayer " << inputLayer[j] << " not having input feature" << i << "\n";
                }
            }
            vector<Size> _n(nfl); 
            fV1_feature.read(reinterpret_cast<char*>(&_n[0]), nfl*sizeof(Size));
            for (int j=0; j<nfl; j++) {
                PosInt iLayer = featureLayer[i][j];
                if (n[iLayer] != _n[j]) {
                    cout << "feature size in layer " << iLayer << " is " << _n[j] << ", inconsistent with the size from V1_allpos.bin, " << n[iLayer] << "\n";
                    return EXIT_FAILURE;
                } else {
                    (*feature[i]).at(iLayer).resize(_n[j]);
	                fV1_feature.read(reinterpret_cast<char*>(&(*feature[i]).at(iLayer)[0]), _n[j] * sizeof(Float));
                }
            }
        }
    }
	fV1_feature.close();


    if (layerMatch.size() != nInputLayer*mLayer) {
        cout << "array size of V1_LGN_layer_match (" << layerMatch.size() << ") is incorrect, should be " << nInputLayer << "(nInputLayer) x " << mLayer << "(mLayer)\n";
        return EXIT_FAILURE;
    }
	cout << "V1 RF properties preset.\n";

    for (PosInt i=0; i<nInputLayer; i++) {
        PosInt qType = 0;
        for (PosInt j=0; j<nType[i]; j++) {
            assert(p_n_LGNeff[qType + j] != 0);
            assert(p_n_LGNeff[qType + j] >= -1);
            qType += nType[i];
        }
    }

	ofstream fV1_RFpreset;
    if (!readFromFile) {
        fV1_RFpreset.open(outputFolder + V1_RFpreset_filename, fstream::out | fstream::binary);
	    if (!fV1_RFpreset) {
	    	cout << "Cannot open or find " << V1_RFpreset_filename <<"\n";
	    	return EXIT_FAILURE;
	    } else {
		    fV1_RFpreset.write((char*)&nInputLayer, sizeof(Size));
		    fV1_RFpreset.write((char*)&inputLayer, nInputLayer*sizeof(Size));
		    fV1_RFpreset.write((char*)&input_n, nInputLayer*sizeof(Size));
		    fV1_RFpreset.write((char*)&typeAccCount[0], totalType*sizeof(Size));
	        fV1_RFpreset.write((char*)&V1Type[0], input_ntotal * sizeof(RFtype_t));
	        fV1_RFpreset.write((char*)&RefType[0], input_ntotal * sizeof(OutputType_t));
        }
    }

	ofstream fV1_RFprop(outputFolder + V1_RFprop_filename + conLGN_suffix, fstream::out | fstream::binary);
	if (!fV1_RFprop) {
		cout << "Cannot open or find V1." << V1_RFprop_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
        fV1_RFprop.write((char*)&nInputLayer, sizeof(Size));
        fV1_RFprop.write((char*)&input_n, nInputLayer*sizeof(Size));
    }

    ofstream f_idList(outputFolder + idList_filename + conLGN_suffix, fstream::out | fstream::binary);
	if (!f_idList) {
		cout << "Cannot open or find V1." << outputFolder + idList_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
        f_idList.write((char*)&input_ntotal, sizeof(Size));
    }
    f_idList.close();

    ofstream f_srList(outputFolder + srList_filename + conLGN_suffix, fstream::out | fstream::binary);
	if (!f_srList) {
		cout << "Cannot open or find V1." << outputFolder + srList_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
        f_srList.write((char*)&input_ntotal, sizeof(Size));
    }
    f_srList.close();

    vector<Float> meanPool(totalType);
    vector<Float> meanSum(totalType);
    Size qn = 0;
	Size qType = 0;
    for (PosInt iLayer = 0; iLayer < nInputLayer; iLayer++) {
        PosInt jLayer = inputLayer[iLayer];
        cout << "layer " << jLayer << "\n";
        Size nL = 0;
        Size nR = 0;
        Size nB = 0;
	    vector<Int> LR(n[jLayer]);
        if (OD[jLayer].size() != input_n[iLayer] || theta[jLayer].size() != input_n[iLayer]) {
            cout << "input layers have to have OD and theta features, size of OD/theta is " << OD[jLayer].size() << "\n";
            assert(OD[jLayer].size() == theta[jLayer].size());
            return EXIT_FAILURE;
        }
        vector<PosInt> type(n[jLayer]);
        for (PosInt i = 0; i<n[jLayer]; i++) {
            for (PosInt j=0; j<nType[iLayer]; j++) {
                if (i%neuronPerBlock[jLayer] < typeAccCount[qType + j]) {
                    type[i] = j;
                    break;
                }
            }
            if (OD[jLayer][i] == 0) {
                LR[i] = -1;
                nL++;
            } else {
                if (OD[jLayer][i] == 1.0) {
                    LR[i] = 1;
                    nR++;
                } else {
                    LR[i] = 0;
                    nB++;
                }
            }
            theta[jLayer][i] = (theta[jLayer][i]-0.5)*M_PI;
        }
        cout << "   left eye: " << nL << " V1 neurons\n";
        cout << "   right eye: " << nR << " V1 neurons\n";
        cout << "   binocular neurons: " << nB << "\n";
	    vector<Float> baRatio = generate_baRatio(n[jLayer], rGen);
        for (PosInt i=0; i<nType[iLayer]; i++) {
            cout << "   max pool of LGN = " << maxLGNperV1pool[qType+i] << " for type " << i << "\n";
        }
        vector<Float> a; // radius of the VF, to be filled
        vector<vector<Size>> poolList = retinotopic_vf_pool(cart[jLayer], cart0, &layerMatch[mLayer*iLayer], VFposEcc[jLayer], useCuda, rGen, baRatio, a, theta[jLayer], type, LR, nL, nB, mL, mR, mLayer, &maxLGNperV1pool[qType], seed, LGN_V1_RFratio, nType[iLayer], iLayer);


        for (PosInt iType=0; iType<nType[iLayer]; iType++) {
            cout << " pool stats for type " << iType << "\n";
            Size minPool_L = maxLGNperV1pool[qType + iType];
            Size maxPool_L = 0;
            Float meanPool_L = 0;
            Size zeroPool_L = 0;
            Float rzeroL = 0; // RF radius
            Float rmeanL = 0;

            Size minPool_R = maxLGNperV1pool[qType + iType];
            Size maxPool_R = 0;
            Float meanPool_R = 0;
            Size zeroPool_R = 0;
            Float rzeroR = 0;
            Float rmeanR = 0;
	        
	        Int jL = 0;
	        Int jR = 0;
            for (PosInt i=0; i<n[jLayer]; i++) {
                if (i%neuronPerBlock[jLayer] >= typeAccCount[qType+iType]) {// pass if not the type
                    continue;            
                }
	        	Int iLR;
	        	if (LR[i] <= 0) {
	        		iLR = jL;
	        		jL++;
	        	}
	        	if (LR[i] >= 0) {
	        		iLR = nL + jR;
	        		jR++;
	        	}
                Size iSize = poolList[i].size();
	        	for (PosInt j=0; j<iSize; j++) {
	        		Float dx = (cart0.first[poolList[i][j]] - cart[jLayer].first[i]);
	        		Float dy = (cart0.second[poolList[i][j]] - cart[jLayer].second[i]);
	        		PosInt lgn_id = poolList[i][j];
	        		Float value;
	        		if (!inside_ellipse(dx, dy, theta[jLayer][i], a[i], a[i]*baRatio[i], value)) {
            			Float tx = cosine(theta[jLayer][i]) * dx + sine(theta[jLayer][i]) * dy;
	        			Float ty = -sine(theta[jLayer][i]) * dx + cosine(theta[jLayer][i]) * dy;
	        			cout << "#" << iLR << "-" << j << "(" << poolList[i][j]<< "): x = " << dx << ", y = " << dy << ", theta = " << theta[jLayer][i]*180/M_PI << ", a = " << a[i] << ", b = " << a[i]*baRatio[i] << "\n";
	        			cout << "1-value = " << 1-value << "\n";
	        			cout << "lgnx = " << cart0.first[lgn_id] << ", lgny = " << cart0.second[lgn_id] << "\n";
	        			cout << "v1x = " << cart[jLayer].first[i] << ", v1y = " << cart[jLayer].second[i] << "\n";
	        			cout << "tx = " << tx << ", ty = " << ty << "\n";
	        			assert(inside_ellipse(dx, dy, theta[jLayer][i], a[i], a[i]*baRatio[i], value));
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
            meanPool_L /= jL;
            rzeroL /= zeroPool_L;
            rmeanL /= jL;
            meanPool_R /= jL;
            rzeroR /= zeroPool_R;
            rmeanR /= jR;
            if (jR > 0) {
                cout << "   right poolSizes: [" << minPool_R << ", " << meanPool_R << ", " << maxPool_R << " < " << maxLGNperV1pool[qType+iType] << "] for " << jR << "cells \n";
                cout << "   among them " << zeroPool_R << " would have no connection from LGN, whose average radius is " << rzeroR << ", compared to population mean " <<  rmeanR << "\n";
            }
            if (jL > 0) {

                cout << "   left poolSizes: [" << minPool_L << ", " << meanPool_L << ", " << maxPool_L << " < " << maxLGNperV1pool[qType+iType] << "] for " << jL << "cells\n";
                cout << "   among them " << zeroPool_L << " would have no connection from LGN, whose average radius is " << rzeroL << ", compared to population mean " <<  rmeanL << "\n";
            }
        }

	    cout << "   poolList and R ready for layer " << iLayer << "\n";
	    vector<Float> sfreq = generate_sfreq(n[jLayer], rGen);
        
        Float mean_sfreq = accumulate(sfreq.begin(), sfreq.end(), 0.0)/n[jLayer];
        Float mean_a = accumulate(a.begin(), a.end(), 0.0)/n[jLayer];
        Float suggesting_SF = mean_sfreq/mean_a/2;
        cout << "   mean a: " << mean_a << " degPerRF\n";
        cout << "   mean sfreq: " << mean_sfreq << " cpRF\n";
        cout << "   mean SF suggested based on preset: " << suggesting_SF << " cpd\n";

	    vector<Float> cx(n[jLayer]);
	    vector<Float> cy(n[jLayer]);
        vector<vector<Float>> srList = retinotopic_connection(poolList, rGen, &p_n_LGNeff[qType], &max_LGNeff[qType], envelopeSig, qn, nType[iLayer], input_n[iLayer], cart[jLayer], cart0, V1Type, type, theta[jLayer], phase, sfreq, modAmp_nCon, baRatio, a, RefType, LGNtype, cx, cy, VFposEcc[jLayer], SimpleComplex, conThres, ori_tol, disLGN, strictStrength, top_pick);

        if (!readFromFile) {
	        fV1_RFpreset.write((char*)&phase[qn], n[jLayer] * sizeof(Float));
	        fV1_RFpreset.write((char*)&modAmp_nCon[qn], n[jLayer] * sizeof(Float));
        }


        fV1_RFprop.write((char*)&cx[0], n[jLayer] * sizeof(Float));
	    fV1_RFprop.write((char*)&cy[0], n[jLayer] * sizeof(Float));
        fV1_RFprop.write((char*)&a[0], n[jLayer] * sizeof(Float));
        fV1_RFprop.write((char*)&phase[qn], n[jLayer] * sizeof(Float));
	    fV1_RFprop.write((char*)&sfreq[0], n[jLayer] * sizeof(Float));
        fV1_RFprop.write((char*)&baRatio[0], n[jLayer] * sizeof(Float));


        for (PosInt iType = 0; iType<nType[iLayer]; iType++) {
            cout << "   connected stats for type " << iType << ":\n";
	        Float sfreq2 = 0;
	        Float min_sfreq = 100.0; // 60 as max sfreq
	        Float max_sfreq = 0.0;
            Size nonZero = 0;
	        for (PosInt i = 0; i<input_n[iLayer]; i++) {
                if (i%neuronPerBlock[jLayer] >= typeAccCount[qType+iType]) {// pass if not the type
                    continue;            
                }
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
            cout << "       mean SF suggested after connection: " << mean_sfreq << " cpd\n";
            cout << "       min non-zero SF suggested after connection: " << min_sfreq << " cpd\n";
            cout << "       max SF suggested after connection: " << max_sfreq << " cpd\n";
	        sfreq2 /=        nonZero;
            cout << "       std SF suggested after connection: " << square_root(sfreq2 - mean_sfreq*mean_sfreq) << " cpd\n";

            Size nonZeroMinPool = maxLGNperV1pool[qType+iType]; 
            Float nonZeroMeanPool = 0; 
            Size nonZeroMaxPool = 0;

            Size maxPool = 0; 
            Size zeroPool = 0;
            Size minPool = maxLGNperV1pool[qType+iType];
            meanPool[qType + iType] = 0; 
	        Float minSum = max_LGNeff[qType + iType];;
	        meanSum[qType + iType] = 0;
            
	        Float maxSum = 0;
	        Float sum2 = 0;
	        Float pool2 = 0;
	        PosInt ic_max = 0;
	        PosInt is_max = 0;
            for (PosInt i=0; i<input_n[iLayer]; i++) {
                if (i%neuronPerBlock[jLayer] >= typeAccCount[qType+iType]) {// pass if not the type
                    continue;            
                }

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

                meanPool[qType+iType] += iSize;
                if (iSize == 0) zeroPool++;
	        	assert(poolList[i].size() == srList[i].size());
	        	Float strength = accumulate(srList[i].begin(), srList[i].end(), 0.0);
	        	if (strength > maxSum) {
	        		maxSum = strength;
	        		is_max = i;
	        	}
	        	if (strength < minSum) minSum = strength;
	        	meanSum[iLayer] += strength;
	        	sum2 += strength * strength;
	        	pool2 += iSize * iSize;
            }
            Size iTypeN = nblock[jLayer]*(typeAcc0[iLayer][iType+1]-typeAcc0[iLayer][iType]);
	        if (SimpleComplex == 1) {
	        	Float nonzeroN = iTypeN*(1-pureComplexRatio[qType + iType]);
	        	cout << "       " << nonzeroN << " nonzero-LGN V1 cells\n";
	        	meanSum[qType+iType] /= nonzeroN;
            	meanPool[qType+iType] /= nonzeroN;
	        	sum2 /= nonzeroN;
	        	pool2 /= nonzeroN;
	        } else {
	        	meanSum[qType+iType] /= iTypeN;
            	meanPool[qType+iType] /= iTypeN;
	        	sum2 /= iTypeN;
	        	pool2 /= iTypeN;
	        }
            cout << "       # connections: [" << minPool << ", " << meanPool[qType+iType] << ", " << maxPool << "] for " << iTypeN << "cells\n";
            cout << "       among them " << zeroPool << " would have no connection from LGN\n";
            cout << "       # nonzero connections: [" << nonZeroMinPool << ", " << nonZeroMeanPool/(iTypeN-zeroPool) << ", " << nonZeroMaxPool << "]\n";
            cout << "       totalConnectionStd: " << square_root(pool2 - meanPool[qType+iType]*meanPool[qType+iType]) << "\n";
            cout << "       totalStrength: [" << minSum << ", " << meanSum[qType+iType] << ", " << maxSum << "]\n";
            cout << "       totalStrengthStd: " << square_root(sum2 - meanSum[qType+iType]*meanSum[qType+iType]) << "\n";
        }

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

	    write_listOfList<Size>(outputFolder + idList_filename + conLGN_suffix, poolList, true);
	    write_listOfListForArray<Float>(outputFolder + srList_filename + conLGN_suffix, srList, true); // read with read_listOfListToArray
		qType += nType[iLayer];
        qn += n[jLayer];
    }
    fV1_RFpreset.close();
	fV1_RFprop.close();

	ofstream fLGN_V1_cfg(outputFolder + output_cfg_filename + conLGN_suffix, fstream::out | fstream::binary);
	if (!fLGN_V1_cfg) {
		cout << "Cannot open or find " << output_cfg_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
		fLGN_V1_cfg.write((char*) &nInputLayer, sizeof(Size));
		fLGN_V1_cfg.write((char*) &inputLayer, nInputLayer*sizeof(Size));
		fLGN_V1_cfg.write((char*) &nTypeEI[0], 2*nInputLayer*sizeof(Size));
		fLGN_V1_cfg.write((char*) &typeAccCount[0], totalType*sizeof(Size));
		fLGN_V1_cfg.write((char*) &p_n_LGNeff[0], totalType*sizeof(Float));
		fLGN_V1_cfg.write((char*) &max_LGNeff[0], totalType*sizeof(Size));
		fLGN_V1_cfg.write((char*) &meanPool[0], nInputLayer*sizeof(Float));
		fLGN_V1_cfg.write((char*) &meanSum[0], nInputLayer*sizeof(Float));
		fLGN_V1_cfg.close();
	}
    return 0;
}
