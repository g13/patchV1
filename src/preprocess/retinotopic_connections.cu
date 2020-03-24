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

vector<vector<Float>> retinotopic_connection(
        vector<vector<Size>> &poolList,
        RandomEngine &rGen,
        const Float p_n_LGNeff,
        const Size n,
		const pair<vector<Float>, vector<Float>> &cart, // V1 VF position (tentative)
        const pair<vector<Float>,vector<Float>> &cart0, // LGN VF position
		const vector<RFtype> &V1Type,
        const vector<Float> &theta,
        const vector<Float> &phase,
        const vector<Float> &sfreq,
        const vector<Float> &modAmp_nCon,
        const vector<Float> &baRatio,
        const vector<Float> &a,
		const vector<OutputType> &RefType,
        const vector<InputType> &LGNtype,
        vector<Float> &cx, // V1 VF position (final)
        vector<Float> &cy,
        Int SimpleComplex) {
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
            if (SimpleComplex == 0) {
			    RF->setup_param(m, sfreq[i], phase[i], modAmp_nCon[i], theta[i], a[i], baRatio[i], RefType[i]);
			    m = RF->construct_connection(x, y, iType, poolList[i], strengthList, rGen, p_n_LGNeff*1.0, p_n_LGNeff < 0);
            } else {
			    RF->setup_param(m, sfreq[i], phase[i], 1.0, theta[i], a[i], baRatio[i], RefType[i]);
			    m = RF->construct_connection(x, y, iType, poolList[i], strengthList, rGen, p_n_LGNeff*modAmp_nCon[i], p_n_LGNeff < 0);
            }
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
			// reset reusable variables
			RF->clear();
		} else {
			// keep tentative VF position
			cx[i] = cart.first[i];
			cy[i] = cart.second[i];
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

__device__
__forceinline__
Float get_distance(Float x, Float y) {
	return square_root(x*x + y*y);
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
    Size nPatch = (m + blockSize - 1)/blockSize;
    Size nRemain = m%blockSize;
    assert((nPatch-1)*blockSize + nRemain == m);
    curandStateMRG32k3a rGen;
    Float V1_x, V1_y, r;
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
        Float b = R*sqr/M_PI;
        r = (local_a > b)? local_a: b;
        // scan and pooling LGN
    }
    Size iPool = 0;
    for (Size iPatch = 0; iPatch < nPatch; iPatch++) {
        // load LGN pos to __shared__
        Size nLGN;
        if (iPatch < nPatch - 1 || threadIdx.x < nRemain) {
            Size LGN_id = j0 + iPatch*blockSize + threadIdx.x;
            sx[threadIdx.x] = x0[LGN_id];
            sy[threadIdx.x] = y0[LGN_id];
            if (iPatch < nPatch - 1) {
                nLGN = blockSize;
            } else {
                nLGN = nRemain;
            }
        }
        __syncthreads();
        // check for distance
        if (V1_id < i0+n) {
            for (Size i = 0; i < nLGN; i++) {
                Size iLGN = (threadIdx.x + i)%nLGN;
                Float dis = get_distance(sx[iLGN]-V1_x, sy[iLGN]-V1_y);
                if (dis < r) {
                    PosInt LGN_id = j0 + iPatch*blockSize + iLGN;
                    PosIntL pid = static_cast<PosIntL>(V1_id)*maxLGNperV1pool + iPool;
                    poolList[pid] = LGN_id;
                    if (LGN_id >= j0+m) {
                        printf("LGN_id:%u < %u\n", LGN_id, j0+m); 
                        assert(LGN_id < j0+m);
                    }
                    iPool++;
                    if (iPool > maxLGNperV1pool) {
                        printf("V1_id:%u, r = %f\n", V1_id, r);
                        assert(iPool <= maxLGNperV1pool);
                    }
                }
            }
        }
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
        vector<Float> &VFposEcc,
        bool useCuda,
        RandomEngine &rGen,
        vector<Float> &baRatio,
        vector<Float> &a,
        vector<Int> &LR,
        Size nL,
        Size mL,
        Size m,
        Size maxLGNperV1pool,
		BigSize seed,
		Float LGN_V1_RFratio
) {
    vector<vector<PosInt>> poolList;
    const Size n = cart.first.size();
        
    poolList.reserve(n);
    if (useCuda) {
		vector<Float> xLR, yLR, baRatioLR, VFposEccLR;
		original2LR(cart.first, xLR, LR, nL);
		original2LR(cart.second, yLR, LR, nL);
		original2LR(baRatio, baRatioLR, LR, nL);
		original2LR(VFposEcc, VFposEccLR, LR, nL);
        Float *d_memblock;
        size_t d_memSize = ((2+3)*n + 2*m)*sizeof(Float) + static_cast<size_t>(n)*maxLGNperV1pool*sizeof(PosInt) + n*sizeof(Size);
		cout << "poolList need memory of " << n << "x" << maxLGNperV1pool << "x" << sizeof(PosInt) << " = " << static_cast<PosIntL>(n)*maxLGNperV1pool*sizeof(PosInt) << " = " << static_cast<PosIntL>(n)*maxLGNperV1pool*sizeof(PosInt) / 1024.0 / 1024.0 << "mb\n";
        checkCudaErrors(cudaMalloc((void **) &d_memblock, d_memSize));
		cout << "need global memory of " << d_memSize / 1024.0 / 1024.0 << "mb in total\n";
        Float* d_x = d_memblock;
        Float* d_y = d_x + n;
        Float* d_x0 = d_y + n;
        Float* d_y0 = d_x0 + m;
        Float* d_baRatio = d_y0 + m;
        Float* d_a = d_baRatio + n; // to be filled
        Float* d_VFposEcc = d_baRatio + n;
		// to be filled
        PosInt* d_poolList = (PosInt*) (d_VFposEcc + n);
        Size* d_nPool = (Size*) (d_poolList + n*maxLGNperV1pool);

        checkCudaErrors(cudaMemcpy(d_x, &(xLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, &(yLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_x0, &(cart0.first[0]), m*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y0, &(cart0.second[0]), m*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_baRatio, &(baRatioLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_VFposEcc, &(VFposEccLR[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
        // TODO: stream kernels in chunks for large network
        Size nblock = (nL + blockSize-1)/blockSize;
        cout <<  "<<<" << nblock << ", " << blockSize << ">>>\n";
        vf_pool_CUDA<<<nblock, blockSize>>>(d_x, d_y, d_x0, d_y0, d_VFposEcc, d_baRatio, d_a, d_poolList, d_nPool, 0, nL, 0, mL, seed, LGN_V1_RFratio, maxLGNperV1pool);
        getLastCudaError("vf_pool for the left eye failed");

		nblock = (n-nL + blockSize-1) / blockSize;
        cout <<  "<<<" << nblock << ", " << blockSize << ">>>\n";
        vf_pool_CUDA<<<nblock, blockSize>>>(d_x, d_y, d_x0, d_y0, d_VFposEcc, d_baRatio, d_a, d_poolList, d_nPool, nL, n-nL, mL, m-mL, seed, LGN_V1_RFratio, maxLGNperV1pool);
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
    return poolList;
}
 
int main(int argc, char *argv[]) {
	namespace po = boost::program_options;
    bool readFromFile, useCuda;
	Float p_n_LGNeff;
	Size maxLGNperV1pool;
    Int SimpleComplex;
    Float LGN_V1_RFratio;
    BigSize seed;
    vector<Size> nRefTypeV1_RF, V1_RefTypeID;
    vector<Float> V1_RFtypeAccDist, V1_RefTypeDist;
	string LGN_vpos_filename, V1_vpos_filename, V1_RFprop_filename, V1_feature_filename;
	po::options_description generic("generic options");
	generic.add_options()
		("help,h", "print usage")
		("seed,s", po::value<BigSize>(&seed)->default_value(1885124), "random seed")
		("readFromFile,f", po::value<bool>(&readFromFile)->default_value(false), "whether to read V1 RF properties from file")
		("useCuda,u", po::value<bool>(&useCuda)->default_value(false), "whether to use cuda")
		("cfg_file,c", po::value<string>()->default_value("LGN_V1.cfg"), "filename for configuration file");

	po::options_description input_opt("input options");
	input_opt.add_options()
		("p_n_LGNeff", po::value<Float>(&p_n_LGNeff)->default_value(10), "LGN conneciton probability [0,1], or number of connections [0,n]")
		("LGN_V1_RFratio,r", po::value<Float>(&LGN_V1_RFratio)->default_value(1.0), "LGN's contribution to the total RF size")
		("maxLGNperV1pool,m", po::value<Size>(&maxLGNperV1pool)->default_value(100), "maximum pooling of LGN neurons per V1 neuron")
		("SimpleComplex", po::value<Int>(&SimpleComplex)->default_value(0), "determine how simple complex is implemented, through modulation modAmp_nCon(0) or number of LGN connection(1)")
		("V1_RFtypeAccDist", po::value<vector<Float>>(&V1_RFtypeAccDist), "determine the relative portion of each V1 RF type")
		("nRefTypeV1_RF", po::value<vector<Size>>(&nRefTypeV1_RF), "determine the number of cone/ON-OFF combinations for each V1 RF type")
		("V1_RefTypeID", po::value<vector<Size>>(&V1_RefTypeID), "determine the ID of the available cone/ON-OFF combinations in each V1 RF type")
		("V1_RefTypeDist", po::value<vector<Float>>(&V1_RefTypeDist), "determine the relative portion of the available cone/ON-OFF combinations in each V1 RF type")
		("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file that stores V1 neurons' parameters")
		("fV1_RFprop", po::value<string>(&V1_RFprop_filename)->default_value("V1_RFprop.bin"), "file that stores V1 neurons' parameters")
		("fLGN_vpos", po::value<string>(&LGN_vpos_filename)->default_value("LGN_vpos.bin"), "file that stores LGN position in visual field (and on-cell off-cell label)")
		("fV1_vpos", po::value<string>(&V1_vpos_filename)->default_value("V1_vpos.bin"), "file that stores V1 position in visual field)");

    string V1_filename, idList_filename, sList_filename;
	po::options_description output_opt("output options");
	output_opt.add_options()
		("fV1", po::value<string>(&V1_filename)->default_value("V1RF.bin"), "file that stores V1 neurons' information")
		("fLGN_vpos_V1_ID", po::value<string>(&idList_filename)->default_value("LGN_V1_idList.bin"), "file stores LGN to V1 connections")
		("fLGN_vpos_V1_s", po::value<string>(&sList_filename)->default_value("LGN_V1_sList.bin"), "file stores LGN to V1 connection strengths");

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
    
	ifstream fV1_vpos;
	fV1_vpos.open(V1_vpos_filename, fstream::in | fstream::binary);
	if (!fV1_vpos) {
		cout << "Cannot open or find " << V1_vpos_filename <<"\n";
		return EXIT_FAILURE;
	}
	Size n;
	Size* size_pointer = &n;
	fV1_vpos.read(reinterpret_cast<char*>(size_pointer), sizeof(Size));
    cout << n << " post-synaptic neurons\n";
	//temporary vectors to get coordinate pairs
	vector<double> decc(n);
	vector<double> dpolar(n);
	// ecc first, polar second; opposite to LGN
	fV1_vpos.read(reinterpret_cast<char*>(&decc[0]), n * sizeof(double)); 
	fV1_vpos.read(reinterpret_cast<char*>(&dpolar[0]), n * sizeof(double));
    auto double2Float = [](double x) {
        return static_cast<Float>(x);
	};

	vector<Float> VFposEcc(n);
    transform(decc.begin(), decc.end(), VFposEcc.begin(), double2Float);
	vector<Float> x(n);
	vector<Float> y(n);
	auto polar2xD2F = [] (double ecc, double polar) {
		return static_cast<Float>(ecc*cos(polar));
	};
	auto polar2yD2F = [] (double ecc, double polar) {
		return static_cast<Float>(ecc*sin(polar));
	};
	transform(decc.begin(), decc.end(), dpolar.begin(), x.begin(), polar2xD2F);
	transform(decc.begin(), decc.end(), dpolar.begin(), y.begin(), polar2yD2F);
	vector<double>().swap(decc); // release memory from temporary vectors
	vector<double>().swap(dpolar);

    cout << "V1_x: [" << *min_element(x.begin(), x.end()) << ", " << *max_element(x.begin(), x.end()) << "]\n";
    cout << "V1_y: [" << *min_element(y.begin(), y.end()) << ", " << *max_element(y.begin(), y.end()) << "]\n";
	auto cart = make_pair(x, y);
	vector<Float>().swap(x); // release memory from temporary vectors
	vector<Float>().swap(y);
	fV1_vpos.close();


	ifstream fLGN_vpos;
	fLGN_vpos.open(LGN_vpos_filename, fstream::in | fstream::binary);
	if (!fLGN_vpos) {
		cout << "Cannot open or find " << LGN_vpos_filename << "\n";
		return EXIT_FAILURE;
	}
	Size mL;
	Size mR;
    Size m;
	Float max_ecc;
	size_pointer = &mL;
	fLGN_vpos.read(reinterpret_cast<char*>(size_pointer), sizeof(Size));
	size_pointer = &mR;
	fLGN_vpos.read(reinterpret_cast<char*>(size_pointer), sizeof(Size));
	fLGN_vpos.read(reinterpret_cast<char*>(&max_ecc), sizeof(Float));
    m = mL + mR;
	{// not used
		Float tmp;
		fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // x0
		fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // xspan
		fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // y0
		fLGN_vpos.read(reinterpret_cast<char*>(&tmp), sizeof(Float)); // yspan
	}
    cout << m << " LGN neurons, " << mL << " from left eye, " << mR << " from right eye.\n";
	cout << "need " << 3 * m * sizeof(Float) / 1024 / 1024 << "mb\n";
	vector<Float> x0(m);
	vector<Float> y0(m);
	fLGN_vpos.read(reinterpret_cast<char*>(&x0[0]), m*sizeof(Float));
	fLGN_vpos.read(reinterpret_cast<char*>(&y0[0]), m*sizeof(Float));

	vector<InputType> LGNtype(m);
	fLGN_vpos.read(reinterpret_cast<char*>(&LGNtype[0]), m * sizeof(PosInt));
	/*** now read from file directly
		//temporary vectors to get cartesian coordinate pairs
		vector<Float> polar0(m);
		vector<Float> ecc0(m);
		fLGN_vpos.read(reinterpret_cast<char*>(&polar0[0]), m*sizeof(Float));
		fLGN_vpos.read(reinterpret_cast<char*>(&ecc0[0]), m*sizeof(Float));
		auto polar2x = [] (Float polar, Float ecc) {
			return ecc*cos(polar);
		};
		auto polar2y = [] (Float polar, Float ecc) {
			return ecc*sin(polar);
		};
		transform(polar0.begin(), polar0.end(), ecc0.begin(), x0.begin(), polar2x);
		transform(polar0.begin(), polar0.end(), ecc0.begin(), y0.begin(), polar2y);
		// release memory from temporary vectors
		vector<Float>().swap(polar0);
		vector<Float>().swap(ecc0);
	*/
    cout << "LGN_x: [" << *min_element(x0.begin(), x0.end()) << ", " << *max_element(x0.begin(), x0.end()) << "]\n";
    cout << "LGN_y: [" << *min_element(y0.begin(), y0.end()) << ", " << *max_element(y0.begin(), y0.end()) << "]\n";
	auto cart0 = make_pair(x0, y0);
	vector<Float>().swap(x0);
	vector<Float>().swap(y0); 
	fLGN_vpos.close();

	cout << "carts ready\n";
    vector<BigSize> seeds{seed,seed+13};
    seed_seq seq(seeds.begin(), seeds.end());
    RandomEngine rGen(seq);

    ifstream fV1_feature;
    fV1_feature.open(V1_feature_filename, ios::in|ios::binary);
	if (!fV1_feature) {
		cout << "failed to open pos file:" << V1_feature_filename << "\n";
		return EXIT_FAILURE;
	}
	Size nFeature; // not used here
    fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
	vector<Float> OD(n);
	fV1_feature.read(reinterpret_cast<char*>(&OD[0]), n * sizeof(Float));
	vector<Int> LR(n);
	vector<Float> theta(n); // [0, 1] control the LGN->V1 RF orientation
	fV1_feature.read(reinterpret_cast<char*>(&theta[0]), n * sizeof(Float));
    Size nL = 0;
    Size nR = 0;
    for (Size i = 0; i<n; i++) {
        if (OD[i] < 0) {
            LR[i] = -1;
            nL++;
        } else {
            LR[i] = 1;
            nR++;
        }
        theta[i] = (theta[i] - 0.5) * M_PI;
    }
	fV1_feature.close();
    cout << "left eye: " << nL << " V1 neurons\n";
    cout << "right eye: " << nR << " V1 neurons\n";

    vector<Float> a; // radius of the VF, to be filled
	vector<Float> baRatio = generate_baRatio(n, rGen);
    cout << "max pool of LGN = " << maxLGNperV1pool << "\n";
    vector<vector<Size>> poolList = retinotopic_vf_pool(cart, cart0, VFposEcc, useCuda, rGen, baRatio, a, LR, nL, mL, m, maxLGNperV1pool, seed, LGN_V1_RFratio);
    Size minPool = maxLGNperV1pool; 
    Size maxPool = 0; 
    Size meanPool = 0; 
    Size zeroPool = 0;
    Float zeroR = 0;
    Float meanR = 0;
    for (PosInt i=0; i<n; i++) {
        Size iSize = poolList[i].size();
        Float r = a[i]*square_root(baRatio[i]*baRatio[i] + 1);
        if (iSize > maxPool) maxPool = iSize;
        if (iSize < minPool) minPool = iSize;
        if (iSize == 0) {
            zeroPool++;
            zeroR += r;
        }
        meanR += r;
        meanPool += iSize;
    }
    meanPool /= n;
    zeroR /= zeroPool;
    meanR /= n;
    cout << "poolSizes: [" << minPool << ", " << meanPool << ", " << maxPool << " < " << maxLGNperV1pool << "]\n";
    cout << "among them " << zeroPool << " would have no connection from LGN, whose average radius is " << zeroR << ", compared to population mean " <<  meanR << "\n";
    
	cout << "poolList and R ready\n";

	vector<RFtype> V1Type(n); // defined in RFtype.h [0..4], RF shapes
	vector<OutputType> RefType(n); // defined in in RFtype.h [0..3] conetype placements
	vector<Float> phase(n); // [0, 2*pi], control the phase
    // theta is read from fV1_feature
	vector<Float> modAmp_nCon(n); // [0,1] controls simple complex ratio, through subregion overlap ratio, or number of LGN connected.
    if (readFromFile) {
        ifstream fV1_RFprop(V1_RFprop_filename, fstream::in | fstream::binary);
	    if (!fV1_RFprop) {
	    	cout << "Cannot open or find " << V1_RFprop_filename <<"\n";
	    	return EXIT_FAILURE;
	    }
	    fV1_RFprop.read(reinterpret_cast<char*>(&V1Type[0]), n * sizeof(RFtype_t));
	    fV1_RFprop.read(reinterpret_cast<char*>(&RefType[0]), n * sizeof(OutputType_t));
	    fV1_RFprop.read(reinterpret_cast<char*>(&phase[0]), n * sizeof(Float));
	    fV1_RFprop.read(reinterpret_cast<char*>(&modAmp_nCon[0]), n * sizeof(Float));
        fV1_RFprop.close();
    } else {
        // discrete portions randomly distributed
        uniform_real_distribution<Float> uniform_01(0,1.0);

        assert(V1_RFtypeAccDist.size() == nRFtype  && V1_RFtypeAccDist.back() == 1.0);
        auto genV1_RFtype = [&uniform_01, &V1_RFtypeAccDist, &rGen] () {
            Float rand = uniform_01(rGen);
            for (Size i=0; i<V1_RFtypeAccDist.size(); i++) {
                if (rand < V1_RFtypeAccDist[i]) {
                    return static_cast<RFtype>(i);
                }
            }
        };
        generate(V1Type.begin(), V1Type.end(), genV1_RFtype);

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
        };
        generate(phase.begin(), phase.end(), genPhase);
        auto genModAmp_nCon = [&uniform_01, &rGen] () {
            return uniform_01(rGen);
        };
        generate(modAmp_nCon.begin(), modAmp_nCon.end(), genModAmp_nCon);

        ofstream fV1_RFprop(V1_RFprop_filename, fstream::out | fstream::binary);
	    if (!fV1_RFprop) {
	    	cout << "Cannot open or find " << V1_RFprop_filename <<"\n";
	    	return EXIT_FAILURE;
	    }
	    fV1_RFprop.write((char*)&V1Type[0], n * sizeof(RFtype_t));
	    fV1_RFprop.write((char*)&RefType[0], n * sizeof(OutputType_t));
	    fV1_RFprop.write((char*)&phase[0], n * sizeof(Float));
	    fV1_RFprop.write((char*)&modAmp_nCon[0], n * sizeof(Float));
        fV1_RFprop.close();
    }
	cout << "V1 RF properties ready.\n";

    assert(p_n_LGNeff > 0);

	vector<Float> sfreq = generate_sfreq(n, rGen);
	vector<Float> cx(n);
	vector<Float> cy(n);
    vector<vector<Float>> srList = retinotopic_connection(poolList, rGen, p_n_LGNeff, n, cart, cart0, V1Type, theta, phase, sfreq, modAmp_nCon, baRatio, a, RefType, LGNtype, cx, cy, SimpleComplex);

	ofstream fV1(V1_filename, fstream::out | fstream::binary);
	if (!fV1) {
		cout << "Cannot open or find V1." << V1_filename <<"\n";
		return EXIT_FAILURE;
	}
    fV1.write((char*)&n, sizeof(Size));
    fV1.write((char*)&cx[0], n * sizeof(Float));
	fV1.write((char*)&cy[0], n * sizeof(Float));
    fV1.write((char*)&a[0], n * sizeof(Float));
    fV1.write((char*)&baRatio[0], n * sizeof(Float));
	fV1.write((char*)&sfreq[0], n * sizeof(Float));
    fV1.close();

    minPool = maxLGNperV1pool; 
    maxPool = 0; 
    meanPool = 0; 
    zeroPool = 0;
    for (PosInt i=0; i<n; i++) {
        Size iSize = poolList[i].size();
        if (iSize > maxPool) maxPool = iSize;
        if (iSize < minPool) minPool = iSize;
        meanPool += iSize;
        if (iSize == 0) zeroPool++;
    }
    meanPool /= n;
    cout << "# connections: [" << minPool << ", " << meanPool << ", " << maxPool << "]\n";
    cout << "among them " << zeroPool << " would have no connection from LGN\n";

    // write poolList to disk, to be used in ext_input.cu and genCon.cu
	write_listOfList<Size>(idList_filename, poolList, false);
	write_listOfListForArray<Float>(sList_filename, srList, false); // read with read_listOfListToArray
    return 0;
}
