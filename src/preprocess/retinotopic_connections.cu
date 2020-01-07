#include "retinotopic_connections.h"
#include <boost/program_options.hpp>
using namespace std;

/* 
    Purpose:
        Connect neurons with visual field centered at (eccentricity, polar) in retinotopic Sheet1 :pre-synaptically: to neurons in another retinotopic Sheet2, with visual field centered near the same spot.
    
    Inputs:
        1. visual field positions of neurons in Sheet1 in ((e)ccentricity, (p)olar angle) of size (2,N1) 
            vf1 --- ([Float], [Float]) 
        2. index list of Sheet1 neurons as an presynaptic pool to corresponding Sheet2 neurons:
            poolList --- [[Int]];
        3. List of Sheet2 neuron properties of size (7,N2): 
           columns: orientation (theta), spatial phase(phase), spatial frequency(sfreq), amplitude* (amp), 2D-envelope (sig, ecc), elipse horizontal diameter (a).
           prop --- [[Float]]
           * works as a spatial sensitivity parameter.
        4. Threshold of connection:
            P_th --- Float 
        5. Noise applied to connection probability between On and Off:
            P_noise --- Float 
    Output:
        1. visual field positions of neurons in Sheet2 in ((e)ccentricity, (p)olar angle) of size (2,N1) 
            vf2 --- ([Float], [Float])
        2. List of connection ID and strength per Sheet2 neuron
            strList --- [[Int], [Float]]
    
    Source file structure:

        "retinotopic_connection.h"
            declare retinotopic_connection
            declare show_connections 

        "retinotopic_connection.cpp"
            implement retinotopic_connection
                for i-th Sheet2 neuron
                transform properties to connection probability of the j-th neurons in the poolList[i] based on their vf positions.
                    mx = mean(vf1[i,0])
                    my = mean(vf1[i,0])
                    x_i, y_i = (x - mx)/a, (y - my)/a
                    x = cos(theta) *  x_i + sin(theta) * y_i
                    y = -sin(theta) * x_i + cos(theta) * y_i
                    envelope(x,y) = exp(-0.5((x/sig_h)^2+(y/sig_v)^2))
                    P(x,y) = envelope(x,y) * amp * cos(2pi/sfreq * x + sphase)
                    P_on(x,y) = envelope(x,y) * (1-amp) + heavside(P(x,y))*P(x,y)
                    P_off(x,y) = envelope(x,y) * (1-amp) + heavside(-P(x,y))*P(x,y)
                normalize P_on and P_off by sum P_on and P_off over the poolList[i]
                cutoff connections for P_k(x,y) < P_th, k = on/off
*/

vector<vector<Float>> retinotopic_connection(
        vector<vector<Size>> &poolList,
        RandomEngine &rGen,
        const Float percent,
        const Size n,
		const pair<vector<Float>, vector<Float>> &cart, // V1 VF position (tentative)
        const pair<vector<Float>,vector<Float>> &cart0, // LGN VF position
		const vector<RFtype> &V1Type,
        const vector<Float> &theta,
        const vector<Float> &phase,
        const vector<Float> &sfreq,
        const vector<Float> &amp,
        const vector<Float> &sig,
        const vector<Float> &baRatio,
        const vector<Float> &a,
		const vector<OutputType> &RefType,
        const vector<InputType> &LGNtype,
        vector<Float> &cx, // V1 VF position (final)
        vector<Float> &cy) {
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
			RF->setup_param(m, sfreq[i], phase[i], amp[i], theta[i], a[i], baRatio[i], RefType[i], sig[i]);
			vector<Float> strengthList;
			// construct pooled LGN neurons' connection to the V1 neuron
            /* DEBUG: if (i==959) {
                std::cout << static_cast<Size>(V1Type[i]) << ": " << static_cast<Size>(RefType[i]) << "\n";
                std::cout << theta[i]*180.0/M_PI << ", " << phase[i]*180/M_PI << "\n";
                std::cout << poolList[i][0] << ": " << static_cast<Size>(iType[0]) << "\n";
                std::cout << poolList[i][1] << ": " << static_cast<Size>(iType[1]) << "\n";
                std::cout << poolList[i][2] << ": " << static_cast<Size>(iType[2]) << "\n";
            } */
			m = RF->construct_connection(x, y, iType, poolList[i], strengthList, rGen, percent);
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

    Float mx = 0.0f;
    Float my = 0.0f;
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
            mx = mx + cart.first[i];
            my = my + cart.second[i];
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
        const pair<vector<Float>,vector<Float>> &cart,
        const pair<vector<Float>,vector<Float>> &cart0,
        const vector<Float> &VFposEcc,
        const bool use_cuda,
        RandomEngine &rGen,
        vector<Float> &baRatio,
        vector<Float> &a,
        vector<Int> &LR,
        Size mL,
        Size m
) {
    vector<vector<Size>> poolList;
    const Size n = cart.first.size();
    poolList.reserve(n);
    if (use_cuda) {
        if (n > 0) {
            Float *d_cart;
            checkCudaErrors(cudaMalloc((void **) &d_cart, 2*2*n*sizeof(Float)));
            checkCudaErrors(cudaMemcpy(d_cart, &(cart.first[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_cart+n, &(cart.second[0]), n*sizeof(Float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaFree(d_cart));
            cout << "cuda not implemented\n";
            assert(false);
        } else {
            cout << "no coordinate provided\n";
            assert(false);
        }
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
            Float R = mapping_rule(VFposEcc[i]*60, normRand[i], rGen)/60.0;
            a.push_back(R/sqrt(M_PI*baRatio[i]));
            Float b = R*R/M_PI/a[i];
            if (a[i] > b) {
                rMap.push_back(a[i]);
            } else {
                rMap.push_back(b);
            }
        }
        // next nearest neuron j to (e,p) in sheet 1
        for (Size i=0; i<n; i++) {
            if (LR[i] > 0) {
                poolList.push_back(draw_from_radius(cart.first[i], cart.second[i], cart0, mL, m, rMap[i]));
            } else {
                poolList.push_back(draw_from_radius(cart.first[i], cart.second[i], cart0, 0, mL, rMap[i]));
            }
        }
    }
    return poolList;
}

// unit test
int main(int argc, char *argv[]) {
	namespace po = boost::program_options;
    Float percent;
	string LGN_vpos_filename, V1_vpos_filename, V1_prop_filename;
	po::options_description input_opt("input options");
	input_opt.add_options()
		("percent", po::value<Float>(&percent)->default_value(0.5), "LGN conneciton probability")
		("fV1_prop", po::value<string>(&V1_prop_filename)->default_value("V1_prop.bin"), "file that stores V1 neurons' parameters")
		("fLGN", po::value<string>(&LGN_vpos_filename)->default_value("LGN(vpos).bin"), "file that stores LGN position in visual field (and on-cell off-cell label)")
		("fV1_vpos", po::value<string>(&V1_vpos_filename)->default_value("V1_vpos.bin"), "file that stores V1 position in visual field)");

    string V1_filename, idList_filename, sList_filename; 
	po::options_description output_opt("output options");
	output_opt.add_options()
		("fV1", po::value<string>(&V1_filename)->default_value("V1.bin"), "file that stores V1 neurons' information")
		("fLGN_V1_ID", po::value<string>(&idList_filename)->default_value("LGN_V1_idList.bin"), "file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&sList_filename)->default_value("LGN_V1_sList.bin"), "file stores LGN to V1 connection strengths");

	po::options_description cmdline_options;
	cmdline_options.add(input_opt).add(output_opt);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
	po::notify(vm);

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
	vector<Float> x(n);
	vector<Float> y(n);
	fV1_vpos.read(reinterpret_cast<char*>(&x[0]), n * sizeof(Float));
	fV1_vpos.read(reinterpret_cast<char*>(&y[0]), n * sizeof(Float));
	auto cart = make_pair(x, y);
	// release memory from temporary vectors
	x.swap(vector<Float>());
	y.swap(vector<Float>());
	vector<Int> LR(n);
	fV1_vpos.read(reinterpret_cast<char*>(&LR[0]), n * sizeof(Int));
	fV1_vpos.close();
    /*
    Size nL = 0;
    Size nR = 0;
    for (auto lr: LR) {
        if (lr > 0) {
            nR++;
        } else {
            nL++;
        }
    } */

	ifstream fLGN;
	fLGN.open(LGN_vpos_filename, fstream::in | fstream::binary);
	if (!fLGN) {
		cout << "Cannot open or find " << LGN_vpos_filename << "\n";
		return EXIT_FAILURE;
	}
	Size mL;
	Size mR;
    Size m;
	Float max_ecc;
	size_pointer = &mL;
	fLGN.read(reinterpret_cast<char*>(size_pointer), sizeof(Size));
	size_pointer = &mR;
	fLGN.read(reinterpret_cast<char*>(size_pointer), sizeof(Size));
	fLGN.read(reinterpret_cast<char*>(&max_ecc), sizeof(Float));
    m = mL + mR;
    cout << m << " LGN neurons, " << mL << " from left eye, " << mR << " from right eye.\n";
	cout << "need " << 3 * m * sizeof(Float) / 1024 / 1024 << "mb\n";

	vector<InputType> LGNtype(m);
	fLGN.read(reinterpret_cast<char*>(&LGNtype[0]), m * sizeof(Size));
	//temporary vectors to get coordinate pairs
	vector<Float> polar0(m);
	vector<Float> ecc0(m);
	fLGN.read(reinterpret_cast<char*>(&polar0[0]), m*sizeof(Float));
	fLGN.read(reinterpret_cast<char*>(&ecc0[0]), m*sizeof(Float));
	vector<Float> x0(m);
	vector<Float> y0(m);
	auto polar2x = [] (Float polar, Float ecc) {
		return ecc*cos(polar);
	};
	auto polar2y = [] (Float polar, Float ecc) {
		return ecc*sin(polar);
	};
	transform(polar0.begin(), polar0.end(), ecc0.begin(), x0.begin(), polar2x);
	transform(polar0.begin(), polar0.end(), ecc0.begin(), y0.begin(), polar2y);
	auto cart0 = make_pair(x0, y0);
	// release memory from temporary vectors
	x0.swap(vector<Float>());
	y0.swap(vector<Float>()); 
	polar0.swap(vector<Float>());
	fLGN.close();

	cout << "carts ready\n";
    vector<Int> seed{820,702};
    seed_seq seq(seed.begin(), seed.end());
    RandomEngine rGen(seq);
    vector<Float> a; // radius of the VF
	vector<Float> baRatio = generate_baRatio(n, rGen);
    vector<vector<Size>> poolList = retinotopic_vf_pool(cart, cart0, ecc0, false, rGen, baRatio, a, LR, mL, m);
	cout << "poolList and R ready\n";

	vector<RFtype> V1Type(n);
	vector<OutputType> RefType(n);
	vector<Float> theta(n);
	vector<Float> phase(n);
	vector<Float> amp(n);
	vector<Float> sig(n);
    ifstream fV1_prop(V1_prop_filename, fstream::in | fstream::binary);
	if (!fV1_prop) {
		cout << "Cannot open or find " << V1_prop_filename <<"\n";
		return EXIT_FAILURE;
	}
	fV1_prop.read(reinterpret_cast<char*>(&V1Type[0]), n * sizeof(RFtype_t));
	fV1_prop.read(reinterpret_cast<char*>(&RefType[0]), n * sizeof(OutputType_t));
	fV1_prop.read(reinterpret_cast<char*>(&theta[0]), n * sizeof(Float));
	fV1_prop.read(reinterpret_cast<char*>(&phase[0]), n * sizeof(Float));
	fV1_prop.read(reinterpret_cast<char*>(&amp[0]), n * sizeof(Float));
	fV1_prop.read(reinterpret_cast<char*>(&sig[0]), n * sizeof(Float));
    fV1_prop.close();

	vector<Float> sfreq = generate_sfreq(n, rGen);
	vector<Float> cx(n);
	vector<Float> cy(n);
    vector<vector<Float>> srList = retinotopic_connection(poolList, rGen, percent, n, cart, cart0, V1Type, theta, phase, sfreq, amp, sig, baRatio, a, RefType, LGNtype, cx, cy);

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
	fV1.write((char*)&theta[0], n * sizeof(Float));
	fV1.write((char*)&phase[0], n * sizeof(Float));
	fV1.write((char*)&amp[0], n * sizeof(Float));
	fV1.write((char*)&sig[0], n * sizeof(Float));
	fV1.write((char*)&V1Type[0], n * sizeof(RFtype_t));
	fV1.write((char*)&RefType[0], n * sizeof(OutputType_t));
    fV1.close();

    // write poolList to disk
	//print_listOfList<Size>(poolList);
	write_listOfList<Size>(idList_filename, poolList, false);
	write_listOfList<Float>(sList_filename, srList, false);
    return 0;
}
