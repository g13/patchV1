#include "retinotopic_connections.h"
#include <boost/program_options.hpp>
using namespace std;

auto average(vector<Float> x, vector<Float> y) {
    Float mx = accumulate(x.begin(), x.end(), 0.0f)/x.size();
    Float my = accumulate(y.begin(), y.end(), 0.0f)/y.size();
    return make_pair(mx,my);
}

// normalize x to [-1,1], y to baRatio*[-1,1]
auto transform_coord_to_unitRF(Float x, Float y, const Float mx, const Float my, const Float theta, const Float a) {
    // a is half-width at the x-axis
    x = (x - mx)/a;
    y = (y - my)/a;
    Float new_x, new_y;
    new_x = cos(theta) * x + sin(theta) * y;
	new_y = -sin(theta) * x + cos(theta) * y;
    return make_pair(new_x, new_y);
}

// probability distribution
Float prob_dist(Float x, Float y, Float phase, Float sfreq, Float amp, Float baRatio, Int LM_phaseRef, Int LM_OF, Float sig = 1.1775) {
    // sfreq should be given as a percentage of the width
    // baRatio comes from Dow et al., 1981
	Int OF, LM;
	if (LM_OF % 2 == 1) {
		OF = 1;
	} else {
		OF = -1 ;
	}
	// check LGN 
	if (LM_OF < 2) {
		LM = 1;
	} else {
		LM = -1;
	}
    Float envelope;
    envelope = exp(-0.5*(pow(x/sig,2)+pow(y/(sig*baRatio),2)));
    // when sfreq == 1, the RF contain a full cycle of cos(x), x\in[-pi, pi]
    Float modulation = amp * cos(sfreq * x * M_PI + phase);
    //if (LM * LM_phaseRef * OF * modulation < 0) {
		// mismatch between L/M cone or On/Off
        //modulation = 0;
    //} // else a mismatch between none or both have positive modulation on connection probability
    Float prob = envelope * (1 + LM * LM_phaseRef * OF *modulation);
 	return prob;
}

void normalize_prob(vector<Float> &prob, Float percent) {
	// average connection probability is controlled at percent.
    const Float norm = accumulate(prob.begin(), prob.end(), 0.0) / (percent * prob.size());
	//cout << "norm = " << norm << "\n";
	//print_list<Float>(prob);
	assert(!isnan(norm));
    Float sum = 0.0;
    for (Size i=0; i<prob.size(); i++) {
        prob[i] = prob[i] / norm;
        sum += prob[i];
    }
    //cout << percent*prob.size() << " ~ " << sum << "\n";
}

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
        vector<vector<Int>> &poolList,
        RandomEngine &rGen,
        const Float percent,
        const Size n,
		const pair<vector<Float>, vector<Float>> &cart, // V1 VF position (tentative)
        const pair<vector<Float>,vector<Float>> &cart0, // LGN VF position
        const vector<Float> &theta,
        const vector<Float> &phase,
        const vector<Float> &sfreq,
        const vector<Float> &amp,
        const vector<Float> &sig,
        const vector<Float> &baRatio,
        const vector<Float> &a,
		const vector<Int> &LM_ref,
        const vector<Int> &on_off,
        vector<Float> &cx, // V1 VF position (final)
        vector<Float> &cy) {
    uniform_real_distribution<Float> dist(0,1);
    vector<vector<Float>> srList;
    // for each neuron i in sheet 2 at (e, p)
    for (Size i=0; i<n; i++) {
        const Int m = poolList[i].size();
		if (m > 0) { // if the pool has LGN neuron
			// intermdiate V1 VF position (tentative)
			vector<Float> x, y;
			x.reserve(m);
			assert(x.size() == 0);
			for (Size j = 0; j < m; j++) {
				x.push_back(cart0.first[poolList[i][j]]);
				y.push_back(cart0.second[poolList[i][j]]);
			}
			tie(cx[i], cy[i]) = average(x, y);
			// calculate pooled LGN neurons' connection probability to the V1 neuron
			vector<Float> prob;
			prob.reserve(m);
			for (Size j = 0; j < m; j++) {
                Float norm_x, norm_y;
				tie(norm_x, norm_y) = transform_coord_to_unitRF(x[j], y[j], cx[i], cy[i], theta[i], a[i]);
				prob.push_back(prob_dist(norm_x, norm_y, phase[i], sfreq[i], amp[i], baRatio[i], LM_ref[i], on_off[poolList[i][j]], sig[i]));
			}
			normalize_prob(prob, percent);
			// make connections and strength (normalized i.e., if prob > 1 then s = 1 else s = prob)
			vector<Int> newList;
			newList.reserve(m);
			vector<Float> sr;
			sr.reserve(m);
			for (Size j = 0; j < m; j++) {
				if (dist(rGen) < prob[j]) {
					newList.push_back(poolList[i][j]);
					if (prob[j] > 1) {
						sr.push_back(prob[j]);
					} else {
						sr.push_back(1);
					}
				}
			}
			srList.push_back(sr);
			//cout << newList.size() << " <= " << poolList[i].size() << "\n";
			poolList[i] = newList;
			poolList[i].shrink_to_fit();
			// calculate vf position (final)
			if (newList.size() > 0) { 
				x.clear();
				y.clear();
				for (Size j = 0; j < newList.size(); j++) {
					x.push_back(cart0.first[newList[j]]);
					y.push_back(cart0.second[newList[j]]);
				}
				tie(cx[i], cy[i]) = average(x, y);
			} else {
				cx[i] = cart.first[i];
				cy[i] = cart.second[i];
			}
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
vector<Int> draw_from_radius(
        const Float x0,
        const Float y0, 
        const Size n,
        const pair<vector<Float>,vector<Float>> &cart,
        const Float radius) {
    vector<Int> index;

    Float mx = 0.0f;
    Float my = 0.0f;
	Float min_dis;
	Size min_id;
    for (Size i=0; i<n; i++) {
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

vector<vector<Int>> retinotopic_vf_pool(
        const pair<vector<Float>,vector<Float>> &cart,
        const pair<vector<Float>,vector<Float>> &cart0,
        const bool use_cuda,
        RandomEngine &rGen,
        vector<Float> &baRatio,
        vector<Float> &a) {
    vector<vector<Int>> poolList;
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
        vector<Float> VFposEcc;
        vector<Float> normRand;
        vector<Float> rMap;
        VFposEcc.reserve(n);
        normRand.reserve(n);
        rMap.reserve(n);
        a.reserve(n);
        normal_distribution<Float> dist(0, 1);

        // transform cart to pol coords
        for (Size i=0; i<n; i++) {
            VFposEcc.push_back(norm_vec(cart.first[i], cart.second[i]));
        }
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
		const Size m = cart0.first.size();
        for (Size i=0; i<n; i++) {
            poolList.push_back(draw_from_radius(cart.first[i], cart.second[i], m, cart0, rMap[i]));
        }
    }
    return poolList;
}

// unit test
int main(int argc, char *argv[]) {
	namespace po = boost::program_options;
    Float percent = 0.50;
    
	string LGN_vpos_filename, V1_vpos_filename, V1_prop_filename;
	po::options_description input_opt("input options");
	input_opt.add_options()
		("fV1_prop", po::value<string>(&V1_prop_filename)->default_value("V1_prop.bin"),"file that stores V1 neurons' parameters")
		("fLGN_vpos", po::value<string>(&LGN_vpos_filename)->default_value("LGN_vpos.bin"),"file that stores LGN position in visual field (and on-cell off-cell label)")
		("fV1_vpos", po::value<string>(&V1_vpos_filename)->default_value("V1_vpos.bin"),"file that stores V1 position in visual field)");

    string V1_filename, idList_filename, sList_filename; 
	po::options_description output_opt("output options");
	output_opt.add_options()
		("fV1", po::value<string>(&V1_filename)->default_value("V1.bin"),"file that stores V1 neurons' information")
		("fLGN_V1_ID", po::value<string>(&idList_filename)->default_value("LGN_V1_idList.bin"),"file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&sList_filename)->default_value("LGN_V1_sList.bin"),"file stores LGN to V1 connection strengths");

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
	fV1_vpos.read(reinterpret_cast<char*>(&n), sizeof(Size));
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

	ifstream fLGN_vpos;
	fLGN_vpos.open(LGN_vpos_filename, fstream::in | fstream::binary);
	if (!fLGN_vpos) {
		cout << "Cannot open or find " << LGN_vpos_filename << "\n";
		return EXIT_FAILURE;
	}
	Size m;
	fLGN_vpos.read(reinterpret_cast<char*>(&m), sizeof(Size));
	cout << m << " pre-synaptic neurons\n";
	cout << "need " << 3 * m * sizeof(Float) / 1024 / 1024 << "mb\n";
	//temporary vectors to get coordinate pairs
	vector<Float> x0(m);
	vector<Float> y0(m);
	fLGN_vpos.read(reinterpret_cast<char*>(&x0[0]), m*sizeof(Float));
	fLGN_vpos.read(reinterpret_cast<char*>(&y0[0]), m*sizeof(Float));
	auto cart0 = make_pair(x0, y0);
	// release memory from temporary vectors
	x0.swap(vector<Float>());
	y0.swap(vector<Float>()); 
	vector<Int> LM_OF(m);
	fLGN_vpos.read(reinterpret_cast<char*>(&LM_OF[0]), m * sizeof(Int));
	fLGN_vpos.close();

	cout << "carts ready\n";
    vector<Int> seed{820,702};
    seed_seq seq(seed.begin(), seed.end());
    RandomEngine rGen(seq);
    vector<Float> a; // radius of the VF
	vector<Float> baRatio = generate_baRatio(n, rGen);
    vector<vector<Int>> poolList = retinotopic_vf_pool(cart, cart0, false, rGen, baRatio, a);
	cout << "poolList and R ready\n";

	vector<Int> LM_phaseRef(n);
	vector<Float> theta(n);
	vector<Float> phase(n);
	vector<Float> amp(n);
	vector<Float> sig(n);
    ifstream fV1_prop(V1_prop_filename, fstream::in | fstream::binary);
	if (!fV1_prop) {
		cout << "Cannot open or find " << V1_prop_filename <<"\n";
		return EXIT_FAILURE;
	}
	fV1_prop.read(reinterpret_cast<char*>(&LM_phaseRef[0]), n * sizeof(Int));
	fV1_prop.read(reinterpret_cast<char*>(&theta[0]), n * sizeof(Float));
	fV1_prop.read(reinterpret_cast<char*>(&phase[0]), n * sizeof(Float));
	fV1_prop.read(reinterpret_cast<char*>(&amp[0]), n * sizeof(Float));
	fV1_prop.read(reinterpret_cast<char*>(&sig[0]), n * sizeof(Float));
    fV1_prop.close();

	vector<Float> sfreq = generate_sfreq(n, rGen);
	vector<Float> cx(n);
	vector<Float> cy(n);
    vector<vector<Float>> srList = retinotopic_connection(poolList, rGen, percent, n, cart, cart0, theta, phase, sfreq, amp, sig, baRatio, a, LM_phaseRef, LM_OF, cx, cy);

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
	fV1.write((char*)&LM_phaseRef[0], n * sizeof(Int));
    fV1.close();

    // write poolList to disk
	//print_listOfList<Int>(poolList);
	write_listOfList<Int>(idList_filename, poolList, false);
	write_listOfList<Float>(sList_filename, srList, false);
    return 0;
}
