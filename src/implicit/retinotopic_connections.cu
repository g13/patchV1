#include "retinotopic_connections.h"
using namespace std;

pair<Float, Float> average(vector<Float> x, vector<Float> y) {
    Float mx = accumulate(x.begin(), x.end(), 0.0f)/x.size();
    Float my = accumulate(y.begin(), y.end(), 0.0f)/y.size();
    return make_pair(mx,my);
}

pair<Float, Float> transform_coord_to_unitRF(Float x, Float y, const Float mx, const Float my, const Float theta, const Float a) {
    // a is half-width at the x-axis
    x = (x - mx)/a;
    y = (y - my)/a;
    Float new_x, new_y;
    new_x = cos(theta) * x + sin(theta) * y;
    new_y = -sin(theta) * y + cos(theta) * y;
    return make_pair(new_x, new_y);
}

// probability distribution
Float prob_dist(Float x, Float y, Float phase, Float sfreq, Float amp, Float baRatio, int on_off, Float sig = 1.1775) {
    // sfreq should be given as a percentage of the width
    // baRatio comes from Dow et al., 1981
    assert(abs(on_off) == 1);
    Float envelope;
    envelope = exp(-0.5*(pow(x/sig,2)+pow(y/(sig*baRatio),2)));
    // when sfreq == 1, the RF contain a full cycle of cos(x) x\in[-pi, pi]
    Float modulation = amp * cos(M_PI*sfreq * x + phase);
    if (on_off * modulation < 0) {
        modulation = 0;
    }
    Float prob = envelope * (1 - amp + on_off*modulation);
 	return prob;
}

void normalize_prob(vector<Float> &prob, Float percent) {
    const Float norm = accumulate(prob.begin(), prob.end(), 0.0) / (percent * prob.size());
	cout << "norm = " << norm << "\n";
    Float sum = 0.0;
    for (int i=0; i<prob.size(); i++) {
        prob[i] = prob[i] / norm;
        sum += prob[i];
    }
    cout << percent*prob.size() << " ~ " << sum << "\n";
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
        const pair<vector<Float>,vector<Float>> &cart0, // LGN VF position
        const vector<Float> &theta,
        const vector<Float> &phase,
        const vector<Float> &sfreq,
        const vector<Float> &amp,
        const vector<Float> &sig,
        const vector<Float> &baRatio,
        const vector<Float> &a,
        const vector<Int> &on_off,
        vector<Float> &cx,
        vector<Float> &cy) {
    uniform_real_distribution<Float> dist(0,1);
    vector<vector<Float>> srList;
    // for each neuron i in sheet 2 at (e, p)
    for (Size i=0; i<n; i++) {
        const Int m = poolList[i].size();
        vector<Float> sr;
        vector<Float> x, y;
        x.reserve(m);
        assert(x.size() == 0);
        for (Size j=0; j<m; j++) {
            x.push_back(cart0.first[poolList[i][j]]);
            y.push_back(cart0.second[poolList[i][j]]);
        }
        pair<Float, Float> center = average(x, y);
        cx[i] = center.first;
        cy[i] = center.second;
		cout << "(" << cx[i] << ", " << cy[i] << ")\n";
        vector<Float> prob;
        prob.reserve(m);
        for (Size j=0; j<m; j++) {
            pair<Float, Float> norm_coord = transform_coord_to_unitRF(x[j], y[j], center.first, center.second, theta[i], a[i]);
            prob.push_back(prob_dist(norm_coord.first, norm_coord.second, phase[i], sfreq[i], amp[i], baRatio[i], on_off[poolList[i][j]], sig[i]));
        }
        normalize_prob(prob, percent);
        //print_list(prob);
        vector<Int> newList;
        newList.reserve(poolList[i].size());
        sr.reserve(m);
        for (Size j=0; j<m; j++) {
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
        cout << newList.size() << " <= " << poolList[i].size() << "\n";
        poolList[i].resize(newList.size());
        poolList[i] = newList;
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
    for (Size i=0; i<n; i++) {
        if (norm_vec(cart.first[i] - x0, cart.second[i] - y0) < radius) {
            index.push_back(i);
            mx = mx + cart.first[i];
            my = my + cart.second[i];
        }
    }
    cout << "(" << mx/index.size() << ", " << my/index.size() << ")\n";
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
            checkCudaErrors(cudaMalloc((void **) &d_cart, 2*2*n*sizeof(float)));
            checkCudaErrors(cudaMemcpy(d_cart, &(cart.first[0]), n*sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_cart+n, &(cart.second[0]), n*sizeof(float), cudaMemcpyHostToDevice));
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
            Float R = mapping_rule(VFposEcc[i]*60, normRand[i], rGen)/120.0f;
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
    for (int i = 0; i<argc; i++) {
        cout << argv[i] << " ";
    }
    cout << "\n";
	char tmp1[101];
	char tmp2[101];
    Float percent;
	if (argc == 4) {
	  	sscanf(argv[argc-1], "%f", &percent);
        sscanf(argv[argc-2], "%100s", tmp2);
        sscanf(argv[argc-3], "%100s", tmp1);
    }
	string fname = tmp1;
    Size n, m;
	ifstream p_file;
	p_file.open(fname, fstream::in | fstream::binary);
	if (!p_file.is_open()) {
		cout << "Cannot open or find " << fname <<"\n";
		return EXIT_FAILURE;
	}
	p_file.read(reinterpret_cast<char*>(&n), sizeof(int));
    cout << n << " post-synaptic neurons\n";
	vector<Float> x(n);
	vector<Float> y(n);
	p_file.read(reinterpret_cast<char*>(&x[0]), n * sizeof(float));
	p_file.read(reinterpret_cast<char*>(&y[0]), n * sizeof(float));
	auto cart = make_pair(x, y);
	//print_pair(cart);
	x.swap(vector<Float>());
	y.swap(vector<Float>());

	p_file.read(reinterpret_cast<char*>(&m), sizeof(int));
	vector<Float> x0(m);
	vector<Float> y0(m);
	p_file.read(reinterpret_cast<char*>(&x0[0]), m*sizeof(float));
	p_file.read(reinterpret_cast<char*>(&y0[0]), m*sizeof(float));
	vector<Int> on_off(m);
	p_file.read(reinterpret_cast<char*>(&on_off[0]), m*sizeof(int));
	auto cart0 = make_pair(x0, y0);
	//print_pair(cart0);
	x0.swap(vector<Float>());
	y0.swap(vector<Float>());
	p_file.close();

	cout << "carts ready\n";
    vector<Int> seed{820,702};
    seed_seq seq(seed.begin(), seed.end());
    RandomEngine rGen(seq);
    vector<Float> a; // radius of the VF
	vector<Float> baRatio = generate_baRatio(n, rGen);
    vector<vector<Int>> poolList = retinotopic_vf_pool(cart, cart0, false, rGen, baRatio, a);
	cout << "poolList and R ready\n";
	string pname = tmp2;
    fstream prop_file;
	prop_file.open(pname, fstream::in | fstream::binary);
	if (!prop_file.is_open()) {
		cout << "Cannot open or find " << pname <<"\n";
		return EXIT_FAILURE;
	}
	vector<Float> theta(n);
	vector<Float> phase(n);
	vector<Float> amp(n);
	vector<Float> sig(n);
	prop_file.read(reinterpret_cast<char*>(&theta[0]), n * sizeof(float));
	prop_file.read(reinterpret_cast<char*>(&phase[0]), n * sizeof(float));
	prop_file.read(reinterpret_cast<char*>(&amp[0]), n * sizeof(float));
	prop_file.read(reinterpret_cast<char*>(&sig[0]), n * sizeof(float));
    prop_file.close();
	vector<Float> sfreq = generate_sfreq(n, rGen);
	vector<Float> cx(n);
	vector<Float> cy(n);
    vector<vector<Float>> srList = retinotopic_connection(poolList, rGen, percent, n, cart0, theta, phase, sfreq, amp, sig, baRatio, a, on_off, cx, cy);

	prop_file.open(pname, fstream::app | fstream::binary);
	if (!prop_file.is_open()) {
		cout << "Cannot open or find " << pname <<"\n";
		return EXIT_FAILURE;
	}
    prop_file.write((char*)&a[0], n * sizeof(float));
    prop_file.write((char*)&baRatio[0], n * sizeof(float));
	prop_file.write((char*)&sfreq[0], n * sizeof(float));
    prop_file.write((char*)&cx[0], n * sizeof(float));
	prop_file.write((char*)&cy[0], n * sizeof(float));
    prop_file.close();

    // write poolList to disk
	print_listOfList<Int>(poolList);
    ofstream r_file("LGNtoV1.bin", std::ios::out|std::ios::binary);
    for (Size i=0; i<n; i++) {
        Int listSize = poolList[i].size();
        r_file.write((char*)&listSize, sizeof(int));
        r_file.write((char*)&poolList[i][0], listSize * sizeof(int));
        r_file.write((char*)&srList[i][0], listSize * sizeof(float));
    }
    return 0;
}
