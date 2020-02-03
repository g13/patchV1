#include "Dow_et_al_1981.h"
using namespace std;

// Dow et al., 1981

vector<Float> generate_sfreq(Size n, RandomEngine &rGen) {
	// width to bar ratio ~ spatial frequency doubled
	vector<Float> sfreq;
	sfreq.reserve(n);
	const Float ratio = 0.12; // ratio between two distributions
	const Size n1 = static_cast<Size> (n*ratio);
	const Size n2 = n - n1;
	// mean and std for the first distribution
	const Float m1 = 1.0;
	const Float std = 0.2;
	// mean and std for the second distribution
	const Float m2 = (4.05*n - n1 * m1) / n2;
	const Float s2 = sqrt(m2);

	// generate from the first distribution
	normal_distribution<Float> normal(m1, std);
	for (Size i = 0; i < n1; i++) {
		Float v = normal(rGen);
		if (v < 1) {
			sfreq.push_back(1.0/2.0);
		}
		else {
			sfreq.push_back(v/2.0);
		}
	}

	// generate from the second distribution
	pair<Float, Float> logs = lognstats<Float>(m2, s2);
	lognormal_distribution<Float> lognormal(logs.first, logs.second);
	for (Size i = 0; i < n2; i++) {
		sfreq.push_back(lognormal(rGen)/2.0);
	}
	// mix the two distribution
	random_shuffle(sfreq.begin(), sfreq.end());
	return vector<Float>(n, 1.0);
}

vector<Float> generate_baRatio(Size n, RandomEngine &rGen) {
	vector<Float> baRatio;
	baRatio.reserve(n);
	const Float ratio = 0.6; // ratio between two distributions
	const Size n1 = static_cast<Size> (n*ratio);
	const Size n2 = n - n1;
	// mean and std for the first distribution
	const Float m1 = 1.8;
	const Float std = 0.4;
	// mean and std for the second distribution
	const Float m2 = (1.71*n - n1 * m1) / n2;
	const Float s2 = 0.8367;

	// generate from the first distribution
	normal_distribution<Float> normal(m1, std);
	for (Size i = 0; i < n1; i++) {
		Float v = normal(rGen);
		if (v <= 0) {
			baRatio.push_back(1);
		}
		else {
			baRatio.push_back(v);
		}
	}
	// generate from the second distribution
	pair<Float, Float> logs = lognstats<Float>(m2 - 1, s2);
	lognormal_distribution<Float> lognormal(logs.first, logs.second);
	for (Size i = 0; i < n2; i++) {
		Float v = lognormal(rGen) + 1;
		if (v < 1) {
			baRatio.push_back(1);
		}
		else {
			baRatio.push_back(v);
		}
	}
	// mix the two distribution
	random_shuffle(baRatio.begin(), baRatio.end());
	return baRatio;
}

// map ecc to radius
Float mapping_rule(Float ecc, Float normalRand, RandomEngine &rGen, Float LGN_V1_RFratio) {
	// *** set LGN contribution of the total RF size
	const Float ratio = sqrt(LGN_V1_RFratio / M_PI);
	// Dow et al., 1981 Fig. 7 TODO: consider generalized extreme value distribution
	const Float a = 20.0f; //13.32f;
	const Float b = 0.037f;
	const Float mean = a + b * ecc;
	const Float std = 3.0; // it is NOT the scatter in the paper, which means the VF center's scatter around an electrode
	const Float lower_bound = 3.0; // TODO: look for the min(LGN RF size, V1 neuron RF size)
	Float R = std  * normalRand + mean;
	
	if (R < lower_bound) {
		// rarely rethrow
		normal_distribution<Float> dist(mean, std);
		do {
			R = dist(rGen);
		} while (R < lower_bound);
	}
	return ratio*R;
}

