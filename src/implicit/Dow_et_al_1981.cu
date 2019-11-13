#include "Dow_et_al_1981.h"
using namespace std;

// Dow et al., 1981
pair<Float, Float> lognstats(const Float m, const Float s) {
	Float mean = log(m*m / sqrt(s + m * m));
	Float std = sqrt(log(s / (m*m) + 1));
	return make_pair(mean, std);
}

vector<Float> generate_sfreq(Size n, RandomEngine &rGen) {
	vector<Float> sfreq;
	sfreq.reserve(n);
	const Float ratio = 0.12; // ratio between two distributions
	const Size n1 = static_cast<Size> (n*ratio);
	const Size n2 = n - n1;
	const Float m1 = 1.0;
	const Float s1 = 0.2;
	const Float m2 = (4.05*n - n1 * m1) / n2;
	const Float s2 = m2 * 1.0;

	normal_distribution<Float> normal(m1, s1);
	for (Size i = 0; i < n1; i++) {
		Float v = normal(rGen);
		if (v < 1) {
			sfreq.push_back(1);
		}
		else {
			sfreq.push_back(v);
		}
	}

	pair<Float, Float> logs = lognstats(m2, s2);
	lognormal_distribution<Float> lognormal(logs.first, logs.second);
	for (Size i = 0; i < n2; i++) {
		sfreq.push_back(lognormal(rGen) / 2.0);
	}
	random_shuffle(sfreq.begin(), sfreq.end());
	return sfreq;
}

vector<Float> generate_baRatio(Size n, RandomEngine &rGen) {
	vector<Float> baRatio;
	baRatio.reserve(n);
	const Float ratio = 0.6; // ratio between two distributions
	const Size n1 = static_cast<Size> (n*ratio);
	const Size n2 = n - n1;
	const Float m1 = 1.8;
	const Float s1 = 0.4;
	const Float m2 = (1.71*n - n1 * m1) / n2;
	const Float s2 = 0.7;

	normal_distribution<Float> normal(m1, s1);
	for (Size i = 0; i < n1; i++) {
		Float v = normal(rGen);
		if (v <= 0) {
			baRatio.push_back(1);
		}
		else {
			baRatio.push_back(v);
		}
	}
	pair<Float, Float> logs = lognstats(m2 - 1, s2);
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
	random_shuffle(baRatio.begin(), baRatio.end());
	cout << "baRatio size: " << baRatio.size() << "\n";
	return baRatio;
}

// map ecc to radius
Float mapping_rule(Float ecc, Float normalRand, RandomEngine &rGen) {
	// *** set LGN contribution of the total RF size
	const Float ratio = sqrt(0.8 / M_PI);
	// Dow et al., 1981
	const Float a = 13.32f;
	const Float b = 0.037f;
	const Float mean = a + b * ecc;
	const Float c = 3.32f;
	const Float d = 0.0116f;
	const Float scatter = c + d * ecc;
	Float R = scatter * normalRand + mean;
	if (R < 0) {
		// rarely rethrow
		normal_distribution<Float> dist(mean, scatter);
		do {
			R = dist(rGen);
		} while (R < 0);
	}
	return ratio*R;
}
