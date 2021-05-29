#ifndef DOW_ET_AL_1981_H
#define DOW_ET_AL_1981_H

#include <vector>
#include <iostream>
#include <utility>
#include <cuda.h>
#include <curand_kernel.h>
#include <random>
#include <algorithm>
#include "../util/util.h"
#include "../types.h"
std::vector<Float> generate_sfreq(const Size n, RandomEngine &rGen);
std::vector<Float> generate_baRatio(const Size n, RandomEngine &rGen);
Float mapping_rule(const Float ecc, const Float normalRand, RandomEngine &rGen, Float LGN_V1_RFratio);

// map ecc to radius
__device__
__forceinline__
Float mapping_rule_CUDA(Float ecc, curandStateMRG32k3a rGen, Float LGN_V1_RFratio) {
	// *** set LGN contribution of the total RF size
	const Float ratio = sqrt(LGN_V1_RFratio / M_PI);
	// Dow et al., 1981 Fig. 7 TODO: consider generalized extreme value distribution
	//const Float a = 13.32f;
	const Float a = 6.0;
	const Float b = 0.037f;
	const Float mean = a + b * ecc;
	const Float std = 0.01; //3.0; // it is NOT the scatter in the paper, which means the VF center's scatter around an electrode
	const Float lower_bound = 3.0; // TODO: look for the min(LGN RF size, V1 neuron RF size)
    // R = sqrt(area)
	Float R = curand_normal(&rGen)*std + mean;
	
	if (R < lower_bound) {
		// rarely rethrow
		do {
			R = curand_normal(&rGen)*std + mean;
		} while (R < lower_bound);
	}
	//return ratio*R;
	return ratio*mean;
}
#endif
