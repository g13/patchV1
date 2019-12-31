#ifndef DOW_ET_AL_1981_H
#define DOW_ET_AL_1981_H

#include <vector>
#include <iostream>
#include <utility>
#include "../util/util.h"
#include "../types.h"

std::pair<Float, Float> lognstats(const Float m, const Float s);
std::vector<Float> generate_sfreq(const Size n, RandomEngine &rGen);
std::vector<Float> generate_baRatio(const Size n, RandomEngine &rGen);
Float mapping_rule(const Float ecc, const Float normalRand, RandomEngine &rGen);
#endif
