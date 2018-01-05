#include "poisson_process.h"
#include <fstream>

linear_distribution::linear_distribution(double r0, double r1, double dt) {
    r_ = (r0+r1)/2;
    double k = 1/(dt*r_);
    double a = (r1-r0)/(2*dt)*k;
    b = k*r0;
    b_2 = b*b;
    a2 = a*2;
    a4 = a*4;
}
double linear_distribution::roll(double rand) {
    return (-b+sqrt(b_2+a4*rand))/a2;
}
void linear_distribution::roll_all(vector<double> &rand, vector<double> &result) {
    result.reserve(rand.size());
    for (unsigned int i=0; i<rand.size(); i++) {
        result.push_back(roll(rand[i]));
    }
}

int predetermine_poisson_spikes(vector<double> &spikeTime, double maxPoissonRate, double endTime, unsigned long seed, string outputFn, bool saveInputToFile, const vector<double> &inputRate, double dt, unsigned long nt) { 
    std::minstd_rand poiGen, ranGen;
    std::uniform_real_distribution<double> uniform0_1=std::uniform_real_distribution<double>(0.0,1.0);
    poiGen.seed(seed);
    spikeTime.reserve(static_cast<unsigned long>(maxPoissonRate*1.5*endTime));
    if (nt==0) {
        assert(dt==0 && &inputRate == NULL);
        do spikeTime.push_back(next_poisson_const(poiGen, spikeTime.back(), maxPoissonRate, uniform0_1));
        while (spikeTime.back() < endTime);
        spikeTime.pop_back();
    } else {
        ranGen.seed(seed+poiGen());
        assert(inputRate.size()==nt+1);
        for (unsigned long i=0; i<nt; i++) {
            next_poisson_non_const(spikeTime, inputRate[i], inputRate[i+1], dt, dt*i, maxPoissonRate, poiGen, ranGen, uniform0_1);
        }
    }
    if (saveInputToFile) {
        std::ofstream file;
        file.open(outputFn, std::ios::binary);
        if (file) {
            file.write((char*)&(spikeTime[0]),spikeTime.size()*sizeof(double));
            file.close();
        } else {
            return 0;
        }
    }
}
