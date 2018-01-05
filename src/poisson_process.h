#ifndef POISSON_PROCESS_H
#define POISSON_PROCESS_H
// generate poisson process
//
#include<cmath>
#include<random> 
#include<vector>
#include<cstring>
#include<cassert>
using std::string;
using std::vector;

struct linear_distribution {
    // just a quadratic solution....
    double b;
    double b_2; //b^2 
    double a2;  // a*2
    double a4;  // a*4
    double r_;  // (r0+r1)/2
    linear_distribution(double r0, double r1, double dt);
    double roll(double rand);
    void roll_all(vector<double> &rand, vector<double> &result);
};
typedef linear_distribution linearDist;

inline double next_poisson_const(std::minstd_rand &poiGen, double lastSpikeTime, double poissonRate, std::uniform_real_distribution<double> &uniform0_1) {
    return lastSpikeTime - log(uniform0_1(poiGen))/poissonRate;
}

inline int next_poisson_non_const(vector<double> &spikeTime, double pR0, double pR1, double dt, double t0, double maxPoissonRate, std::minstd_rand &poiGen, std::minstd_rand &ranGen, std::uniform_real_distribution<double> &uniform0_1) { 
    linearDist linear(pR0,pR1,dt);
    if (uniform0_1(poiGen) < linear.r_/maxPoissonRate) {
        spikeTime.push_back(t0 + linear.roll(uniform0_1(ranGen)));
    } else {
        return 0;
    }
}

int predetermine_poisson_spikes(vector<double> &spikeTime, double maxPoissonRate, double endTime, unsigned long seed, string outputFn="", bool saveInputToFile = false, const vector<double>& inputRate = vector<double>(), double dt = 0, unsigned long nt = 0); 

#endif
