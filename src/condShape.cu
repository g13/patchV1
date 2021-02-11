#include "condShape.h"

void learnQ(LearnVarShapeQ &l, Float tauQ[], Float A_Q[], Float gmax, Float gmin, Size n) {
    l.n = n;
    for (PosInt i=0; i<n; i++) {
        l.tau[2*i+0] = tauQ[2*i+0]; // post E
        l.tau[2*i+1] = tauQ[2*i+1]; // pre I
        l.A_LTP[i] = A_Q[2*i+0];
        l.A_LTD[i] = A_Q[2*i+1];
    }
    l.gmax = gmax;
    l.gmin = gmin;
}
void printQ(LearnVarShapeQ &l) {
    std::cout << "I -> E learning time scales\n";
    std::cout << "#     preQ    postQ    rateQLTP   rateQLTD\n";
    for (PosInt i=0; i<l.n; i++) {
        std::cout << i << ":    " << l.tau[2*i+0] << ",     " << l.tau[2*i+1] << ",      " << l.A_LTP[i] << ",     " << l.A_LTD[i] << "\n";
    }
    std::cout << "strength range from: " << l.gmin << " to " << l.gmax << "\n";
}
void learnE(LearnVarShapeE &l, Float tauLTP[], Float tauLTD[], Float tauTrip[], Float tauAvg, Float targetFR, Float A_LTP[], Float gmax, Float gmin, Size n) {
    l.n = n;
    for (PosInt i=0; i<n; i++) {
        l.tau[3*i+0] = tauLTP[i];
        l.tau[3*i+1] = tauLTD[i];
        l.tau[3*i+2] = tauTrip[i];
        l.A_LTP[i] = A_LTP[i];
        Float tauAvg_in_sec = tauAvg/1000.0;
        l.A_ratio[i] = tauTrip[i]*l.A_LTP[i]/(tauLTD[i]*(tauAvg_in_sec*tauAvg_in_sec))/1000.0; // * tau_LTP * filtered spike avg^2 / target firing rate = A_LTD
    }
    l.tau[3*n] = tauAvg;
    l.targetFR = targetFR;
    l.gmax = gmax;
    l.gmin = gmin;
}
void printE(LearnVarShapeE &l) {
    std::cout << "E -> E learning time scales, sp_avg: " << l.tau[3*l.n] << "\n";
    std::cout << "#     LTP     LTD     trip    rateLTP\n";
    for (PosInt i=0; i<l.n; i++) {
        std::cout << i << ":    " << l.tau[3*i+0] << ",    " << l.tau[3*i+1] << ",    " << l.tau[3*i+2] << ",    " << l.A_LTP[i] << "\n";
    }
    std::cout << "strength range from: " << l.gmin << " to " << l.gmax << "\n";
}
