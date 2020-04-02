#include "condShape.h"

void learnQ(LearnVarShapeQ &l, Float tauQ[], Float A_Q[], Size n) {
    l.n = n;
    for (PosInt i=0; i<n; i++) {
        l.tau[2*i+0] = tauQ[2*i+0]; // post E
        l.tau[2*i+1] = tauQ[2*i+1]; // pre I
        l.A_LTP[i] = A_Q[2*i+0];
        l.A_LTD[i] = A_Q[2*i+1];
    }
}
void printQ(LearnVarShapeQ &l) {
    std::cout << "I -> E learning time scales\n";
    std::cout << "#     preQ    postQ    rateQLTP   rateQLTD\n";
    for (PosInt i=0; i<l.n; i++) {
        std::cout << i << ":    " << l.tau[2*i+0] << ",     " << l.tau[2*i+1] << ",      " << l.A_LTP[i] << ",     " << l.A_LTD[i] << "\n";
    }
}
void learnE(LearnVarShapeE &l, Float tauLTP[], Float tauLTD[], Float tau_trip[], Float tau_avg, Float A_LTP[], Size n) {
    l.n = n;
    for (PosInt i=0; i<n; i++) {
        l.tau[3*i+0] = tauLTP[i];
        l.tau[3*i+1] = tauLTD[i];
        l.tau[3*i+2] = tau_trip[i];
        l.A_LTP[i] = A_LTP[i];
        l.A_ratio[i] = tau_trip[i]*A_LTP[i]/(tauLTD[i]*(tau_avg*tau_avg)); // * tau_LTP * filtered spike avg^2 / target firing rate = A_LTD
    }
    l.tau[3*n] = tau_avg;
}
void printE(LearnVarShapeE &l) {
    std::cout << "E -> E learning time scales, sp_avg: " << l.tau[3*n] << "\n";
    std::cout << "#     LTP     LTD     trip    rateLTP\n";
    for (PosInt i=0; i<l.n; i++) {
        std::cout << i << ":    " << l.tau[3*i+0] << ",    " << l.tau[3*i+1] << ",    " << l.tau[3*i+2] << ",    " << l.A_LTP[i] << "\n";
    }
}
