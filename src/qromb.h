#ifndef QROMB_H
#define QROMB_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "cuda_precision.h"
#include "func.h"
/* Romberg Quadrature */

// Numerical Analysis 9th edition, Burden & Faires
std::ofstream dummy;
float_point qromb(Func1 *func, int iter, float_point h, int verbose = 0, std::ofstream &outputData = dummy) {
    float_point *R1 = new float_point[iter];
    float_point *R2 = new float_point[iter];
    float_point left = func->eval(0);
    float_point right = func->eval(h);
    R1[0] = 0.5*h*(left+right);
    if (verbose) {
        float_point t = 0;
        outputData.write((char*)&t, sizeof(float_point));
        outputData.write((char*)&left, sizeof(float_point));
        t = h;
        outputData.write((char*)&t, sizeof(float_point));
        outputData.write((char*)&right, sizeof(float_point));
    }
    for (int i=2; i<=iter; i++){
        float_point sum = 0;
        for (int j=1; j<=pow(2,i-2); j++){
            float_point t = (j-0.5)*h;
            float_point value = func->eval((j-0.5)*h);
            if (verbose) {
                outputData.write((char*)&t, sizeof(float_point));
                outputData.write((char*)&value, sizeof(float_point));
            }
            sum += value;
        }
        R2[0] = 0.5*(R1[0]+ sum*h);
        for (int j=1; j<i; j++) {
            R2[j] = R2[j-1] + (R2[j-1]-R1[j-1])/(pow(4,j)-1.0);
        }
        h = h*0.5;
        if (verbose) {
            if (i == iter || verbose > 1) {
                std::cout << "converging error: " << std::setprecision(6) << R2[i-1]-R2[i-2] << ", # iterations: " << i << ", # evaluations: " << pow(2,i-1) + 1 << "\n";
            }
            if (verbose > 2) {
                std::cout << "    R1: ";
                for (int j=0; j<i-1; j++) {
                    std::cout << std::setprecision(6) << R1[j];
                    if (j<i-2) std::cout << ", ";
                    else std::cout << "\n";
                }
                std::cout << "    R2: ";
                for (int j=0; j<i; j++) {
                    std::cout << std::setprecision(6) << R2[j];
                    if (j<i-1) std::cout << ", ";
                    else std::cout << "\n";
                }
            }
        }
        for (int j=0; j<i; j++) {
            R1[j] = R2[j];
        }
    }
    float_point integral = R1[iter-1];
    delete []R1;
    delete []R2;
    return integral;
}

#endif
