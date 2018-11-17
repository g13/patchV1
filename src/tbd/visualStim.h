#include "FreeImagePlus.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
using std::cout;
using std::endl;
using std::vector;
using std::string;
template<typename T>
void getBWi(vector<vector<float>> R, vector<vector<float>> G, vector<vector<float>> B, BYTE *bits, unsigned width, unsigned height, unsigned pitch, unsigned bpp, bool minIsBlack = true) {
    for (unsigned i=0; i<height; i++) {
        T *pixel = (T *) bits;
        for (unsigned j=0; j<width; j++) {
            float BW = static_cast<float>(pixel[j])/(pow(2,bpp)-1);
            if (!minIsBlack) {
                BW = 1-BW;
            }
            R[i][j] = BW;
            G[i][j] = BW;
            B[i][j] = BW; 
        }
        bits += pitch;
    }
}
template<typename T>
void getBWf(vector<vector<float>> R, vector<vector<float>> G, vector<vector<float>> B, BYTE *bits, unsigned width, unsigned height, unsigned pitch) {
    for (unsigned i=0; i<height; i++) {
        T *pixel = (T *) bits;
        for (unsigned j=0; j<width; j++) {
            float BW = static_cast<float>(pixel[j]);
            assert(BW <=1 && BW > 0);
            R[i][j] = BW;
            G[i][j] = BW;
            B[i][j] = BW; 
        }
        bits += pitch;
    }
}

void ErrorHandler(FREE_IMAGE_FORMAT fif, const char *message){
    cout << "\n";
    cout << "***\n";
    if(fif != FIF_UNKNOWN) {
        cout<< FreeImage_GetFormatFromFIF(fif) << " format\n";
    }
    cout << message << "\n";
    cout << "***\n";
}
