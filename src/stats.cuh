#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "types.h"
#include "MACRO.h"

__global__
void pixelizeOutput(
        Float* __restrict__ fr,
        Float* __restrict__ output,
        PosInt* __restrict__ pid, 
		Size* __restrict__ m, // within one pixel
		Size nPerPixel_I, Size nPerPixel_C, Size nPixel_I, Size nPixel, Size n, Float odt
);

void reshape_chunk_and_write(Float chunk[], std::ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, Size neuronPerBlock, Size nGap, bool hWrite);

void getLGN_V1_surface(std::vector<int> &xy, std::vector<std::vector<PosInt>> &LGN_V1_ID, int surface_xy[], Size nLGNperV1[], Size max_LGNperV1, Size nLGN);

// VisLGN, VisV1 (visual field)  or PhyV1 (physical position) with mixed [C]ontralateral and [I]psilateral
template<typename T>
std::vector<std::vector<PosInt>> getUnderlyingID(T x[], T y[], Int* pick, Size n0, Size n, Size width, Size height, T x0, T xspan, T y0, T yspan, Size* maxPerPixel, Size checkN) {
	// offset normally is the column's ID that separate left and right
	std::vector<std::vector<PosInt>> uid(height*width, std::vector<PosInt>());
    *maxPerPixel = 1;
    for (PosInt i=0; i<n-n0; i++) {
        if (pick[i] > 0) {
            PosInt idx = static_cast<PosInt>(((x[i]-x0)/xspan)*width);
            PosInt idy = static_cast<PosInt>(((y[i]-y0)/yspan)*height);
            if (idx == width) idx--;
            if (idy == height) idy--;
            PosInt id = idx+idy*width;
            if (idx >= width) {
                std::cout << "x[" << i << "]: " << x[i] << ", width: " << width << ", x0: " << x0 << ", xspan: " << xspan << "\n";
                std::cout << "idx = " << idx << " < " << width << "\n";
                assert(idx<width);
            }
            if (idy >= height) {
                std::cout << "y[" << i << "]: " << y[i] << ", height: " << height << ", y0: " << y0 << ", yspan: " << yspan << "\n";
                std::cout << "idy = " << idy << " < " << height << "\n";
                assert(idy<height);
            }
            uid[id].push_back(i+n0);
            if (uid[id].size() > *maxPerPixel) *maxPerPixel = uid[id].size();
        }
    }
    if (n-n0 > 0) {
        std::vector<bool> picked(n-n0, false);
        for (PosInt i=0; i<height*width; i++) {
            for (PosInt j=0; j<uid[i].size(); j++) {
                assert(uid[i][j] >= n0);
                assert(uid[i][j] < n0+n);
                picked[uid[i][j]-n0] = true;
            }
        }
        assert(std::accumulate(picked.begin(), picked.end(), 0) == checkN);
    }
    return uid;
}

template<typename T>
void flattenBlock(Size nblock, Size neuronPerBlock, T *pos) {
    Size networkSize = nblock*neuronPerBlock;
    std::vector<T> x;
    std::vector<T> y;
	x.reserve(networkSize);
	y.reserve(networkSize);
    for (PosInt i=0; i<nblock; i++) {
        PosInt offset = i*2*neuronPerBlock;
        for (PosInt j=0; j<neuronPerBlock; j++) {
            x.push_back(pos[offset + j]);
        }
        for (PosInt j=0; j<neuronPerBlock; j++) {
            y.push_back(pos[offset + neuronPerBlock + j]);
        }
    }
    memcpy(pos, &x[0], networkSize*sizeof(T));
    memcpy(pos+networkSize, &y[0], networkSize*sizeof(T));
}

bool fill_fSpikeTrain(std::vector<std::vector<std::vector<Float>>> &fsp, Float sp[], std::vector<std::vector<PosInt>> &fcs, std::vector<std::vector<PosInt>> &vecID, std::vector<Size> nVec, Size nV1);

void fill_fGapTrain(std::vector<std::vector<std::vector<Float>>> &fv, Float sp[], std::vector<std::vector<PosInt>> &gap_fcs, std::vector<std::vector<PosInt>> &gapVecID, std::vector<Size> nGapVec, Size mI);
