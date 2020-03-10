#include <fstream>
#include <vector>
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
		Size trainDepth, PosInt currentTimeSlot, Size nPerPixel_I, Size nPerPixel_C, Size nPixel_I, Size nPixel
);

void reshape_chunk_and_write(Float chunk[], std::ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1);

void getLGN_V1_surface(std::vector<PosInt> &xy, std::vector<std::vector<PosInt>> &LGN_V1_ID, PosInt* surface_xy, Size* nLGNperV1, Size max_LGNperV1, Size nLGN);

// VisLGN, VisV1 (visual field)  or PhyV1 (physical position) with mixed [C]ontralateral and [I]psilateral
template<typename T>
std::vector<std::vector<PosInt>> getUnderlyingID(T* x, T* y, Int* pick, Size n, Size width, Size height, T x0, T xspan, T y0, T yspan, Size* maxPerPixel) {
	// offset normally is the column's ID that separate left and right
	std::vector<std::vector<PosInt>> uid(height*width, std::vector<PosInt>());
    *maxPerPixel = 1;
    for (PosIntL i=0; i<n; i++) {
        if (pick[i] > 0) {
            PosInt idx = static_cast<PosInt>(((x[i]-x0)/xspan)*width);
            PosInt idy = static_cast<PosInt>(((y[i]-y0)/yspan)*height);
            PosInt id = idx+idy*width;
            uid[id].push_back(i);
            if (uid[id].size() > *maxPerPixel) *maxPerPixel = uid[id].size();
        }
    }
    return uid;
}

template<typename T>
void flattenBlock(Size nblock, Size neuronPerBlock, T *pos) {
    Size networkSize = nblock*neuronPerBlock;
    std::vector<T> x(networkSize);
    std::vector<T> y(networkSize);
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
