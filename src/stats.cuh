#include <fstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "types.h"
#include "MACRO.h"

void reshape_chunk_and_write(Float chunk[], std::ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1);

void getLGN_V1_surface(std::vector<PosInt> &xy, std::vector<std::vector<PosInt>> &LGN_V1_ID, PosInt* surface_xy, Size* nLGNperV1, Size max_LGNperV1, Size nLGN);

template<typename T>
std::vector<std::vector<PosInt>> getUnderlyingID(T *x, T *y, bool pick[] Size n, Size width, Size height,  T x0, T xspan, T y0, T yspan, Size &maxPerPixel) {
    vector<vector<PosInt>> uid(height*width, vector<PosInt>());
    Size maxPerPixel = 1;
    for (PosIntL i=0; i<n; i++) {
        if (pick[i]) {
            PosInt idx = static_cast<PosInt>(((x[i]-x0)/xspan)*width);
            PosInt idy = static_cast<PosInt>(((y[i]-y0)/yspan)*height);
            PosInt id = idx+idy*width;
            uid[id].push_back(i);
            if (uid[id].size() > maxPerPixel) maxPerPixel = uid[id].size();
        }
    }
    return uid;
}
