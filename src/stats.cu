#include "stats.cuh"
using namespace std;

__global__
void pixelize(
        Float* __restrict__ array,
        double* __restrict__ x,
        double* __restrict__ y,
        Float* __restrict__ frame,
        Size width, Size height) 
{
}

// From nChunks of [chunkSize, ngTypeE+ngTypeI, blockSize] -> [ngTypeE+ngTypeI, nV1], where nV1 = nChunk*chunkSize*blockSize
void reshape_chunk_and_write(Float chunk[], ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1) {
    PosIntL offset = 0;
    size_t gSize = nV1*(nE+nI);
    Float *flatten = new Float[gSize];
    Size chunkSize = maxChunkSize;
    for (PosInt i=0; i<nChunk; i++) {
        PosIntL offset_f;
        if (i > iSizeSplit - 1) {
            chunkSize = remainChunkSize;
            offset_f = iSizeSplit*maxChunkSize + (i-iSizeSplit)*chunkSize;
        } else {
            offset_f = i*maxChunkSize;
        }
        for (PosInt j=0; j<nE; j++) {
            for (PosInt k=0; k<chunkSize*blockSize; k++) {
                flatten[j*nV1 + f_offset + k] = chunk[offset];
                offset++;
            }
        }
        for (PosInt j=0; j<nI; j++) {
            for (PosInt k=0; k<chunkSize*blockSize; k++) {
                flatten[(nE+j)*nV1 + f_offset + k] = chunk[offset];
                offset++;
            }
        }
    }
    assert(offset == (iSizeSplit*maxChunkSize + (nChunk - iSizeSplit)*remainChunkSize)*blockSize*(nE+nI));
    fRawData.write((char*) flatten, gSize*sizeof(Float));
    delete []flatten;
}

void getLGN_V1_surface(vector<PosInt> &xy, vector<vector<PosInt>> &LGN_V1_ID, PosInt* surface_xy, Size* nLGNperV1, Size max_LGNperV1, Size nLGN) {
    Size nV1 = LGN_V1_ID.size();
    for (PosInt i=0; i<nV1; i++) {
        nLGNperV1[i] = LGN_V1_ID[i].size();
        for (PosInt j=0; j<nLGNperV1[i]; j++) {
            PosInt xid = i*max_LGNperV1 + j;
            surface_xy[xid] = xy[LGN_V1_ID[i][j]]; // x
            PosInt yid = nV1*max_LGNperV1 + xid;
            surface_xy[yid] = xy[nLGN + LGN_V1_ID[i][j]];
        }
    }
}
