#include "stats.cuh"
using namespace std;

__global__
void pixelizeOutput(
        Float* __restrict__ fr,
        Float* __restrict__ output,
        PosInt* __restrict__ pid, Size* __restrict__ m, // within one pixel
		Size nPerPixel_I, Size nPerPixel_C, Size nPixel_I, Size nPixel, Size n, Float odt)
{
	PosInt tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < nPixel) {
		Size m_local = m[tid];
		Float value = 0;
		if (m_local > 0) {
			Size nPerPixel = tid < nPixel_I? nPerPixel_I: nPerPixel_C;
			PosInt offset = tid < nPixel_I? 0: (nPixel_I*nPerPixel_I);
			PosInt ICid = tid - (tid >= nPixel_I)*nPixel_I;
            offset += ICid*nPerPixel;

			for (PosInt i=0; i<m_local; i++) {
				PosInt id = pid[offset + i];
				Float sInfo = fr[id];
				if (sInfo >= 0) {
					value += flooring(sInfo);
                    #ifdef DEBUG
                        if (id == 0) {
                            printf("i fired, sInfo = %f\n", sInfo);
                        }
                    #endif
				}
			}
			value /= m_local*odt;
		}
		__syncwarp();
		output[tid] += value;
	}
}

// From nChunks of [chunkSize, ngTypeE+ngTypeI, blockSize] -> [ngTypeE+ngTypeI, nV1], where nV1 = nChunk*chunkSize*blockSize
void reshape_chunk_and_write(Float chunk[], ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, Size nGap, bool hWrite)
{
    PosIntL offset = 0;
    size_t outputSize;
    Float *flatten;

	if (hWrite) {
		outputSize = nV1*(nE+nI)*2 + nGap; // g and h
	} else {
		outputSize = nV1*(nE+nI) + nGap; // g only
	}
	flatten = new Float[outputSize];
    Size chunkSize = maxChunkSize;
    for (PosInt i=0; i<nChunk; i++) {
        PosIntL offset_f; // flattened neuron id offset before current chunk
        if (i >= iSizeSplit) {
            chunkSize = remainChunkSize;
            offset_f = (iSizeSplit*maxChunkSize + (i-iSizeSplit)*chunkSize)*blockSize;
        } else {
            offset_f = i*maxChunkSize*blockSize;
        }
        for (PosInt j=0; j<nE; j++) {
            for (PosInt k=0; k<chunkSize*blockSize; k++) {
                flatten[j*nV1 + offset_f + k] = chunk[offset];
                offset++;
            }
        }
        for (PosInt j=0; j<nI; j++) {
            for (PosInt k=0; k<chunkSize*blockSize; k++) {
                flatten[(nE+j)*nV1 + offset_f + k] = chunk[offset];
                offset++;
            }
        }
		if (hWrite) {
        	for (PosInt j=0; j<nE; j++) {
        	    for (PosInt k=0; k<chunkSize*blockSize; k++) {
        	        flatten[(nE+nI+j)*nV1 + offset_f + k] = chunk[offset];
        	        offset++;
        	    }
        	}
        	for (PosInt j=0; j<nI; j++) {
        	    for (PosInt k=0; k<chunkSize*blockSize; k++) {
        	        flatten[(2*nE+nI+j)*nV1 + offset_f + k] = chunk[offset];
        	        offset++;
        	    }
        	}
		} else {
			offset += chunkSize*blockSize*(nE+nI);
		}
    }
    assert(offset == nV1*(nE+nI)*2);

	Float* gap = flatten + nV1*(nE+nI)*2;
    for (PosInt i=0; i<nGap; i++) {
    	gap[i] = chunk[offset+i];
	}
    fRawData.write((char*) flatten, outputSize*sizeof(Float));
    delete []flatten;
}

void getLGN_V1_surface(vector<int> &xy, vector<vector<PosInt>> &LGN_V1_ID, int surface_xy[], Size nLGNperV1[], Size max_LGNperV1, Size nLGN)
{
    Size nV1 = LGN_V1_ID.size();
    for (PosInt i=0; i<nV1; i++) {
        nLGNperV1[i] = LGN_V1_ID[i].size();
        assert(nLGNperV1[i] <= max_LGNperV1);
        for (PosInt j=0; j<nLGNperV1[i]; j++) {
            PosInt xid = i*max_LGNperV1 + j;
            surface_xy[xid] = xy[LGN_V1_ID[i][j]]; // x
            PosInt yid = nV1*max_LGNperV1 + xid;
            surface_xy[yid] = xy[nLGN + LGN_V1_ID[i][j]]; // y
        }
    }
}

bool fill_fSpikeTrain(std::vector<std::vector<std::vector<Float>>> &fsp, Float sp[], std::vector<std::vector<PosInt>> &fcs, std::vector<std::vector<PosInt>> &vecID, std::vector<Size> nVec, Size nV1) {
    bool outsideSpiked = false;
    for (PosInt i=0; i<nV1; i++) {
        for (PosInt j=0; j<nVec[i]; j++) {
            Float sInfo = sp[vecID[i][j]];
            if (sInfo > 0 && !outsideSpiked) {
                outsideSpiked = true;
            }
            fsp[i][j][fcs[i][j]] = sInfo;
        }
    }
    return outsideSpiked;
}

void fill_fGapTrain(std::vector<std::vector<std::vector<Float>>> &fv, Float sp[], std::vector<std::vector<PosInt>> &gap_fcs, std::vector<std::vector<PosInt>> &gapVecID, std::vector<Size> nGapVec, Size mI) {
    for (PosInt i=0; i<mI; i++) {
        for (PosInt j=0; j<nGapVec[i]; j++) {
            fv[i][j][gap_fcs[i][j]] = sp[gapVecID[i][j]];
        }
    }
}
