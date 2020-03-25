#include "stats.cuh"
using namespace std;

__global__
void pixelizeOutput(
        Float* __restrict__ fr,
        Float* __restrict__ output,
        PosInt* __restrict__ pid, 
		Size* __restrict__ m, // within one pixel
		Size nPerPixel_I, Size nPerPixel_C, Size nPixel_I, Size nPixel, Size n, bool debug)
{
	PosInt tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (debug && tid == 0) {
        printf("im here\n");
    }
	if (tid < nPixel) {
		Size m_local = m[tid];
		Float value = 0;
        bool fired = false;
		if (m_local > 0) {
			Size nPerPixel = tid < nPixel_I? nPerPixel_I: nPerPixel_C;
			PosInt offset = tid < nPixel_I? 0: (nPixel_I*nPerPixel_I);
			PosInt ICid = tid - (tid >= nPixel_I)*nPixel_I;
            offset += ICid*nPerPixel;

			for (PosInt i=0; i<m_local; i++) {
				PosInt id = pid[offset + i];
                //DEBUG
                if (debug) {
                    assert(id < 32768);
                }
				PosInt sInfo = fr[id];
				if (sInfo > 0) {
					value += ceiling(sInfo);
                    if (debug) {
                        printf("i fired\n");
                    }
                    assert(value > 0);
                    fired = true;
				}
			}
			value /= m_local;
            if (fired) {
                assert(value > 0);
            }
		}
		__syncwarp();
		output[tid] += value;
        if (debug && fired) {
            printf("frame output: %f at half\n", output[tid]);
            assert(output[tid] > 0);
        }
	}
}

// From nChunks of [chunkSize, ngTypeE+ngTypeI, blockSize] -> [ngTypeE+ngTypeI, nV1], where nV1 = nChunk*chunkSize*blockSize
void reshape_chunk_and_write(Float chunk[], ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, bool hWrite)
{
    PosIntL offset = 0;
    size_t outputSize;
    Float *flatten;

	if (hWrite) {
		outputSize = nV1*(nE+nI)*2; // g and h
	} else {
		outputSize = nV1*(nE+nI); // g only
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
    fRawData.write((char*) flatten, outputSize*sizeof(Float));
    delete []flatten;
}

void getLGN_V1_surface(vector<PosInt> &xy, vector<vector<PosInt>> &LGN_V1_ID, PosInt surface_xy[], Size nLGNperV1[], Size max_LGNperV1, Size nLGN)
{
    Size nV1 = LGN_V1_ID.size();
    for (PosInt i=0; i<nV1; i++) {
        nLGNperV1[i] = LGN_V1_ID[i].size();
        assert(nLGNperV1[i] <= max_LGNperV1);
        for (PosInt j=0; j<nLGNperV1[i]; j++) {
            PosInt xid = i*max_LGNperV1 + j;
            surface_xy[xid] = xy[LGN_V1_ID[i][j]]; // x
            PosInt yid = nV1*max_LGNperV1 + xid;
            surface_xy[yid] = xy[nLGN + LGN_V1_ID[i][j]];
        }
    }
}

void fill_fSpikeTrain(std::vector<std::vector<std::vector<Float>>> &fsp, Float sp[], std::vector<std::vector<PosInt>> &fcs, std::vector<std::vector<PosInt>> &vecID, std::vector<Size> nVec, Size nV1) {
    for (PosInt i=0; i<nV1; i++) {
        for (PosInt j=0; j<nVec[i]; j++) {
            fsp[i][j][fcs[i][j]] = sp[vecID[i][j]];
        }
    }
}
