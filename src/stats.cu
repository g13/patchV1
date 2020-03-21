#include "stats.cuh"
using namespace std;

__global__
void pixelizeOutput(
        Float* __restrict__ fr,
        Float* __restrict__ output,
        PosInt* __restrict__ pid, 
		Size* __restrict__ m, // within one pixel
		Size trainDepth, PosInt currentTimeSlot, Size nPerPixel_I, Size nPerPixel_C, Size nPixel_I, Size nPixel, Size n)
{
	PosInt tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < nPixel) {
		Size m_local = m[tid];
		Float value = 0;
		if (m_local > 0) {
			Size nPerPixel = tid < nPixel_I? nPerPixel_I: nPerPixel_C;
			PosInt offset = tid < nPixel_I? 0: (nPixel_I*nPerPixel_I);
			PosInt ICid = tid - (tid >= nPixel_I)*nPixel_I;
			for (PosInt i=0; i<m_local; i++) {
				PosInt id = pid[offset + ICid*nPerPixel + i];
                if (id >= n) {
                    printf("offset:%u + ICid:%u*nPerPixel:%u + %u\n", offset, ICid, nPerPixel, i);
                    assert(id < n);
                }
                if (trainDepth*id + currentTimeSlot >= n*trainDepth) {
                    printf("trainDepth: %u, currentTimeSlot:%u, id:%u < n:%u\n", trainDepth, currentTimeSlot, id, n);
                    assert(trainDepth*id + currentTimeSlot < n*trainDepth);
                }
				PosInt sInfo = fr[trainDepth*id + currentTimeSlot];
				if (sInfo > 0) {
					value += ceiling(sInfo);
				}
			}
			value = value/m_local;
		}
		__syncwarp();
		output[tid] += value;
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
		outputSize = nV1*(nE+nI);
	}
	flatten = new Float[outputSize];
    Size chunkSize = maxChunkSize;
    for (PosInt i=0; i<nChunk; i++) {
        PosIntL offset_f;
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
