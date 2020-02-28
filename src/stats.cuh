#include <fstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "types.h"
#include "MACRO.h"

void reshape_chunk_and_write(Float chunk[], std::ofstream &fRawData, Size maxChunkSize, Size remainChunkSize, Size nChunk, Size nE, Size nI, Size nV1);

void getLGN_V1_surface(std::vector<PosInt> &xy, std::vector<std::vector<PosInt>> &LGN_V1_ID, PosInt* surface_xy, Size* nLGNperV1, Size max_LGNperV1, Size nLGN);
