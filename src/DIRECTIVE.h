#ifndef DIRECTIVE_H
#define DIRECTIVE_H

#define MAX_FFINPUT_PER_DT 500 // dt in ms
#define HALF_MEMORY_OCCUPANCY 1 // in Mb
#define KERNEL_PERFORMANCE
#define TEST_WITH_MANUAL_FFINPUT
//#define DEBUG 
//#define GPU_ONLY
#define RECLAIM
#define CPU_ONLY
//#define NAIVE // for naive summation of conductance

#define timeNow() std::chrono::high_resolution_clock::now()

#endif
