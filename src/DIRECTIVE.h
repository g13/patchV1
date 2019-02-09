#ifndef DIRECTIVE_H
#define DIRECTIVE_H

#define MAX_FFINPUT_PER_DT 100 // dt in ms
#define HALF_MEMORY_OCCUPANCY 100 // in Mb
#define KERNEL_PERFORMANCE
#define TEST_CONVERGENCE_NO_ROUNDING_ERR
//#define SKIP_IO
//#define TEST_WITH_MANUAL_FFINPUT
//#define DEBUG 
//#define GPU_ONLY
#define RECLAIM
#define CPU_ONLY
#define FULL_SPEED
//#define NAIVE // for naive summation of conductance

#define timeNow() std::chrono::high_resolution_clock::now()

#endif
