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

#define SINGLE_PRECISION
#ifdef SINGLE_PRECISION
	#define square_root sqrtf
	#define atan atan2f
	#define uniform curand_uniform
	#define expp expf
	#define power powf
	#define abs fabsf 
    #define copy copysignf
#else
	#define square_root sqrt
	#define atan atan2
	#define uniform curand_uniform_double
	#define expp exp 
	#define power pow
	#define abs fabs 
    #define copy copysign
#endif

#endif
