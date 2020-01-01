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
	#define abs fabsf 
    #define copy copysignf
	#define power powf
	#define square_root sqrtf
    #define logrithm logf
	#define exponential expf
    #define log_gamma lgammaf

    #define tangent tanf
    #define sine sinf 
    #define cosine cosf 
	#define atan atan2f
    #define arccos acosf 

	#define uniform curand_uniform
    #define normal curand_normal
    #define log_normal curand_log_normal
#else
	#define abs fabs 
    #define copy copysign
	#define power pow
	#define square_root sqrt
    #define logrithm log
	#define exponential exp 
    #define log_gamma lgamma

    #define tangent tan
    #define sine sin 
    #define cosine cos 
	#define atan atan2
    #define arccos acos 

	#define uniform curand_uniform_double
    #define normal curand_normal_double
    #define log_normal curand_log_normal_double
#endif

#endif
