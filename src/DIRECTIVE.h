#include "types.h"
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
#include "types.h"
#define RECLAIM
#define CPU_ONLY
#define FULL_SPEED
//#define NAIVE // for naive summation of conductance

#define timeNow() std::chrono::high_resolution_clock::now()

#ifdef DOUBLE_PRECISION
	#define absb fabs 
    #define copymsb copysign
	#define powerb pow
	#define square_rootb sqrt
    #define logrithmb log
	#define exponentialb exp 
    #define log_gammab lgamma

    #define tangentb tan
    #define sineb sin 
    #define cosineb cos 
	#define atanb atan2
	#define arctanb atan
    #define arccosb acos 
    #define arcsinb asin 

	#define uniformb curand_uniform_double
    #define normalb curand_normal_double
    #define log_normalb curand_log_normal_double
#else
	#define absb fabsf 
    #define copymsb copysignf
	#define powerb powf
	#define square_rootb sqrtf
    #define logrithmb logf
	#define exponentialb expf
    #define log_gammab lgammaf

    #define tangentb tanf
    #define sineb sinf 
    #define cosineb cosf 
	#define atanb atan2f
	#define arctanb atanf
    #define arccosb acosf 
    #define arcsinb asinf 

	#define uniformb curand_uniform
    #define normalb curand_normal
    #define log_normalb curand_log_normal
#endif

#ifdef SINGLE_PRECISION
	#define abs fabsf 
    #define copyms copysignf
	#define power powf
	#define square_root sqrtf
    #define logrithm logf
	#define exponential expf
    #define log_gamma lgammaf

    #define tangent tanf
    #define sine sinf 
    #define cosine cosf 
	#define atan atan2f
	#define arctan atanf
    #define arccos acosf 
    #define arcsin asinf 

	#define uniform curand_uniform
    #define normal curand_normal
    #define log_normal curand_log_normal
#else
	#define abs fabs 
    #define copyms copysign
	#define power pow
	#define square_root sqrt
    #define logrithm log
	#define exponential exp 
    #define log_gamma lgamma

    #define tangent tan
    #define sine sin 
    #define cosine cos 
	#define atan atan2
	#define arctan atan
    #define arccos acos 
    #define arcsin asin 

	#define uniform curand_uniform_double
    #define normal curand_normal_double
    #define log_normal curand_log_normal_double
#endif

#endif
