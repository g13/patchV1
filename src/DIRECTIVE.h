#ifndef DIRECTIVE_H
#define DIRECTIVE_H
#include "types.h"

//#define DEBUG 
//#define CHECK
//#define SYNC

#ifdef DOUBLE_PRECISION
	#define B_LOG_MAXIMUM 709.7827128933827
    #define roundingb round 
    #define flooringb floor 
    #define ceilingb ceil
    #define modb fmod
	#define absb fabs 
    #define copymsb copysign
	#define powerb pow
	#define square_rootb sqrt
    #define logarithmb log
	#define exponentialb exp 
    #define log_gammab lgamma
    #define minimumb fmin

    #define tangentb tan
    #define sineb sin 
    #define cosineb cos 
	#define atanb atan2
	#define arctanb atan
    #define arccosb acos 
    #define arcsinb asin 
    #define bessel0b cyl_bessel_i0

	#define uniformb curand_uniform_double
    #define normalb curand_normal_double
    #define log_normalb curand_log_normal_double
#else
	#define B_LOG_MAXIMUM 88.72284
    #define roundingb roundf 
    #define flooringb floorf 
    #define ceilingb ceilf
    #define modb fmodf
	#define absb fabsf 
    #define copymsb copysignf
	#define powerb powf
	#define square_rootb sqrtf
    #define logarithmb logf
	#define exponentialb expf
    #define log_gammab lgammaf
    #define minimumb fminf

    #define tangentb tanf
    #define sineb sinf 
    #define cosineb cosf 
	#define atanb atan2f
	#define arctanb atanf
    #define arccosb acosf 
    #define arcsinb asinf 
    #define bessel0b cyl_bessel_i0f

	#define uniformb curand_uniform
    #define normalb curand_normal
    #define log_normalb curand_log_normal
#endif

#ifdef SINGLE_PRECISION
	#define LOG_MAXIMUM 88.72284
    #define rounding roundf
    #define flooring floorf
    #define ceiling ceilf
    #define mod fmodf
	#define abs fabsf 
    #define copyms copysignf
	#define power powf
	#define square_root sqrtf
    #define logarithm logf
	#define exponential expf
    #define log_gamma lgammaf
    #define minimum fminf

    #define tangent tanf
    #define sine sinf 
    #define cosine cosf 
	#define atan atan2f
	#define arctan atanf
    #define arccos acosf 
    #define arcsin asinf 
    #define bessel0 cyl_bessel_i0f

	#define uniform curand_uniform
    #define normal curand_normal
    #define log_normal curand_log_normal
#else
	#define LOG_MAXIMUM 709.7827128933827
    #define rounding round 
    #define flooring floor
    #define ceiling ceil
    #define mod fmod
	#define abs fabs 
    #define copyms copysign
	#define power pow
	#define square_root sqrt
    #define logarithm log
	#define exponential exp 
    #define log_gamma lgamma
    #define minimum fmin

    #define tangent tan
    #define sine sin 
    #define cosine cos 
	#define atan atan2
	#define arctan atan
    #define arccos acos 
    #define arcsin asin 
    #define bessel0 cyl_bessel_i0

	#define uniform curand_uniform_double
    #define normal curand_normal_double
    #define log_normal curand_log_normal_double
#endif

#endif
