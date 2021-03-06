#ifndef DIRECTIVE_H
#define DIRECTIVE_H

#define MAX_FFINPUT_PER_DT 100// dt in ms
#define MAX_SPIKE_PER_DT 100// dt in ms
#define HALF_MEMORY_OCCUPANCY 1 // in Mb
#define KERNEL_PERFORMANCE
#define TEST_CONVERGENCE_NO_ROUNDING_ERR
#define SCHEME 0
#define SKIP_IO
//#define TEST_WITH_MANUAL_FFINPUT
//#define GPU_ONLY
//#define NAIVE
//#define DEBUG
//#define FULL_SPEED
//#define SPIKE_CORRECTION
//#define SPEED_TEST  // not much use, maybe try when there's a lot of spikes per dt
//#define VOLTA
#define SINGLE_PRECISION
#ifdef SINGLE_PRECISION
	typedef float _float;
#else
	typedef double _float;
#endif

#define timeNow() std::chrono::high_resolution_clock::now()

#endif
