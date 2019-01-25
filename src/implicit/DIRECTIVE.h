#ifndef DIRECTIVE_H
#define DIRECTIVE_H

#define HALF_MEMORY_OCCUPANCY 1 // in Mb
#define KERNEL_PERFORMANCE
//#define TEST_WITH_MANUAL_FFINPUT
//#define GPU_ONLY
//#define NAIVE
#define DEBUG
//#define FULL_SPEED
//#define SPIKE_CORRECTION
//#define SPEED_TEST  // not much use, maybe try when there's a lot of spikes per dt
//#define VOLTA

#define timeNow() std::chrono::high_resolution_clock::now()

#endif
