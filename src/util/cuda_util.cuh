#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H
#include <cuda.h>
#include "../DIRECTIVE.h"
#include "../CUDA_MACRO.h"
#include "../types.h"

__host__
__device__
inline void axisRotate3D(Float theta0, Float, phi0, Float ceta, Float seta, Float &theta, Float &phi) {
	// view the globe from the eye as the origin looking along the z-axis pointing out, with x-y-z axis
	// the stimulus plane will have a vertical x-axis, horizontal y-axis
	// theta0 is the angle formed by the rotation axis's projection on x-y plane and the y-axis
	// phi0 is the angle formed by the rotation axis's and the z-axis
	// eta is the angle rotating along the axis counter-clockwisely view from the eye.
	// ceta = cos(eta)
	// seta = cos(eta)
	Float x_rot = sine(phi0) * sine(theta0);
	Float y_rot = sine(phi0) * cosine(theta0);
	Float z_rot = cosine(phi0);
	Float x = sine(phi) * sine(theta);
	Float y = sine(phi) * cosine(theta);
	Float z = cosine(phi);
	// wiki 3d rotation matrix along an axis
	z = (z_rot*x_rot*(1-ceta) - y_rot*seta) * x + (z_rot*y_rot*(1-ceta) + x_rot*seta) * y + (ceta + z_rot*z_rot*(1-ceta)) * z;
	x = (x_rot*x_rot*(1-ceta) + ceta) * x + (x_rot*y_rot*(1-ceta) - z_rot*seta) * y + (x_rot*z_rot*(1-ceta) + y_rot*seta) * z;

	phi = arccos(phi);
	theta = arcsin(x/sine(phi));
}

__host__
__device__
inline void orthPhiRotate3D(Float theta0, Float, phi0, Float eta, Float &theta, Float &phi) {
	// phi (0, pi)
	// theta [-pi,pi]
	// eta [-pi/2, pi/2] the rotating angle, orhogonal to radius*dPhi
	//
	//cosine(phi) = cosine(phi0) * cosine(eta);
	//sine(dtheta) = sine(eta)/sine(phi);
	
	phi = arccos(cosine(phi0) * cosine(eta));
	theta = theta0 + arcsin(sine(eta) / sine(phi));
}


// 1D
template <typename T>
__global__ void init(T *array, T value, PosInt nData) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nData) {
        array[id] = value;
    }
}

__device__ void warp0_min(Float array[], PosInt id[]);

__device__ void warps_min(Float array[], Float data, PosInt id[]);

__device__ void find_min(Float array[], Float data, PosInt id[]);

// only reduce is extensively tested: cuda_full_min.cu

template <typename T>
__device__ void warps_reduce(T array[], T data) {
    PosInt tid = blockDim.x*threadIdx.y + threadIdx.x;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
        //if (tid % warpSize == 0) {
        //    printf("#%i at %i: %f\n", tid, offset, data);
        //}
    }
    __syncthreads();
    if (tid % warpSize == 0) {
        array[tid/warpSize] = data;
    }
}

template <typename T>
__device__ void warp0_reduce(T array[]) {
    PosInt tid = blockDim.x*threadIdx.y + threadIdx.x;
    T data = array[tid];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
        //if (tid == 0) {
        //    printf("##%i at %i: %f\n", tid, offset, data);
        //}
    }
    if (tid == 0) {
        array[0] = data;
    }
}

template <typename T>
__device__ void block_reduce(T array[], T data) {
	warps_reduce<T>(array, data);
    __syncthreads();
    //if (blockDim.x*threadIdx.y + threadIdx.x < nWarp) {
    if (blockDim.x*threadIdx.y + threadIdx.x < warpSize) {
        warp0_reduce<T>(array);
    }
    __syncthreads();
}

#endif
