#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H
#include <cuda.h>
#include "../DIRECTIVE.h"
#include "../CUDA_MACRO.h"
#include "../types.h"
#include <float.h>

__host__
__device__
__forceinline__
void orthPhiRotate3D(Float theta0, Float phi0, Float eta, Float &theta, Float &phi) {
	// phi (0, pi)
	// theta [-pi,pi]
	// eta [-pi/2, pi/2] the rotating angle, orhogonal to radius*dPhi
	
    if (phi0 < 0.0) {
        phi0 = -phi0;
        theta0 -= copy(M_PI, theta0);
    }
    Float sinPhi0 = sine(phi0);
    if (sinPhi0 == 0) {
        printf("phi0: %f\n", phi0);
        sinPhi0 = phi0;
    }
    // need by the next step is cosine and sine
	phi = arccos(cosine(phi0) * cos(eta));
	theta = theta0 + atan(tangent(eta), sinPhi0); //atan2(y, x)
    assert(!isnan(phi));
    assert(!isnan(theta));
}

__host__
__device__
__forceinline__
void axisRotate3D(Float theta0, Float phi0, Float ceta, Float seta, Float &cost, Float &sint, Float phi, Float &tanPhi) {
	// view the globe from the eye as the origin looking along the z-axis pointing out, with x-y-z axis
	// the stimulus plane will have a vertical x-axis, horizontal y-axis
	// theta0 is the angle formed by the rotation axis's projection on x-y plane and the y-axis
	// phi0 is the angle formed by the rotation axis's and the z-axis
	// eta is the angle rotating along the axis counter-clockwisely view from the eye.
	// ceta = cos(eta)
	// seta = cos(eta)
    Float sinPhi = sine(phi0);
	Float x_rot = sinPhi * sine(theta0);
	Float y_rot = sinPhi * cosine(theta0);
	Float z_rot = cosine(phi0);
    sinPhi = sine(phi);
	Float x = sinPhi * sint;
	Float y = sinPhi * cost;
	Float z = cosine(phi);
	// wiki 3d rotation matrix along an axis reverse the rotation direction (seta -> -seta)
	Float z_prime = (z_rot*x_rot*(1-ceta) + y_rot*seta) * x +
                    (z_rot*y_rot*(1-ceta) - x_rot*seta) * y +
                    (z_rot*z_rot*(1-ceta) +       ceta) * z;

	Float y_prime = (y_rot*x_rot*(1-ceta) - z_rot*seta) * x +
                    (y_rot*y_rot*(1-ceta) +       ceta) * y +
                    (y_rot*z_rot*(1-ceta) + x_rot*seta) * z;

	Float x_prime = (x_rot*x_rot*(1-ceta) +       ceta) * x +
                    (x_rot*y_rot*(1-ceta) + z_rot*seta) * y +
                    (x_rot*z_rot*(1-ceta) - y_rot*seta) * z;
    
    tanPhi = square_root(x_prime*x_prime + y_prime*y_prime)/z_prime;
    Float theta = atan2(x_prime, y_prime);
    cost = cosine(theta);
    sint = sine(theta);
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
