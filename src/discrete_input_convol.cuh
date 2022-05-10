//#include <cufft.h>
#ifndef DISCRETE_INPUT_CONVOL_CUH
#define DISCRETE_INPUT_CONVOL_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <tuple>
#include "DIRECTIVE.h"
#include "LGN_props.cuh"
#include "types.h"
#include "CONST.h"
#include "condShape.h"
#include "preprocess/RFtype.h"
#include "util/cuda_util.cuh"

// block_reduce works fine when block is not fully occupied
// assuming viewing distance is a single unit length

/*
__global__ 
void testTexture(Float L, Float M, Float S);
*/
    

template<typename T>
__global__
void cudaMemsetNonzero(
        T* array,
        Size n,
        T value) 
{
    Size id =  blockDim.x * blockDim.y * (gridDim.x*blockIdx.y + blockIdx.x) + blockDim.x*threadIdx.y + threadIdx.x;
	/*
    if (id == 0) {
        printf("array initialized to %f\n", value);
    }*/
    if (id < n) {
        array[id] = value;
    }
}

__device__
__forceinline__
Float get_spike(Size &nsp, Float &leftTimeRate, Float &lastNegLogRand, Float dt, Float rate, curandStateMRG32k3a *state);

//template<int ntimes> extern

__global__
void virtual_LGN_convol(
        Float* __restrict__ lum,
        Float* __restrict__ contrast,
		cudaTextureObject_t* __restrict__ linearFrame,
        Float* __restrict__ current_convol,
		float* __restrict__ parvo_center,
		float* __restrict__ magno_center,
		InputType_t* __restrict__ LGN_type,
		Int inputType, Size nParvo_L, Size nMagno_L, Size nParvo_R, Size nLGN, PosInt prev, PosInt next, Float rt, bool saveOutputB4V1
);

__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ current_convol,
        Float convolRatio,
        Float* __restrict__ LGN_fr,
        Float* __restrict__ LGN_sInfo,
        int* __restrict__ sx,
        int* __restrict__ sy,
        Float* __restrict__ leftTimeRate,
        Float* __restrict__ lastNegLogRand,
		curandStateMRG32k3a* __restrict__ state,
		InputType_t* __restrict__ LGN_type,
		Float* __restrict__ switch_value,
        InputActivation typeStatus,
        Float* __restrict__ lVar,
		cudaSurfaceObject_t LGNspikeSurface,
        Float frRatio, int varSlot, LearnVarShapeFF_E_pre lE, LearnVarShapeFF_I_pre lI, Size nFF, Float dt, int learning, bool learnData_FF, bool LGN_switch, bool getLGN_sp, bool virtual_LGN, int switchNow
);

__global__
void store_PM(// weights and max convolution
        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component &spatial,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        float* __restrict__ center,
        Float* __restrict__ max_convol,
		Size nBefore, Size nAfter, Size nL, Size nLGN,
		Float L_x0,
		Float L_y0,
		Float R_x0,
		Float R_y0,
		Float normViewDistance,
        Float nsig, // span of spatialRF sample in units of std
        int PM,
        bool uniform_retina,
        bool virtual_LGN
);

__global__
void parvo_maxConvol(
        Spatial_component &spatial,
        Float* __restrict__ TW_storage,
        Float* __restrict__ covariant,
        Float* __restrict__ max_convol,
        Size nSample1D, Size nParvo_L, Size nMagno_L, Size nLGN, SmallSize nKernelSample, Float kernelSampleDt, Float nsig
);

__global__ 
void LGN_convol_parvo(
        Float* __restrict__ luminance,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ current_convol,
        Float* __restrict__ contrast,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
		cudaTextureObject_t L_retinaInput,
		cudaTextureObject_t M_retinaInput,
		cudaTextureObject_t S_retinaInput,
		Size nParvo_L, Size nMagno_L, Size nLGN,
		Float normViewDistance,
        PosInt currentFrame,
        Size maxFrame,
		Size ntPerFrame,
        PosInt iFramePhase,
        Float Itau,
        Size iKernelSampleT0,
        Size kernelSampleInterval,
        Size nKernelSample,
        Float dt,
        Size denorm,
        bool saveOutputB4V1
);

__global__ 
void LGN_convol_magno(
        Float* __restrict__ luminance,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ current_convol,
        Float* __restrict__ contrast,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
		cudaTextureObject_t L_retinaInput,
		cudaTextureObject_t M_retinaInput,
		cudaTextureObject_t S_retinaInput,
		Size nParvo_L, Size nMagno_L, Size nParvo_R,
		Float normViewDistance,
        PosInt currentFrame,
        Size maxFrame,
		Size ntPerFrame,
        PosInt iFramePhase,
        Float Itau,
        Size iKernelSampleT0,
        Size kernelSampleInterval,
        Size nKernelSample,
        Float dt,
        Size denorm,
        bool saveOutputB4V1
);

__host__
__device__
__forceinline__
void orthPhiRotate3D_arc(Float theta0, Float phi0, Float eta, Float &theta, Float &phi) {
	// phi (0, pi) typically small
	// theta [-pi,pi]
	// eta [-pi/2, pi/2] the rotating angle, orhogonal to radius*dphi, typically small
	
    // need by the next step is cos and sin
	Float sin_phi0 = sine(phi0);
	Float tan_eta = tangent(eta);
    Float r = square_root(tan_eta*tan_eta + sin_phi0*sin_phi0);
    phi = arcsin(r);
    if (phi0 < 0.0) phi = -phi;
    Float dtheta = atan(tan_eta, sin_phi0); //atan2(y, x)
	theta = theta0 + dtheta;
    assert(!isnan(phi));
    assert(!isnan(theta));
}


// rotations are free of arc-trignometric function to avoid precision issues around the origin.
__host__
__device__
__forceinline__
void orthPhiRotate3D(Float theta, Float phi, Float eta, Float &cost, Float &sint, Float &cosPhi, Float &sinPhi) {
	// phi (0, pi)
	// theta [-pi,pi]
	// eta [-pi/2, pi/2] the rotating angle, orhogonal to radius*dPhi

    Float cost1, sint1;
    Float tan_eta = tangentb(eta);
    Float sin_phi = sineb(phi);
    Float r = square_rootb(tan_eta*tan_eta + sin_phi*sin_phi);

    //cosPhi = static_cast<Float>(cosineb(eta) * cosineb(phi));
    cosPhi = cosine(eta) * cosine(phi);
    assert(abs(cosPhi) <= 1);
    //sinPhi = static_cast<Float>(r);
    sinPhi = r;
    assert(abs(sinPhi) <= 1);
    //printf("r = %1.15e, sinPhi = %1.15e\n", sinPhi, square_rootb(1-cosPhi_d*cosPhi_d));
    //assert(false);

    cost1 = sin_phi/r;
    sint1 = tan_eta/r;
    assert(abs(cost1) <= 1);
    assert(abs(sint1) <= 1);
    if (abs(sint1) > 1.0) {
        sint1 = copyms(square_root(1-cost1*cost1), sint1);
    }
    if (abs(cost1) > 1.0) {
        cost1 = copyms(square_root(1-sint1*sint1), cost1);
    }
	/* DEBUG
    if (isnan(cost1) || isnan(sint1) || abs(cost1)>1.0 || abs(sint1) > 1.0) {
        printf("cost1 = %f, sint1 = %f\n", cost1, sint1);
        printf("sin_phi(%f) = %f\n", phi, sin_phi);
        printf("tan_eta(%f) = %f\n", eta, tan_eta);
        printf("cost1^2 + sint1^2 == %e\n", cost1*cost1 + sint1*sint1);
        assert(abs(cost1)<=1.0);
        assert(abs(sint1)<=1.0);
        assert(!isnan(cost1));
        assert(!isnan(sint1));
    }*/

    Float cost0 = cosine(theta);
    Float sint0 = sine(theta);
	/* DEBUG
    if (isnan(cost0) || isnan(sint0) || abs(cost0)>1.0 || abs(sint0) > 1.0) {
        printf("cos(%f) = %f, sin(%f) = %f\n", theta, cost0, theta, sint0);
        assert(abs(cost0)<=1.0);
        assert(abs(sint0)<=1.0);
        assert(!isnan(cost0));
        assert(!isnan(sint0));
    }*/
    cost = cost0*cost1 - sint0*sint1;
    sint = sint0*cost1 + cost0*sint1;
    if (abs(sint) > 1.0) {
        assert(abs(cost) < 1.0);
        sint = copyms(square_root(1-cost*cost), sint);
    }
    if (abs(cost) > 1.0) {
        assert(abs(sint) < 1.0);
        cost = copyms(square_root(1-sint*sint), cost);
    }
	/* DEBUG
    if (isnan(cost) || isnan(sint) || abs(cost)>1.0 || abs(sint) > 1.0) {
        printf("cost = %f, sint = %f\n", cost, sint);
        printf("cost0 = %f, sint0 = %f\n", cost0, sint0);
        printf("cost1 = %f, sint1 = %f\n", cost1, sint1);
        assert(abs(cost)<=1.0);
        assert(abs(sint)<=1.0);
        assert(!isnan(cost));
        assert(!isnan(sint));
    }*/
}

__host__
__device__
__forceinline__
void axisRotate3D(Float theta0, Float phi0, Float ceta, Float seta, Float &cost, Float &sint, Float cosPhi, Float sinPhi, Float &tanPhi) {
	// view the globe from the eye as the origin looking along the z-axis pointing out, with x-y-z axis
	// the stimulus plane will have a vertical x-axis, horizontal y-axis
	// theta0 is the angle formed by the rotation axis's projection on x-y plane and the y-axis
	// phi0 is the angle formed by the rotation axis's and the z-axis
	// eta is the angle rotating along the axis counter-clockwisely view from the eye.
	// ceta = cos(eta)
	// seta = cos(eta)

    Float sinPhi0 = sine(phi0);
	Float x_rot = sinPhi0 * sine(theta0);
	Float y_rot = sinPhi0 * cosine(theta0);
	Float z_rot = cosineb(phi0);

	Float x = sinPhi * sint;
	Float y = sinPhi * cost;
	Float z = cosPhi;
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
    Float theta = atan(x_prime, y_prime);
    if (isnan(theta)) {
        printf("tan(%f/%f), theta = %f\n", x_prime, y_prime, theta);
        assert(!isnan(theta));
    }
    cost = cosine(theta);
    sint = sine(theta);
}

__host__
__device__
__forceinline__
void retina_to_plane(Float cosp, Float sinp, Float tanEcc, float &x, float &y, const Float normViewDistance, Float LR_x0, Float LR_y0) {
    Float r = tanEcc*normViewDistance;
    x = LR_x0 + static_cast<float>(r*cosp);
    y = LR_y0 + static_cast<float>(r*sinp);
}
#endif
