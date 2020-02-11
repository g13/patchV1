//#include <cufft.h>
#ifndef DISCRETE_INPUT_CONVOL_CUH
#define DISCRETE_INPUT_CONVOL_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <device_launch_parameters.h>
#include <tuple>
#include "DIRECTIVE.h"
#include "LGN_props.cuh"
#include "types.h"
#include "util/cuda_util.cuh"

// block_reduce works fine when block is not fully occupied
// assuming viewing distance is a single unit length

__global__ 
void testTexture(Float L, Float M, Float S);
    
__global__
void cudaMemsetNonzero(Float* array, Size n, Float value);

__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ current_convol,
        Float* __restrict__ LGN_fr
);

__global__
void store(// weights and max convolution
        Float* __restrict__ max_convol,

        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component &spatial,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
		Size nLGN_L,
		Float L_x0,
		Float L_y0,
		Float R_x0,
		Float R_y0,
		Float normViewDistance,
        Float nsig // span of spatialRF sample in units of std
);

__global__ 
void LGN_convol_c1s(
        Float* __restrict__ luminance,
        Float* __restrict__ SW_storage,
        float* __restrict__ SC_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ current_convol,
        Float* __restrict__ contrast,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
		Size nLGN_L,
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
        Size denorm
);

__host__
__device__
__forceinline__
void orthPhiRotate3D_arc(Float theta0, Float phi0, Float eta, Float &theta, Float &phi) {
	// phi (0, pi)
	// theta [-pi,pi]
	// eta [-pi/2, pi/2] the rotating angle, orhogonal to radius*dPhi
	
    if (phi0 < 0.0) {
        phi0 = -phi0;
        theta0 -= copyms(M_PI, theta0);
    }
    // need by the next step is cos and sin
	phi =  static_cast<Float>(acos( cos(static_cast<double>(phi0))*cos(static_cast<double>(eta)) ));
	theta = theta0 + static_cast<Float>(atan( tan(static_cast<double>(eta)), sin(static_cast<double>(phi0)) )); //atan2(y, x)
    assert(!isnan(phi));
    assert(!isnan(theta));
}

// rotations are free of arc-trignometric function to avoid precision issues around the origin.
__host__
__device__
__forceinline__
void orthPhiRotate3D(Float theta0, Float phi0, Float eta, Float &cost, Float &sint, Float &cosPhi, Float &sinPhi) {
	// phi (0, pi)
	// theta [-pi,pi]
	// eta [-pi/2, pi/2] the rotating angle, orhogonal to radius*dPhi
	
    // need by the next step is cos and sin
    //cosPhi = cos(phi0) * cos(eta);
    double dcos = cos(static_cast<double>(eta));
	double cosPhi_d = cos(static_cast<double>(phi0)) * dcos;
    double sinPhi_d = sqrt(1.0 - cosPhi_d*cosPhi_d);
	/* DEBUG
    if (abs(sinPhi_d) == 0 || abs(dcos) == 0) {
        printf("cosPhi = %e, sinPhi = %e, cos0(%f) = %e, dcos(%f) = %e\n", cosPhi_d, sinPhi_d, phi0, cos(static_cast<double>(phi0)), eta, cos(static_cast<double>(eta)));
        assert(abs(dcos) > 0);
        assert(abs(sinPhi_d) > 0);
    }*/

	cosPhi = static_cast<Float>(cosPhi_d);
    sinPhi = static_cast<Float>(sinPhi_d);

    Float cost1, sint1;
    cost1 = sine(phi0) * dcos/sinPhi;
    sint1 = sine(eta)/sinPhi;
    if (abs(sint1) > 1.0) {
        assert(abs(cost1) < 1.0);
        sint1 = copyms(square_root(1-cost1*cost1), sint1);
    }
    if (abs(cost1) > 1.0) {
        assert(abs(sint1) < 1.0);
        cost1 = copyms(square_root(1-sint1*sint1), cost1);
    }
	/* DEBUG
    if (isnan(cost1) || isnan(sint1) || abs(cost1)>1.0 || abs(sint1) > 1.0) {
        printf("sinPhi = %f\n", sinPhi);
        printf("cost1 = %f, sint1 = %f\n", cost1, sint1);
        printf("sin0(%f) = %f\n", phi0, sine(phi0));
        printf("dsin(%f) = %f\n", eta, sine(eta));
        printf("dcos(%f) = %f\n", eta, dcos);
        assert(abs(cost1)<=1.0);
        assert(abs(sint1)<=1.0);
        assert(!isnan(cost1));
        assert(!isnan(sint1));
    }*/

    Float cost0 = cosine(theta0);
    Float sint0 = sine(theta0);
	/* DEBUG
    if (isnan(cost0) || isnan(sint0) || abs(cost0)>1.0 || abs(sint0) > 1.0) {
        printf("cos(%f) = %f, sin(%f) = %f\n", theta0, cost0, theta0, sint0);
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
