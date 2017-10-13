/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft_static_scalar_gpu_kernel.: header to declare kernels used by GPU NUFFT calculation 
*/
#ifndef _NUFFT_GPU_KERNEL
#define _NUFFT_GPU_KERNEL
#include "gpu.h"
#include "direct_gpu_kernel.h"
#include "nufft_static_scalar_gpu.h"
//#define _GPU_D_TEST
namespace NBODYFAST_NS{

#ifndef USE_TEXTURE
//#define USE_TEXTURE
#endif

#ifdef USE_TEXTURE
	texture<int,1> tex_box_int;
#endif

// Each NUFFT stage will have 2 kernels, generally. One for linear and another for cubic operations. 
__global__ void nufft_project_static_scalar_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_project_static_scalar_linear(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_fft_prep_static_scalar_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_fft_prep_static_scalar_linear(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_convolution_static_scalar(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_fft_postp_static_scalar_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_fft_postp_static_scalar_linear(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_correct_static_scalar_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_correct_static_scalar_linear(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_interp_static_scalar_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_interp_static_scalar_linear(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
__global__ void nufft_exact_near_field(FP_TYPE *_d_test, NufftArrayGpuStaticScalar *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param);
//__device__ int near_to_glb_idx(int _near_idx, int _box_idx, int near_correct_layer, int* num_boxes);
//__device__ int near_cnt_to_dim( int* _near_box_idx_dim, int _near_idx, unsigned int* _box_idx_dim, int near_correct_layer, int* num_boxes);
}
#endif