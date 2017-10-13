/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_gpu_kernel.h: header to declare kernels used by GPU Direct calculation 
*/
#ifndef _DIRECT_GPU_KERNEL
#define _DIRECT_GPU_KERNEL
#include "gpu.h"
namespace NBODYFAST_NS{
__global__ void direct_onfly_static_scalar_single(FP_TYPE *, FP_TYPE *, FP_TYPE *, FP_TYPE *, FP_TYPE *, int, FP_TYPE);
__global__ void direct_onfly_static_vector_single(FP_TYPE *, FP_TYPE *, FP_TYPE *, FP_TYPE *, FP_TYPE *, int, FP_TYPE);
}
#endif