/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_static_vector_gpu.cu: class definition of Class DirectStaticVectorGpu
*/
#include "nbodyfast.h"
#include "gpu.h"
#include "direct_static_vector_gpu.h"
#include "direct_gpu_kernel.h"
#include "field.h"
#include "field_static_vector_gpu.h"
#include "memory_gpu.h"

// This marco constant is used for transferring _d_test array back to host and write it on the disk
// _d_test array is used for extracting run-time info inside a device kernel
//#define _GPU_D_TEST

namespace NBODYFAST_NS{

DirectStaticVectorGpu :: DirectStaticVectorGpu(class NBODYFAST *n_ptr) : DirectStaticVector(n_ptr)
{
	field_static_vector_gpu = dynamic_cast<FieldStaticVectorGpu*>(n_ptr->field);
}
DirectStaticVectorGpu :: ~DirectStaticVectorGpu()
{
}
int DirectStaticVectorGpu :: execution()
{
	if (nbodyfast->multi_device)
	{
		execution_multi();
	}
	else
	{
		execution_single();
	}

	return 0;
}
int DirectStaticVectorGpu :: execution_single()
{
	cudaError_t _cuda_error;
	int _num_blk;
	dim3 _dim_block;
	_dim_block.x = BLOCK_SIZE_DIRECT; // number of threads per block
	dim3 _dim_grid;
	_num_blk = problem_size / BLOCK_SIZE_DIRECT; // number of blocks in the grid 

	if (problem_size%BLOCK_SIZE_DIRECT != 0) _num_blk += 1;

	/////////////////////////////////////////////////////////////////////////////
	// if number of blocks is greater than 65535, we need a 2D grid
	unsigned int v = _num_blk / 65535; 
	// round v to neareast power of 2
	v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

	_dim_grid.x = v;
	_dim_grid.y = (_num_blk - 1) / v + 1;
	/////////////////////////////////////////////////////////////////////////////

	// _d_test is a test array to designed to extract information from kernels
	// it is enabled/disabled by setting _GPU_D_TEST macro 
	FP_TYPE *_d_test = NULL; 
#ifdef _GPU_D_TEST
	int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", nbodyfast->gpu->dev_list[0].index);
#endif	
	// Set cache/shared memory perference
	cudaFuncSetCacheConfig(direct_onfly_static_vector_single, cudaFuncCachePreferShared);
	// calling direct calculation kernel
	direct_onfly_static_vector_single<<<_dim_grid, _dim_block>>>(field_static_vector_gpu->d_src_coord[0], field_static_vector_gpu->d_obs_coord[0], field_static_vector_gpu->d_src_amp[0], field_static_vector_gpu->d_field_amp[0], _d_test, problem_size, Field::epsilon());
	//cudaDeviceSynchronize();			

#ifdef _GPU_D_TEST
	nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size);
	nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test);	
#endif
	return 0;
}
// MultiGPU direct method has not been integrated to this library yet. 
int DirectStaticVectorGpu :: execution_multi()
{
	return 0;
}
}