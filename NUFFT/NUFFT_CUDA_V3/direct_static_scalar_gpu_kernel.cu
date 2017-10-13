/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_static_scalar_gpu_kernel.cu: device kernels of direct GPU calculation
*/
#include "interp_kernel.h"
#include "direct_gpu_kernel.h"

namespace NBODYFAST_NS{

__global__ void direct_onfly_static_scalar_single(FP_TYPE *_src_coord, FP_TYPE *_obs_coord, FP_TYPE *_src_amp, FP_TYPE *_field_amp, FP_TYPE *_d_test, int _problem_size, FP_TYPE _epsilon)
{
	unsigned int _tidx = threadIdx.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;

	FP_TYPE _P;
	__shared__ FP_TYPE s_src_coord[BLOCK_SIZE_DIRECT * 3];
	__shared__ FP_TYPE s_obs_coord[BLOCK_SIZE_DIRECT * 3];
	__shared__ FP_TYPE s_src_amp[BLOCK_SIZE_DIRECT];
	
	for (int i = 0; i < 3; i++) 
	{
		s_src_coord[BLOCK_SIZE_DIRECT * i + _tidx] = 0.0f;
		s_obs_coord[BLOCK_SIZE_DIRECT * i + _tidx] = 0.0f;
	}

	s_src_amp[_tidx] = 0.0f;

	_P = 0.0f;

	if (_bid * BLOCK_SIZE_DIRECT + _tidx < _problem_size) // number of total threads could be larger than problem size, so we need this limiter
	{
		// load observer coordinates
		for (int k = 0; k < 3; k++)
		{
			s_obs_coord[BLOCK_SIZE_DIRECT * k + _tidx] = _obs_coord[_bid * BLOCK_SIZE_DIRECT + _problem_size * k + _tidx];
		}
	}
	
	for (int j = 0; j * BLOCK_SIZE_DIRECT < _problem_size; j++)
	{
		if (j * BLOCK_SIZE_DIRECT + _tidx < _problem_size)
		{
			// load source observer
			for (int k = 0; k < 3; k++)
			{
				s_src_coord[BLOCK_SIZE_DIRECT * k + _tidx] = _src_coord[j * BLOCK_SIZE_DIRECT + _problem_size * k + _tidx];
			}
			// load source amplitudes
			s_src_amp[_tidx] = _src_amp[(j * BLOCK_SIZE_DIRECT + _tidx)];
		}
		else
		{
			s_src_amp[_tidx] = 0.0f;
		}
	
		__syncthreads();
#pragma unroll 64
		for (int k = 0; k < BLOCK_SIZE_DIRECT; k ++)
		{

			FP_TYPE _r[3];
			FP_TYPE _magn;			
			for (int m = 0; m < 3; m++)
			{
				_r[m] = s_src_coord[k + m * BLOCK_SIZE_DIRECT] - s_obs_coord[_tidx + m* BLOCK_SIZE_DIRECT]; 
			}
			_magn = s_src_amp[k];
			get_field_static_scalar(_r, &_magn, &_P, _epsilon);
		}
	__syncthreads();		
	
	}

	if (_bid * BLOCK_SIZE_DIRECT + _tidx < _problem_size)
		_field_amp[_bid * BLOCK_SIZE_DIRECT + _tidx] = _P;

	//_d_test[_bid * BLOCK_SIZE_DIRECT + _tidx] = 9.9f;
	//_d_test[_bid * BLOCK_SIZE_DIRECT + _tidx] = _src_amp[_problem_size * 0 + _bid * BLOCK_SIZE_DIRECT + _tidx];
}
}