/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft_static_vector_gpu_kernel.cu: kernels used by NufftStaticVectorGpu
*/
#include "interp_kernel.h"
#include "nufft_static_vector_gpu_kernel.h"

//#define BEN_DEBUG
//#define BEN_DEBUG_MULTI
//#define BEN_NEW_METHOD
//#define BEN_DEBUG_FFT
namespace NBODYFAST_NS{
/*
* most kernels have a cubic verison and a linear version, corresponding to two different interpolation scheme 
*/ 
__global__ void nufft_project_static_vector_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	unsigned int _tidx = threadIdx.x; // thread index within a block
	unsigned int _bdim = blockDim.x; // number of threads per block
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x; // block index (the grid might be 2D)
	unsigned int _local_box_idx; // local box index (index of the box among all boxes on the same device)
	unsigned int _box_idx; // global box index (index of the box as in the entire computational domain)
	unsigned int _box_idx_dim[3]; // 3D index of boxes (used to determine the location of box)
	unsigned int _box_sub_idx = 0; // sub index of a box (used when a box is treated by multiple blocks)
	unsigned int _obs_idx; // observer index (this is the index of grid points in this kernel)
 	unsigned int _obs_idx_dim[3]; // 3D index of observers (used to determin the coordinates of each grid points)

	__shared__ FP_TYPE s_src_coord[BLOCK_SIZE_PROJ_INTERP * 3]; // shared memory array to store the source coordinates
	__shared__ FP_TYPE s_src_amp[BLOCK_SIZE_PROJ_INTERP * FIELD_DIM]; //shared memory array to store the source amplitudes

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 64;
	int const src_size_dev = nufft_const->src_size_dev; // number of sources on this device (so the length of source arrays)
	int * const num_boxes = nufft_const->num_boxes; // number of total boxes (across the entire domain)
	FP_TYPE * const box_size = nufft_const->box_size; // as it is shown...the box size
	FP_TYPE * const src_domain_range = nufft_const->src_domain_range; // computational domain boundaries and sizes

	FP_TYPE _P[FIELD_DIM]; // used to store final fields. declared as an array as it might have more than 1 components for vector fields
	for(unsigned int i = 0; i < FIELD_DIM; i++ )	_P[i] = 0.0f;
	int _global_src_idx_start2 = 0; // the offset of source coord/amp in the global source arrays
	int _num_src2 = 0; // number of sources in the current box
	FP_TYPE _box_origin[3]; // the front-top-left corner of the current box
	FP_TYPE _interp_coeff; // interpolation coefficients
	FP_TYPE _r_norm[3]; // normalized coordinates of a source within a box. used to do the Lagrange interpolation
	int _shared_start = 0; // a temporary variable for serializing tasks while number of threads is less than number of sources
	int _shared_offset; // indicate the range of shared memory for current box (when multiple boxes are handled by the same block)

	if (param->num_blk_per_box > 0) // number of block per box is greater than or equal to 1 
	{
		_local_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;
		_shared_offset = 0;
	
		if (_obs_idx >= interp_nodes_per_box) return;
	}

	if (param->num_box_per_blk > 0) // number of boxes per block is greater than 1
	{
		_local_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box;
		_shared_offset = _tidx / interp_nodes_per_box * interp_nodes_per_box;

		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}

	if (_local_box_idx >=  nufft_array->d_src_box_list_inv[0]) return; // if _local_box_idx is greater than total number of boxes the current device should process, then just terminate the current thread 

	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1]; // otherwise, get the global box idx from d_src_box_list
	
	if (_box_idx >= nufft_const->total_num_boxes) return; // I think this is an unnecessary check

	idx_to_idx_dim_cubic(_obs_idx_dim, _obs_idx); // get 3D observer index from 1D observer index 

	// get 3D box index from 1D box index 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];

	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + src_domain_range[m]; // calculate the position of the box
	}

	// The starting index of sources in the current box
	_global_src_idx_start2 = nufft_array->d_src_box_map[_box_idx];

	// Number of sources in the current box
	_num_src2 = nufft_array->d_src_box_map[_box_idx + nufft_const->total_num_boxes * 2];

	while (_shared_start < _num_src2)
	{
		// Load/Calculate the normalized coordinates and amplitudes of sources to shared memory
		if (_shared_start + _obs_idx < _num_src2)
		{
			for (int m = 0; m < 3; m++)	
			{
				// s_src_coord stores the coordinates that have been normalized to [0, 1];
				s_src_coord[_tidx + m * _bdim] = (nufft_array->d_src_coord[_global_src_idx_start2 + (_shared_start + _obs_idx) + src_size_dev * m] - _box_origin[m]) / box_size[m];
			}
			for(unsigned int j = 0; j < FIELD_DIM; j++ )
				s_src_amp[_tidx + j * BLOCK_SIZE_PROJ_INTERP] = nufft_array->d_src_amp[_global_src_idx_start2 + _shared_start + _obs_idx + j * src_size_dev];
		}

		__syncthreads();
 
		// Loop around current piece of source (not more than number of threads per block)
		// From source to source grid
		for (int i = 0; (i < interp_nodes_per_box) && (i + _shared_start < _num_src2); i++)
		{
			for (int m = 0; m < 3; m++)
			{
				_r_norm[m] = s_src_coord[i + _shared_offset + m * _bdim];
			}
			lagrange_project_cubic(_interp_coeff, _r_norm, _obs_idx_dim);    
			for(unsigned int j = 0; j < FIELD_DIM; j++ )
				_P[j] += _interp_coeff * s_src_amp[i + _shared_offset + j * BLOCK_SIZE_PROJ_INTERP];
		} // i
		__syncthreads();
		_shared_start += interp_nodes_per_box;	
	} // _shared_start

	for(unsigned int j = 0; j < FIELD_DIM; j++ )
		nufft_array->d_u_src_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j * nufft_const->total_num_boxes_dev * interp_nodes_per_box] = _P[j];

	#ifdef _GPU_D_TEST
		_d_test[_bid * BLOCK_SIZE_PROJ_INTERP + _tidx] = 0.00f;
	#endif

	#ifdef _GPU_D_TEST
		_d_test[_bid * BLOCK_SIZE_PROJ_INTERP + _tidx] += _P[0]; // * _bdim;//_interp_coeff;
	#endif	


}

__global__ void nufft_project_static_vector_linear(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _local_box_idx;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
 	unsigned int _obs_idx_dim[3];

	__shared__ FP_TYPE s_src_coord[BLOCK_SIZE_PROJ_INTERP * 3];
	__shared__ FP_TYPE s_src_amp[BLOCK_SIZE_PROJ_INTERP * FIELD_DIM];

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 8;
	int const src_size_dev = nufft_const->src_size_dev;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const src_domain_range = nufft_const->src_domain_range;

	FP_TYPE _P[FIELD_DIM];
	for(unsigned int i = 0; i < FIELD_DIM; i++ )	_P[i] = 0.0f;
	int _global_src_idx_start2 = 0;
	int _num_src2 = 0;
	FP_TYPE _box_origin[3];
	FP_TYPE _interp_coeff;
	FP_TYPE _r_norm[3];
	int _shared_start = 0;
	int _shared_offset;

	if (param->num_blk_per_box > 0)
	{
		_local_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;
		_shared_offset = 0;

		if (_obs_idx >= interp_nodes_per_box) return;
	}

	if (param->num_box_per_blk > 0) 
	{
		_local_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box;
		_shared_offset = _tidx / interp_nodes_per_box * interp_nodes_per_box;

		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}

	if (_local_box_idx >= nufft_array->d_src_box_list_inv[0]) return;
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];

	if (_box_idx >= nufft_const->total_num_boxes) return;

	idx_to_idx_dim_linear(_obs_idx_dim, _obs_idx);

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];

	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + src_domain_range[m];
	}

	// The starting index of sources in the current box
	_global_src_idx_start2 = nufft_array->d_src_box_map[_box_idx];

	// Number of sources in the current box
	_num_src2 = nufft_array->d_src_box_map[_box_idx + nufft_const->total_num_boxes * 2];

	while (_shared_start < _num_src2)
	{
		// Load/Calculate the normalized coordinates and amplitudes of sources to shared memory
		if (_shared_start + _obs_idx < _num_src2)
		{
			for (int m = 0; m < 3; m++)	
			{
				// s_src_coord stores the coordinates that have been normalized to [0, 1];
				s_src_coord[_tidx + m * _bdim] = (nufft_array->d_src_coord[_global_src_idx_start2 + (_shared_start + _obs_idx) + src_size_dev * m] - _box_origin[m]) / box_size[m];
			}
			for(unsigned int j = 0; j < FIELD_DIM; j++ )
				s_src_amp[_tidx + j * BLOCK_SIZE_PROJ_INTERP] = nufft_array->d_src_amp[_global_src_idx_start2 + _shared_start + _obs_idx + j * src_size_dev];
		}

		__syncthreads();

		// Loop around current piece of source (not more than number of threads per block)
		// From source to source grid
		for (int i = 0; (i < interp_nodes_per_box) && (i + _shared_start < _num_src2); i++)
		{
			for (int m = 0; m < 3; m++)
			{
				_r_norm[m] = s_src_coord[i + _shared_offset + m * _bdim];
			}
		 
			lagrange_project_linear(_interp_coeff, _r_norm, _obs_idx_dim);
			for(unsigned int j = 0; j < FIELD_DIM; j++ )    
				_P[j] += _interp_coeff * s_src_amp[i + _shared_offset + j * BLOCK_SIZE_PROJ_INTERP];

		} // i

		__syncthreads();
		_shared_start += interp_nodes_per_box;	
	} // _shared_start

	for(unsigned int j = 0; j < FIELD_DIM; j++ )
		nufft_array->d_u_src_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j * nufft_const->total_num_boxes_dev * interp_nodes_per_box] = _P[j];
#ifdef _GPU_D_TEST
	_d_test[_bid * BLOCK_SIZE_PROJ_INTERP + _tidx] = 9.99f;
	_d_test[_bid * BLOCK_SIZE_PROJ_INTERP + _tidx] = _P[0];
#endif		

}

__global__ void nufft_fft_prep_static_vector_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	const unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x; // global thread index
	const unsigned int interp_nodes_per_box = 64;
	FP_TYPE _P[FIELD_DIM];
	for(unsigned int i = 0; i < FIELD_DIM; i++ )	_P[i] = 0.0f;

	if (_gid >= nufft_const->total_num_grid_pts) return; // terminate extra threads launched

	int _grid_idx_dim[3]; // 3D global grid point index
	int _local_idx_dim[6]; // 3D local grid point index (for each grid point, there could be two different boxes contributing to the field along each dimension)
	int _box_idx_dim[6]; // for each grid point, there could be two boxes contributing to it along each dimension
	int _box_idx[9]; // _box_idx[8] stores number of boxes contributing to the current grid points. _box_idx[0~7] stores the contributing box index
	int _local_idx[9]; // _box_idx[0~7] stores the contributing index of the grid points of the contributing box

	for (int i = 0; i < 9; i++)
	{
		_box_idx[i] = -1;
		_local_idx[i] = -1;
	}
	
	// 3D index of current grid points in the entire projection grid
	_grid_idx_dim[2] = _gid / (nufft_const->num_grid_pts[0] * nufft_const->num_grid_pts[1]);
	_grid_idx_dim[1] =  (_gid - _grid_idx_dim[2] * (nufft_const->num_grid_pts[0] * nufft_const->num_grid_pts[1])) / nufft_const->num_grid_pts[0];
	_grid_idx_dim[0] = _gid % nufft_const->num_grid_pts[0];

	// the position of current grid points in the FFT grid (FFT grid is padded so is larger than the projection grid)
	int _cufft_in_addr = _grid_idx_dim[2] * nufft_const->fft_size[0] * nufft_const->fft_size[1]
								+ _grid_idx_dim[1] * nufft_const->fft_size[0]
								+ _grid_idx_dim[0];
	int _u_src_grid_addr = 0;
	
	grid_idx_to_local_idx_cubic(_grid_idx_dim, _box_idx_dim, _local_idx_dim, nufft_const->num_boxes); // get local index 

	int _dim_idx[3];
	int _cnt1 = 0;
	int _box_idx_temp;

	// there will be 8 possible boxes that overlaps at the current grid points
	// however, there might be less if the current grid points is not at the corner of a box, or the box is at the corner, on the edge or surface of the entire computational domain.
	// the following triple loops judges how many boxes are valid and really contributing to the current grid point
	for (_dim_idx[0] = 0; _dim_idx[0] < 2; _dim_idx[0]++)
		for (_dim_idx[1] = 0; _dim_idx[1] < 2; _dim_idx[1]++)
			for (_dim_idx[2] = 0; _dim_idx[2] < 2; _dim_idx[2]++)
			{
				if (_box_idx_dim[_dim_idx[0]] >= 0 && _box_idx_dim[2 + _dim_idx[1]] >= 0 && _box_idx_dim[4 + _dim_idx[2]] >= 0)
				{
					_box_idx_temp = _box_idx_dim[_dim_idx[0]] + _box_idx_dim[2 + _dim_idx[1]] *  nufft_const->num_boxes[0] + _box_idx_dim[4 + _dim_idx[2]] * nufft_const->num_boxes[0] * nufft_const->num_boxes[1];

					_box_idx[_cnt1] = _box_idx_temp;
					int _local_idx_dim_temp[3];
					_local_idx_dim_temp[0] = _local_idx_dim[_dim_idx[0]];
					_local_idx_dim_temp[1] = _local_idx_dim[2 + _dim_idx[1]];
					_local_idx_dim_temp[2] = _local_idx_dim[4 + _dim_idx[2]];

					local_idx_dim_to_local_idx_cubic(&_local_idx[_cnt1], _local_idx_dim_temp);	
					
					_cnt1++;
				}

			}
	_box_idx[8] = _cnt1;

#ifdef _GPU_D_TEST
	_d_test[_gid] = 0.00f;
#endif
	// add the projected amplitudes of all overlapping grid points together
	for (int i = 0; i < _box_idx[8]; i++)
	{
		_u_src_grid_addr = _box_idx[i] * interp_nodes_per_box + _local_idx[i];
		for(unsigned int j = 0; j < FIELD_DIM; j++ ) _P[j] += nufft_array->d_u_src_grid[_u_src_grid_addr + j * nufft_const->total_num_boxes * interp_nodes_per_box];
#ifdef _GPU_D_TEST
	_d_test[_gid] = size_t(_u_src_grid_addr);
#endif
	}

	for(unsigned int j = 0; j < FIELD_DIM; j++ ) nufft_array->d_fft_inplace_r2c_FP[_cufft_in_addr + j*nufft_const->total_num_fft_pts] = _P[j];

}

///FFT CHANGE
__global__ void nufft_fft_prep_static_vector_linear(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	const unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	const unsigned int interp_nodes_per_box = 8;

	FP_TYPE _P[FIELD_DIM];
	for(unsigned int i = 0; i < FIELD_DIM; i++ )	_P[i] = 0.0f;
	if (_gid >= nufft_const->total_num_grid_pts) return;
	int _grid_idx_dim[3];
	int _local_idx_dim[6];
	int _box_idx_dim[6];
	int _box_idx[9];
	int _local_idx[9];

	for (int i = 0; i < 9; i++)
	{
		_box_idx[i] = -1;
		_local_idx[i] = -1;
	}
	_grid_idx_dim[2] = _gid / (nufft_const->num_grid_pts[0] * nufft_const->num_grid_pts[1]);
	_grid_idx_dim[1] =  (_gid - _grid_idx_dim[2] * (nufft_const->num_grid_pts[0] * nufft_const->num_grid_pts[1])) / nufft_const->num_grid_pts[0];
	_grid_idx_dim[0] = _gid % nufft_const->num_grid_pts[0];

	int _cufft_in_addr = _grid_idx_dim[2] * nufft_const->fft_size[0] * nufft_const->fft_size[1]
								+ _grid_idx_dim[1] * nufft_const->fft_size[0]
								+ _grid_idx_dim[0];
	int _u_src_grid_addr = 0;
	
	grid_idx_to_local_idx_linear(_grid_idx_dim, _box_idx_dim, _local_idx_dim, nufft_const->num_boxes);

	int _dim_idx[3];
	int _cnt1 = 0;
	int _box_idx_temp;

	for (_dim_idx[0] = 0; _dim_idx[0] < 2; _dim_idx[0]++)
		for (_dim_idx[1] = 0; _dim_idx[1] < 2; _dim_idx[1]++)
			for (_dim_idx[2] = 0; _dim_idx[2] < 2; _dim_idx[2]++)
			{
				if (_box_idx_dim[_dim_idx[0]] >= 0 && _box_idx_dim[2 + _dim_idx[1]] >= 0 && _box_idx_dim[4 + _dim_idx[2]] >= 0)
				{
					_box_idx_temp = _box_idx_dim[_dim_idx[0]] + _box_idx_dim[2 + _dim_idx[1]] *  nufft_const->num_boxes[0] + _box_idx_dim[4 + _dim_idx[2]] * nufft_const->num_boxes[0] * nufft_const->num_boxes[1];

					_box_idx[_cnt1] = _box_idx_temp;
					int _local_idx_dim_temp[3];
					_local_idx_dim_temp[0] = _local_idx_dim[_dim_idx[0]];
					_local_idx_dim_temp[1] = _local_idx_dim[2 + _dim_idx[1]];
					_local_idx_dim_temp[2] = _local_idx_dim[4 + _dim_idx[2]];

					local_idx_dim_to_local_idx_linear(&_local_idx[_cnt1], _local_idx_dim_temp);

					_cnt1++;
				}

			}
	_box_idx[8] = _cnt1;


	for (int i = 0; i < _box_idx[8]; i++)
	{
		_u_src_grid_addr = _box_idx[i] * interp_nodes_per_box + _local_idx[i];
		for(unsigned int j = 0; j < FIELD_DIM; j++ ) _P[j] += nufft_array->d_u_src_grid[_u_src_grid_addr + j * nufft_const->total_num_boxes * interp_nodes_per_box];
	}

	for(unsigned int j = 0; j < FIELD_DIM; j++ ) nufft_array->d_fft_inplace_r2c_FP[_cufft_in_addr + j*nufft_const->total_num_fft_pts] = _P[j];

#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;
//	_d_test[_gid] = nufft_array->d_u_src_grid[_gid];
#endif
}

//BEN ELIMINATED THE d_fft_inplace_b
__global__ void nufft_convolution_static_vector(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	// convolution is simple. just multiply the transformed matrix and impedance matrix, entry-by-entry
	const unsigned int _tid = threadIdx.x;
	const unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	//const unsigned int S_GREEN_SIZE = 
	__shared__ CUFFT_COMPLEX_TYPE s_u_src_grid_k[FIELD_DIM * BLOCK_SIZE_CONV];
	__shared__ FP_TYPE s_g_grid_k[ FIELD_DIM*(FIELD_DIM+1)/2 * BLOCK_SIZE_CONV];//IMP MAT CHANGE!!!SHOULD BE LESS!!!!IF THIS SUBROUTINE IS LIMITED BY SHARED MEMORY, THEN
													//WE SHOULD SIMPLY NOT USE SHARED MEMORY HERE

	if (_gid >= (nufft_const->total_num_fft_r2c_pts)) 
		return;

	int i = _gid % nufft_const->fft_r2c_size[0];
	int k = _gid / (nufft_const->fft_r2c_size[0]*nufft_const->fft_r2c_size[1]);
	int j = (_gid - k*nufft_const->fft_r2c_size[0]*nufft_const->fft_r2c_size[1])/nufft_const->fft_r2c_size[0];
	int flag_y = 1;	int flag_z = 1;
	if( j >= nufft_const->green_size[1] )
	{	j = nufft_const->fft_size[1] - j; flag_y = -1;}
	if( k >= nufft_const->green_size[2])
	{	k = nufft_const->fft_size[2] - k; flag_z = -1;}
	int _green_id = i + j*nufft_const->green_size[0] + k*nufft_const->green_size[0]*nufft_const->green_size[1];

	for(unsigned int j = 0; j < FIELD_DIM; j++)	s_u_src_grid_k[_tid+j*BLOCK_SIZE_CONV] = nufft_array->d_fft_inplace_r2c[_gid+j*nufft_const->total_num_fft_r2c_pts];
	//the Green's func assignment in shared memory is not compatible to FIELD_DIM != 3
	s_g_grid_k[_tid                  ] = 		     nufft_array->d_k_imp_mat_data_gpu[_green_id]; //xx
	s_g_grid_k[_tid+  BLOCK_SIZE_CONV] = flag_y *		     nufft_array->d_k_imp_mat_data_gpu[_green_id + nufft_const->total_num_green_pts]; //xy
	s_g_grid_k[_tid+2*BLOCK_SIZE_CONV] = flag_z * 	     nufft_array->d_k_imp_mat_data_gpu[_green_id + 2*nufft_const->total_num_green_pts]; //xz
	s_g_grid_k[_tid+3*BLOCK_SIZE_CONV] =  	     	     nufft_array->d_k_imp_mat_data_gpu[_green_id + 3*nufft_const->total_num_green_pts]; //yy
	s_g_grid_k[_tid+4*BLOCK_SIZE_CONV] = flag_y*flag_z * nufft_array->d_k_imp_mat_data_gpu[_green_id + 4*nufft_const->total_num_green_pts]; //yz
	s_g_grid_k[_tid+5*BLOCK_SIZE_CONV] = 		     nufft_array->d_k_imp_mat_data_gpu[_green_id + 5*nufft_const->total_num_green_pts]; //zz

	FP_TYPE real0	= s_u_src_grid_k[_tid].x*s_g_grid_k[_tid]	+ s_u_src_grid_k[_tid+BLOCK_SIZE_CONV].x*s_g_grid_k[_tid+BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+2*BLOCK_SIZE_CONV].x*s_g_grid_k[_tid+2*BLOCK_SIZE_CONV];

	FP_TYPE img0	= s_u_src_grid_k[_tid].y*s_g_grid_k[_tid]	+ s_u_src_grid_k[_tid+BLOCK_SIZE_CONV].y*s_g_grid_k[_tid+BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+2*BLOCK_SIZE_CONV].y*s_g_grid_k[_tid+2*BLOCK_SIZE_CONV];

	FP_TYPE real1	= s_u_src_grid_k[_tid].x*s_g_grid_k[_tid+BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+BLOCK_SIZE_CONV].x*s_g_grid_k[_tid+3*BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+2*BLOCK_SIZE_CONV].x*s_g_grid_k[_tid+4*BLOCK_SIZE_CONV];

	FP_TYPE img1	= s_u_src_grid_k[_tid].y*s_g_grid_k[_tid+BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+BLOCK_SIZE_CONV].y*s_g_grid_k[_tid+3*BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+2*BLOCK_SIZE_CONV].y*s_g_grid_k[_tid+4*BLOCK_SIZE_CONV];
	
	FP_TYPE real2	= s_u_src_grid_k[_tid].x*s_g_grid_k[_tid+2*BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+BLOCK_SIZE_CONV].x*s_g_grid_k[_tid+4*BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+2*BLOCK_SIZE_CONV].x*s_g_grid_k[_tid+5*BLOCK_SIZE_CONV];

	FP_TYPE img2	= s_u_src_grid_k[_tid].y*s_g_grid_k[_tid+2*BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+BLOCK_SIZE_CONV].y*s_g_grid_k[_tid+4*BLOCK_SIZE_CONV]	+ s_u_src_grid_k[_tid+2*BLOCK_SIZE_CONV].y*s_g_grid_k[_tid+5*BLOCK_SIZE_CONV];

	//the division here can be put into the Greens'func preprocessing
	nufft_array->d_fft_inplace_r2c[_gid].x = real0/FP_TYPE(nufft_const->total_num_fft_pts);
	nufft_array->d_fft_inplace_r2c[_gid].y = img0/FP_TYPE(nufft_const->total_num_fft_pts);
	nufft_array->d_fft_inplace_r2c[_gid+nufft_const->total_num_fft_r2c_pts].x = real1/FP_TYPE(nufft_const->total_num_fft_pts);
	nufft_array->d_fft_inplace_r2c[_gid+nufft_const->total_num_fft_r2c_pts].y = img1/FP_TYPE(nufft_const->total_num_fft_pts);
	nufft_array->d_fft_inplace_r2c[_gid+2*nufft_const->total_num_fft_r2c_pts].x = real2/FP_TYPE(nufft_const->total_num_fft_pts);
	nufft_array->d_fft_inplace_r2c[_gid+2*nufft_const->total_num_fft_r2c_pts].y = img2/FP_TYPE(nufft_const->total_num_fft_pts);
#ifdef BEN_DEBUG_FFT
	_d_test[_gid] = nufft_const->green_size[0]*1000+nufft_const->fft_size[0];//s_g_grid_k[_tid];
#endif
#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;
	_d_test[_gid] = nufft_array->d_fft_inplace_b[_gid].x;

#endif
}
//BEN ELIMINATED THE d_fft_inplace_b
__global__ void nufft_fft_postp_static_vector_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	// post processing does opposite thing as the pre processing
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
	unsigned int _obs_idx_dim[3];
	const unsigned int interp_nodes_per_box = 64;

	if (param->num_blk_per_box > 0)
	{
		_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;

		if (_obs_idx >= interp_nodes_per_box) return;
	}

	if (param->num_box_per_blk > 0) 
	{
		_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box;
		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}

	if (_box_idx >= nufft_const->total_num_boxes) return;


	// Get the Index number of the observer box 
	_box_idx_dim[2] = _box_idx / (nufft_const->num_boxes[0] * nufft_const->num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (nufft_const->num_boxes[0] * nufft_const->num_boxes[1])) / nufft_const->num_boxes[0];
	_box_idx_dim[0] = _box_idx % nufft_const->num_boxes[0];
	
	// Get the global index number of the grid point
	unsigned int obs_idx_glb;

	idx_to_idx_dim_cubic(_obs_idx_dim, _obs_idx);
	obs_idx_dim_to_obs_idx_glb_cubic(obs_idx_glb, _box_idx_dim, _obs_idx_dim, nufft_const->fft_size);


	/*nufft_array->d_u_obs_grid[_gid] = nufft_array->d_fft_inplace_b[obs_idx_glb].x;*/
	//nufft_array->d_u_obs_grid[_gid] = nufft_array->d_fft_inplace_f[obs_idx_glb].x;
	for(unsigned int j = 0; j < FIELD_DIM; j++)	
		nufft_array->d_u_obs_grid[_gid+j*nufft_const->total_num_boxes*interp_nodes_per_box] = nufft_array->d_fft_inplace_r2c_FP[obs_idx_glb+j*nufft_const->total_num_fft_pts];

#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;
	_d_test[_gid] = nufft_array->d_u_obs_grid[_gid];

#endif

}
//BEN ELIMINATED THE d_fft_inplace_b
__global__ void nufft_fft_postp_static_vector_linear(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
	unsigned int _obs_idx_dim[3];
	const unsigned int interp_nodes_per_box = 8;

	if (param->num_blk_per_box > 0)
	{
		_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;

		if (_obs_idx >= interp_nodes_per_box) return;

	}

	if (param->num_box_per_blk > 0) 
	{
		_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box;
		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}

	if (_box_idx >= nufft_const->total_num_boxes) return;


	// Get the Index number of the observer box 
	_box_idx_dim[2] = _box_idx / (nufft_const->num_boxes[0] * nufft_const->num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (nufft_const->num_boxes[0] * nufft_const->num_boxes[1])) / nufft_const->num_boxes[0];
	_box_idx_dim[0] = _box_idx % nufft_const->num_boxes[0];
	
	// Get the global index number of the grid point
	unsigned int obs_idx_glb;

	idx_to_idx_dim_linear(_obs_idx_dim, _obs_idx);
	obs_idx_dim_to_obs_idx_glb_linear(obs_idx_glb, _box_idx_dim, _obs_idx_dim, nufft_const->fft_size);

	//nufft_array->d_u_obs_grid[_gid] = nufft_array->d_fft_inplace_b[obs_idx_glb].x;
	//nufft_array->d_u_obs_grid[_gid] = nufft_array->d_fft_inplace_f[obs_idx_glb].x;
	for(unsigned int j = 0; j < FIELD_DIM; j++)	
		nufft_array->d_u_obs_grid[_gid+j*nufft_const->total_num_boxes*interp_nodes_per_box] = nufft_array->d_fft_inplace_r2c_FP[obs_idx_glb+j*nufft_const->total_num_fft_pts];

#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;
	_d_test[_gid] = nufft_array->d_u_obs_grid[_gid];

#endif

}
//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
#ifndef BEN_NEW_METHOD
__global__ void nufft_correct_static_vector_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{ 
	// nufft_correct_static_vector_cubic calculate the inaccurate field generated by near field sources again and subtract them from the total field
	// we don't have to project the amplitudes from source to source grid again since we already have them stored in the array d_u_src_grid_dev
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
 	unsigned int _local_box_idx;
 	unsigned int _local_box_idx2;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
//	unsigned int _obs_idx_glb;
	unsigned int _obs_idx_dim[3];
	
	// Used only when a block handles multiple boxes
	int _shared_offset;

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 64; 
	int const near_correct_layer = nufft_const->near_correct_layer;
	int const total_num_near_box_per_box = (2*near_correct_layer+1)*(2*near_correct_layer+1)*(2*near_correct_layer+1);
	//	int const problem_size = nufft_const->problem_size;
	FP_TYPE const epsilon = nufft_const->epsilon;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const cell_size = nufft_const->cell_size;
	FP_TYPE * const src_domain_range = nufft_const->src_domain_range;

	if (param->num_blk_per_box > 0)
	{
		_local_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;

		if (_obs_idx >= interp_nodes_per_box) return;
		_shared_offset = 0;
	}

	if (param->num_box_per_blk > 0) 
	{
		_local_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box; 
		_shared_offset = _tidx / interp_nodes_per_box * interp_nodes_per_box;

		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}

	if (_local_box_idx >= nufft_const->total_num_boxes_dev) return;
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];

	__shared__ FP_TYPE s_src_amp[BLOCK_SIZE_CORRECT*FIELD_DIM];
      
	FP_TYPE _Q1[FIELD_DIM]; // An array stores far field on observer
	for(unsigned int i = 0; i < FIELD_DIM; i++)	_Q1[i] = 0.0f;

	FP_TYPE _r1[3];
	FP_TYPE _box_origin[3];
	int _near_box_idx = 0;
	int _near_box_idx_dim[3];
	FP_TYPE _near_box_origin[3];

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];

	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + src_domain_range[m];
	}
	
	idx_to_idx_dim_cubic(_obs_idx_dim, _obs_idx);
#ifdef _GPU_D_TEST
	_d_test[_gid] = 0.00f;  
#endif
	// Ben new bound method
	//for (_near_box_idx_dim[2] = nufft_array->d_near_bound_list[6*_box_idx+2]; _near_box_idx_dim[2] <= nufft_array->d_near_bound_list[6*_box_idx+5]; _near_box_idx_dim[2]++){
	//	for(_near_box_idx_dim[1] = nufft_array->d_near_bound_list[6*_box_idx+1]; _near_box_idx_dim[1] <= nufft_array->d_near_bound_list[6*_box_idx+4]; _near_box_idx_dim[1]++){
	//		for(_near_box_idx_dim[0] = nufft_array->d_near_bound_list[6*_box_idx+0]; _near_box_idx_dim[0] <= nufft_array->d_near_bound_list[6*_box_idx+3]; _near_box_idx_dim[0]++){

	//			// For each box, d_NearBoxListThread has 28 entries (maybe empty), while the first entry is 
	//			// the total number of near boxes.
	//			_near_box_idx = _near_box_idx_dim[0] + _near_box_idx_dim[1]*num_boxes[0] + _near_box_idx_dim[2]*num_boxes[0]*num_boxes[1];//near_cnt_to_dim( _near_box_idx_dim, _near_box_counter, _box_idx_dim, near_correct_layer, num_boxes);

	//			// Current near box origins
	//			for (int m = 0; m < 3; m++)
	//				_near_box_origin[m] = _near_box_idx_dim[m] * box_size[m] + src_domain_range[m];

	//			__syncthreads();
	//			_local_box_idx2 = nufft_array->d_src_box_list_inv[_near_box_idx + 1];
	//			s_src_amp[_tidx] = nufft_array->d_u_src_grid_dev[_local_box_idx2 * interp_nodes_per_box + _obs_idx];
	//			__syncthreads();

	//			///////////////////////////////////////////////////////////////////////////////////////
	//			//// Get the inaccurate near field.
	//			//// Source: near-list box grid points
	//			//// Observer: actual box grid points
	//			//// Interaction: direct Green's function
	//			///////////////////////////////////////////////////////////////////////////////////////// 
	//			for (int m = 0; m < 3; m++)
	//			{
	//				// _r1 is the bias between the left-top-front corner of the near-list box currently being handled and the actual box grid point.
	//				_r1[m] = _near_box_origin[m] - (_box_origin[m] + _obs_idx_dim[m] * cell_size[m]) ;
	//			}

	//			// cubic : 28% of computational time of this subroutine goes here
	//			direct_grid_interact_cubic_static_vector(_r1, _Q1, s_src_amp + _shared_offset, cell_size, epsilon);
	//			//__syncthreads();
	//		}
	//	}
	//}// _near_box_counter
	// Loop around all near boxes, presumbly 27 !!!!CHANGED BY BEN
	for (int _near_box_counter = 0; _near_box_counter < total_num_near_box_per_box; _near_box_counter++)
	{

		// For each box, d_NearBoxListThread has 28 entries (maybe empty), while the first entry is 
		// the total number of near boxes.
		_near_box_idx = near_cnt_to_dim( _near_box_idx_dim, _near_box_counter, _box_idx_dim, near_correct_layer, num_boxes);
		if( _near_box_idx < 0 || nufft_array->d_src_box_map[_near_box_idx + nufft_const->total_num_boxes * 2] == 0)
			continue;

		// Current near box origins
		for (int m = 0; m < 3; m++)
		{
			_near_box_origin[m] = _near_box_idx_dim[m] * box_size[m] + src_domain_range[m];
		}
 
		__syncthreads();
		_local_box_idx2 = nufft_array->d_src_box_list_inv[_near_box_idx + 1];

		for(unsigned int j = 0; j < FIELD_DIM; j++ )
			s_src_amp[_tidx+j*BLOCK_SIZE_CORRECT] = nufft_array->d_u_src_grid_dev[_local_box_idx2 * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box];

		__syncthreads();

		///////////////////////////////////////////////////////////////////////////////////////
		//// Get the inaccurate near field.
		//// Source: near-list box grid points
		//// Observer: actual box grid points
		//// Interaction: direct Green's function
		///////////////////////////////////////////////////////////////////////////////////////// 
		for (int m = 0; m < 3; m++)
		{
			// _r1 is the bias between the left-top-front corner of the near-list box currently being handled and the actual box grid point.
			_r1[m] = _near_box_origin[m] - (_box_origin[m] + _obs_idx_dim[m] * cell_size[m]) ;
		}
 
		// cubic : 28% of computational time of this subroutine goes here

		 direct_grid_interact_cubic_static_vector(_r1, _Q1, s_src_amp + _shared_offset, cell_size, epsilon, BLOCK_SIZE_CORRECT);
		 //__syncthreads();
#ifdef _GPU_D_TEST
	_d_test[_gid] += _Q1[0];
#endif
	} // _near_box_counter

	__syncthreads();
	 
	// Field amplitudes on the observer grid
//	s_obs_amp[_tidx] = d_u_obs_grid[_obs_idx_glb] - _Q1[0];
//	s_obs_amp[_tidx] = d_u_obs_grid[_obs_idx_glb]; 

	for(unsigned int j = 0; j < FIELD_DIM; j++ )
		nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box] -= _Q1[j];

#ifdef BEN_DEBUG_MULTI
	//_d_test[_gid] = nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx];
#endif
}
#else
__global__ void nufft_correct_static_vector_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{ 
	// nufft_correct_static_vector_cubic calculate the inaccurate field generated by near field sources again and subtract them from the total field
	// we don't have to project the amplitudes from source to source grid again since we already have them stored in the array d_u_src_grid_dev
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
 	unsigned int _local_box_idx;
 	unsigned int _local_box_idx2;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _obs_idx;
//	unsigned int _obs_idx_glb;
	unsigned int _obs_idx_dim[3];
	
	// Used only when a block handles multiple boxes
	int _shared_offset;

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 64; 
	int const near_correct_layer = nufft_const->near_correct_layer;
	int const total_num_near_box_per_box = (2*near_correct_layer+1)*(2*near_correct_layer+1)*(2*near_correct_layer+1);
	//	int const problem_size = nufft_const->problem_size;
	FP_TYPE const epsilon = nufft_const->epsilon;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const cell_size = nufft_const->cell_size;
	FP_TYPE * const src_domain_range = nufft_const->src_domain_range;

	_local_box_idx = _bid;
	_obs_idx = _tidx % interp_nodes_per_box; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	if (_local_box_idx >= nufft_const->total_num_boxes_dev) 
		return;
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];

	__shared__ FP_TYPE s_src_amp[64];
	__shared__ int s_near_box_list[BLOCK_SIZE_CORRECT];
      
	FP_TYPE _Q1[FIELD_DIM]; // An array stores far field on observer
	for(unsigned int j = 0; j < FIELD_DIM; j++)	_Q1[j] = FP_TYPE(0.0f);

	FP_TYPE _r1[3];
	FP_TYPE _box_origin[3];
	int _near_box_idx = 0;
	int _near_box_idx_dim[3];
	FP_TYPE _near_box_origin[3];

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];

	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + src_domain_range[m];
	}
	
	idx_to_idx_dim_cubic(_obs_idx_dim, _obs_idx);
#ifdef _GPU_D_TEST
	_d_test[_gid] = 0.00f;  
#endif
	int near_box_left = total_num_near_box_per_box;
	int s_near_list_start = 0;
#ifdef BEN_DEBUG
	int tmp_cnt = 0;
#endif
	while( near_box_left > 0 ){
		if( _tidx < near_box_left ){
			s_near_box_list[_tidx] = near_to_glb_idx(s_near_list_start + _tidx, _bid, near_correct_layer, num_boxes);//_bid can be replaced by _box_idx_dim, because it should be only computed once
		}
		else{ // IS IT NECESSARY???
			s_near_box_list[_tidx] = -1;
		}
		__syncthreads();
		// Loop around all near boxes, presumbly 27 !!!!CHANGED BY BEN
		for (int _near_box_counter = 0; _near_box_counter < near_box_left && _near_box_counter < BLOCK_SIZE_CORRECT; _near_box_counter++)
		{

			// For each box, d_NearBoxListThread has 28 entries (maybe empty), while the first entry is 
			// the total number of near boxes.
			_near_box_idx = s_near_box_list[_near_box_counter];
			//_near_box_idx = near_cnt_to_dim( _near_box_idx_dim, _near_box_counter, _box_idx_dim, near_correct_layer, num_boxes);
				
			if( _near_box_idx < 0 )
				continue;
#ifdef BEN_DEBUG
			tmp_cnt++;
#endif
			// Current near box index number
			_near_box_idx_dim[2] = _near_box_idx / (num_boxes[0] * num_boxes[1]);
			_near_box_idx_dim[1] =  (_near_box_idx - _near_box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
			_near_box_idx_dim[0] = _near_box_idx % num_boxes[0];

			// Current near box origins
			for (int m = 0; m < 3; m++)
			{
				_near_box_origin[m] = _near_box_idx_dim[m] * box_size[m] + src_domain_range[m];
			}
 
		
			_local_box_idx2 = nufft_array->d_src_box_list_inv[_near_box_idx + 1];

			if( _tidx < interp_nodes_per_box){
				for(unsigned int j = 0; j < FIELD_DIM; j++ )			
					s_src_amp[_tidx+j*BLOCK_SIZE_CORRECT] = nufft_array->d_u_src_grid_dev[_local_box_idx2 * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box];
			}

			__syncthreads();

			///////////////////////////////////////////////////////////////////////////////////////
			//// Get the inaccurate near field.
			//// Source: near-list box grid points
			//// Observer: actual box grid points
			//// Interaction: direct Green's function
			///////////////////////////////////////////////////////////////////////////////////////// 
			for (int m = 0; m < 3; m++)
			{
				// _r1 is the bias between the left-top-front corner of the near-list box currently being handled and the actual box grid point.
				_r1[m] = _near_box_origin[m] - (_box_origin[m] + _obs_idx_dim[m] * cell_size[m]) ;
			}
 
			// cubic : 28% of computational time of this subroutine goes here
			 direct_grid_interact_cubic_static_vector(_r1, _Q1, s_src_amp + _shared_offset, cell_size, epsilon, BLOCK_SIZE_CORRECT);
			 __syncthreads();
	#ifdef _GPU_D_TEST
		_d_test[_gid] += _Q1[0];
	#endif
		} // _near_box_counter
		near_box_left -= BLOCK_SIZE_CORRECT;
		s_near_list_start += BLOCK_SIZE_CORRECT; 
	}//while(near_box_left > 0)

	__syncthreads();
#ifdef BEN_DEBUG
		//if( s_near_list_start != 0 )
			_d_test[_gid] = tmp_cnt;//s_near_box_list[_tidx];//_Q1[0];//
#endif	
	if( _tidx < interp_nodes_per_box){
		for(unsigned int j = 0; j < FIELD_DIM; j++ )
			nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box] -= _Q1[j];
	}


}
//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
#endif
#ifndef BEN_NEW_METHOD
__global__ void nufft_correct_static_vector_linear(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{ 
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned int _local_box_idx;
 	unsigned int _local_box_idx2;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
	unsigned int _obs_idx_dim[3];
	
	// Used only when a block handles multiple boxes
	int _shared_offset;

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 8;
	int const near_correct_layer = nufft_const->near_correct_layer;
	int const total_num_near_box_per_box = (2*near_correct_layer+1)*(2*near_correct_layer+1)*(2*near_correct_layer+1);
	//	int const problem_size = nufft_const->problem_size;
	FP_TYPE const epsilon = nufft_const->epsilon;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const cell_size = nufft_const->cell_size;
	FP_TYPE * const src_domain_range = nufft_const->src_domain_range;
 

	if (param->num_blk_per_box > 0)
	{
		_local_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;

		if (_obs_idx >= interp_nodes_per_box) return;
		_shared_offset = 0;

	}

	if (param->num_box_per_blk > 0) 
	{
		_local_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box; 
		_shared_offset = _tidx / interp_nodes_per_box * interp_nodes_per_box;

		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}
	
	if (_local_box_idx >= nufft_const->total_num_boxes_dev) //nufft_const->total_num_boxes is the number of boxes in current device
		return;
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];


	__shared__ FP_TYPE s_src_amp[BLOCK_SIZE_CORRECT*FIELD_DIM];
      
	FP_TYPE _Q1[FIELD_DIM]; // An array stores far field on observer
	for(unsigned int j = 0; j < FIELD_DIM; j++)	_Q1[j] = FP_TYPE(0.0f);

	FP_TYPE _r1[3];
	FP_TYPE _box_origin[3];
	int _near_box_idx = 0;
	int _near_box_idx_dim[3];
	FP_TYPE _near_box_origin[3];

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];

	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + src_domain_range[m];
	}

	idx_to_idx_dim_linear(_obs_idx_dim, _obs_idx);

	// Loop around all near boxes, presumbly 27
	/*for (int _near_box_counter = 0; _near_box_counter < nufft_array->d_near_box_list[_local_box_idx * total_num_near_box_per_box_p1]; _near_box_counter++)
	{*/
	for( int _near_box_counter = 0; _near_box_counter < total_num_near_box_per_box; _near_box_counter++){

		// For each box, d_NearBoxListThread has 28 entries (maybe empty), while the first entry is 
		// the total number of near boxes.
		//_near_box_idx = nufft_array->d_near_box_list[_near_box_counter + 1 + _local_box_idx * total_num_near_box_per_box_p1];
		_near_box_idx = near_cnt_to_dim( _near_box_idx_dim, _near_box_counter, _box_idx_dim, near_correct_layer, num_boxes);
		if( _near_box_idx < 0  || nufft_array->d_src_box_map[_near_box_idx + nufft_const->total_num_boxes * 2] == 0)
			continue;
		// Current near box index number
		/*_near_box_idx_dim[2] = _near_box_idx / (num_boxes[0] * num_boxes[1]);
		_near_box_idx_dim[1] =  (_near_box_idx - _near_box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
		_near_box_idx_dim[0] = _near_box_idx % num_boxes[0];*/

		// Current near box origins
		for (int m = 0; m < 3; m++)
		{
			_near_box_origin[m] = _near_box_idx_dim[m] * box_size[m] + src_domain_range[m];
		}
 
		// Field amplitudes on the source grid
		_local_box_idx2 = nufft_array->d_src_box_list_inv[_near_box_idx + 1];//d_src_box_list_inv has a length of total_num_boxes+1
		for(unsigned int j = 0; j < FIELD_DIM; j++ )			
			s_src_amp[_tidx+j*BLOCK_SIZE_CORRECT] = nufft_array->d_u_src_grid_dev[_local_box_idx2 * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box];

		__syncthreads();

		///////////////////////////////////////////////////////////////////////////////////////
		// Get the inaccurate near field.
		// Source: near-list box grid points
		// Observer: actual box grid points
		// Interaction: direct Green's function
		///////////////////////////////////////////////////////////////////////////////////////
		for (int m = 0; m < 3; m++)
		{
			// _r1 is the bias between the left-top-front corner of the near-list box currently being handled and the actual box grid point.
			_r1[m] = _near_box_origin[m] - (_box_origin[m] + _obs_idx_dim[m] * cell_size[m]) ;
		}
		// linear: 0% of computational time of this subroutine goes here
		direct_grid_interact_linear_static_vector(_r1, _Q1, s_src_amp + _shared_offset, cell_size, epsilon, BLOCK_SIZE_CORRECT);

	} // _near_box_counter

	__syncthreads();
	 
	// Field amplitudes on the observer grid
//	s_obs_amp[_tidx] = d_u_obs_grid[_obs_idx_glb] - _Q1[0];
//	s_obs_amp[_tidx] = d_u_obs_grid[_obs_idx_glb]; 

	for(unsigned int j = 0; j < FIELD_DIM; j++ )
		nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box] -= _Q1[j];

#ifdef BEN_DEBUG_MULTI
	//_d_test[_gid] = nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx];
#endif
#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;  
	_d_test[_gid] = _local_box_idx * interp_nodes_per_box + _obs_idx;
#endif

}
#else
__global__ void nufft_correct_static_vector_linear(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{ 
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned int _local_box_idx;
 	unsigned int _local_box_idx2;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
	unsigned int _obs_idx_dim[3];

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 8;
	int const near_correct_layer = nufft_const->near_correct_layer;
	int const total_num_near_box_per_box = (2*near_correct_layer+1)*(2*near_correct_layer+1)*(2*near_correct_layer+1);
	//	int const problem_size = nufft_const->problem_size;
	FP_TYPE const epsilon = nufft_const->epsilon;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const cell_size = nufft_const->cell_size;
	FP_TYPE * const src_domain_range = nufft_const->src_domain_range;
 

	_local_box_idx = _bid;
	_box_sub_idx = 0;
	_obs_idx = _tidx % interp_nodes_per_box; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	if (_local_box_idx >= nufft_const->total_num_boxes_dev) 
		return;

	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];

	__shared__ FP_TYPE s_src_amp[8];
	__shared__ int s_near_box_list[BLOCK_SIZE_CORRECT];
      
	FP_TYPE _Q1[FIELD_DIM]; // An array stores far field on observer
	for(unsigned int j = 0; j < FIELD_DIM; j++)	_Q1[j] = FP_TYPE(0.0f);

	FP_TYPE _r1[3];
	FP_TYPE _box_origin[3];
	int _near_box_idx = 0;
	int _near_box_idx_dim[3];
	FP_TYPE _near_box_origin[3];

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];

	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + src_domain_range[m];
	}

	idx_to_idx_dim_linear(_obs_idx_dim, _obs_idx);

	int near_box_left = total_num_near_box_per_box;
	int s_near_list_start = 0;
#ifdef BEN_DEBUG
	int tmp_cnt = 0;
#endif
	while( near_box_left > 0 ){
		if( _tidx < near_box_left ){
			s_near_box_list[_tidx] = near_to_glb_idx(s_near_list_start + _tidx, _bid, near_correct_layer, num_boxes);//_bid can be replaced by _box_idx_dim, because it should be only computed once
		}
		else{ // IS IT NECESSARY???
			s_near_box_list[_tidx] = -1;
		}
		__syncthreads();

		// Loop around all possible near boxes
		for( int _near_box_counter = 0; _near_box_counter < near_box_left && _near_box_counter < BLOCK_SIZE_CORRECT; _near_box_counter++){

			_near_box_idx = s_near_box_list[_near_box_counter];
			//_near_box_idx = near_cnt_to_dim( _near_box_idx_dim, _near_box_counter, _box_idx_dim, near_correct_layer, num_boxes);
				
			if( _near_box_idx < 0 )
				continue;
			//tmp_cnt ++;
			// Current near box index number
			_near_box_idx_dim[2] = _near_box_idx / (num_boxes[0] * num_boxes[1]);
			_near_box_idx_dim[1] =  (_near_box_idx - _near_box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
			_near_box_idx_dim[0] = _near_box_idx % num_boxes[0];

			// Current near box origins
			for (int m = 0; m < 3; m++)
			{
				_near_box_origin[m] = _near_box_idx_dim[m] * box_size[m] + src_domain_range[m];
			}

			// Field amplitudes on the source grid
			_local_box_idx2 = nufft_array->d_src_box_list_inv[_near_box_idx + 1];

			if( _tidx < interp_nodes_per_box){
				for(unsigned int j = 0; j < FIELD_DIM; j++ )			
					s_src_amp[_tidx+j*BLOCK_SIZE_CORRECT] = nufft_array->d_u_src_grid_dev[_local_box_idx2 * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box];
			}
			__syncthreads();

			///////////////////////////////////////////////////////////////////////////////////////
			// Get the inaccurate near field.
			// Source: near-list box grid points
			// Observer: actual box grid points
			// Interaction: direct Green's function
			///////////////////////////////////////////////////////////////////////////////////////
			for (int m = 0; m < 3; m++)
			{
				// _r1 is the bias between the left-top-front corner of the near-list box currently being handled and the actual box grid point.
				_r1[m] = _near_box_origin[m] - (_box_origin[m] + _obs_idx_dim[m] * cell_size[m]) ;
			}
			// linear: 0% of computational time of this subroutine goes here
			//if( _tidx < interp_nodes_per_box)
				direct_grid_interact_linear_static_vector(_r1, _Q1, s_src_amp, cell_size, epsilon, BLOCK_SIZE_CORRECT);
				__syncthreads();
		} // _near_box_counter
		near_box_left -= BLOCK_SIZE_CORRECT;
		s_near_list_start += BLOCK_SIZE_CORRECT; 
	} //while( near_box_left > 0 )
	__syncthreads();

#ifdef BEN_DEBUG
		//if( s_near_list_start != 0 )
			_d_test[_gid] = tmp_cnt;//s_near_box_list[_tidx];//_Q1[0];//
#endif	

	// Field amplitudes on the observer grid
//	s_obs_amp[_tidx] = d_u_obs_grid[_obs_idx_glb] - _Q1[0];
//	s_obs_amp[_tidx] = d_u_obs_grid[_obs_idx_glb]; 
	if( _tidx < interp_nodes_per_box){
		for(unsigned int j = 0; j < FIELD_DIM; j++ )
			nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box] -= _Q1[j];
	}
 
#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;  
	_d_test[_gid] = _local_box_idx * interp_nodes_per_box + _obs_idx;
#endif

}
#endif
__global__ void nufft_interp_static_vector_cubic(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{ 
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
 	unsigned int _local_box_idx;
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
//	unsigned int _obs_idx_glb;
	unsigned int _obs_idx_dim[3];
	
	// Used only when a block handles multiple boxes
	int _shared_offset;

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 64;
	unsigned int const total_num_boxes = nufft_const->total_num_boxes;
	unsigned int const total_num_boxes_dev = nufft_const->total_num_boxes_dev;
	int const obs_size_dev = nufft_const->obs_size_dev;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const obs_domain_range = nufft_const->obs_domain_range;
 
	if (param->num_blk_per_box > 0)
	{
		_local_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;

		if (_obs_idx >= interp_nodes_per_box) return;
		_shared_offset = 0;

	}

	if (param->num_box_per_blk > 0) 
	{
		_local_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box;
		_shared_offset = _tidx / interp_nodes_per_box * interp_nodes_per_box;

		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}

	if (_local_box_idx >= total_num_boxes_dev) return;	
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];

	__shared__ FP_TYPE s_obs_coord[BLOCK_SIZE_PROJ_INTERP * 3];
	__shared__ FP_TYPE s_obs_amp[BLOCK_SIZE_PROJ_INTERP*FIELD_DIM];

	FP_TYPE _Q1[FIELD_DIM]; // An array stores far field on observer
	for(unsigned int j = 0; j < FIELD_DIM; j++)	_Q1[j] = 0.0f;

	FP_TYPE _r_norm[3];
	int _glb_src_idx_start2 = 0;
	int _num_src2 = 0;
	FP_TYPE _box_origin[3];

	int _shared_start2;

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];
 
	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + obs_domain_range[m];
	}
	
	idx_to_idx_dim_cubic(_obs_idx_dim, _obs_idx);

//	 Field amplitudes on the observer grid, with inaccurate near field subtracted
	for(unsigned int j = 0; j < FIELD_DIM; j++ )	
		s_obs_amp[_tidx+j*BLOCK_SIZE_PROJ_INTERP] = nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box]; 
	__syncthreads();

	// Number of observers in the current box

	_num_src2 = nufft_array->d_obs_box_map[_box_idx + 2 * total_num_boxes];
	_glb_src_idx_start2 = nufft_array->d_obs_box_map[_box_idx];
	_shared_start2 = 0; 

	while (_shared_start2 < _num_src2) // loop around all sources
	{

		///////////////////////////////////////////////////////////////////////////////////////
		// Doing interpolations from grid to actual observers
		///////////////////////////////////////////////////////////////////////////////////////

	//	// Calculating far field, interpolation from the grid points to the observers
		for(unsigned int j = 0; j < FIELD_DIM; j++ ) _Q1[j] = 0.0f;

		if (_shared_start2 + _obs_idx < _num_src2)
		{
			for (int m = 0; m < 3; m++)
			{
				// Unnormalized coordinates of observers, used by near field calculation
				s_obs_coord[_tidx + m * _bdim] = nufft_array->d_obs_coord[_glb_src_idx_start2 + _shared_start2 + _obs_idx + obs_size_dev * m];

				// Normalized coordinates of observers, used by far field interpolation
				_r_norm[m] = (s_obs_coord[_tidx + m * _bdim] - _box_origin[m]) / box_size[m];
			} // m
		} // 

		lagrange_interp_cubic_vector(_Q1, _r_norm, s_obs_amp + _shared_offset, BLOCK_SIZE_PROJ_INTERP);

		if (_shared_start2 + _obs_idx < _num_src2)
		{
			for(unsigned int j = 0; j < FIELD_DIM; j++ )
				nufft_array->d_field_amp[_glb_src_idx_start2 + _shared_start2 + _obs_idx + j*obs_size_dev] = _Q1[j];
		}

		__syncthreads();
		_shared_start2 += interp_nodes_per_box;

	} // shared_start2
#ifdef _GPU_D_TEST
	_d_test[_gid] = 0.00f;  
	_d_test[_gid] =s_obs_amp[_tidx];
#endif


}
__global__ void nufft_interp_static_vector_linear(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{  
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
 	unsigned int _local_box_idx;	 
	unsigned int _box_idx;
	unsigned int _box_idx_dim[3];
	unsigned int _box_sub_idx = 0;
	unsigned int _obs_idx;
//	unsigned int _obs_idx_glb;
	unsigned int _obs_idx_dim[3];
	
	// Used only when a block handles multiple boxes
	int _shared_offset;

	// Local copies of global constant variables
	unsigned int const interp_nodes_per_box = 8;
	unsigned int const total_num_boxes = nufft_const->total_num_boxes;
	unsigned int const total_num_boxes_dev = nufft_const->total_num_boxes_dev;
//	int const problem_size = nufft_const->problem_size;
//	int const src_size_dev = nufft_const->src_size_dev;
	int const obs_size_dev = nufft_const->obs_size_dev;
	int * const num_boxes = nufft_const->num_boxes;
	FP_TYPE * const box_size = nufft_const->box_size;
	FP_TYPE * const obs_domain_range = nufft_const->obs_domain_range;


	if (param->num_blk_per_box > 0)
	{
		_local_box_idx = _bid / param->num_blk_per_box;
		_box_sub_idx = _bid % param->num_blk_per_box;
		_obs_idx = _tidx + _box_sub_idx * _bdim;

		if (_obs_idx >= interp_nodes_per_box) return;
		_shared_offset = 0;

	}

	if (param->num_box_per_blk > 0) 
	{
		_local_box_idx = _bid * param->num_box_per_blk + _tidx / interp_nodes_per_box;
		_box_sub_idx = 0;
		_obs_idx = _tidx % interp_nodes_per_box;
		_shared_offset = _tidx / interp_nodes_per_box * interp_nodes_per_box;

		if (_tidx >= interp_nodes_per_box * param->num_box_per_blk) return;
	}
	
	if (_local_box_idx >= total_num_boxes_dev) return;	
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];

	__shared__ FP_TYPE s_obs_coord[BLOCK_SIZE_PROJ_INTERP * 3];
	__shared__ FP_TYPE s_obs_amp[BLOCK_SIZE_PROJ_INTERP * FIELD_DIM];

	FP_TYPE _Q1[FIELD_DIM]; // An array stores far field on observer
	for(unsigned int j = 0; j < FIELD_DIM; j++ ) _Q1[j] = 0.0f;

	FP_TYPE _r_norm[3];
	int _glb_src_idx_start2 = 0;
	int _num_src2 = 0;
	FP_TYPE _box_origin[3];

	int _shared_start2;

	// Get the index number of the observer box 
	_box_idx_dim[2] = _box_idx / (num_boxes[0] * num_boxes[1]);
	_box_idx_dim[1] =  (_box_idx - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];
 
	for (int m = 0; m < 3; m++)
	{
		_box_origin[m] = _box_idx_dim[m] * box_size[m] + obs_domain_range[m];
	}
	
	idx_to_idx_dim_linear(_obs_idx_dim, _obs_idx);

	// Field amplitudes on the observer grid
	for(unsigned int j = 0; j < FIELD_DIM; j++ )	
		s_obs_amp[_tidx+j*BLOCK_SIZE_PROJ_INTERP] = nufft_array->d_u_obs_grid_dev[_local_box_idx * interp_nodes_per_box + _obs_idx + j*nufft_const->total_num_boxes_dev*interp_nodes_per_box]; 	
	__syncthreads();

	///////////////////////////////////////////////////////////////////////////////////////
	// The following is a combined loop to get far field
	// Sources: far-field grid
	// Observers: actual observers
	// Operations: Lagrange interpolation
	///////////////////////////////////////////////////////////////////////////////////////	

	// Number of observers in the current box

	_num_src2 = nufft_array->d_obs_box_map[_box_idx + 2 * total_num_boxes];

	_glb_src_idx_start2 = nufft_array->d_obs_box_map[_box_idx];

	_shared_start2 = 0; 
	while (_shared_start2 < _num_src2)
	{

		///////////////////////////////////////////////////////////////////////////////////////
		// Doing interpolations from grid to actual observers
		///////////////////////////////////////////////////////////////////////////////////////

		// Calculating far field, interpolation from the grid points to the observers

		for(unsigned int j = 0; j < FIELD_DIM; j++ ) _Q1[j] = 0.0f;

		if (_shared_start2 + _obs_idx < _num_src2)
		{
			for (int m = 0; m < 3; m++)
			{				// Unnormalized coordinates of observers, used by near field calculation
				s_obs_coord[_tidx + m * _bdim] = nufft_array->d_obs_coord[_glb_src_idx_start2 + _shared_start2 + _obs_idx + obs_size_dev * m];

				// Normalized coordinates of observers, used by far field interpolation
				_r_norm[m] = (s_obs_coord[_tidx + m * _bdim] - _box_origin[m]) / box_size[m];
			}
		}

		lagrange_interp_linear_vector(_Q1, _r_norm, s_obs_amp + _shared_offset, BLOCK_SIZE_PROJ_INTERP);

		if (_shared_start2 + _obs_idx < _num_src2)
		{
			for(unsigned int j = 0; j < FIELD_DIM; j++ )
				nufft_array->d_field_amp[_glb_src_idx_start2 + _shared_start2 + _obs_idx + j*obs_size_dev] = _Q1[j];
		}

		__syncthreads();
		_shared_start2 += interp_nodes_per_box;
	} // shared_start2
#ifdef BEN_DEBUG_MULTI
	//_d_test[_gid] = _Q1[0];
#endif
#ifdef _GPU_D_TEST
	_d_test[_gid] = 9.99f;  
	_d_test[_gid] = _Q1[0];
#endif


}
//BEN ELIMINATED THE D_NEAR_BOX_LIST IN THIS SUBROUTINE
//THE MULTIPROCESSOR OCCUPANCY LIMITATION IS REGISTER=40 -> 75% OCCUPANCY. REGISTER = 32 WILL LEAD TO 100% OCCUPANCY
__global__ void nufft_exact_near_field(FP_TYPE *_d_test, NufftArrayGpuStaticVector *nufft_array, NufftParamGpu *nufft_const, GpuExecParam *param)
{  
	unsigned int _tidx = threadIdx.x;
	unsigned int _bdim = blockDim.x;
	unsigned int _bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int _gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
 	unsigned int _local_box_idx;
	unsigned int _box_idx;
	int const _block_size = BLOCK_SIZE_NEAR;
	int const src_size_dev = nufft_const->src_size_dev;
	int const obs_size_dev = src_size_dev;
	//	int const obs_size_dev = nufft_const->obs_size_dev;
	FP_TYPE const epsilon = nufft_const->epsilon;

	_local_box_idx = _bid;

	if (_local_box_idx >= nufft_const->total_num_boxes_dev) return;	
	_box_idx = nufft_array->d_src_box_list[_local_box_idx + 1];
	
	__shared__ FP_TYPE s_src_coord[BLOCK_SIZE_NEAR * 3];
	__shared__ FP_TYPE s_obs_coord[BLOCK_SIZE_NEAR * 3];
	__shared__ FP_TYPE s_src_amp[BLOCK_SIZE_NEAR * FIELD_DIM];
	__shared__ int s_near_box_list[BLOCK_SIZE_NEAR];

	//Stores near field on observer
	FP_TYPE _Q2[FIELD_DIM];
	for(unsigned int j = 0; j < FIELD_DIM; j++)	_Q2[j] = 0.0f;

	FP_TYPE _r1[3];
	int _glb_src_idx_start2 = 0;
	int _num_src2 = 0;
	FP_TYPE _magn[FIELD_DIM];

	int _shared_start2 = 0;
	//BEN ADDED THESE TWO PARAMETERS FOR NEAR BOX LIST
	unsigned int const near_correct_layer = nufft_const->near_correct_layer;
	int _total_num_near_box = (2*near_correct_layer+1)*(2*near_correct_layer+1)*(2*near_correct_layer+1);
	int _near_box_idx = 0;
	//int _total_near_box = nufft_array->d_near_box_list[_local_box_idx * total_num_near_box_per_box_p1];

	int _near_box_counter = 0;
	int _g_addr_start = 0; 
	int _s_addr_start = 0; 
	int _s_addr_end = 0; 
	int _near_src = 0;
	bool _fetch;


	/////////////////////////////////////////////////////////////////////////////////////
	// The following is a loop to get accurate near field on observers
	// Sources: sources in near-field boxes
	// Observers: actual observers
	// Pay attention that the sources in either source box or observer box or both 
	// can be greater than number of thread per box
	///////////////////////////////////////////////////////////////////////////////////////	

	// Number of observers in the current box

	_num_src2 = nufft_array->d_obs_box_map[_box_idx + 2 * nufft_const->total_num_boxes];
	_glb_src_idx_start2 = nufft_array->d_obs_box_map[_box_idx];

	while (_shared_start2 < _num_src2)//if shared start is smaller than the num of srcs inside current box
									//empty src box will stop working here
	{
#ifdef _GPU_D_TEST
	_d_test[_gid] = 0.00f;  
#endif 
		for(unsigned int j = 0; j < FIELD_DIM; j++)	_Q2[j] = 0.0f;

		if (_shared_start2 + _tidx < _num_src2)
		{
			for (int m = 0; m < 3; m++)
			{
				// Unnormalized coordinates of observers, used by near field calculation
				s_obs_coord[_tidx + m * _bdim] = nufft_array->d_obs_coord[_glb_src_idx_start2 + _shared_start2 + _tidx + obs_size_dev * m];
			} // m
		} // 

		int s_near_list_start = 0;
		///////////////////////////////////////////////////////////////////////////////////
		// Doing direct summation to get accurate near field
		/////////////////////////////////////////////////////////////////////////////////////
		while( s_near_list_start < _total_num_near_box ){

			if( s_near_list_start + _tidx < _total_num_near_box ){
				_near_box_idx = near_to_glb_idx(s_near_list_start + _tidx, _box_idx, near_correct_layer, nufft_const->num_boxes);//if out of computational domain, return -1
				if( _near_box_idx < 0 )
					s_near_box_list[_tidx] = -1;
				else{
					if( nufft_array->d_obs_box_map[_near_box_idx + 2 * nufft_const->total_num_boxes] > 0)//this function has not been tested
						s_near_box_list[_tidx] = _near_box_idx;
					else
						s_near_box_list[_tidx] = -1;
				}
			}
			else{ // because this is checked inside the inner while loop
				s_near_box_list[_tidx] = -1;
			}
			
			__syncthreads();

			_near_box_counter = -1;
			_g_addr_start = 0; 
			_s_addr_start = 0; 
			_s_addr_end = 0; 
			_fetch = false;

			// The grand loop
			while (_s_addr_end >= 0)//fetch all near srcs
			{
				__syncthreads();
				if (_tidx < _s_addr_end) 
				{
					for (int m = 0; m < 3; m++)
					{
						s_src_coord[_tidx + m * _block_size] = nufft_array->d_obs_coord[_g_addr_start + _tidx + src_size_dev * m];
					} // m
					for(unsigned int j = 0; j < FIELD_DIM; j++)	
						s_src_amp[_tidx+j*BLOCK_SIZE_NEAR] = nufft_array->d_src_amp[_g_addr_start + _tidx + j*src_size_dev];
				}


#ifdef _GPU_D_TEST
				for (int i = _s_addr_start; i < _block_size && i < _s_addr_end; i++)
				{
					_d_test[_gid] += s_src_amp[i];
				}
#endif
				// ready to move on to next near box?
				//NEED MORE CONSIDERATION IF _near_box_counter needs the last two conditions!!!//////////////////////s_near_list_start + _tidx < _total_num_near_box
				if (_s_addr_end - _block_size < 0 && _near_box_counter + 1 < _block_size && _near_box_counter+1 <= _total_num_near_box-s_near_list_start)//_near_box_counter + 1 < _total_num_near_box && 
				{
					_fetch = true;
				}

				while (_fetch)// until shared mem is full or run out of near box
				{

					// fetch new boxes until the shared memory is full
					// update status variables
					_near_box_counter++;//move on to next near box
					//_near_box_idx = nufft_array->d_near_box_list[_near_box_counter + 1 + _local_box_idx * total_num_near_box_per_box_p1];
					_near_box_idx = s_near_box_list[_near_box_counter];
					if( _near_box_counter + 1 >= _block_size || _near_box_counter + 1 == _total_num_near_box-s_near_list_start)//_near_box_counter+1 >= _total_num_near_box || SHOULD CHECK IT IS LARGER THAN BLOCK SIZE??
						_fetch = false;
					if(_near_box_idx < 0)
						continue;
					//_near_src: total num of srcs in the near box
					_near_src = nufft_array->d_obs_box_map[_near_box_idx + nufft_const->total_num_boxes * 2];
					//_g_addr_start: start index of srcs in the near box
					_g_addr_start = nufft_array->d_obs_box_map[_near_box_idx];
					//record the start and end index in the context of shared memory
					_s_addr_start = _s_addr_end;
					_s_addr_end = _s_addr_end + _near_src;

					// Shared memory is full
					if (_s_addr_end >= _block_size)// || _near_box_counter + 1 >= _total_num_near_box
					{
						_fetch = false;
					}

					if (_tidx >= _s_addr_start && _tidx < _s_addr_end) 
					{
						for (int m = 0; m < 3; m++)
						{
							s_src_coord[_tidx + m * _block_size] = nufft_array->d_obs_coord[_g_addr_start + _tidx - _s_addr_start + src_size_dev * m];
						} // m
						for(unsigned int j = 0; j < FIELD_DIM; j++)
							s_src_amp[_tidx+j*BLOCK_SIZE_NEAR] = nufft_array->d_src_amp[_g_addr_start - _s_addr_start + _tidx + j*src_size_dev];
					}
				} // fetch
				__syncthreads();

				// do calculations
				for (int i = 0; i < _block_size && i < _s_addr_end; i++)
				{
					for (int m = 0; m < 3; m++)
					{
						_r1[m] = s_obs_coord[_tidx + m * _block_size] - s_src_coord[i + m * _block_size];
					} // m
					for(unsigned int j = 0; j < FIELD_DIM; j++) _magn[j] = s_src_amp[i+j*BLOCK_SIZE_NEAR];

					get_field_static_vector(_r1, _magn, _Q2, epsilon);
				} // i 

				_s_addr_end -= _block_size;
				_g_addr_start += _near_src - _s_addr_end > _block_size ? _block_size : _near_src - _s_addr_end;
			} // _loop to fetch more srcs into shared memory
//#ifdef BEN_DEBUG
//			_d_test[_gid] = _total_num_near_box-s_near_list_start;
//#endif
			s_near_list_start += BLOCK_SIZE_NEAR;
		}//while loop to fetch more near boxes
		if (_shared_start2 + _tidx < _num_src2)
		{
			for(unsigned int j = 0; j < FIELD_DIM; j++)
				nufft_array->d_field_amp[_glb_src_idx_start2 + _shared_start2 + _tidx + j*obs_size_dev] += _Q2[j];
		} // _loop to fetch more obs into shared memory

		__syncthreads();
		_shared_start2 += _block_size;

	} // shared_start2
} 
}
