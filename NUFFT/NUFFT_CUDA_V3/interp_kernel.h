/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* interp_kernel.h: class declaration of Class Gpu
*/
#ifndef _INTERP_KERNEL
#define _INTERP_KERNEL
#include "fp_precision.h"
// Lagrange interpolation coefficient
__device__ const FP_TYPE _lagrange_coeff_c[4][4] = {1.0f, -5.5f, 9.0f, -4.5f, 0.0f, 9.0f, -22.5f, 13.5f, 0.0f, -4.5f, 18.0f, -13.5f, 0.0f, 1.0f, -4.5f, 4.5f};
__device__ const FP_TYPE _lagrange_coeff_l[2][2] = {1.0f, -1.0f, 0.0f, 1.0f};
// These are inline functions inside GPU kernels to make the code reads better
namespace NBODYFAST_NS{
// this is the Green's function of static scalar field, it has to be the same as FieldStaticScalar::get_G()
inline __device__ void get_field_static_scalar(const FP_TYPE *r, const FP_TYPE *magn, FP_TYPE *Q1, const FP_TYPE epsilon)
{
	FP_TYPE r_amp2, inv_r_amp1;
	r_amp2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

	if (r_amp2 > epsilon)	
	{
		inv_r_amp1 = rsqrtf(r_amp2);
		Q1[0] += inv_r_amp1 * magn[0];
	}
}
inline __device__ void get_field_static_vector(const FP_TYPE *r, const FP_TYPE *magn, FP_TYPE *Q1, const FP_TYPE epsilon)
{
	FP_TYPE r_amp2, inv_r_amp1,inv_r_amp3,inv_r_amp5;
	FP_TYPE mdotr = FP_TYPE(0.f);
	for(unsigned int j = 0; j < FIELD_DIM; j++)	mdotr += r[j]*magn[j];
	r_amp2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

	if (r_amp2 > epsilon)	
	{
		inv_r_amp1 = rsqrtf(r_amp2);
		inv_r_amp3 = inv_r_amp1 * inv_r_amp1 * inv_r_amp1;
		inv_r_amp5 = inv_r_amp3 * inv_r_amp1 * inv_r_amp1;
		for(unsigned int j =0; j< FIELD_DIM; j++)	
			Q1[j] += 3 * mdotr * r[j] * inv_r_amp5 - magn[j] * inv_r_amp3;
	}
}

inline __device__ int near_to_glb_idx(int _near_idx, int _box_idx, int near_correct_layer, int* num_boxes){
	int _near_box_idx_dim[3];
	int _box_idx_dim[3];
	int near_num = 2*near_correct_layer+1;

	_near_box_idx_dim[2] = _near_idx / (near_num*near_num) - near_correct_layer;
	_near_box_idx_dim[1] = (_near_idx % (near_num*near_num)) / near_num - near_correct_layer;
	_near_box_idx_dim[0] = _near_idx % near_num - near_correct_layer;

	_box_idx_dim[2] = _box_idx / (num_boxes[0]*num_boxes[1]);
	_box_idx_dim[1] = (_box_idx - _box_idx_dim[2]*(num_boxes[0]*num_boxes[1])) / num_boxes[0];
	_box_idx_dim[0] = _box_idx % num_boxes[0];
	// if current near box is not valid, go to the next
	if (_box_idx_dim[0] +  _near_box_idx_dim[0] < 0 || _box_idx_dim[1] +  _near_box_idx_dim[1]  < 0 || _box_idx_dim[2] +  _near_box_idx_dim[2]  < 0 ||
		_box_idx_dim[0] +  _near_box_idx_dim[0] >= num_boxes[0] || _box_idx_dim[1] + _near_box_idx_dim[1] >= num_boxes[1] || _box_idx_dim[2] +  _near_box_idx_dim[2] >= num_boxes[2]) 
	{
		return -1;
	}
	// if current near box presents but number of sources inside is 0, treat it as not presenting
	return _box_idx + (_near_box_idx_dim[0]) + (_near_box_idx_dim[1]) * num_boxes[0] + (_near_box_idx_dim[2]) * num_boxes[0] * num_boxes[1]; 
}
inline __device__ int near_cnt_to_dim( int* _near_box_idx_dim, int _near_idx, unsigned int* _box_idx_dim, int near_correct_layer, int* num_boxes){
	int _near_box_idx_dim_tmp[3];
	int near_num = 2*near_correct_layer+1;

	_near_box_idx_dim_tmp[2] = _near_idx / (near_num*near_num) - near_correct_layer;
	_near_box_idx_dim_tmp[1] = (_near_idx % (near_num*near_num)) / near_num - near_correct_layer;
	_near_box_idx_dim_tmp[0] = _near_idx % near_num - near_correct_layer;

	for( int m = 0; m < 3; m++ ){
		_near_box_idx_dim[m] = _box_idx_dim[m] + _near_box_idx_dim_tmp[m];
		// if current near box is not valid, go to the next
		if( _near_box_idx_dim[m] < 0 || _near_box_idx_dim[m] >= num_boxes[m] )
			return -1;
	}
	return _near_box_idx_dim[0] + _near_box_idx_dim[1]*num_boxes[0] + _near_box_idx_dim[2]*num_boxes[0]*num_boxes[1];
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Cubic version of inline functions

// transfer 1D index to 3D index
inline __device__ void idx_to_idx_dim_cubic(unsigned int *_idx_dim, unsigned int _idx)
{
	_idx_dim[2] = _idx >> 4;
	_idx_dim[1] = ((_idx & 15) >> 2);
	_idx_dim[0] = _idx & 3;
}

// Lagrange projection, cubic order
inline __device__ void lagrange_project_cubic(FP_TYPE& coeff, FP_TYPE r[3], unsigned int * _obs_idx_dim)
{

	FP_TYPE _interp_coeff_dim[3];
	FP_TYPE r1, r2, r3;
	for (int m = 0; m < 3; m++)
	{
		r1 = r[m];
		r2 = r1 * r1;
		r3 = r1 * r2;
		_interp_coeff_dim[m] =  _lagrange_coeff_c[_obs_idx_dim[m]][3] * r3 + _lagrange_coeff_c[_obs_idx_dim[m]][2] * r2 + _lagrange_coeff_c[_obs_idx_dim[m]][1] * r1 + _lagrange_coeff_c[_obs_idx_dim[m]][0];
	}

	coeff = _interp_coeff_dim[0] * _interp_coeff_dim[1] * _interp_coeff_dim[2];

}
// Lagrange interpolation, cubic order, for scalar case
inline __device__ void lagrange_interp_cubic(FP_TYPE *_Q1, FP_TYPE *r, FP_TYPE *_src_amp)
{

	FP_TYPE _r1, _r2, _r3;
	FP_TYPE _interp_coeff_dim[12];
	FP_TYPE _interp_coeff;

	for (int m = 0; m < 3; m++)
	{
		_r1 = r[m];
		_r2 = _r1 * _r1;
		_r3 = _r2 * _r1;
		for (int k = 0; k < 4; k++)
		{
			_interp_coeff_dim[m * 4 + k] =  _lagrange_coeff_c[k][3] * _r3 + _lagrange_coeff_c[k][2] * _r2 + _lagrange_coeff_c[k][1] * _r1 + _lagrange_coeff_c[k][0];
		}
	}
	for (int i = 0; i < 64; i++)
	{
		unsigned int _ridx = i;
		unsigned int _ridx_dim[3];
		idx_to_idx_dim_cubic(_ridx_dim, _ridx);
		_interp_coeff = _interp_coeff_dim[_ridx_dim[0]] * _interp_coeff_dim[_ridx_dim[1] + 4] * _interp_coeff_dim[_ridx_dim[2] + 8];
		
		_Q1[0] += _interp_coeff * _src_amp[_ridx];  
	}
}
// Lagrange interpolation, cubic order, for vector case
inline __device__ void lagrange_interp_cubic_vector(FP_TYPE *_Q1, FP_TYPE *r, FP_TYPE *_src_amp, const int _src_amp_size)
{

	FP_TYPE _r1, _r2, _r3;
	FP_TYPE _interp_coeff_dim[12];
	FP_TYPE _interp_coeff;

	for (int m = 0; m < 3; m++)
	{
		_r1 = r[m];
		_r2 = _r1 * _r1;
		_r3 = _r2 * _r1;
		for (int k = 0; k < 4; k++)
		{
			_interp_coeff_dim[m * 4 + k] =  _lagrange_coeff_c[k][3] * _r3 + _lagrange_coeff_c[k][2] * _r2 + _lagrange_coeff_c[k][1] * _r1 + _lagrange_coeff_c[k][0];
		}
	}
	for (int i = 0; i < 64; i++)
	{
		unsigned int _ridx = i;
		unsigned int _ridx_dim[3];
		idx_to_idx_dim_cubic(_ridx_dim, _ridx);
		_interp_coeff = _interp_coeff_dim[_ridx_dim[0]] * _interp_coeff_dim[_ridx_dim[1] + 4] * _interp_coeff_dim[_ridx_dim[2] + 8];
		
		for(unsigned int j = 0; j < FIELD_DIM; j++)
			_Q1[j] += _interp_coeff * _src_amp[_ridx+j*_src_amp_size];  
	}
}


// 3D observer grid index (in local box) to 1D observer grid index (global)
inline __device__ void obs_idx_dim_to_obs_idx_glb_cubic(unsigned int& obs_idx_glb, unsigned int* _box_idx_dim, unsigned int* _obs_idx_dim, int*fft_size)
{
	obs_idx_glb = (_box_idx_dim[2]  * 3 + _obs_idx_dim[2]) * fft_size[1] * fft_size[0] +
	(_box_idx_dim[1]  * 3 + _obs_idx_dim[1]) * fft_size[0] +
	_box_idx_dim[0] * 3 + _obs_idx_dim[0];
}

// 3D observer grid index (global) to 3D box index and 3D local grid index
inline __device__ void grid_idx_to_local_idx_cubic(int *_grid_idx_dim, int *_box_idx_dim, int *_local_idx_dim, int *_num_boxes)
{
	for (int k = 0; k < 3; k++)
	{
		if (_grid_idx_dim[k] % 3 == 0)
		{
			_box_idx_dim[k * 2] = _grid_idx_dim[k] / 3 - 1;
			_box_idx_dim[k * 2 + 1] = _grid_idx_dim[k] / 3;

			_local_idx_dim[k * 2] = 3;
			_local_idx_dim[k * 2 + 1] = 0;

			if (_box_idx_dim[k * 2 + 1] > _num_boxes[k] - 1) _box_idx_dim[k * 2 + 1] = -1;
		}
		else
		{
			_box_idx_dim[k * 2] = -1;
			_box_idx_dim[k * 2 + 1] = _grid_idx_dim[k] / 3;

			_local_idx_dim[k * 2] = -1;
			_local_idx_dim[k * 2 + 1] = _grid_idx_dim[k] % 3;
		}
	}

}

// 3D local grid inex to 1D local grid index
inline __device__ void local_idx_dim_to_local_idx_cubic(int* _local_idx, int *_local_idx_dim_temp)
{
	*_local_idx = _local_idx_dim_temp[0] + (_local_idx_dim_temp[1] << 2) + (_local_idx_dim_temp[2] << 4);
}
// direct field calculation between near-field grid to near-field grid
inline __device__ void direct_grid_interact_cubic_static_scalar(const FP_TYPE *_r1, FP_TYPE *_Q1, const FP_TYPE *s_obs_amp,
	const FP_TYPE *_cell_size, const FP_TYPE _epsilon)
{

	unsigned int _near_obs_idx_dim[3];
	FP_TYPE _r2[3];
	FP_TYPE _magn[1];

	for (int i = 0; i < 64; i++)
	{
		int _near_obs_idx = i; // & 63;

		idx_to_idx_dim_cubic(_near_obs_idx_dim, _near_obs_idx);
		for (int m = 0; m < 3; m++)
		{		
			_r2[m] = _r1[m] + _near_obs_idx_dim[m] * _cell_size[m];
		}
		_magn[0] = s_obs_amp[i];
		get_field_static_scalar(_r2, _magn, _Q1, _epsilon);
	} // i 
}
// direct field calculation between near-field grid to near-field grid, for vector case
inline __device__ void direct_grid_interact_cubic_static_vector(const FP_TYPE *_r1, FP_TYPE *_Q1, const FP_TYPE *s_obs_amp,
	const FP_TYPE *_cell_size, const FP_TYPE _epsilon, const int s_obs_amp_size)
{

	unsigned int _near_obs_idx_dim[3];
	FP_TYPE _r2[3];
	FP_TYPE _magn[FIELD_DIM];

	for (int i = 0; i < 64; i++)
	{
		int _near_obs_idx = i; // & 63;

		idx_to_idx_dim_cubic(_near_obs_idx_dim, _near_obs_idx);
		for (int m = 0; m < 3; m++)
		{		
			_r2[m] = _r1[m] + _near_obs_idx_dim[m] * _cell_size[m];
		}
		for(unsigned int j = 0; j < FIELD_DIM; j++)	_magn[j] = s_obs_amp[i+j*s_obs_amp_size];
		get_field_static_vector(_r2, _magn, _Q1, _epsilon);
	} // i 
}






/////////////////////////////////////////////////////////////////////////////////////////////////
// Linear version of inline functions
inline __device__ void idx_to_idx_dim_linear(unsigned int *_idx_dim, unsigned int _idx)
{
	_idx_dim[2] = _idx >> 2;
	_idx_dim[1] = ((_idx & 3) >> 1);
	_idx_dim[0] = _idx & 1;
}
inline __device__ void lagrange_project_linear(FP_TYPE& coeff, FP_TYPE r[3], unsigned int * _obs_idx_dim)
{
	FP_TYPE _interp_coeff_dim[3];
	FP_TYPE r1;
	for (int m = 0; m < 3; m++)
	{
		r1 = r[m];
		_interp_coeff_dim[m] =  _lagrange_coeff_l[_obs_idx_dim[m]][1] * r1 + _lagrange_coeff_l[_obs_idx_dim[m]][0];
	}

	coeff = _interp_coeff_dim[0] * _interp_coeff_dim[1] * _interp_coeff_dim[2];
}


inline __device__ void lagrange_interp_linear(FP_TYPE *_Q1, FP_TYPE *r, FP_TYPE *_src_amp)
{

	FP_TYPE _r1;
	FP_TYPE _interp_coeff_dim[6];
	FP_TYPE _interp_coeff;

	unsigned int  _ridx_dim[3];
	for (int m = 0; m < 3; m++)
	{
		_r1 = r[m];
		for (int k = 0; k < 2; k++)
		{
			_interp_coeff_dim[m * 2 + k] =  _lagrange_coeff_l[k][1] * _r1 + _lagrange_coeff_l[k][0];
		}
	}
	for (int i = 0; i < 8; i++)
	{
		unsigned int _ridx = i;

		idx_to_idx_dim_linear(_ridx_dim, _ridx);
		_interp_coeff = _interp_coeff_dim[_ridx_dim[0]] * _interp_coeff_dim[_ridx_dim[1] + 2] * _interp_coeff_dim[_ridx_dim[2] + 4];
		
		_Q1[0] += _interp_coeff * _src_amp[_ridx];  
	}
}
inline __device__ void lagrange_interp_linear_vector(FP_TYPE *_Q1, FP_TYPE *r, FP_TYPE *_src_amp, const int _src_amp_size)
{

	FP_TYPE _r1;
	FP_TYPE _interp_coeff_dim[6];
	FP_TYPE _interp_coeff;

	unsigned int  _ridx_dim[3];
	for (int m = 0; m < 3; m++)
	{
		_r1 = r[m];
		for (int k = 0; k < 2; k++)
		{
			_interp_coeff_dim[m * 2 + k] =  _lagrange_coeff_l[k][1] * _r1 + _lagrange_coeff_l[k][0];
		}
	}
	for (int i = 0; i < 8; i++)
	{
		unsigned int _ridx = i;

		idx_to_idx_dim_linear(_ridx_dim, _ridx);
		_interp_coeff = _interp_coeff_dim[_ridx_dim[0]] * _interp_coeff_dim[_ridx_dim[1] + 2] * _interp_coeff_dim[_ridx_dim[2] + 4];
		
		for(unsigned int j = 0; j < FIELD_DIM; j++)
			_Q1[j] += _interp_coeff * _src_amp[_ridx+j*_src_amp_size];  
	}
}


inline __device__ void obs_idx_dim_to_obs_idx_glb_linear(unsigned int& obs_idx_glb, unsigned int* _box_idx_dim, unsigned int* _obs_idx_dim, int*fft_size)
{
	obs_idx_glb = (_box_idx_dim[2]  * 1 + _obs_idx_dim[2]) * fft_size[1] * fft_size[0] +
	(_box_idx_dim[1]  * 1 + _obs_idx_dim[1]) * fft_size[0] +
	_box_idx_dim[0] * 1 + _obs_idx_dim[0];
}


inline __device__ void grid_idx_to_local_idx_linear(int *_grid_idx_dim, int *_box_idx_dim, int *_local_idx_dim, int *_num_boxes)
{
	for (int k = 0; k < 3; k++)
	{
		_box_idx_dim[k * 2] = _grid_idx_dim[k] - 1;
		_box_idx_dim[k * 2 + 1] = _grid_idx_dim[k];

		_local_idx_dim[k * 2] = 1;
		_local_idx_dim[k * 2 + 1] = 0;

		if (_box_idx_dim[k * 2 + 1] > _num_boxes[k] - 1) _box_idx_dim[k * 2 + 1] = -1;
	}

}
inline __device__ void local_idx_dim_to_local_idx_linear(int* _local_idx, int *_local_idx_dim_temp)
{
	*_local_idx = _local_idx_dim_temp[0] + (_local_idx_dim_temp[1] << 1) + (_local_idx_dim_temp[2] << 2);
}


inline __device__ void direct_grid_interact_linear_static_scalar(const FP_TYPE *_r1, FP_TYPE *_Q1, const FP_TYPE *s_obs_amp,
	 const FP_TYPE *_cell_size, const FP_TYPE _epsilon)
{

	unsigned int _near_obs_idx_dim[3];
	FP_TYPE _r2[3];
	FP_TYPE _magn[1];

	for (int i = 0; i < 8; i++)
	{
		int _near_obs_idx = i & 7;
		idx_to_idx_dim_linear(_near_obs_idx_dim, _near_obs_idx);
		for (int m = 0; m < 3; m++)
		{		
			_r2[m] = _r1[m] + _near_obs_idx_dim[m] * _cell_size[m];
		}
		_magn[0] = s_obs_amp[i];
		get_field_static_scalar(_r2, _magn, _Q1, _epsilon);
	} // i 
}
inline __device__ void direct_grid_interact_linear_static_vector(const FP_TYPE *_r1, FP_TYPE *_Q1, const FP_TYPE *s_obs_amp,
	 const FP_TYPE *_cell_size, const FP_TYPE _epsilon, const int s_obs_amp_size)
{

	unsigned int _near_obs_idx_dim[3];
	FP_TYPE _r2[3];
	FP_TYPE _magn[FIELD_DIM];

	for (int i = 0; i < 8; i++)
	{
		int _near_obs_idx = i & 7;
		idx_to_idx_dim_linear(_near_obs_idx_dim, _near_obs_idx);
		for (int m = 0; m < 3; m++)
		{		
			_r2[m] = _r1[m] + _near_obs_idx_dim[m] * _cell_size[m];
		}
		for(unsigned int j = 0; j < FIELD_DIM; j++)	_magn[j] = s_obs_amp[i+j*s_obs_amp_size];
		get_field_static_vector(_r2, _magn, _Q1, _epsilon);
	} // i 
}




/////////////////////////////////////////////////////////////////////////////////////////////////

}
#endif
