/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft_static_scalar_gpu.cu: class definition of Class NufftStaticScalarGpu
*/
#include "error.h"
#include "field_static_scalar_gpu.h"
#include "gpu.h"
#include "imp_matrix_static_scalar.h"
#include "memory.h"
#include "memory_gpu.h"
#include "nbodyfast.h"
#include "nufft_static_scalar_gpu.h"
#include "nufft_static_scalar_gpu_kernel.h"
#include "timer.h"

//#define BEN_DEBUG
//#define BEN_DEBUG_MULTI
//#define BEN_NEW_METHOD
//#define BEN_DEBUG_FFT
namespace NBODYFAST_NS{
/* reorder_u_src_grid() and reorder_u_obs_grid() are reserved for more complex task division scheme in mult GPU calculation
 * the boxes being processed by any GPU device always have continuous box index number, currently
 * this situation might be changed in the future, so we might need to reorder data to correcly put them together in sequence
*/
void inline reorder_u_src_grid(FP_TYPE *u_src_grid)
{
}
void inline reorder_u_obs_grid(FP_TYPE *u_obs_grid)
{
}
/* the two functions below could be implemented as copy constructor...but I can't remember why I didn't do so....
*/
// set_value() for copying between NufftArrayGpuStaticScalar objects
void NufftArrayGpuStaticScalar :: set_value(NufftStaticScalarGpu *nufft_ptr)
{
	NufftArrayGpu::set_value(nufft_ptr);

	d_u_src_grid = NULL;
	d_u_obs_grid = NULL;
	//d_k_imp_mat_data = NULL;//IMP MAT CHANGE!!!!!!!!!
	d_k_imp_mat_data_gpu = NULL;
	d_k_u_src_grid = NULL;
	d_k_u_obs_grid = NULL;
	d_u_near_src_grid = NULL;
	d_u_near_obs_grid = NULL;
	d_k_u_near_src_grid = NULL;

	d_u_src_grid_dev = NULL;
	d_u_obs_grid_dev = NULL;


//	d_src_coord = nufft_ptr->field_static_scalar_gpu->d_src_coord[0];
//	d_obs_coord =nufft_ptr->field_static_scalar_gpu->d_obs_coord[0];

}
// set_value_base() for copying between NufftArrayGpuStaticScalar objects and its base class object
void NufftArrayGpuStaticScalar :: set_value_base(NufftArrayGpu *nufft_array_gpu)
{
	d_src_box_map = nufft_array_gpu->d_src_box_map;
	d_obs_box_map = nufft_array_gpu->d_obs_box_map;

	d_src_trans_idx = nufft_array_gpu->d_src_trans_idx;
	d_obs_trans_idx = nufft_array_gpu->d_obs_trans_idx;

	//d_near_box_list = nufft_array_gpu->d_near_box_list;
	d_near_bound_list = nufft_array_gpu->d_near_bound_list;

	d_src_box_list = nufft_array_gpu->d_src_box_list;
	d_obs_box_list = nufft_array_gpu->d_obs_box_list;

	d_src_box_list_inv = nufft_array_gpu->d_src_box_list_inv;
	d_obs_box_list_inv = nufft_array_gpu->d_obs_box_list_inv;

	d_src_grid_coord = nufft_array_gpu->d_src_grid_coord;
	d_obs_grid_coord = nufft_array_gpu->d_obs_grid_coord;

	d_fft_inplace_r2c = nufft_array_gpu->d_fft_inplace_r2c;
	d_fft_inplace_r2c_FP = nufft_array_gpu->d_fft_inplace_r2c_FP;
	//d_fft_inplace_b = nufft_array_gpu->d_fft_inplace_b;

}
NufftStaticScalarGpu :: NufftStaticScalarGpu(class NBODYFAST *n_ptr) : NufftStaticScalar(n_ptr), NufftGpu(n_ptr), Nufft(n_ptr)
{
//	_nbodyfast_test_ptr = n_ptr;

	nufft_array_gpu_static_scalar = NULL;
	d_nufft_array_gpu_static_scalar = NULL;
	imp_matrix_temp = NULL;

	field_static_scalar_gpu = dynamic_cast<FieldStaticScalarGpu*>(n_ptr->field);

}

NufftStaticScalarGpu :: ~NufftStaticScalarGpu()
{
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;

		nbodyfast->gpu->memory_gpu->free_device<NufftArrayGpuStaticScalar>(_cuda_error, &d_nufft_array_gpu_static_scalar[_thread_id], _cur_dev);	

		//nbodyfast->gpu->memory_gpu->free_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_k_imp_mat_data), _cur_dev);	//IMP MAT CHANGE!!!
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_k_imp_mat_data_gpu), _cur_dev);	
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid), _cur_dev);	
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid), _cur_dev);	

		if (nbodyfast->multi_device == true)
		{
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid_dev), _cur_dev);
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid_dev), _cur_dev);	
		}
		if (nbodyfast->field->src_obs_overlap == false)
		{
			//nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid), _cur_dev);
			//nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid_dev), _cur_dev);	
		}
	}


	for (int i = 0; i < nbodyfast->num_devices; i++)
	{
		nbodyfast->memory->free_host<NufftArrayGpuStaticScalar>(&nufft_array_gpu_static_scalar[i]);
	}
	nbodyfast->memory->free_host<NufftArrayGpuStaticScalar*>(&nufft_array_gpu_static_scalar);
	nbodyfast->memory->free_host<NufftArrayGpuStaticScalar*>(&d_nufft_array_gpu_static_scalar);

}

int NufftStaticScalarGpu :: preprocessing()
{
	// call the preprocessing subroutines of its parents (NufftStaticScalarGpu has two parents)
	NufftStaticScalar::preprocessing();
	NufftGpu::preprocessing();
	
	nbodyfast->error->last_error = nbodyfast->multi_device ? preprocessing_multi() : preprocessing_single();

	return 0;
}
int NufftStaticScalarGpu :: preprocessing_single()
{

	int _thread_id = 0;
	int _cur_dev = nbodyfast->device_name[_thread_id];
	cudaSetDevice(_cur_dev);
	cudaError_t _cuda_error;

	// Host copy of NufftArrayGpuStaticScalar. this is actually a table on the host that contains pointers to other working arrays
	nbodyfast->memory->alloc_host<NufftArrayGpuStaticScalar*>(&nufft_array_gpu_static_scalar, 1, "nufft->nufft_array_gpu_static_scalar");	
	nufft_array_gpu_static_scalar[0] = NULL;

	nbodyfast->memory->alloc_host<NufftArrayGpuStaticScalar>(&nufft_array_gpu_static_scalar[0], 1, "nufft->nufft_array_gpu_static_scalar[0]");	

	// Set the values of the host copy of NufftArrayGpuStaticScalar
	nufft_array_gpu_static_scalar[0]->set_value(this);
	nufft_array_gpu_static_scalar[0]->set_value_base(nufft_array_gpu[0]);

	// Get the address of field->d_src_amp and field->d_field_amp and put it to nufft_array_gpu_static_scalar
	nufft_array_gpu_static_scalar[0]->d_src_coord = field_static_scalar_gpu->d_src_coord[0];
	nufft_array_gpu_static_scalar[0]->d_obs_coord = field_static_scalar_gpu->d_obs_coord[0];	
	nufft_array_gpu_static_scalar[0]->d_src_amp = field_static_scalar_gpu->d_src_amp[0];
	nufft_array_gpu_static_scalar[0]->d_field_amp = field_static_scalar_gpu->d_field_amp[0];

	// u_src_grid is for storing projection results
	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &nufft_array_gpu_static_scalar[0]->d_u_src_grid, total_num_boxes * interp_nodes_per_box, "nufft->nufft_array_gpu_static_scalar[0]->d_u_src_grid", _cur_dev);

	// d_obs_grid is for storing interpolation sources
	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &nufft_array_gpu_static_scalar[0]->d_u_obs_grid, total_num_boxes * interp_nodes_per_box, "nufft->nufft_array_gpu_static_scalar[0]->d_u_obs_grid", _cur_dev);

	nufft_array_gpu_static_scalar[0]->d_u_src_grid_dev = nufft_array_gpu_static_scalar[0]->d_u_src_grid;
	nufft_array_gpu_static_scalar[0]->d_u_obs_grid_dev = nufft_array_gpu_static_scalar[0]->d_u_obs_grid;
	
	// Impedance matrix transfer, only the k_space version is needed
	int _num_padded_grid_pts = fft_size[0] * fft_size[1] * fft_size[2]; // this is actually the same as total_num_fft_pts
	
	
	//IMP MAT CHANGE!!!!!!!!!
	/*nbodyfast->memory->alloc_host<CUFFT_COMPLEX_TYPE>(&imp_matrix_temp, _num_padded_grid_pts, "nufft->imp_matrix_temp");	
	
	for (int i = 0; i < _num_padded_grid_pts; i++)
	{
		imp_matrix_temp[i].x = g_grid->k_imp_mat_data[i].real();
		imp_matrix_temp[i].y = g_grid->k_imp_mat_data[i].imag();
	}

	
	nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data), _num_padded_grid_pts, "nufft->nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data", _cur_dev);
	
	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<CUFFT_COMPLEX_TYPE>(nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data, imp_matrix_temp,_num_padded_grid_pts, _cur_dev);
	
	nbodyfast->memory->free_host<CUFFT_COMPLEX_TYPE>(&imp_matrix_temp);*/
	//IMP MAT CHANGE!!!!!!!!!ABOVE!!!!!!!!!!!!!!!!!!!


	//BEN ADDED
	int _total_num_green_pts = g_grid->_padded_green_dim[0]*g_grid->_padded_green_dim[1]*g_grid->_padded_green_dim[2];
	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data_gpu), _total_num_green_pts, "nufft->nufft_array_gpu_static_scalar[0]->k_imp_mat_data_gpu", _cur_dev);
	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data_gpu, g_grid->k_imp_mat_data_gpu, _total_num_green_pts, _cur_dev);

	// device copy of NufftArrayGpuStaticScalar. this is actually a table on device that contains pointers to other working arrays
	nbodyfast->memory->alloc_host<NufftArrayGpuStaticScalar*>(&d_nufft_array_gpu_static_scalar, 1, "nufft->d_nufft_array_gpu_static_scalar");
	d_nufft_array_gpu_static_scalar[0] = NULL;

	nbodyfast->gpu->memory_gpu->alloc_device<NufftArrayGpuStaticScalar>(_cuda_error, &d_nufft_array_gpu_static_scalar[0], 1, "nufft->d_nufft_array_gpu_static_scalar[0]", _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<NufftArrayGpuStaticScalar>(d_nufft_array_gpu_static_scalar[0], nufft_array_gpu_static_scalar[0],  1, _cur_dev);

	// Then thats' it! GPU calculation is ready to go!

	return 0;
}
int NufftStaticScalarGpu :: preprocessing_multi()
{
	cudaError_t _cuda_error;
	
	// To store temporary grid amplitudes copied after projection
	nbodyfast->memory->alloc_host<FP_TYPE>(&u_src_grid, total_num_boxes * interp_nodes_per_box, "nufft->u_src_grid");
	if (field_gpu->src_obs_overlap == true)
	{
		u_obs_grid = u_src_grid;
	}

	// Pointer to the host copy
	nbodyfast->memory->alloc_host<NufftArrayGpuStaticScalar*>(&nufft_array_gpu_static_scalar, nbodyfast->num_devices, "nufft->nufft_array_gpu_static_scalar");	

	// Pointer to the device copy
	nbodyfast->memory->alloc_host<NufftArrayGpuStaticScalar*>(&d_nufft_array_gpu_static_scalar, nbodyfast->num_devices, "nufft->d_nufft_array_gpu_static_scalar");

#pragma omp barrier
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		std::stringstream _array_name;

		nufft_array_gpu_static_scalar[_thread_id] = NULL;
		
		// host copy
		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu_static_scalar[" << _thread_id << "]";
		{
			nbodyfast->memory->alloc_host<NufftArrayGpuStaticScalar>(&nufft_array_gpu_static_scalar[_thread_id], 1, _array_name.str());
		}
		nufft_array_gpu_static_scalar[_thread_id]->set_value(this);
		nufft_array_gpu_static_scalar[_thread_id]->set_value_base(nufft_array_gpu[_thread_id]);

		// allocate arrays inside of nufft_array_gpu_static_scalar 
		// Get the address of field->d_src_amp and field->d_field_amp
		nufft_array_gpu_static_scalar[_thread_id]->d_src_coord = field_static_scalar_gpu->d_src_coord[_thread_id];
		nufft_array_gpu_static_scalar[_thread_id]->d_obs_coord = field_static_scalar_gpu->d_obs_coord[_thread_id];	
		nufft_array_gpu_static_scalar[_thread_id]->d_src_amp = field_static_scalar_gpu->d_src_amp[_thread_id];
		nufft_array_gpu_static_scalar[_thread_id]->d_field_amp = field_static_scalar_gpu->d_field_amp[_thread_id];

		//std::cout << "nufft_array_gpu_static_scalar[_thread_id]->d_src_amp" << nufft_array_gpu_static_scalar[_thread_id]->d_src_amp << std::endl;
		// d_u_src_grid is for storing projection results
		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu_static_scalar[" << _thread_id << "]->d_u_src_grid_dev";
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid_dev, src_box_list_inv_dev[_thread_id][0] * interp_nodes_per_box, _array_name.str(), _cur_dev);

		// d_u_obs_grid is for storing interpolation sources
		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu_static_scalar[" << _thread_id << "]->d_u_obs_grid_dev";
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid_dev, obs_box_list_dev[_thread_id][0] * interp_nodes_per_box, _array_name.str(), _cur_dev);

		if (_thread_id == 0)
		{
			// d_u_src_grid is for storing combined projection results
			_array_name.str("");
			_array_name <<"nufft->nufft_array_gpu_static_scalar[" << _thread_id << "]->d_u_src_grid";
			nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid, total_num_boxes * interp_nodes_per_box, _array_name.str(), _cur_dev);
			nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid = nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid;


			// Impedance matrix, only the k_space version is needed
			int _num_padded_grid_pts = fft_size[0] * fft_size[1] * fft_size[2];

			//IMP MAT CHANGE!!!!!!!!!!!!!!!!!!!!!
			/*nbodyfast->memory->alloc_host<CUFFT_COMPLEX_TYPE>(&imp_matrix_temp, _num_padded_grid_pts, "nufft->imp_matrix_temp");	

			for (int i = 0; i < _num_padded_grid_pts; i++)
			{
				imp_matrix_temp[i].x = g_grid->k_imp_mat_data[i].real();
				imp_matrix_temp[i].y = g_grid->k_imp_mat_data[i].imag();
			}

			nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_k_imp_mat_data), _num_padded_grid_pts, "nufft->nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data", _cur_dev);

			nbodyfast->gpu->memory_gpu->memcpy_host_to_device<CUFFT_COMPLEX_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_k_imp_mat_data, imp_matrix_temp,_num_padded_grid_pts, _cur_dev);

			nbodyfast->memory->free_host<CUFFT_COMPLEX_TYPE>(&imp_matrix_temp);*/
			//IMP MAT CHANGE!!!!!!!!!!!!!!!!!!!!!ABOVE

			//BEN ADDED
			int _total_num_green_pts = g_grid->_padded_green_dim[0]*g_grid->_padded_green_dim[1]*g_grid->_padded_green_dim[2];
			nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu_static_scalar[_thread_id]->d_k_imp_mat_data_gpu), _total_num_green_pts, "nufft->nufft_array_gpu_static_scalar[0]->k_imp_mat_data_gpu", _cur_dev);
			nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_k_imp_mat_data_gpu, g_grid->k_imp_mat_data_gpu, _total_num_green_pts, _cur_dev);
		}

		// Device data
		d_nufft_array_gpu_static_scalar[_thread_id] = NULL;

		{
			_array_name.str("");
			_array_name <<"nufft->d_nufft_array_gpu_static_scalar[" << _thread_id << "]";

			nbodyfast->gpu->memory_gpu->alloc_device<NufftArrayGpuStaticScalar>(_cuda_error, &d_nufft_array_gpu_static_scalar[_thread_id], 1, _array_name.str(), _cur_dev);
		}

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<NufftArrayGpuStaticScalar>(d_nufft_array_gpu_static_scalar[_thread_id], nufft_array_gpu_static_scalar[_thread_id],  1, _cur_dev);

		// Then thats' it! GPU calculation is ready to go!

	}
#pragma omp barrier
	return 0;
}
int NufftStaticScalarGpu :: projection()
{
	if (nbodyfast->multi_device)
	{
		nbodyfast->error->last_error = projection_multi();
	}
	else
	{
		nbodyfast->error->last_error = projection_single();
	}

	return 0;
}

int NufftStaticScalarGpu :: projection_single()
{
	int _thread_id = 0;
	int _cur_dev = nbodyfast->device_name[_thread_id];
	cudaSetDevice(_cur_dev);
	cudaError_t _cuda_error;

	/* Projection
	 * project source amplitudes to grid.
	*/
	{
		cudaDeviceSynchronize();		
		GpuExecParam exec_param;

		/* this section is used to calculate number of block per box, or number of box per block
		 * and total number of blocks needed for the current kernel
		*/

		if (interp_nodes_per_box >= BLOCK_SIZE_PROJ_INTERP)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_PROJ_INTERP + 1;
			exec_param.num_box_per_blk = -1;
			// number of the blocks are "total number boxes on the current device" times "number of blocks per box"
			exec_param.num_blk = nufft_const_gpu[_thread_id]->total_num_boxes_dev * exec_param.num_blk_per_box; 
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_PROJ_INTERP - 1) / interp_nodes_per_box + 1;
			// number of the blocks are "total number boxes on the current device" divided by "number of boxes per block"
			exec_param.num_blk = (nufft_const_gpu[_thread_id]->total_num_boxes_dev - 1) / exec_param.num_box_per_blk + 1;
		}
		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	

		nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_u_src_grid_dev, 0, nufft_const_gpu[0]->total_num_boxes_dev * interp_nodes_per_box, _cur_dev);

		cudaFuncSetCacheConfig(nufft_project_static_scalar_cubic, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(nufft_project_static_scalar_linear, cudaFuncCachePreferShared);

		if (interp_order[0] == 3)
		{
			nufft_project_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_project_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

		cudaDeviceSynchronize();			

#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_projection[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);	
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_src_amp[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_src_amp, problem_size, _filename.str());
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_src_coord[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_src_coord, problem_size * 3, _filename.str());
		}
#endif

	}
	// Projection finished

	/* FFT grid preprocessing
	 * The local grids generated by the projection kernel are ordered by box indices
	 * this needs to be combined to generate the global grid 
	*/
	{

		GpuExecParam exec_param;

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		exec_param.num_blk = (total_num_grid_pts -1) / BLOCK_SIZE_PROJ_INTERP + 1;

		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	

		nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_fft_inplace_r2c_FP, 0, 2*total_num_fft_r2c_pts, _cur_dev);

		cudaFuncSetCacheConfig(nufft_fft_prep_static_scalar_cubic, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(nufft_fft_prep_static_scalar_linear, cudaFuncCachePreferShared);

		cudaDeviceSynchronize();			
		if (interp_order[0] == 3)
		{
			nufft_fft_prep_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_fft_prep_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		cudaDeviceSynchronize();			
		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

#ifdef _GPU_D_TEST
		std::stringstream _filename;
		_filename << "..\\Temp\\test_array_preprop[" << _thread_id << "].out"; 
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);	
#endif
#ifdef BEN_DEBUG_FFT
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_f, total_num_fft_pts, "..\\Temp\\d_fft_pre_old.out");
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu[0]->d_fft_inplace_r2c_FP, 2*total_num_fft_r2c_pts, "..\\Temp\\d_fft_pre.out");
#endif

	}
	// FFT grid preprocessing finished

	return 0;
}

int NufftStaticScalarGpu :: projection_multi()
{
	// Projection
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
//		double _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();
		cudaSetDevice(_cur_dev);
		cudaDeviceSynchronize();		
		GpuExecParam exec_param;
		if (interp_nodes_per_box >= BLOCK_SIZE_PROJ_INTERP)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_PROJ_INTERP + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = src_box_list_inv_dev[_thread_id][0] * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_PROJ_INTERP - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (src_box_list_inv_dev[_thread_id][0] - 1) / exec_param.num_box_per_blk + 1;
		}
		GpuExecParam *d_exec_param = NULL;

		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		//std::cout << "exec_param.num_blk_per_box: " << exec_param.num_blk_per_box << std::endl;
		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;

#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
#endif	

		cudaFuncSetCacheConfig(nufft_project_static_scalar_cubic, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(nufft_project_static_scalar_linear, cudaFuncCachePreferShared);

		nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid_dev, 0, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _cur_dev);

		cudaDeviceSynchronize();	
		if (interp_order[0] == 3)
		{
			nufft_project_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_project_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}


		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_projection[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);	
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_src_amp[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_src_amp, nbodyfast->src_size_dev[_thread_id], _filename.str());
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_src_coord[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_src_coord, nbodyfast->src_size_dev[_thread_id] * 3, _filename.str());
		}
#endif

		nbodyfast->gpu->memory_gpu->memcpy_device_to_host<FP_TYPE>(u_src_grid + src_grid_start_dev[_thread_id], nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid_dev, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _cur_dev);
		
		// Reorder u_src_grid to have box lined in sequence 
		reorder_u_src_grid(u_src_grid);
		cudaDeviceSynchronize();			
//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout << "Projection : thread_id: " << _thread_id <<". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;
	}
	// Projection finished


#pragma omp barrier

	// FFT preprocessing, this is done on device 0 only;
	{
		int _thread_id = 0;
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
//		double _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();

		cudaSetDevice(_cur_dev);
		cudaDeviceSynchronize();			
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_u_src_grid, u_src_grid, total_num_boxes * interp_nodes_per_box, _cur_dev);


		GpuExecParam exec_param;

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		exec_param.num_blk = (total_num_grid_pts -1) / BLOCK_SIZE_PROJ_INTERP + 1;
		
		GpuExecParam *d_exec_param = NULL;

		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;

#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
#endif	

		cudaFuncSetCacheConfig(nufft_fft_prep_static_scalar_cubic, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(nufft_fft_prep_static_scalar_linear, cudaFuncCachePreferShared);

		nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_fft_inplace_r2c_FP, 0, 2*total_num_fft_r2c_pts, _cur_dev);

		if (interp_order[0] == 3)
		{
			nufft_fft_prep_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_fft_prep_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}

		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

#ifdef _GPU_D_TEST
		std::stringstream _filename;
		_filename << "..\\Temp\\test_array_preprop[" << _thread_id << "].out"; 
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);	
#endif

//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout << "FFT preprocessing: thread_id: " << _thread_id << ". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;
		cudaDeviceSynchronize();			

	}
	// FFT grid preprocessing finished

	return 0;
}

int NufftStaticScalarGpu :: fft_real_to_k()
{
	/* single and multi GPU subroutines are the same for FFT.
	 * it is always done on the device 0
	*/
	if (nbodyfast->multi_device)
	{
		nbodyfast->error->last_error = fft_real_to_k_multi();
	}
	else
	{
		nbodyfast->error->last_error = fft_real_to_k_single();
	}

	return 0;
}

int NufftStaticScalarGpu :: fft_real_to_k_single()
{
	int _thread_id = 0;
	int _cur_dev = nbodyfast->device_name[_thread_id];
	cudaSetDevice(_cur_dev);
	// Forward FFT
	{
		cudaDeviceSynchronize();			
		cudaError_t _cuda_error;

	#ifdef _GPU_D_TEST
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_f, total_num_fft_pts / 8, "..\\Temp\\d_fft_in_f.out");
	#endif
		// the macro "CUFFT_EXEC is defined in "fp_precision.h". This shouldn't be done as a macro but instead through function pointer or something...I am lazy...
		CUFFT_EXEC_F(nufft_array_gpu[0]->cufftPlan_r2c, nufft_array_gpu[0]->d_fft_inplace_r2c_FP, nufft_array_gpu[0]->d_fft_inplace_r2c);
		cudaDeviceSynchronize();			

	#ifdef _GPU_D_TEST
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_f, total_num_fft_pts / 8, "..\\Temp\\d_fft_out_f.out");
	#endif
	#ifdef BEN_DEBUG_FFT
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\d_fft_out_f.out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_r2c, total_num_fft_r2c_pts, _filename.str());
			nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_f, total_num_fft_pts, "..\\Temp\\d_fft_out_f_old.out");
		}
	#endif
	cudaDeviceSynchronize();			

	}
	return 0;
}

int NufftStaticScalarGpu :: fft_real_to_k_multi()
{
	// Forward FFT
	{
		int _thread_id = 0;
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		cudaSetDevice(_cur_dev);
		cudaDeviceSynchronize();			

//		double _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();

	#ifdef _GPU_D_TEST
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[_thread_id]->d_fft_inplace_f, total_num_fft_pts / 8, "..\\Temp\\d_fft_in_f.out");
	#endif
		// the macro "CUFFT_EXEC is defined in "fp_precision.h". This shouldn't be done as a macro but instead through function pointer or something...I am lazy...
		CUFFT_EXEC_F(nufft_array_gpu[_thread_id]->cufftPlan_r2c, nufft_array_gpu[_thread_id]->d_fft_inplace_r2c_FP, nufft_array_gpu[_thread_id]->d_fft_inplace_r2c);
		cudaDeviceSynchronize();			

	#ifdef _GPU_D_TEST
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_f, total_num_fft_pts / 8, "..\\Temp\\d_fft_out_f.out");
	#endif

//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout<< "FFT : thread_id: " << _thread_id << ". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;
	}
	return 0;
}

int NufftStaticScalarGpu :: convolution_k()
{
	/* single and multi GPU subroutines are the same for convolution.
	 * it is always done on the device 0
	*/
	if (nbodyfast->multi_device)
	{
		nbodyfast->error->last_error = convolution_k_multi();
	}
	else
	{
		nbodyfast->error->last_error = convolution_k_single();
	}
	return 0;
}

int NufftStaticScalarGpu :: convolution_k_single()
{
	int _thread_id = 0;
	int _cur_dev = nbodyfast->device_name[_thread_id];
	cudaSetDevice(_cur_dev);
	cudaError_t _cuda_error;

	// Tensor-vector multiplication
	//ben new method
	{

		GpuExecParam exec_param;

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_CONV;
		dim3 _dim_grid;

		exec_param.num_blk = (total_num_fft_r2c_pts - 1) / BLOCK_SIZE_CONV + 1;//BEN MADE THE CHANGE HERE

		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	
		cudaFuncSetCacheConfig(nufft_convolution_static_scalar, cudaFuncCachePreferShared);

		cudaDeviceSynchronize();			
		nufft_convolution_static_scalar<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		cudaDeviceSynchronize();			

		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

	#ifdef _GPU_D_TEST
		std::stringstream _filename;
		_filename << "..\\Temp\\test_array_conv[" << _thread_id << "].out"; 
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size / 8, _filename.str());
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);	
	#endif
	#ifdef BEN_DEBUG_FFT
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_r2c, total_num_fft_r2c_pts, "..\\Temp\\d_conv_out.out");
	#endif
	}

	return 0;
}

int NufftStaticScalarGpu :: convolution_k_multi()
{
	// Tensor-vector multiplication
	//BEN NEW METHOD
	{
		int _thread_id = 0;
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		cudaSetDevice(_cur_dev);

//		double _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();

		GpuExecParam exec_param;

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_CONV;
		dim3 _dim_grid;

		exec_param.num_blk = (total_num_fft_r2c_pts - 1) / BLOCK_SIZE_CONV + 1;

		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	

		cudaFuncSetCacheConfig(nufft_convolution_static_scalar, cudaFuncCachePreferShared);

		cudaDeviceSynchronize();			
		nufft_convolution_static_scalar<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		cudaDeviceSynchronize();			
		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

	#ifdef _GPU_D_TEST
		std::stringstream _filename;
		_filename << "..\\Temp\\test_array_conv[" << _thread_id << "].out"; 
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size / 8, _filename.str());
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);	
	#endif

//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout << "Convolution : thread_id: " << _thread_id << ". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;
	}
	return 0;
}

int NufftStaticScalarGpu :: fft_k_to_real()
{
	/* single and multi GPU subroutines are the same for FFT.
	 * it is always done on the device 0
	*/
	if (nbodyfast->multi_device)
	{
		nbodyfast->error->last_error = fft_k_to_real_multi();
	}
	else
	{
		nbodyfast->error->last_error = fft_k_to_real_single();
	}
	return 0;
}

int NufftStaticScalarGpu :: fft_k_to_real_single()
{
	int _thread_id = 0;
	int _cur_dev = nbodyfast->device_name[_thread_id];
	cudaSetDevice(_cur_dev);
	cudaError_t _cuda_error;

#ifdef _GPU_D_TEST
	nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_b, total_num_grid_pts * 8 / 8, "..\\Temp\\d_fft_in_b.out");
#endif
		// the macro "CUFFT_EXEC is defined in "fp_precision.h". This shouldn't be done as a macro but instead through function pointer or something...I am lazy...
	CUFFT_EXEC_B(nufft_array_gpu[0]->cufftPlan_c2r, nufft_array_gpu[0]->d_fft_inplace_r2c, nufft_array_gpu[0]->d_fft_inplace_r2c_FP);
	cudaDeviceSynchronize();

#ifdef _GPU_D_TEST
	nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_b, total_num_grid_pts * 8 / 8, "..\\Temp\\d_fft_out_b.out");
#endif
	#ifdef BEN_DEBUG_FFT
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\d_fft_out_b.out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu[0]->d_fft_inplace_r2c_FP, total_num_fft_r2c_pts*2, _filename.str());
			nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_f, total_num_fft_pts, "..\\Temp\\d_fft_out_b_old.out");
		}
		{
			/*std::stringstream _filename;
			_filename << "..\\Temp\\standard_array_conv[" << _thread_id << "].out"; */
			//nbodyfast->gpu->memory_gpu->device_array_to_disk_complex<CUFFT_COMPLEX_TYPE>(d_nufft_array_gpu_static_scalar[0]->d_k_imp_mat_data, total_num_fft_pts, _filename.str());
		}
	#endif
	return 0;
}

int NufftStaticScalarGpu :: fft_k_to_real_multi()
{
	// Inverse FFT
	{
		int _thread_id = 0;
		int _cur_dev = nbodyfast->device_name[_thread_id];

		cudaError_t _cuda_error;
		cudaSetDevice(_cur_dev);
		cudaDeviceSynchronize();			

//		double _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();

	#ifdef _GPU_D_TEST
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[_thread_id]->d_fft_inplace_b, total_num_grid_pts * 8 / 8, "..\\Temp\\d_fft_in_i.out");
	#endif
		// the macro "CUFFT_EXEC is defined in "fp_precision.h". This shouldn't be done as a macro but instead through function pointer or something...I am lazy...
		CUFFT_EXEC_B(nufft_array_gpu[_thread_id]->cufftPlan_c2r, nufft_array_gpu[_thread_id]->d_fft_inplace_r2c, nufft_array_gpu[_thread_id]->d_fft_inplace_r2c_FP);
		cudaDeviceSynchronize();			

	#ifdef _GPU_D_TEST
		nbodyfast->gpu->memory_gpu->device_array_to_disk<CUFFT_COMPLEX_TYPE>(nufft_array_gpu[0]->d_fft_inplace_b, total_num_grid_pts * 8 / 8, "..\\Temp\\d_fft_out_i.out");
	#endif

//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout << "Inverse FFT : thread_id: " << _thread_id << ". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;
	}
	return 0;
}

int NufftStaticScalarGpu :: correction_interpolation()
{
//	double _temp_time1, _temp_time2;
//	_temp_time1 = nbodyfast->timer->get_time();
	//nbodyfast->gpu->memory_gpu->output_allocated_list_file();
	
	if (nbodyfast->multi_device)
	{
		nbodyfast->error->last_error = correction_interpolation_multi();
	}
	else
	{
		nbodyfast->error->last_error = correction_interpolation_single();
	}
//	_temp_time2 = nbodyfast->timer->get_time();
//	std::cout << "NufftStaticScalarGpu :: correction_interpolation(). Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;

	return 0;
}

int NufftStaticScalarGpu :: correction_interpolation_single()
{
	int _thread_id = 0;
	int _cur_dev = nbodyfast->device_name[_thread_id];
	cudaSetDevice(_cur_dev);
	cudaError_t _cuda_error;

	nbodyfast->timer->time_stamp("DEBUG1");
	// inverse-FFT post-processing. rearrange field amplitudes on the grid points box by box.
	{
		cudaDeviceSynchronize();			

		GpuExecParam exec_param;

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		if (interp_nodes_per_box >= BLOCK_SIZE_PROJ_INTERP)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_PROJ_INTERP + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = nufft_const_gpu[_thread_id]->total_num_boxes_dev * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_PROJ_INTERP - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (nufft_const_gpu[_thread_id]->total_num_boxes_dev - 1) / exec_param.num_box_per_blk + 1;
		}

		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);


		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	
 
		cudaFuncSetCacheConfig(nufft_fft_postp_static_scalar_cubic, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(nufft_fft_postp_static_scalar_linear, cudaFuncCachePreferShared);

		if (interp_order[0] == 3)
		{
			nufft_fft_postp_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_fft_postp_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);
		cudaDeviceSynchronize();			

#ifdef _GPU_D_TEST
		std::stringstream _filename;
		_filename << "..\\Temp\\test_array_postpro[" << _thread_id << "].out"; 
		std::cout << _filename.str() << std::endl;
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
#endif
	}

#ifdef BEN_DEBUG
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_u_obs_grid_dev_pre_correction.out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_u_obs_grid_dev, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _filename.str());
		}
#endif
	nbodyfast->timer->time_stamp("DEBUG1");
	nbodyfast->timer->time_stamp("DEBUG2");
	// Correction. for each observer box, recalculate the inaccurate field and subtract them from the field grid
	{//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
		cudaDeviceSynchronize();			
		GpuExecParam exec_param;
#ifndef BEN_NEW_METHOD
		if (interp_nodes_per_box >= BLOCK_SIZE_CORRECT)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_CORRECT + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = nufft_const_gpu[_thread_id]->total_num_boxes_dev * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_CORRECT - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (nufft_const_gpu[_thread_id]->total_num_boxes_dev - 1) / exec_param.num_box_per_blk + 1;
		}
#else
		exec_param.num_blk_per_box = -1;
		exec_param.num_box_per_blk = 1;
		exec_param.num_blk = nufft_const_gpu[_thread_id]->total_num_boxes_dev;
#endif

		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);
 
		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_CORRECT;
		dim3 _dim_grid;

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	
#ifdef BEN_DEBUG
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
#endif

		cudaFuncSetCacheConfig(nufft_correct_static_scalar_cubic, cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(nufft_correct_static_scalar_linear, cudaFuncCachePreferL1);

		if (interp_order[0] == 3)
		{
			nufft_correct_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_correct_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}

		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);
		cudaDeviceSynchronize();			
		

#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_correction[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
		}
#endif

//#ifdef BEN_DEBUG
//		{
//			std::stringstream _filename;
//			_filename << "..\\Temp\\test_array_post_corrections.out"; 
//			std::cout << _filename.str() << std::endl;
//			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
//			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
//		}
//#endif
	}
#ifdef BEN_DEBUG
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_u_obs_grid_dev_post_correction.out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_u_obs_grid_dev, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _filename.str());
		}
		//{
		//	std::stringstream _filename;
		//	_filename << "..\\Temp\\test_array_d_near_box_list.out"; 
		//	nbodyfast->gpu->memory_gpu->device_array_to_disk<int>(nufft_array_gpu_static_scalar[0]->d_near_box_list, nufft_const_gpu[_thread_id]->total_num_boxes_dev * 126, _filename.str());
		//	//nufft_array->d_near_box_list[_local_box_idx * total_num_near_box_per_box_p1]
		//}
#endif
		nbodyfast->timer->time_stamp("DEBUG2");
		nbodyfast->timer->time_stamp("DEBUG3");
	// Interpolation. interpolate grid field amplitudes to observers 
	{
		cudaDeviceSynchronize();			
		GpuExecParam exec_param;

		if (interp_nodes_per_box >= BLOCK_SIZE_PROJ_INTERP)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_PROJ_INTERP + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = total_num_boxes * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_PROJ_INTERP - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (total_num_boxes - 1) / exec_param.num_box_per_blk + 1;
		}
		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	

		cudaFuncSetCacheConfig(nufft_interp_static_scalar_cubic, cudaFuncCachePreferL1); 
		cudaFuncSetCacheConfig(nufft_interp_static_scalar_linear, cudaFuncCachePreferL1); 

		//		nbodyfast->timer->time_stamp();

		if (interp_order[0] == 3)
		{
			nufft_interp_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_interp_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		}

		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

#ifdef BEN_DEBUG
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_obs_field_post_interp.out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_field_amp, problem_size, _filename.str());
		}
#endif
#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_interp[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_u_obs_grid_dev[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_u_obs_grid_dev, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _filename.str());
		}
#endif
		cudaDeviceSynchronize();		
	}
	nbodyfast->timer->time_stamp("DEBUG3");
	nbodyfast->timer->time_stamp("DEBUG4");

	// Near-field. calculate and add the accurate near field to the final field values of each observer
	{//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
		cudaDeviceSynchronize();		
		GpuExecParam exec_param;

		exec_param.num_blk_per_box = -1;
		exec_param.num_box_per_blk = 1;
		exec_param.num_blk = total_num_boxes * exec_param.num_box_per_blk;

		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
//		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_NEAR;
		dim3 _dim_grid;

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	
	#ifdef BEN_DEBUG
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif

		cudaFuncSetCacheConfig(nufft_exact_near_field, cudaFuncCachePreferShared); 

		//		nbodyfast->timer->time_stamp();
#ifdef USE_TEXTURE
		size_t offset = size_t(-1);
		if( (_cuda_error = cudaBindTexture(&offset, tex_box_int, nufft_array_gpu_static_scalar[0]->d_obs_box_map)  ) != cudaSuccess)
			std::cout << "cudaBindTexure error: " << cudaGetErrorString(_cuda_error) << std::endl;
		if (offset != 0)
			std::cout << "memory is not aligned, refusing to use texture cache" << std::endl;
#endif
		nufft_exact_near_field<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[0], d_nufft_const_gpu[0], d_exec_param);
		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

#ifdef USE_TEXTURE
		cudaUnbindTexture(tex_box_int);
#endif

#ifdef BEN_DEBUG
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_nearfield.out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
		}
		//{
		//	std::stringstream _filename;
		//	_filename << "..\\Temp\\test_array_d_near_box_list.out"; 
		//	nbodyfast->gpu->memory_gpu->device_array_to_disk<int>(nufft_array_gpu_static_scalar[0]->d_near_box_list, nufft_const_gpu[_thread_id]->total_num_boxes_dev * 27, _filename.str());
		//	//nufft_array->d_near_box_list[_local_box_idx * total_num_near_box_per_box_p1]
		//}
#endif
#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_nearfield[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_field_amp[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[0]->d_field_amp, problem_size, _filename.str());
		}
#endif
		cudaDeviceSynchronize();		
	}
	nbodyfast->timer->time_stamp("DEBUG4");

	return 0;
}

int NufftStaticScalarGpu :: correction_interpolation_multi()
{
	// FFT postprocessing
	{
		int _thread_id = 0;
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		cudaSetDevice(_cur_dev);
		cudaDeviceSynchronize();		
//		FP_TdoubleYPE _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();

		GpuExecParam exec_param;

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;
		dim3 _dim_grid;

		if (interp_nodes_per_box >= BLOCK_SIZE_PROJ_INTERP)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_PROJ_INTERP + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = total_num_boxes * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_PROJ_INTERP - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (total_num_boxes - 1) / exec_param.num_box_per_blk + 1;
		}
		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	
 
		cudaFuncSetCacheConfig(nufft_fft_postp_static_scalar_cubic, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(nufft_fft_postp_static_scalar_linear, cudaFuncCachePreferShared);

		if (interp_order[0] == 3)
		{
			nufft_fft_postp_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_fft_postp_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}

		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

		nbodyfast->gpu->memory_gpu->memcpy_device_to_host<FP_TYPE>(u_obs_grid, nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid, total_num_boxes * interp_nodes_per_box, _cur_dev);

#ifdef _GPU_D_TEST
		std::stringstream _filename;
		_filename << "..\\Temp\\test_array_postpro[" << _thread_id << "].out"; 
		std::cout << _filename.str() << std::endl;
		nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
#endif
		cudaDeviceSynchronize();		
//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout << "Inverse FFT postprocessing : thread_id: " << _thread_id << ". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;

	}

	// Correction
#pragma omp barrier
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		cudaSetDevice(_cur_dev);
		cudaDeviceSynchronize();		
//		double _temp_time1, _temp_time2;
//		_temp_time1 = nbodyfast->timer->get_time();

		// Reorder u_obs_grid to have box lined in sequence 
		reorder_u_obs_grid(u_obs_grid);
		
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid_dev, u_obs_grid + obs_grid_start_dev[_thread_id], nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _cur_dev);

		//std::stringstream _filename;
		//_filename << "..\\Temp\\TestArrayDevice[" << _thread_id << "].out"; 
		//std::cout << _filename.str() << std::endl;
		//nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid_dev, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _filename.str());


		GpuExecParam exec_param;

		if (interp_nodes_per_box >= BLOCK_SIZE_CORRECT)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_CORRECT + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = nufft_const_gpu[_thread_id]->total_num_boxes_dev * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_CORRECT - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (nufft_const_gpu[_thread_id]->total_num_boxes_dev - 1) / exec_param.num_box_per_blk + 1;
		}
		GpuExecParam *d_exec_param = NULL;
		nbodyfast->gpu->memory_gpu->alloc_device<GpuExecParam>(_cuda_error, &d_exec_param, 1, "d_exec_param", _cur_dev);
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);
	
		//for debug
#ifdef BEN_DEBUG_MULTI
		std::cout << "nufft_const_gpu[" << _thread_id << "]->total_num_boxes_dev: " << nufft_const_gpu[_thread_id]->total_num_boxes_dev << std::endl;
		std::cout << "exec_param.num_blk[" << _thread_id << "]: " << exec_param.num_blk << std::endl;
#endif
		//std::cout << "nufft_array->d_src_box_list_inv[" << _thread_id << "][0]: " << nufft_array_gpu_static_scalar[_thread_id]->src_box_list_inv_dev[0] << std::endl;
		//for debug

		dim3 _dim_block;
		_dim_block.x = BLOCK_SIZE_CORRECT;
		dim3 _dim_grid;

		unsigned int _temp_x;
		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

		
		FP_TYPE *_d_test = NULL;
	#ifdef _GPU_D_TEST
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	

		cudaFuncSetCacheConfig(nufft_correct_static_scalar_cubic, cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(nufft_correct_static_scalar_linear, cudaFuncCachePreferL1);


		if (interp_order[0] == 3)
		{
			nufft_correct_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_correct_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}
	

#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_correction[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
		}
#endif


		// interpolation
		if (interp_nodes_per_box >= BLOCK_SIZE_PROJ_INTERP)
		{
			exec_param.num_blk_per_box = (interp_nodes_per_box - 1) / BLOCK_SIZE_PROJ_INTERP + 1;
			exec_param.num_box_per_blk = -1;
			exec_param.num_blk = nufft_const_gpu[_thread_id]->total_num_boxes_dev * exec_param.num_blk_per_box;
		}
		else
		{
			exec_param.num_blk_per_box = -1;
			exec_param.num_box_per_blk = (BLOCK_SIZE_PROJ_INTERP - 1) / interp_nodes_per_box + 1;
			exec_param.num_blk = (nufft_const_gpu[_thread_id]->total_num_boxes_dev - 1) / exec_param.num_box_per_blk + 1;
		}

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<GpuExecParam>(d_exec_param, &exec_param, 1, _cur_dev);
	
		//std::cout << "exec_param.num_blk_per_box: " << exec_param.num_blk_per_box << std::endl;

		_dim_block.x = BLOCK_SIZE_PROJ_INTERP;

		_temp_x = exec_param.num_blk / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (exec_param.num_blk - 1) / _temp_x + 1;

	#ifdef _GPU_D_TEST
		_mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif	
	#ifdef BEN_DEBUG_MULTI
		int _mem_test_size = _dim_block.x * _dim_grid.x * _dim_grid.y;
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &_d_test, _mem_test_size, "_d_test", _cur_dev);
	#endif
		cudaFuncSetCacheConfig(nufft_interp_static_scalar_cubic, cudaFuncCachePreferL1); 
		cudaFuncSetCacheConfig(nufft_interp_static_scalar_linear, cudaFuncCachePreferL1); 


		if (interp_order[0] == 3)
		{
			nufft_interp_static_scalar_cubic<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}
		if (interp_order[0] == 1)
		{
			nufft_interp_static_scalar_linear<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);
		}

#ifdef BEN_DEBUG_MULTI
		{
			/*std::stringstream _filename;
			_filename << "..\\Temp\\test_array_interp_multi[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());*/
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
		}
#endif
#ifdef BEN_DEBUG_MULTI
        {
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_interp_multi[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			//nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(d_nufft_array_gpu_static_scalar[_thread_id]->d_field_amp, nbodyfast->obs_size_act_dev[_thread_id], _filename.str());
		}
#endif
#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_interp[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
		}
#endif
		// exact near field
		_dim_block.x = BLOCK_SIZE_NEAR;
		_temp_x = nufft_const_gpu[_thread_id]->total_num_boxes_dev / 65535;

		// round v to neareast power of 2
		_temp_x |= _temp_x >> 1;
		_temp_x |= _temp_x >> 2;
		_temp_x |= _temp_x >> 4;
		_temp_x |= _temp_x >> 8;
		_temp_x |= _temp_x >> 16;
		_temp_x++;

		_dim_grid.x = _temp_x;
		_dim_grid.y = (nufft_const_gpu[_thread_id]->total_num_boxes_dev - 1) / _temp_x + 1;

		cudaFuncSetCacheConfig(nufft_exact_near_field, cudaFuncCachePreferShared); 

		nufft_exact_near_field<<<_dim_grid, _dim_block>>>(_d_test, d_nufft_array_gpu_static_scalar[_thread_id], d_nufft_const_gpu[_thread_id], d_exec_param);


		nbodyfast->gpu->memory_gpu->free_device<GpuExecParam>(_cuda_error, &d_exec_param, _cur_dev);

#ifdef _GPU_D_TEST
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_nearfield[" << _thread_id << "].out"; 
			std::cout << _filename.str() << std::endl;
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(_d_test, _mem_test_size, _filename.str());
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &_d_test, _cur_dev);
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_u_obs_grid_dev[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_u_obs_grid_dev, nufft_const_gpu[_thread_id]->total_num_boxes_dev * interp_nodes_per_box, _filename.str());
		}
		{
			std::stringstream _filename;
			_filename << "..\\Temp\\test_array_d_field_amp[" << _thread_id << "].out"; 
			nbodyfast->gpu->memory_gpu->device_array_to_disk<FP_TYPE>(nufft_array_gpu_static_scalar[_thread_id]->d_field_amp, nbodyfast->obs_size_dev[_thread_id], _filename.str());
		}
#endif

//		_temp_time2 = nbodyfast->timer->get_time();
//		std::cout << "Correction : thread_id: " << _thread_id << ". Elapsed time: " << _temp_time2 - _temp_time1 << std::endl;
		cudaDeviceSynchronize();		
	}
#pragma omp barrier

	return 0;
}




}