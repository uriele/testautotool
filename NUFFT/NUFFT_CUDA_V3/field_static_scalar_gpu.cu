/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_static_scalar_gpu.cu: class definition of Class FieldStaticScalarGpu
*/
#include "field_static_scalar_gpu.h"
#include "direct.h"
#include "error.h"
#include "gpu.h"
#include "memory.h"
#include "memory_gpu.h"
#include "nbodyfast.h"
#include "nufft.h"

namespace NBODYFAST_NS{
FieldStaticScalarGpu :: FieldStaticScalarGpu(class NBODYFAST *n_ptr) : FieldStaticScalar(n_ptr), FieldGpu(n_ptr), Field(n_ptr)
{
	d_src_amp = NULL;
	d_field_amp = NULL;
}

FieldStaticScalarGpu :: ~FieldStaticScalarGpu()
{
	cudaError_t _cuda_error;
	for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id++)
	{
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;

		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &d_field_amp[_thread_id], _cur_dev);
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &d_src_amp[_thread_id], _cur_dev);
	}

	nbodyfast->memory->free_host<FP_TYPE*>(&d_field_amp);
	nbodyfast->memory->free_host<FP_TYPE*>(&d_src_amp);

}

int FieldStaticScalarGpu :: array_alloc_multi_interface()
{
	FieldStaticScalar::array_alloc_multi_interface(); 
	FieldGpu::array_alloc_multi_interface();

	nbodyfast->error->last_error = amp_field_alloc_multi();

	return 0;
}
int FieldStaticScalarGpu :: amp_field_alloc()
{

	FieldStaticScalar::amp_field_alloc();
	
	nbodyfast->memory->alloc_host<FP_TYPE*>(&d_src_amp, nbodyfast->num_devices, "field->d_src_amp");
	nbodyfast->memory->alloc_host<FP_TYPE*>(&d_field_amp, nbodyfast->num_devices, "field->d_field_amp");

	for (int i = 0; i < nbodyfast->num_devices; i++)
	{
		d_src_amp[i] = NULL;
		d_field_amp[i] = NULL;
	}

	if (nbodyfast->multi_device == false) amp_field_alloc_single();
	// similar to coordinate arrays in the Class FieldGpu, source and amplitude arrays for multi GPU calculation are allocated after preprocessing
	return 0;
}

int FieldStaticScalarGpu :: amp_field_alloc_single()
{	
	cudaError_t _cuda_error;

	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_src_amp[0], nbodyfast->problem_size, "field->d_src_amp[0]", nbodyfast->gpu->dev_list[0].index);
	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_field_amp[0], nbodyfast->problem_size, "field->d_field_amp[0]", nbodyfast->gpu->dev_list[0].index);

	return 0;
}
int FieldStaticScalarGpu :: amp_field_alloc_multi()
{
#pragma omp barrier
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		std::stringstream _array_name;
		_array_name <<"nufft->d_src_amp[" << _thread_id << "]";
		{
			nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_src_amp[_thread_id], nbodyfast->src_size_dev[_thread_id], _array_name.str(), _cur_dev);
		}
		_array_name.str("");
		_array_name <<"nufft->d_field_amp[" << _thread_id << "]";
		{
			nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_field_amp[_thread_id], nbodyfast->obs_size_dev[_thread_id], _array_name.str(), _cur_dev);
		}
	}
#pragma omp barrier
//
//	nbodyfast->memory->output_allocated_list();
//	nbodyfast->gpu->memory_gpu->output_allocated_list();

	return 0;
}


// Get source amplitudes from outside, the set_src_amp is usually called at every iteration, at the beginning of execution
int FieldStaticScalarGpu :: set_src_amp(double *_charge)
{
	nbodyfast->error->last_error = nbodyfast->multi_device ? set_src_amp_multi(_charge) : set_src_amp_single(_charge);
	return 0;
}
// single GPU calculation goes here
int FieldStaticScalarGpu :: set_src_amp_single(double *_charge)
{
	cudaDeviceSynchronize();		
	if (nbodyfast->algo_name == "nufft")
	{
		int *src_trans_idx = nbodyfast->nufft->get_src_trans_idx();
#pragma omp parallel for
		for (int i = 0; i < nbodyfast->problem_size; i++)
		{
			src_amp[i] = _charge[src_trans_idx[i]]; // to copy data for NUFFT, we need src_trans_idx[] as the reordering table
			field_amp[i] = 0.0f;
		}

		// copy source amplitudes to device and set field amplitudes to 0	
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_src_amp[0], src_amp, nbodyfast->problem_size, nbodyfast->device_name[0]);
		nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(d_field_amp[0], 0, nbodyfast->problem_size, nbodyfast->device_name[0]); 
		cudaDeviceSynchronize();			

		return 0;
	}

	// default source copying for direct methods
	if (nbodyfast->algo_name == "direct")
	{
		for (int i = 0; i < nbodyfast->problem_size; i++)
		{	
			src_amp[i] = FP_TYPE(_charge[i]);
			field_amp[i] = 0.0f;
		}

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_src_amp[0], src_amp, nbodyfast->problem_size, nbodyfast->gpu->dev_list[0].index);
		nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(d_field_amp[0], 0, nbodyfast->problem_size, nbodyfast->gpu->dev_list[0].index);
		cudaDeviceSynchronize();			

		return 0;

	}
	
	std::cout << "unknown algorithm..." << std::endl;
	exit(0);
	return -1;

}
// multi GPU calculation goes here
int FieldStaticScalarGpu :: set_src_amp_multi(double *_charge)
{
	cudaDeviceSynchronize();		
	if (nbodyfast->algo_name == "nufft")
	{
		int **src_trans_idx_dev = nbodyfast->nufft->get_src_trans_idx_dev();
		int *src_trans_idx = nbodyfast->nufft->get_src_trans_idx();
#pragma omp barrier
#pragma omp parallel
		{
			int _thread_id = omp_get_thread_num();
			int _cur_dev = nbodyfast->device_name[_thread_id];
			cudaError_t _cuda_error;

			for (int i = 0; i < nbodyfast->src_size_dev[_thread_id]; i++)
			{
				src_amp_dev[_thread_id][i] = _charge[src_trans_idx[src_trans_idx_dev[_thread_id][i]]];

//				//src_amp_dev[_thread_id][i] = src_amp[src_trans_idx_dev[_thread_id][i]];
//				//src_amp_dev[_thread_id][i] = _charge[src_trans_idx_dev[_thread_id][i]];

				field_amp_dev[_thread_id][i] = 0.0f;
			}
			
			// copy source amplitudes to device and set field amplitudes to 0
			nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_src_amp[_thread_id], src_amp_dev[_thread_id], nbodyfast->src_size_dev[_thread_id], _cur_dev);
			nbodyfast->gpu->memory_gpu->memset_device<FP_TYPE>(d_field_amp[_thread_id], 0, nbodyfast->obs_size_dev[_thread_id], _cur_dev); 
		}
		cudaDeviceSynchronize();		
#pragma omp barrier
		return 0;
	}

	// default source copying for direct methods
	if (nbodyfast->algo_name == "direct")
	{
		std::cout << "direct method on multi GPU is not ready yet" << std::endl;
		std::cout << "stopped at \"int FieldStaticScalarGpu :: set_src_amp_single(double *_charge)\" " << std::endl;
		
		exit(0);

		return 0;

	}
	
	std::cout << "unknown algorithm..." << std::endl;
	exit(0);
	return -1;

}

// Transfer field amplitudes to outside, the set_fld_amp is usually called at every iteration, at the end of execution. They are symetrical to set_src series of subroutines
int FieldStaticScalarGpu :: set_fld_amp(double *_field)
{
	nbodyfast->error->last_error = nbodyfast->multi_device ? set_fld_amp_multi(_field) : set_fld_amp_single(_field);
	return 0;
}
// single GPU calculation goes here
int FieldStaticScalarGpu :: set_fld_amp_single(double *_field)
{
	cudaDeviceSynchronize();		
	if (nbodyfast->algo_name == "nufft")
	{
		nbodyfast->gpu->memory_gpu->memcpy_device_to_host<FP_TYPE>(field_amp, d_field_amp[0], nbodyfast->problem_size, nbodyfast->gpu->dev_list[0].index);
		int *obs_trans_idx = nbodyfast->nufft->get_obs_trans_idx();
#pragma omp parallel for
		for (int i = 0; i < nbodyfast->problem_size; i++)
		{
			_field[obs_trans_idx[i]] = field_amp[i];
		}	
		cudaDeviceSynchronize();		
		return 0;
	}

	// default source copying for direct methods
	if (nbodyfast->algo_name == "direct")
	{
		nbodyfast->gpu->memory_gpu->memcpy_device_to_host<FP_TYPE>(field_amp, d_field_amp[0], nbodyfast->problem_size, nbodyfast->gpu->dev_list[0].index);

		for (int i = 0; i < nbodyfast->problem_size; i++)
		{	
			_field[i] = double(field_amp[i]);
			src_amp[i] = 0.0f;
			field_amp[i] = 0.0f;
		}

		//cudaDeviceSynchronize();			

		return 0;
	}
	
	std::cout << "unknown algorithm..." << std::endl;
	exit(0);
	return -1;
}
// multi GPU calculation goes here
int FieldStaticScalarGpu :: set_fld_amp_multi(double *_field)
{
	cudaDeviceSynchronize();		
	if (nbodyfast->algo_name == "nufft")
	{
		int **obs_trans_idx_dev = nbodyfast->nufft->get_obs_trans_idx_dev();
		int *obs_trans_idx = nbodyfast->nufft->get_obs_trans_idx();
#pragma omp barrier
#pragma omp parallel
		{

			int _thread_id = omp_get_thread_num();
			int _cur_dev = nbodyfast->device_name[_thread_id];
			cudaError_t _cuda_error;
			nbodyfast->gpu->memory_gpu->memcpy_device_to_host<FP_TYPE>(field_amp_dev[_thread_id], d_field_amp[_thread_id], nbodyfast->obs_size_dev[_thread_id], _cur_dev);

			for (int i = 0; i < nbodyfast->obs_size_act_dev[_thread_id]; i++)
			{
				//field_amp[obs_trans_idx_dev[_thread_id][i]] = field_amp_dev[_thread_id][i];
				_field[obs_trans_idx[obs_trans_idx_dev[_thread_id][i]]] = field_amp_dev[_thread_id][i];
				//_field[obs_trans_idx_dev[_thread_id][i]] = field_amp_dev[_thread_id][i];

			}
			//cudaDeviceSynchronize();			
		}
//#pragma omp barrier
//#pragma omp parallel for
//		for (int i = 0; i < problem_size; i++)
//		{
//			_field[obs_trans_idx[i]] = field_amp[i];
//		}
#pragma omp barrier
		cudaDeviceSynchronize();		
		return 0;
	}

	// default source copying for direct methods
	if (nbodyfast->algo_name == "direct")
	{
		std::cout << "direct method on multi GPU is not ready yet" << std::endl;
		std::cout << "stopped at \"int FieldStaticScalarGpu :: set_src_amp_single(double *_charge)\" " << std::endl;
		
		exit(0);

		return 0;

	}
	
	std::cout << "unknown algorithm..." << std::endl;
	exit(0);
	return -1;
}


}
