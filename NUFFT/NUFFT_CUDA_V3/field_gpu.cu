/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_gpu.cu: class definition of Class FieldGpu
*/
#include "error.h"
#include "memory.h"
#include "memory_gpu.h"
#include "field_gpu.h"
#include "gpu.h"

namespace NBODYFAST_NS{
FieldGpu :: FieldGpu(class NBODYFAST *n_ptr) : Field(n_ptr)
{
	d_src_coord = NULL;
	d_obs_coord = NULL;	
}

FieldGpu::~FieldGpu()
{
	for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id++)
	{
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &d_src_coord[_thread_id], _cur_dev);

		if (src_obs_overlap == false)
		{
			nbodyfast->gpu->memory_gpu->free_device<FP_TYPE>(_cuda_error, &d_obs_coord[_thread_id], _cur_dev);
		}
	}

	nbodyfast->memory->free_host<FP_TYPE*>(&d_src_coord);
	if (src_obs_overlap == false)
	{
		nbodyfast->memory->free_host<FP_TYPE*>(&d_obs_coord);
	}
}

int FieldGpu :: array_alloc_multi_interface()
{
	nbodyfast->error->last_error = Field :: array_alloc_multi_interface();
	nbodyfast->error->last_error = coord_alloc_multi(); // just transfer the programm to coord_alloc_multi().
	return 0;
}
int FieldGpu :: coord_alloc(double *SrcCoord, double *ObsCoord)
{
	Field::coord_alloc(SrcCoord, ObsCoord); // call base class coord_alloc

	// allocate the list
	nbodyfast->memory->alloc_host<FP_TYPE*>(&d_src_coord, nbodyfast->num_devices, "field->d_src_coord");
	for (int i = 0; i < nbodyfast->num_devices; i++)
	{
		d_src_coord[i] = NULL;
	}

	if (src_obs_overlap == false)
	{
		nbodyfast->memory->alloc_host<FP_TYPE*>(&d_obs_coord, nbodyfast->num_devices, "field->d_obs_coord");
		for (int i = 0; i < nbodyfast->num_devices; i++)
		{
			d_obs_coord[i] = NULL;
		}
	}
	else
	{
		d_obs_coord = d_src_coord;
	}

	if (nbodyfast->multi_device == false) coord_alloc_single(); // for single GPU calculation
	
	// If multi GPU, more coord allocation will happen after the preprocessing, via array_alloc_multi_interface()=>coord_alloc_multi()

	return 0;
}

int FieldGpu :: coord_alloc_single()
{
	cudaError_t _cuda_error;

	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_src_coord[0], nbodyfast->problem_size * 3, "field->d_src_coord[0]", nbodyfast->gpu->dev_list[0].index);
	if (nbodyfast->algo_name == "direct")
	{
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_src_coord[0], src_coord, nbodyfast->problem_size * 3, nbodyfast->gpu->dev_list[0].index);
	}

	if (src_obs_overlap == false)
	{
		nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_obs_coord[0], nbodyfast->problem_size * 3, "field->d_obs_coord[0]", nbodyfast->gpu->dev_list[0].index);
	
		if (nbodyfast->algo_name == "direct")
		{
			nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_obs_coord[0], obs_coord, nbodyfast->problem_size * 3, nbodyfast->gpu->dev_list[0].index);
		}

	}

	// Setting the values of d_src_coord and d_obs_coord will happen after the preprocessing if using NUFFT. 


	return 0;

}
int FieldGpu :: coord_alloc_multi()
{
#pragma omp barrier
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];

		cudaError_t _cuda_error;
		std::stringstream _array_name;
		_array_name <<"nufft->d_src_coord[" << _thread_id << "]";
		{
			nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_src_coord[_thread_id], nbodyfast->src_size_dev[_thread_id] * 3, _array_name.str(), _cur_dev); 
		}
		if (nbodyfast->algo_name == "direct")
		{
			//nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_src_coord[_thread_id], src_coord_dev[_thread_id], problem_size * 3, _cur_dev);
		}

		// if sources and observers are not overlapping
		// this has not been implemented yet
		if (src_obs_overlap == false)
		{
			std::cout << "sources and observers should overlap with each other. exit in function FieldGpu :: coord_alloc_multi() " << std::endl;
			exit(0);
			/*
			_array_name.str("");
			_array_name <<"nufft->d_obs_coord[" << _thread_id << "]";
			{
				nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &d_obs_coord[_thread_id], nbodyfast->obs_size_dev[_thread_id] * 3, _array_name.str(), _cur_dev);
			}
			if (nbodyfast->algo_name == "direct")
			{
				nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(d_obs_coord[_thread_id], obs_coord_dev[_thread_id], nbodyfast->problem_size * 3, _cur_dev);
			}
			*/
		}

	}	
#pragma omp barrier
//	nbodyfast->gpu->memory_gpu->output_allocated_list();

	// Setting the values of d_src_coord and d_obs_coord will happen after the preprocessing if using NUFFT. 
	return 0;
}

}




