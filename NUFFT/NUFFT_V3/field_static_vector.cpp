/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_static_vector.cpp: class definition of Class FieldStaticVector
*/
#include "error.h"
#include "field_static_vector.h"
#include "memory.h"
#include "nbodyfast.h"
#include "nufft.h"
#include "direct.h"

namespace NBODYFAST_NS{
FieldStaticVector :: FieldStaticVector(class NBODYFAST *n_ptr) : Field(n_ptr)
{
	src_amp = NULL;
	field_amp = NULL;
	src_amp_dev = NULL;
	field_amp_dev = NULL;
}
FieldStaticVector :: ~FieldStaticVector()
{
	if (nbodyfast->multi_device == true)
	{
		for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id ++)
		{
			nbodyfast->memory->free_host<FP_TYPE>(&src_amp_dev[_thread_id]);
			nbodyfast->memory->free_host<FP_TYPE>(&field_amp_dev[_thread_id]);
		}

		nbodyfast->memory->free_host<FP_TYPE*>(&field_amp_dev);
		nbodyfast->memory->free_host<FP_TYPE*>(&src_amp_dev);
	}
	nbodyfast->memory->free_host<FP_TYPE>(&field_amp);
	nbodyfast->memory->free_host<FP_TYPE>(&src_amp);
}
int FieldStaticVector :: array_alloc_multi_interface()
{
	nbodyfast->error->last_error = amp_field_alloc_multi();
	return 0;
}
int FieldStaticVector :: amp_field_alloc()
{
	nbodyfast->memory->alloc_host<FP_TYPE>(&src_amp, FIELD_DIM*nbodyfast->problem_size, "field->src_amp");
	nbodyfast->memory->alloc_host<FP_TYPE>(&field_amp, FIELD_DIM*nbodyfast->problem_size, "field->field_amp");


	return 0;
}

int FieldStaticVector :: set_src_amp(double *_charge)
{
	// source copying for nufft method
	// sources are reordered according to src_trans_idx matrix
	if (nbodyfast->algo_name == "nufft")
	{
		int *src_trans_idx = nbodyfast->nufft->get_src_trans_idx();
#pragma omp parallel for
		for (int i = 0; i < nbodyfast->problem_size; i++)
		{
			for(unsigned int j = 0; j < FIELD_DIM; j++){
				src_amp[i+j*nbodyfast->problem_size] = static_cast<FP_TYPE>(_charge[src_trans_idx[i]+j*nbodyfast->problem_size]);
				field_amp[i+j*nbodyfast->problem_size] = 0.0f;
			}
		}
		return 0;
	}

	// default source copying for direct methods
	if (nbodyfast->algo_name == "direct")
	{
#pragma omp parallel for
		for (int i = 0; i < FIELD_DIM*nbodyfast->problem_size; i++)
		{	
				src_amp[i] = static_cast<FP_TYPE>(_charge[i]);
				field_amp[i] = 0.0f;
		}
		return 0;
	}
	
	std::cout << "unknown algorithm...exiting..." << std::endl;
	exit(0);
	return -1;
}
int FieldStaticVector :: set_fld_amp(double *_field)
{
	std::cout << "arrive in set_fid_amp in field_static_scalar.cpp" << std::endl;
	// source copying for nufft method
	// sources are reordered according to src_trans_idx matrix
	if (nbodyfast->algo_name == "nufft")
	{
		int *obs_trans_idx = nbodyfast->nufft->get_obs_trans_idx();
#pragma omp parallel for
		for (int i = 0; i < nbodyfast->problem_size; i++)
		{
			for(unsigned int j = 0; j < FIELD_DIM; j++)
				_field[obs_trans_idx[i]+j*nbodyfast->problem_size] = static_cast<double>(field_amp[i+j*nbodyfast->problem_size]);
		}
		return 0;
	}

	// default source copying for direct methods
	if (nbodyfast->algo_name == "direct")
	{
#pragma omp parallel for
		for (int i = 0; i < FIELD_DIM*nbodyfast->problem_size; i++)
		{	
			_field[i] = static_cast<double>(field_amp[i]);
		}
		return 0;
	}
	
	std::cout << "unknown algorithm..." << std::endl;
	exit(0);
	return -1;
}

int FieldStaticVector :: amp_field_alloc_multi()
{
	// allocating arrays for multGPU calculation
	// these arrays are host copies of arrays that will be transferred to each GPU
	nbodyfast->memory->alloc_host<FP_TYPE*>(&src_amp_dev, nbodyfast->num_devices, "field->src_amp_dev");
	nbodyfast->memory->alloc_host<FP_TYPE*>(&field_amp_dev, nbodyfast->num_devices, "field->field_amp_dev");

	for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id ++)
	{
		std::stringstream _array_name;
		_array_name <<"nufft->src_amp_dev[" << _thread_id << "]";

		src_amp_dev[_thread_id] = NULL;
		{
			nbodyfast->memory->alloc_host<FP_TYPE>(&src_amp_dev[_thread_id], FIELD_DIM*nbodyfast->src_size_dev[_thread_id], _array_name.str());
		}

		_array_name.str("");
		_array_name <<"nufft->field_amp_dev[" << _thread_id << "]";

		field_amp_dev[_thread_id] = NULL;
		{
			nbodyfast->memory->alloc_host<FP_TYPE>(&field_amp_dev[_thread_id], FIELD_DIM*nbodyfast->obs_size_dev[_thread_id], _array_name.str());
		}
	}
	return 0;
}
}
