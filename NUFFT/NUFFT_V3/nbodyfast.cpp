/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nbodyfast.cpp: class definition of Class NBODYFAST
*/
#include <iostream>
#include "nbodyfast.h"
#include "domain.h"
#include "error.h"
#include "field.h"
#include "input.h"
#include "memory.h"
#include "nufft.h"
#include "output.h"
#include "timer.h"
#include "field_static_scalar.h"
#include "direct_static_scalar.h"
#include "nufft_static_scalar.h"
#include "field_static_vector.h"
#include "direct_static_vector.h"
#include "nufft_static_vector.h"

#ifdef _USE_CUDA
#include "gpu_headers.h"
#endif

namespace NBODYFAST_NS{
NBODYFAST :: NBODYFAST()
{
	memory = NULL; domain = NULL; error = NULL; input = NULL;
	output = NULL; timer = NULL; 	gpu = NULL; direct = NULL; nufft = NULL; 
	src_size_dev = NULL;	obs_size_dev = NULL;
	src_size_act_dev = NULL;	obs_size_act_dev = NULL;
	algo_name = "not-set"; gpu_on = false;
	multi_device = false;
}
NBODYFAST :: ~NBODYFAST()
{

}

void NBODYFAST::global_init(double *_src_coord, double *_obs_coord, int *_problem_size, double *_wavenumber, int *_field_type, int *_gpu_on, int *_algo, int *_num_devices, int *_gpu_list)
{
	// Set calculation parameters
	problem_size = *_problem_size;
	device_name = new int[*_num_devices];
	memory = new Memory(this);
	domain = new Domain(this);
	error = new Error(this);
	input = new Input(this);
	output = new Output(this);
	timer = new Timer(this);

	num_devices = *_num_devices;
	// the number of OpenMP threads before calling NUFFT
	_num_threads_sys = 1;
#ifdef _OPEN_MP
	_num_threads_sys = omp_get_num_threads();
	
	omp_set_num_threads(num_devices); // Set number of OpenMP threads to num_devices
#endif

#ifdef _USE_CUDA
	if (*_gpu_on > 0) 	// If _gpu_on, create Gpu object
	{
		gpu = new Gpu(this, _num_devices, _gpu_list);
		gpu_on = true;
	}
	else
	{
		gpu = NULL;
		gpu_on = false;
		std::cout << "Running NBODYFAST on CPU." << std::endl;
	}

	if (num_devices > 1)
	{
		multi_device = true;
	}
	
	field_type = *_field_type;
	// switch among field types
	switch(*_field_type)
	{
	case 1: // case 1 is static scalar field
		if (gpu_on == true)
		{
			field = new FieldStaticScalarGpu(this);
		}
		else
		{
			field = new FieldStaticScalar(this);
		}
		// switch among algorithms
		switch(*_algo)
		{
		case 0:
			algo_name = "direct";
			if (gpu_on == true)
			{
				direct = new DirectStaticScalarGpu(this);
			}
			else
			{
				direct = new DirectStaticScalar(this);
			}
			break;
		case 1:
			algo_name = "nufft";
			if (gpu_on == true)
			{
				nufft = new NufftStaticScalarGpu(this);
			}
			else
			{
				nufft = new NufftStaticScalar(this);
			}
			break;
		}
	break;
	case 2: // case 2 is static vector field
		if (gpu_on == true)
		{
			field = new FieldStaticVectorGpu(this);
		}
		else
		{
			field = new FieldStaticVector(this);
		}
		// switch among algorithms
		switch(*_algo)
		{
		case 0:
			algo_name = "direct";
			if (gpu_on == true)
			{
				direct = new DirectStaticVectorGpu(this);
			}
			else
			{
				direct = new DirectStaticVector(this);
			}
			break;
		case 1:
			algo_name = "nufft";
			if (gpu_on == true)
			{
				nufft = new NufftStaticVectorGpu(this);
			}
			else
			{
#if !S_V_IMPLEMENT
			std::cout << "static vector field on CPU has not been implemented...exiting...";
			system("pause");
#else
				nufft = new NufftStaticVector(this);
#endif
			}
			break;
		}
		break;
	}
#else
	if (*_gpu_on > 0)
	{
		std::cout << "Warning: please define _USE_CUDA if you want the code to run on GPUs." << std::endl;
		std::cout << "Parameter gpu_on and _USE_CUDA are conflicting." << std::endl;
	}

	std::cout << "Running NBODYFAST on CPU using" << *_num_devices << " devices." << std::endl;

	switch(*_field_type)
	{
	case 1:
		field = new FieldStaticScalar(this);

		switch(*_algo)
		{
		case 0:
			algo_name = "direct";
			direct = new DirectStaticScalar(this);
			break;
		case 1:
			algo_name = "nufft";
			nufft = new NufftStaticScalar(this);
			break;
		}
		break;
	}
#endif

	//timer->time_stamp("MEMO");
	error->last_error = field->coord_alloc(_src_coord, _obs_coord); // allocating arrays for storing coordinates
	error->last_error = field->amp_field_alloc(); // allocating arrays for storing source amp. and field amp.
	
	//timer->time_stamp("MEMO");

	timer->time_stamp("PREP");
	error->last_error = domain->setup(); // computational domain setup

	if (algo_name == "nufft")
	{
		error->last_error = nufft->preprocessing(); // nufft preprocessing, including box division, reordering sources, etc.
	}
	timer->time_stamp("PREP");

#ifdef _OPEN_MP
	omp_set_num_threads(_num_threads_sys); // while exiting the NUFFT library, set no. of omp threads to what it used to be before entering the NUFFT library
#endif
}

void NBODYFAST::execute(double *src_amp, double *fld_amp)
{

#ifdef _OPEN_MP
	omp_set_num_threads(num_devices); // set the no. of omp threads
#endif

	timer->time_stamp("MEMO");
	error->last_error = field->set_src_amp(src_amp); // set source amp. would require transfer data from CPU to GPU is GPU is on.
	timer->time_stamp("MEMO");

	timer->time_stamp("EXEC");
	error->last_error = field->execute(); // execution
	timer->time_stamp("EXEC");

	timer->time_stamp("MEMO");
	error->last_error = field->set_fld_amp(fld_amp); // set field amp. would require transfer data from CPU to GPU is GPU is on.
	timer->time_stamp("MEMO");
#ifdef _OPEN_MP
	omp_set_num_threads(_num_threads_sys); // restore the no. of omp threads
#endif

}

void NBODYFAST::global_deinit()
{
#ifdef _OPEN_MP
	omp_set_num_threads(num_devices); // set the no. of omp threads
#endif
	
	// delete all the dynamically allocated data on the heap
	if (algo_name == "nufft")
	{
		delete nufft;
		nufft = NULL;
	}
	if (algo_name == "direct")
	{
		delete direct;
		direct = NULL;
	}

	delete field;
	field = NULL;

	if (gpu_on == true)
	{
		delete gpu;
		gpu = NULL;
	}
	delete timer;
	timer = NULL;

	delete output;
	output = NULL;

	delete input;
	input = NULL;

	delete error;
	error = NULL;

	delete domain;
	domain = NULL;

	delete memory;
	memory = NULL;


#ifdef _OPEN_MP
	omp_set_num_threads(_num_threads_sys); // restore the no. of omp threads
#endif
}
}
