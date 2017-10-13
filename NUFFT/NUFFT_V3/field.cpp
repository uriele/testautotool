/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field.cpp: class definition of Class Field
*/
#include "error.h"
#include "field.h"
#include "memory.h"
#include "nbodyfast.h"
#include "nufft.h"
#include "direct.h"

namespace NBODYFAST_NS{
FP_TYPE Field::max_domain_size = 0;
int Field::coord_alloc(double *SrcCoord, double *ObsCoord) 
{
	// allocating source coordinates array
 	nbodyfast->memory->alloc_host<FP_TYPE>(&src_coord, nbodyfast->problem_size * 3, "field->src_coord");
	
	// if the source and observer coincides, set obs_cood the same with src_coord
	if (SrcCoord == ObsCoord)
	{
		src_obs_overlap = true;
		obs_coord = src_coord;
	}
	else // otherwise allocate a seperate obs_coord
	{
		src_obs_overlap = false;
		nbodyfast->memory->alloc_host<FP_TYPE>(&obs_coord, nbodyfast->problem_size * 3, "field->obs_coord");
	}
	
	// copy the coordinates value from outside
	for (int i = 0; i < nbodyfast->problem_size * 3; i++)
	{
		src_coord[i] = SrcCoord[i];
		if (src_obs_overlap == false)
		{
			obs_coord[i] = ObsCoord[i];
		}
	}	
	return 0;
}
// can't remember why I have this wrapper here...but what it does is only transfer the process to Field::coord_alloc_multi()
int Field::array_alloc_multi_interface()
{
	nbodyfast->error->last_error = coord_alloc_multi();
	return 0;
}

// dummy subroutine just for symmetry...
int Field::coord_alloc_single()
{
	return 0;
}

// allocating arrays for calculations on multiGPUs
// this subroutine is called after NUFFT has completed it preprocessing because the tasks division and mapping are at box level
// if we are not using NUFFT, this subroutine will never be called
int Field::coord_alloc_multi()
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// allocating the arrays on the host, one copy for each device
	int **src_trans_idx_dev;		
	int **obs_trans_idx_dev;		
	
	nbodyfast->memory->alloc_host<FP_TYPE*>(&src_coord_dev, nbodyfast->num_devices, "field->src_coord_dev");
	src_trans_idx_dev = nbodyfast->nufft->get_src_trans_idx_dev();		
	
	if (src_obs_overlap == false)
	{
		nbodyfast->memory->alloc_host<FP_TYPE*>(&obs_coord_dev, nbodyfast->num_devices, "field->obs_coord_dev");
		obs_trans_idx_dev = nbodyfast->nufft->get_obs_trans_idx_dev();		
	}
	else
	{
		obs_coord_dev = src_coord_dev;
		obs_trans_idx_dev = src_trans_idx_dev;
	}
#pragma omp barrier

#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];

		std::stringstream _array_name;
		_array_name <<"nufft->src_coord_dev[" << _cur_dev << "]";
		
		src_coord_dev[_thread_id] = NULL;
//#pragma omp critical
		{
			nbodyfast->memory->alloc_host<FP_TYPE>(&src_coord_dev[_thread_id], nbodyfast->src_size_dev[_thread_id] * 3, _array_name.str());
		}
		
		if (src_obs_overlap == false)
		{
			std::cout << "sources and observers should overlap with each other. exit in function int Field::coord_alloc_multi() " << std::endl;
			exit(0);
/*
			_array_name.str("");
			_array_name <<"nufft->obs_coord_dev[" << _cur_dev << "]";

			obs_coord_dev[_thread_id] = NULL;
	//#pragma omp critical
			{
				nbodyfast->memory->alloc_host<FP_TYPE>(&obs_coord_dev[_thread_id], nbodyfast->obs_size_dev[_thread_id] * 3, _array_name.str());
			}
*/
		}


		for (int j = 0; j < nbodyfast->src_size_dev[_thread_id]; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				src_coord_dev[_thread_id][j + k * nbodyfast->src_size_dev[_thread_id]] = src_coord[src_trans_idx_dev[_thread_id][j] + k * nbodyfast->problem_size];
			}
		}
	}
#pragma omp barrier
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//	nbodyfast->memory->output_allocated_list();
	return 0;
}

Field::Field(class NBODYFAST *n_ptr)
{
	nbodyfast = n_ptr;
	src_coord = NULL;
	obs_coord = NULL;
	src_coord_dev = NULL;
	obs_coord_dev = NULL;
}
Field::~Field()
{
	if (nbodyfast->multi_device == true)
	{
		for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id ++)
		{
			nbodyfast->memory->free_host<FP_TYPE>(&src_coord_dev[_thread_id]);
		}

		nbodyfast->memory->free_host<FP_TYPE*>(&src_coord_dev);
	}
	nbodyfast->memory->free_host<FP_TYPE>(&src_coord);

	if (src_obs_overlap == false)
	{
		if (nbodyfast->multi_device == true)
		{

			for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id ++)
			{
				nbodyfast->memory->free_host<FP_TYPE>(&obs_coord_dev[_thread_id]);
			}
			nbodyfast->memory->free_host<FP_TYPE*>(&obs_coord_dev);
		}
		nbodyfast->memory->free_host<FP_TYPE>(&obs_coord);
	
	}
}
// Field::execute() calls the correct execute() in corresponding algorithm class
int Field :: execute()
{
	if (nbodyfast->algo_name == "nufft")
	{
		nbodyfast->nufft->execution();
		return 0;
	}
	if (nbodyfast->algo_name == "direct")
	{
		nbodyfast->direct->execution();
		return 0;
	}
	std::cout << "unknown algorithm..." << std::endl;
	exit(0);
	return 0;
}
}