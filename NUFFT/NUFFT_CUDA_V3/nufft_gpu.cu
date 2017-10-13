/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft_gpu.cu: class declaration of Class NufftGpu, Class NufftParamGpu, Class NufftArrayGpu
*/
#include "domain.h"
#include "error.h"
#include "field_gpu.h"
#include "gpu.h"
#include "memory_gpu.h"
#include "nbodyfast.h"
#include "nufft_gpu.h"
namespace NBODYFAST_NS{
// this is used to copy data from an existing NufftParamGpu object to a newly generated NufftParamGpu object
void NufftParamGpu :: set_value(NufftGpu *nufft_ptr)
{
	nbodyfast = nufft_ptr->nbodyfast;

	epsilon = Field::epsilon();
	
	if (nufft_ptr->interp_order[0] == 1)
	{
		linear_interp = true;
	}
	else if (nufft_ptr->interp_order[0] == 3)
	{
		linear_interp = false;
	}
	else
	{
		std::cout << "The library does not support interpolation order other than linear and cubic" << std::endl;
		exit(0);
	}

	problem_size = nufft_ptr->problem_size;
	interp_nodes_per_box = nufft_ptr->interp_nodes_per_box;
	total_num_boxes = nufft_ptr->total_num_boxes;
	src_num_nonempty_boxes = nufft_ptr->src_num_nonempty_boxes;
	obs_num_nonempty_boxes = nufft_ptr->obs_num_nonempty_boxes;
	total_num_grid_pts = nufft_ptr->total_num_grid_pts;
	total_num_near_grid_pts = nufft_ptr->total_num_near_grid_pts;
	near_box_cnt = nufft_ptr->near_box_cnt;//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
	max_near_src_cnt = nufft_ptr->max_near_src_cnt;
	near_correct_layer = nufft_ptr->near_correct_layer;

	total_num_fft_pts = nufft_ptr->total_num_fft_pts;
	total_num_green_pts = nufft_ptr->total_num_green_pts;
	total_num_fft_r2c_pts = nufft_ptr->total_num_fft_r2c_pts;

	for (int i = 0; i < 3; i++)
	{
		interp_order[i] = nufft_ptr->interp_order[i];//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
		num_nodes[i] = nufft_ptr->num_nodes[i];
		num_boxes[i] = nufft_ptr->num_boxes[i];
		num_grid_pts[i] = nufft_ptr->num_grid_pts[i];
		near_num_grid_pts[i] = nufft_ptr->near_num_grid_pts[i];

		fft_size[i] = nufft_ptr->fft_size[i];
		green_size[i] = nufft_ptr->green_size[i];
		fft_r2c_size[i] = nufft_ptr->fft_r2c_size[i];

		cell_size[i] = nufft_ptr->cell_size[i];
		box_size[i] = nufft_ptr->box_size[i];

		src_domain_center[i] = nbodyfast->domain->src_domain_center[i];
		obs_domain_center[i]  = nbodyfast->domain->obs_domain_center[i];
		unified_domain_size[i] = nbodyfast->domain->unified_domain_size[i];
	}
	for (int i = 0; i < 9; i++)
	{
		src_domain_range[i] = nbodyfast->domain->src_domain_range[i];
		obs_domain_range[i] = nbodyfast->domain->obs_domain_range[i];
	}
}

void NufftArrayGpu :: set_value(NufftGpu *nufft_ptr)
{
	nbodyfast = nufft_ptr->nbodyfast;

	d_src_box_map = NULL;
	d_obs_box_map = NULL;

	d_src_trans_idx = NULL;
	d_obs_trans_idx = NULL;

	//d_near_box_list = NULL;
	d_near_bound_list = NULL;
	d_src_box_list = NULL;

	d_src_grid_coord = NULL;
	d_obs_grid_coord = NULL;
}
NufftGpu :: NufftGpu(NBODYFAST *n_ptr) : Nufft(n_ptr)
{
	nbodyfast = n_ptr;
	nufft_const_gpu = NULL;
	d_nufft_const_gpu = NULL;
	nufft_array_gpu = NULL;
	d_nufft_array_gpu = NULL;

	field_gpu = dynamic_cast<FieldGpu*>(nbodyfast->field);
}

NufftGpu :: ~NufftGpu()
{
	cudaError_t _cuda_error;
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;

		nbodyfast->gpu->memory_gpu->free_device<NufftParamGpu>(_cuda_error, &d_nufft_const_gpu[_thread_id], _cur_dev);	
		if (_thread_id == 0)
		{
			//nbodyfast->gpu->memory_gpu->free_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_fft_inplace_b), _cur_dev);	
			nbodyfast->gpu->memory_gpu->free_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_fft_inplace_r2c), _cur_dev);	
			cufftDestroy(nufft_array_gpu[_thread_id]->cufftPlan_r2c);
			cufftDestroy(nufft_array_gpu[_thread_id]->cufftPlan_c2r);
		}

		//nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_near_box_list), _cur_dev);	
		nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_near_bound_list), _cur_dev);	
		
		nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_trans_idx), _cur_dev);	
		nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_box_map), _cur_dev);	
		nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_box_list), _cur_dev);	
		nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_box_list_inv), _cur_dev);	

		if (nbodyfast->field->src_obs_overlap == false)
		{
			nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_obs_trans_idx), _cur_dev);	
			nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_obs_box_map), _cur_dev);
			nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_obs_box_list), _cur_dev);
			nbodyfast->gpu->memory_gpu->free_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_obs_box_list_inv), _cur_dev);
		}
	}

	for (int i = 0; i < nbodyfast->num_devices; i++)
	{
		nbodyfast->memory->free_host<NufftParamGpu>(&nufft_const_gpu[i]);
		nbodyfast->memory->free_host<NufftArrayGpu>(&nufft_array_gpu[i]);
		nbodyfast->memory->free_host<NufftParamGpu>(&d_nufft_const_gpu[i]);

	}
	nbodyfast->memory->free_host<NufftParamGpu*>(&nufft_const_gpu);
	nbodyfast->memory->free_host<NufftParamGpu*>(&d_nufft_const_gpu);
	nbodyfast->memory->free_host<NufftArrayGpu*>(&nufft_array_gpu);

}

int NufftGpu :: preprocessing()
{
	nbodyfast->error->last_error = nbodyfast->multi_device ? preprocessing_multi() : preprocessing_single();
	return 0;
}
int NufftGpu :: preprocessing_single()//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
{
	int _cur_dev = nbodyfast->device_name[0];
	cudaError_t _cuda_error;

	// Copy sorted coordinates to GPU
	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(field_gpu->d_src_coord[0], field_gpu->src_coord, problem_size * 3, _cur_dev);

	if (field_gpu->src_obs_overlap == false)
	{
		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(field_gpu->d_obs_coord[0], field_gpu->obs_coord, problem_size * 3, _cur_dev);
	}

	// Allocate and copy domain and grid related parameters to GPU
	// This is the host copy of this data
	nbodyfast->memory->alloc_host<NufftParamGpu*>(&nufft_const_gpu, 1, "nufft->nufft_const_gpu");
	nufft_const_gpu[0] = NULL;
	nbodyfast->memory->alloc_host<NufftParamGpu>(&nufft_const_gpu[0], 1, "nufft->nufft_const_gpu[0]");
	nufft_const_gpu[0]->set_value(this);

	// total_num_boxes is set to number of boxes processed by the current device instead of "total"
	nufft_const_gpu[0]->total_num_boxes_dev = total_num_boxes;
	nufft_const_gpu[0]->src_size_dev = nbodyfast->problem_size;
	nufft_const_gpu[0]->obs_size_dev = nbodyfast->problem_size;

	// Allocate grid related arrays on GPU
	nbodyfast->memory->alloc_host<NufftArrayGpu*>(&nufft_array_gpu, 1, "nufft->nufft_array_gpu");	
	nufft_array_gpu[0] = NULL;
	nbodyfast->memory->alloc_host<NufftArrayGpu>(&nufft_array_gpu[0], 1, "nufft->nufft_array_gpu[0]");
	nufft_array_gpu[0]->set_value(this);

	nufft_array_gpu[0]->d_src_coord = field_gpu->d_src_coord[0];
	nufft_array_gpu[0]->d_obs_coord = field_gpu->d_obs_coord[0];

	// Allocation
	int num_near_box_per_dim = 2*near_correct_layer+1;
	int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;

	nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_src_box_list), total_num_boxes + 1, "nufft->nufft_array_gpu[0]->d_src_box_list", _cur_dev);

	nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_src_box_map), total_num_boxes * 3, "nufft->nufft_array_gpu[0]->d_src_box_map", _cur_dev);

	nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_src_trans_idx), problem_size, "nufft->nufft_array_gpu[0]->d_src_trans_idx", _cur_dev);

	//nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_near_box_list), total_num_boxes * (total_num_near_box_per_box+1), "nufft->nufft_array_gpu[0]->d_near_box_list", _cur_dev);//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
	nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_near_bound_list), total_num_boxes * 6, "nufft->nufft_array_gpu[0]->d_near_bound_list", _cur_dev);

	nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_src_box_list_inv), total_num_boxes + 1, "nufft->nufft_array_gpu[0]->d_src_box_list_inv", _cur_dev);

	//nufft_array_gpu[0]->d_fft_inplace_r2c should have been allocated in nufft_gpu_static_scalar_gpu.cu/nufft_gpu_static_vector_gpu.cu...
	//which means it should have been a member of nufft_array_gpu_static_scalar/nufft_array_gpu_static_vector
	//lets keep this and fix it after we got correct results.
	if(nbodyfast->get_field_type() == 1)
		nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[0]->d_fft_inplace_r2c), total_num_fft_r2c_pts, "nufft->nufft_array_gpu[0]->d_fft_inplace_r2c", _cur_dev);
	else if(nbodyfast->get_field_type() == 2)
		nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[0]->d_fft_inplace_r2c), FIELD_DIM*total_num_fft_r2c_pts, "nufft->nufft_array_gpu[0]->d_fft_inplace_r2c", _cur_dev);
	nufft_array_gpu[0]->d_fft_inplace_r2c_FP = (FP_TYPE*)(nufft_array_gpu[0]->d_fft_inplace_r2c);
	//nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[0]->d_fft_inplace_b), total_num_fft_pts, "nufft->nufft_array_gpu[0]->d_fft_inplace_b", _cur_dev);

	// build CUFFT plan
	int _fft_size[3];
	for (int m = 0; m < 3; m ++)
	{
		_fft_size[m] = fft_size[2 - m];
	}

	int fft_batch = -1;
	if(nbodyfast->get_field_type() == 1)	fft_batch = 1;
	else if(nbodyfast->get_field_type() == 2)	fft_batch = FIELD_DIM;
	else { std::cout << "unsupported field type in NUFFT...exiting..." << std::endl; exit(0); }

	cufftPlanMany(&nufft_array_gpu[0]->cufftPlan_r2c, 3, _fft_size, NULL, 0, 0, NULL, 0, 0, CUFFT_TYPE_F, fft_batch);
	cufftPlanMany(&nufft_array_gpu[0]->cufftPlan_c2r, 3, _fft_size, NULL, 1, 0, NULL, 1, 0, CUFFT_TYPE_B, fft_batch);

	cufftSetCompatibilityMode(nufft_array_gpu[0]->cufftPlan_r2c, CUFFT_COMPATIBILITY_NATIVE);
	cufftSetCompatibilityMode(nufft_array_gpu[0]->cufftPlan_c2r, CUFFT_COMPATIBILITY_NATIVE);

	
	// Copy

	nufft_array_gpu[0]->d_src_coord = field_gpu->d_src_coord[0];
	nufft_array_gpu[0]->d_obs_coord = field_gpu->d_obs_coord[0];

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_src_box_list, src_box_list, total_num_boxes + 1, _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_src_box_map, src_box_map, total_num_boxes * 3, _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_src_trans_idx, src_trans_idx, problem_size, _cur_dev);

	//nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_near_box_list, near_box_list,  total_num_boxes * (total_num_near_box_per_box+1), _cur_dev);//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_near_bound_list, near_bound_list,  total_num_boxes*6, _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_src_box_list_inv, src_box_list, total_num_boxes + 1, _cur_dev);

	if (field_gpu->src_obs_overlap == false)
	{
			std::cout << "sources and observers should overlap with each other. exit in function int NufftGpu :: preprocessing_single()" << std::endl;
			exit(0);
		/*
		nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_obs_box_map), total_num_boxes * 3, "nufft->nufft_array_gpu[0]->d_obs_box_map", _cur_dev);

		nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[0]->d_obs_trans_idx), problem_size, "nufft->nufft_array_gpu[0]->d_obs_trans_idx", _cur_dev);

	nbodyfast->gpu->memory_gpu->alloc_device<FP_TYPE>(_cuda_error, &(nufft_array_gpu[0]->d_obs_grid_coord), total_num_grid_pts * 3, "nufft->nufft_array_gpu[0]->d_obs_grid_coord", _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_obs_box_map, obs_box_map, total_num_boxes * 3, _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[0]->d_obs_trans_idx, obs_trans_idx, problem_size, _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(nufft_array_gpu[0]->d_obs_grid_coord, obs_grid_coord,  total_num_boxes * 27, _cur_dev);
	*/
	}
	else
	{
		nufft_array_gpu[0]->d_obs_box_map = nufft_array_gpu[0]->d_src_box_map;
		nufft_array_gpu[0]->d_obs_trans_idx = nufft_array_gpu[0]->d_src_trans_idx;
	}

	// This is the device copy of this data
	nbodyfast->memory->alloc_host<NufftParamGpu*>(&d_nufft_const_gpu, 1, "nufft->d_nufft_const_gpu");

	d_nufft_const_gpu[0] = NULL;

	nbodyfast->gpu->memory_gpu->alloc_device<NufftParamGpu>(_cuda_error, &d_nufft_const_gpu[0], 1, "nufft->d_nufft_const_gpu[0]", _cur_dev);

	nbodyfast->gpu->memory_gpu->memcpy_host_to_device<NufftParamGpu>(d_nufft_const_gpu[0], nufft_const_gpu[0],  1, _cur_dev);

	return 0;
}
int NufftGpu :: preprocessing_multi()//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
{

	// This is the host copy of nufft_const_gpu and nufft_array_gpu
	nbodyfast->memory->alloc_host<NufftParamGpu*>(&nufft_const_gpu, nbodyfast->num_devices, "nufft->nufft_const_gpu");

	nbodyfast->memory->alloc_host<NufftArrayGpu*>(&nufft_array_gpu, nbodyfast->num_devices, "nufft->nufft_array_gpu");	

	// This is the device copy of nufft_const_gpu
	// The device copy of nufft_array_gpu will be allocated and set in the functions of the derived classes of nufft_gpu. ("such as nufft_static_scalar_gpu")
	nbodyfast->memory->alloc_host<NufftParamGpu*>(&d_nufft_const_gpu, nbodyfast->num_devices, "nufft->d_nufft_const_gpu");
#pragma omp barrier
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();

		int _cur_dev = nbodyfast->device_name[_thread_id];
		cudaError_t _cuda_error;
		std::stringstream _array_name;

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(field_gpu->d_src_coord[_thread_id], field_gpu->src_coord_dev[_thread_id], nbodyfast->src_size_dev[_thread_id] * 3, _cur_dev);	 

		if (field_gpu->src_obs_overlap == false)
		{
			nbodyfast->gpu->memory_gpu->memcpy_host_to_device<FP_TYPE>(field_gpu->d_obs_coord[_thread_id], field_gpu->src_coord_dev[_thread_id], nbodyfast->src_size_dev[_thread_id] * 3, _cur_dev);
		}

		// Allocate and copy domain and grid related parameters to GPU
		// This is the host copy of this data
		
		nufft_const_gpu[_thread_id] = NULL;
		_array_name.str("");
		_array_name <<"nufft->nufft_const_gpu[" << _thread_id << "]";
		{
			nbodyfast->memory->alloc_host<NufftParamGpu>(&nufft_const_gpu[_thread_id], 1, _array_name.str());
		}
		nufft_const_gpu[_thread_id]->set_value(this);

		//for debug
		std::cout << "nufft_const_gpu[" << _thread_id << "]->near_correct_layer = " << nufft_const_gpu[_thread_id]->near_correct_layer << std::endl;
		//std::cout << "nufft_const_gpu[" << _thread_id << "]->total_num_box_dev = " << nufft_const_gpu[_thread_id]->total_num_boxes_dev << std::endl;
		//for debug

		// total_num_boxes is set to number of boxes processed by the current device instead of "total"
		nufft_const_gpu[_thread_id]->total_num_boxes_dev = src_box_list_dev[_thread_id][0];
		nufft_const_gpu[_thread_id]->src_size_dev = nbodyfast->src_size_dev[_thread_id];
		nufft_const_gpu[_thread_id]->obs_size_dev = nbodyfast->obs_size_dev[_thread_id];

		// Allocate grid related arrays on GPU

		nufft_array_gpu[_thread_id] = NULL;
		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]";
		{
			nbodyfast->memory->alloc_host<NufftArrayGpu>(&nufft_array_gpu[_thread_id], 1, _array_name.str());
		}
		nufft_array_gpu[_thread_id]->set_value(this);

		//std::cout << "field_gpu->d_src_coord[_thread_id]: " << field_gpu->d_src_coord[_thread_id] << std::endl;
		// Allocate arrays inside of nufft_const_gpu and nufft_array_gpu
		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_src_box_list";
		{
			//NOTE: TOTAL_NUM_BOXES is the real glb total_num_boxes, not only on this device
			nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_box_list), total_num_boxes + 1, _array_name.str(), _cur_dev);
		}

		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_src_box_list_inv";
		{
			//NOTE: TOTAL_NUM_BOXES is the real glb total_num_boxes, not only on this device
			nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_box_list_inv), total_num_boxes + 1, _array_name.str(), _cur_dev);
		}

		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_src_box_map";
		{
			//NOTE: TOTAL_NUM_BOXES is the real glb total_num_boxes, not only on this device
			nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_box_map), total_num_boxes * 3, _array_name.str(), _cur_dev);
		}

		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_src_trans_idx";
		{
			nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_src_trans_idx), nbodyfast->src_size_dev[_thread_id], _array_name.str(), _cur_dev);
		}


		if (field_gpu->src_obs_overlap == false)
		{
			// Not yet implemented
			std::cout << "The code is not yet working for src_obs_overalp = false at int NufftGpu :: preprocessing_multi()" << std::endl;
		}
		else
		{
			nufft_array_gpu[_thread_id]->d_obs_box_list = nufft_array_gpu[_thread_id]->d_src_box_list;
			nufft_array_gpu[_thread_id]->d_obs_box_map = nufft_array_gpu[_thread_id]->d_src_box_map;
			nufft_array_gpu[_thread_id]->d_obs_trans_idx = nufft_array_gpu[_thread_id]->d_src_trans_idx;
			nufft_array_gpu[_thread_id]->d_obs_box_list_inv = nufft_array_gpu[_thread_id]->d_src_box_list_inv;

		}

		_array_name.str("");
		/*_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_near_box_list";
		{
			nbodyfast->gpu->memory_gpu->alloc_device<int>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_near_box_list), src_box_list_dev[_thread_id][0] * 28, _array_name.str(), _cur_dev);
		}*/
		
		// Only one copy of the FFT array are needed
		if (_thread_id == 0)
		{
			_array_name.str("");
			_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_fft_inplace_r2c";
			{
				if(nbodyfast->get_field_type() == 1)
					nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_fft_inplace_r2c),
						 total_num_fft_r2c_pts, _array_name.str(), _cur_dev);
				else if(nbodyfast->get_field_type() == 2)	
					nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_fft_inplace_r2c),
						 FIELD_DIM * total_num_fft_r2c_pts, _array_name.str(), _cur_dev);
				else { std::cout << "unsupported field type in NUFFT...exiting..." << std::endl; exit(0); }
			}
			nufft_array_gpu[_thread_id]->d_fft_inplace_r2c_FP = (FP_TYPE*)(nufft_array_gpu[_thread_id]->d_fft_inplace_r2c);
			/*_array_name.str("");
			_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_fft_inplace_b";
			{
				nbodyfast->gpu->memory_gpu->alloc_device<CUFFT_COMPLEX_TYPE>(_cuda_error, &(nufft_array_gpu[_thread_id]->d_fft_inplace_b), total_num_fft_pts, _array_name.str(), _cur_dev);
			}*/
			int _fft_size[3];
			for (int m = 0; m < 3; m ++)
			{
				_fft_size[m] = fft_size[2 - m];
			}
			int fft_batch = -1;
			if(nbodyfast->get_field_type() == 1)	fft_batch = 1;
			else if(nbodyfast->get_field_type() == 2)	fft_batch = FIELD_DIM;
			else { std::cout << "unsupported field type in NUFFT...exiting..." << std::endl; exit(0); }

			cufftPlanMany(&nufft_array_gpu[_thread_id]->cufftPlan_r2c, 3, _fft_size, NULL, 1, 0, NULL, 1, 0, CUFFT_TYPE_F, fft_batch);
			cufftPlanMany(&nufft_array_gpu[_thread_id]->cufftPlan_c2r, 3, _fft_size, NULL, 1, 0, NULL, 1, 0, CUFFT_TYPE_B, fft_batch);
			
			cufftSetCompatibilityMode(nufft_array_gpu[_thread_id]->cufftPlan_r2c, CUFFT_COMPATIBILITY_NATIVE);
			cufftSetCompatibilityMode(nufft_array_gpu[_thread_id]->cufftPlan_c2r, CUFFT_COMPATIBILITY_NATIVE);
		}

		// Copy
		nufft_array_gpu[_thread_id]->d_src_coord = field_gpu->d_src_coord[_thread_id];

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[_thread_id]->d_src_box_list, src_box_list_dev[_thread_id], total_num_boxes + 1, _cur_dev);

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[_thread_id]->d_src_box_list_inv, src_box_list_inv_dev[_thread_id], total_num_boxes + 1, _cur_dev);

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[_thread_id]->d_src_box_map, src_box_map_dev[_thread_id], total_num_boxes * 3, _cur_dev);

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[_thread_id]->d_src_trans_idx, src_trans_idx_dev[_thread_id], nbodyfast->src_size_dev[_thread_id], _cur_dev);

		//nbodyfast->gpu->memory_gpu->memcpy_host_to_device<int>(nufft_array_gpu[_thread_id]->d_near_box_list, near_box_list_dev[_thread_id],  28 * src_box_list_dev[_thread_id][0], _cur_dev);


		if (field_gpu->src_obs_overlap == false)
		{
			// Not yet implemented
			std::cout << "The code is not yet working for src_obs_overalp = false at int NufftGpu :: preprocessing_multi()" << std::endl;
			exit(0);
		}
		else
		{
			nufft_array_gpu[_thread_id]->d_obs_coord = field_gpu->d_obs_coord[_thread_id];
		}

		// Set values of device copy of nufft_const_gpu
		d_nufft_const_gpu[_thread_id] = NULL;
		_array_name.str("");
		_array_name <<"nufft->nufft_array_gpu[" << _thread_id << "]->d_nufft_const_gpu";
		{

			nbodyfast->gpu->memory_gpu->alloc_device<NufftParamGpu>(_cuda_error, &d_nufft_const_gpu[_thread_id], 1, _array_name.str(), _cur_dev);
		}

		nbodyfast->gpu->memory_gpu->memcpy_host_to_device<NufftParamGpu>(d_nufft_const_gpu[_thread_id], nufft_const_gpu[_thread_id],  1, _cur_dev);

	}
	#pragma omp barrier
//	nbodyfast->memory->output_allocated_list();
//	nbodyfast->gpu->memory_gpu->output_allocated_list();
	return 0;
}

}
