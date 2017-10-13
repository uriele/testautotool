/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_static_scalar.cpp: class definition of Class FieldStaticScalar
*/
#include "nbodyfast.h"
#include "domain.h"
#include "error.h"
#include "field.h"
#include "imp_matrix.h"
#include "memory.h"
#include "nufft.h"
#include "timer.h"
#include <fstream>
#include <iostream>

namespace NBODYFAST_NS{
Nufft :: Nufft(class NBODYFAST *n_ptr)
{
	nbodyfast = n_ptr;
	problem_size = nbodyfast->problem_size;
	
	gpu_on = false;
	cpu_near_onfly = true;

	imp_mat_ptr = NULL;
	near_imp_mat_ptr = NULL;

	src_box_map = NULL;src_trans_idx = NULL;obs_box_map = NULL;obs_trans_idx = NULL;
	src_box_list = NULL;obs_box_list = NULL;
	near_box_list = NULL;
	near_bound_list = NULL;
	src_grid_coord = NULL;obs_grid_coord = NULL;
	
	src_interp_coeff = NULL;obs_interp_coeff = NULL;	src_interp_idx = NULL;obs_interp_idx = NULL;

	src_box_list_dev = NULL;obs_box_list_dev = NULL;
	src_start_dev = NULL;obs_start_dev = NULL;
	src_grid_start_dev = NULL;obs_grid_start_dev = NULL;
	near_box_list_dev = NULL;
	src_box_map_dev = NULL;obs_box_map_dev = NULL;
	src_trans_idx_dev = NULL;obs_trans_idx_dev = NULL;
	src_box_list_inv_dev = NULL;obs_box_list_inv_dev = NULL;
	fft_in_f = NULL;fft_out_f = NULL;fft_in_b = NULL;fft_out_b = NULL;
	fft_in_f_near = NULL;fft_out_f_near = NULL;fft_in_b_near = NULL;fft_out_b_near = NULL;

	plan_grid = NULL;plan_grid_near = NULL;
}
Nufft :: ~Nufft()
{
	if (nbodyfast->multi_device == true)
	{
		for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id ++)
		{
			nbodyfast->memory->free_host<int>(&src_trans_idx_dev[_thread_id]);
			nbodyfast->memory->free_host<int>(&src_box_map_dev[_thread_id]);

			nbodyfast->memory->free_host<int>(&near_box_list_dev[_thread_id]);
			nbodyfast->memory->free_host<int>(&src_box_list_dev[_thread_id]);
			nbodyfast->memory->free_host<int>(&src_box_list_inv_dev[_thread_id]);
		}

		nbodyfast->memory->free_host<int>(&nbodyfast->src_size_dev);
		nbodyfast->memory->free_host<int>(&nbodyfast->src_size_act_dev);
		nbodyfast->memory->free_host<int*>(&src_trans_idx_dev);
		nbodyfast->memory->free_host<int*>(&src_box_map_dev);
		nbodyfast->memory->free_host<int*>(&src_box_list_dev);
		nbodyfast->memory->free_host<int*>(&src_box_list_inv_dev);
		nbodyfast->memory->free_host<int*>(&near_box_list_dev);

		nbodyfast->memory->free_host<int>(&src_start_dev);
		nbodyfast->memory->free_host<int>(&src_grid_start_dev);
	}

	if (gpu_on == false)
	{
		nbodyfast->memory->free_host<ImpMatrix>(&near_imp_mat_ptr);

		//fftw_destroy_plan(plan_grid_near[1]);
		//fftw_destroy_plan(plan_grid_near[0]);
		nbodyfast->memory->free_host<fftw_plan>(&plan_grid_near);
		
		nbodyfast->memory->free_host<fftw_complex>(&fft_out_b_near);
		nbodyfast->memory->free_host<fftw_complex>(&fft_in_b_near);
		nbodyfast->memory->free_host<fftw_complex>(&fft_out_f_near);
		nbodyfast->memory->free_host<fftw_complex>(&fft_in_f_near);
	}

	nbodyfast->memory->free_host<ImpMatrix>(&imp_mat_ptr);

	//fftw_destroy_plan(plan_grid[1]);
	//fftw_destroy_plan(plan_grid[0]);
	nbodyfast->memory->free_host<fftw_plan>(&plan_grid);
	nbodyfast->memory->free_host<fftw_complex>(&fft_out_b);
	nbodyfast->memory->free_host<fftw_complex>(&fft_in_b);
	nbodyfast->memory->free_host<fftw_complex>(&fft_out_f);
	nbodyfast->memory->free_host<fftw_complex>(&fft_in_f);

	nbodyfast->memory->free_host<int>(&src_interp_idx);
	nbodyfast->memory->free_host<FP_TYPE>(&src_interp_coeff);
	nbodyfast->memory->free_host<FP_TYPE>(&src_grid_coord);
	nbodyfast->memory->free_host<int>(&near_box_list);
	nbodyfast->memory->free_host<int>(&near_bound_list);
	nbodyfast->memory->free_host<int>(&src_box_map);
	nbodyfast->memory->free_host<int>(&src_trans_idx);
	nbodyfast->memory->free_host<int>(&src_box_list);

	if (nbodyfast->field->src_obs_overlap == false)
	{
		if (nbodyfast->multi_device == true)
		{
			for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id ++)
			{
				nbodyfast->memory->free_host<int>(&obs_trans_idx_dev[_thread_id]);
				nbodyfast->memory->free_host<int>(&obs_box_map_dev[_thread_id]);
				nbodyfast->memory->free_host<int>(&obs_box_list_dev[_thread_id]);
				nbodyfast->memory->free_host<int>(&obs_box_list_inv_dev[_thread_id]);

			}
			nbodyfast->memory->free_host<int>(&nbodyfast->obs_size_dev);
			nbodyfast->memory->free_host<int>(&nbodyfast->obs_size_act_dev);
			nbodyfast->memory->free_host<int*>(&obs_trans_idx_dev);
			nbodyfast->memory->free_host<int*>(&obs_box_map_dev);
			nbodyfast->memory->free_host<int*>(&obs_box_list_dev);
			nbodyfast->memory->free_host<int*>(&obs_box_list_inv_dev);

		}
		nbodyfast->memory->free_host<int>(&obs_interp_idx);
		nbodyfast->memory->free_host<FP_TYPE>(&obs_interp_coeff);
		nbodyfast->memory->free_host<FP_TYPE>(&obs_grid_coord);
		nbodyfast->memory->free_host<int>(&obs_box_map);
		nbodyfast->memory->free_host<int>(&obs_trans_idx);
		nbodyfast->memory->free_host<int>(&obs_box_list);

	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// These inline functions are not supposed to be reused
// They are separated out from Nufft::preprocessing only for the reason of code readability
// parameter file reading
inline int Nufft::input_file_reading()
{
	std::ifstream in("./InterpParam.in");

	if (!in) // if there is no InterpParam.in on the disk, set the parameters to default
	{
		for (int k = 0; k < 3; k++)
		{
			num_box_inp = 3;
			box_size_inp = 0;
			interp_order[k] = 3;
			num_nodes[k] = interp_order[k] + 1;
			near_correct_layer = 1;
		}
		std::cout << "No \"InterpParam.in\" existing on the disk...using automatic settings...might not be optimal \n";
	}
	else // if yes, use the values from file
	{
		std::cout << "Reading from \"InterpParam.in\" for the settings (interp order, number of boxes, etc...) \n";
		in >> num_box_inp;
		in >> box_size_inp;
		in >> interp_order_inp;
		in >> near_correct_layer;

		for (int k = 0; k < 3; k++)
		{
			interp_order[k] = interp_order_inp;
			num_nodes[k] = interp_order_inp + 1;
		}
//		adaptive_decomp = false;
	}

	interp_nodes_per_box = num_nodes[0] * num_nodes[1] * num_nodes[2];
	in.close();
	return 0;
}

// domain resizing
inline int Nufft::domain_resize()
{
	std::cout << "Calculating domain sizes... \n";
	
	// if the input file does not specify box size (which is usually the case)
	// calculate appropriate box size by dividing the domain size by number of boxes
	if (box_size_inp == 0.0f || box_size_inp < nbodyfast->domain->max_size / num_box_inp)
	{
		box_size_inp = nbodyfast->domain->max_size / num_box_inp * (1 + 1e-3);
	}
	else // just use the box size provided by the input file
	{
		num_box_inp = nbodyfast->domain->max_size / box_size_inp;
	}

	for (int k = 0; k < 3; k++)
	{
		// set number of boxes
		num_boxes[k] = nbodyfast->domain->unified_domain_size[k] / box_size_inp + 1;

		// Resize the computational domain according to the number of boxes settings
		nbodyfast->domain->src_domain_range[k] = nbodyfast->domain->src_domain_center[k] - num_boxes[k] * box_size_inp * 0.5f;
		nbodyfast->domain->src_domain_range[k + 3] = nbodyfast->domain->src_domain_center[k] + num_boxes[k] * box_size_inp * 0.5f;
		nbodyfast->domain->obs_domain_range[k] = nbodyfast->domain->obs_domain_center[k] - num_boxes[k] * box_size_inp * 0.5f;
		nbodyfast->domain->obs_domain_range[k + 3] = nbodyfast->domain->obs_domain_center[k] + num_boxes[k] * box_size_inp * 0.5f;
		nbodyfast->domain->unified_domain_size[k] = nbodyfast->domain->obs_domain_range[k + 3] - nbodyfast->domain->obs_domain_range[k];
		
		num_grid_pts[k] = interp_order[k] * num_boxes[k] + 1;
		cell_size[k] = nbodyfast->domain->unified_domain_size[k] / (num_grid_pts[k] - 1);
		box_size[k] = box_size_inp;

	}
	//std::cout<<"Average #srcs per box: " << problem_size/(num_boxes[0]*num_boxes[1]*num_boxes[2])<<std::endl;
	std::cout<<"Total #box: " << num_boxes[0] << " X " << num_boxes[1] << " X " << num_boxes[2] << std::endl;

	return 0;
}
// source ordering, making sources belongs to the same box together
inline int Nufft::source_ordering()
{
	std::cout << "Reordering sources... \n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Make a copy of src and obs coord if adaptive domain division is used (this has not been implemented yet)
	FP_TYPE *_src_coord_temp = NULL;
	int *_src_trans_idx = NULL;
	nbodyfast->memory->alloc_host<FP_TYPE>(&_src_coord_temp, problem_size * 3, "nufft->_src_coord_temp");
	nbodyfast->memory->alloc_host<int>(&_src_trans_idx, problem_size, "nufft->_src_trans_idx");

	for (int i = 0; i < problem_size * 3; i++)
	{
		_src_coord_temp[i] = nbodyfast->field->src_coord[i];
	}	

	for (int i = 0; i < problem_size; i++)
	{
		_src_trans_idx[i] = i;
	}

	int *_src_box_map = NULL;
	nbodyfast->memory->alloc_host<int>(&_src_box_map,total_num_boxes * 2, "nufft->_src_box_map");

	for (int i = 0; i < total_num_boxes * 2; i++)
	{
		_src_box_map[i] = 0;
	}

	// try to order sources into boxes	
	source_swap(_src_coord_temp, problem_size, _src_box_map, _src_trans_idx, nbodyfast->domain->src_domain_range, box_size, num_boxes);

	// calculating average number of sources per box (for adaptive domain division, but not very useful for non-uniform domains)
	src_num_nonempty_boxes = 0;

	for (int i = 0; i < total_num_boxes; i++)
	{
		if (_src_box_map[i + total_num_boxes] - _src_box_map[i] != 0) src_num_nonempty_boxes++;
	}

	std::cout << "nonempty boxes/total num boxes is " << src_num_nonempty_boxes << "/" << total_num_boxes << std::endl;
	std::cout << "avarage num of srcs in nonempty boxes is " << problem_size/src_num_nonempty_boxes << std::endl;
	std::cout << "avarage num of srcs in boxes is " << problem_size/total_num_boxes << std::endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// if we are happy we current box setup, then go ahead update the source coordinate, box-source mapping, etc...
	for (int i = 0; i < problem_size * 3; i++)
	{
		nbodyfast->field->src_coord[i] = _src_coord_temp[i];
	}	

	for (int i = 0; i < total_num_boxes; i++)
	{
		src_box_map[i] = _src_box_map[i];
		src_box_map[i + total_num_boxes] = _src_box_map[i + total_num_boxes];
		src_box_map[i + total_num_boxes * 2] = _src_box_map[i + total_num_boxes] - _src_box_map[i];
	}

	for (int i = 0; i < problem_size; i++)
	{
		src_trans_idx[i] = _src_trans_idx[i];
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// free up temporary arrays
	nbodyfast->memory->free_host<int>(&_src_box_map);
	nbodyfast->memory->free_host<int>(&_src_trans_idx);
	nbodyfast->memory->free_host<FP_TYPE>(&_src_coord_temp);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// if the source and observers are not overlapped, we have to do it all over again for observers
	if (nbodyfast->field->src_obs_overlap == false)
	{
		FP_TYPE *_obs_coord_temp = NULL;
		int *_obs_trans_idx = NULL;
		nbodyfast->memory->alloc_host<FP_TYPE>(&_obs_coord_temp, problem_size * 3, "nufft->_obs_coord_temp");
		nbodyfast->memory->alloc_host<int>(&_obs_trans_idx, problem_size, "nufft->_obs_trans_idx");
		for (int i = 0; i < problem_size * 3; i++)
		{
			_obs_coord_temp[i] = nbodyfast->field->obs_coord[i];
		}	

		for (int i = 0; i < problem_size; i++)
		{
			_obs_trans_idx[i] = i;
		}

		int *_obs_box_map = NULL;
		nbodyfast->memory->alloc_host<int>(&_obs_box_map, total_num_boxes * 2, "nufft->_obs_box_map");

		for (int i = 0; i < total_num_boxes * 2; i++)
		{
			_obs_box_map[i] = 0;
		}
		
		source_swap(_obs_coord_temp, problem_size, _obs_box_map, _obs_trans_idx, nbodyfast->domain->obs_domain_range, box_size, num_boxes);
		obs_num_nonempty_boxes = 0;

		for (int i = 0; i < total_num_boxes; i++)
		{
			if (_obs_box_map[i + total_num_boxes] - _obs_box_map[i] != 0) obs_num_nonempty_boxes++;
		}
		
		int _average_per_box = problem_size * 2 / (src_num_nonempty_boxes + obs_num_nonempty_boxes);
		std::cout << "avarage num of srcs in nonempty boxes is " << _average_per_box << std::endl;
		std::cout << "avarage num of srcs in boxes is " << problem_size/total_num_boxes << std::endl;

		for (int i = 0; i < problem_size * 3; i++)
		{
			nbodyfast->field->obs_coord[i] = _obs_coord_temp[i];
		}	
		for (int i = 0; i < total_num_boxes; i++)
		{
			obs_box_map[i] = _obs_box_map[i];
			obs_box_map[i + total_num_boxes] = _obs_box_map[i + total_num_boxes];
			obs_box_map[i + total_num_boxes * 2] = _obs_box_map[i + total_num_boxes] - _obs_box_map[i];

		}

		for (int i = 0; i < problem_size; i++)
		{
			obs_trans_idx[i] = _obs_trans_idx[i];
		}

		nbodyfast->memory->free_host<int>(&_obs_box_map);
		nbodyfast->memory->free_host<int>(&_obs_trans_idx);
		nbodyfast->memory->free_host<FP_TYPE>(&_obs_coord_temp);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// if they are overlapped, then nice! we just set the pointers...
	else
	{
		obs_box_map = src_box_map;
		obs_trans_idx = src_trans_idx;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	return 0;
}


// source list generation... this is a dummy subroutine that generates "src_box_list" (for single GPU it should contains all boxes..)
// src_box_list is for single and multiGPU to reuse the same interface 
inline int Nufft::source_list_gen()
{
	std::cout << "Generating source box list... \n";

	if (nbodyfast->field->src_obs_overlap == false)
	{
		// Set all near field related arrays to NULL 
		// To be done!!!
		std::cout << "The code is not yet working for src_obs_overalp = false. Exiting at inline int Nufft::near_list_gen()" << std::endl;
		exit(0);
	}
	for (int i = 0; i < total_num_boxes + 1; i++)
	{
		src_box_list[i] = -1;
	}
	src_box_list[0] = total_num_boxes;
	//int _cur_box = 1;
	for (int i = 0; i < total_num_boxes; i++)
	{
		src_box_list[i+1] = i;
	}
	return 0;
}
// near list generation
inline int Nufft::near_list_gen()//!!!!!!!!CHANGED BY BEN!!!!!NEAR CHANGE!!!!!!!!!!!!!
{
	std::cout << "Generating near box list with " << near_correct_layer << " layers... \n";

	if (nbodyfast->field->src_obs_overlap == false)
	{
		// Set all near field related arrays to NULL 
		// To be done!!!
		std::cout << "The code is not yet working for src_obs_overalp = false. Exiting at inline int Nufft::near_list_gen()" << std::endl;
		exit(0);
	}

	//!!!!!!!!!!!!!BEN ADDED THESE TEMP VARIABLES!!!!!!!!!!!!!!!!!!!!!!!
	int num_near_box_per_dim = 2*near_correct_layer+1;
	int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;
	// initialize the near_box_list to all -1
	for (int i = 0; i < (total_num_near_box_per_box+1) * total_num_boxes; i++) near_box_list[i] = -1;
	near_box_cnt = 0;
	max_near_src_cnt = 0;
	// _total_near_src is for finding max number of near source a box can have. this max number of near source is used only by CPU NUFFT if near field correction is done in the pre-computed fashion	
	int *_total_near_src = NULL;
	nbodyfast->memory->alloc_host<int>(&_total_near_src, total_num_boxes, "nufft->_total_near_src");
	
	// loop around all boxes
	for (int i = 0; i < total_num_boxes; i++)
	{
		// box index in each dimension
		int _box_idx_dim[3];
		_box_idx_dim[2] = i / (num_boxes[0] * num_boxes[1]);
		_box_idx_dim[1] = (i - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
		_box_idx_dim[0] = i % num_boxes[0];	
		_total_near_src[i] = 0;
		near_box_cnt = 0;

		// assume there are 27 near boxes//!!!!!!!!!!CHANGED BY BEN
		// but if current near box does not present (when the main box is on the corner, edge and surface of the computational domain, some of its near boxes will not present)
		for (int j = 0; j < total_num_near_box_per_box; j++)
		{
			int _near_box_idx_dim[3];
			_near_box_idx_dim[2] = j / (num_near_box_per_dim*num_near_box_per_dim) - near_correct_layer;
			_near_box_idx_dim[1] = (j % (num_near_box_per_dim*num_near_box_per_dim)) / num_near_box_per_dim - near_correct_layer;
			_near_box_idx_dim[0] = j % num_near_box_per_dim - near_correct_layer;
			// if current near box is not valid, go to the next
			if (_box_idx_dim[0] +  _near_box_idx_dim[0] < 0 || _box_idx_dim[1] +  _near_box_idx_dim[1]  < 0 || _box_idx_dim[2] +  _near_box_idx_dim[2]  < 0 ||
				_box_idx_dim[0] +  _near_box_idx_dim[0] >= num_boxes[0] || _box_idx_dim[1] + _near_box_idx_dim[1] >= num_boxes[1] || _box_idx_dim[2] +  _near_box_idx_dim[2] >= num_boxes[2]) 
			{
				continue;
			}
			// if current near box presents but number of sources inside is 0, treat it as not presenting
			int _current_near_box_idx = i + (_near_box_idx_dim[0]) + (_near_box_idx_dim[1]) * num_boxes[0] + (_near_box_idx_dim[2]) * num_boxes[0] * num_boxes[1]; 

			if (src_box_map[_current_near_box_idx + total_num_boxes] - src_box_map[_current_near_box_idx] == 0)
			{
				continue;
			}
			
			// if it is a valide nearbox, push it to the end of near_box_list
			near_box_list[near_box_cnt + 1 + i * (total_num_near_box_per_box+1)] = _current_near_box_idx;
			near_box_cnt++; 
			_total_near_src[i] += src_box_map[_current_near_box_idx + total_num_boxes] - src_box_map[_current_near_box_idx];
		}
		
		// put number of near boxes into the first of near box list
		near_box_list[i * (total_num_near_box_per_box+1)] = near_box_cnt;

		if (_total_near_src[i] > max_near_src_cnt) max_near_src_cnt = _total_near_src[i];
	}

	nbodyfast->memory->free_host<int>(&_total_near_src);


	return 0;
}

// near_bound generation
inline int Nufft::near_bound_gen(){
		
	std::cout << "Generating near box bound list with " << near_correct_layer << " layers... \n";

	if (nbodyfast->field->src_obs_overlap == false)
	{
		// Set all near field related arrays to NULL 
		// To be done!!!
		std::cout << "The code is not yet working for src_obs_overalp = false. Exiting at inline int Nufft::near_bound_gen()" << std::endl;
		exit(0);
	}

	//!!!!!!!!!!!!!BEN ADDED THESE TEMP VARIABLES!!!!!!!!!!!!!!!!!!!!!!!
	/*int num_near_box_per_dim = 2*near_correct_layer+1;
	int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;*/
	
	// initialize the near_box_list to all -1
	for (int i = 0; i < 6 * total_num_boxes; i++) 
		near_bound_list[i] = -1;
	
	// box index in each dimension
	int _box_idx_dim[3];

	// loop around all boxes
	for (int i = 0; i < total_num_boxes; i++)
	{
		// box index in each dimension
		_box_idx_dim[2] = i / (num_boxes[0] * num_boxes[1]);
		_box_idx_dim[1] = (i - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
		_box_idx_dim[0] = i % num_boxes[0];	

		for( int m = 0; m < 3; m++){
			near_bound_list[i*6+m] = _box_idx_dim[m]-near_correct_layer > 0 ? _box_idx_dim[m]-near_correct_layer : 0;
			near_bound_list[i*6+m+3] = _box_idx_dim[m]+near_correct_layer <= num_boxes[m]-1 ? _box_idx_dim[m]+near_correct_layer : num_boxes[m]-1;
		}
		
	}

	return 0;
}
// interpolation grid generationg, this grid is where the source amplitude projects to and also where FFT applies
inline int Nufft::interp_grid_gen()
{
	std::cout << "Generating interpolation grid... \n";

	// building a regular grid
	int _pp = 0;
	for (int i = 0; i < num_grid_pts[2]; i++)
		for (int j = 0; j < num_grid_pts[1]; j++)
			for (int k = 0; k < num_grid_pts[0]; k++)
			{
				src_grid_coord[_pp] =  nbodyfast->domain->src_domain_range[0] + k * cell_size[0];
				src_grid_coord[_pp + total_num_grid_pts] = nbodyfast->domain->src_domain_range[1] + j * cell_size[1];
				src_grid_coord[_pp + total_num_grid_pts * 2] = nbodyfast->domain->src_domain_range[2] + i * cell_size[2];
				_pp++;
			}

	// if source and observer not overlapping, build a grid for observers too
	if (nbodyfast->field->src_obs_overlap == false)
	{
		int _pp = 0;
		for (int i = 0; i < num_grid_pts[2]; i++)
			for (int j = 0; j < num_grid_pts[1]; j++)
				for (int k = 0; k < num_grid_pts[0]; k++)
				{
					obs_grid_coord[_pp] =  nbodyfast->domain->obs_domain_range[0] + k * cell_size[0];
					obs_grid_coord[_pp + total_num_grid_pts] = nbodyfast->domain->obs_domain_range[1] + j * cell_size[1];
					obs_grid_coord[_pp + total_num_grid_pts * 2] = nbodyfast->domain->obs_domain_range[2] + i * cell_size[2];
					_pp++;
				}
	}
	else
	{
		obs_grid_coord = src_grid_coord;
	}
	return 0;
}

// interplation coefficient generation, used by CPU NUFFT only
inline int Nufft::interp_coeff_gen()
{
	std::cout << "Generating interpolation coefficients... \n";

	// Calculating normalized Lagrange interp nodes, and w(j)
	FP_TYPE *_interp_nodes_norm = NULL;
	FP_TYPE *_interp_w = NULL;
	nbodyfast->memory->alloc_host<FP_TYPE>(&_interp_nodes_norm, num_nodes[0] * 3, "nufft->_interp_nodes_norm");
	nbodyfast->memory->alloc_host<FP_TYPE>(&_interp_w, num_nodes[0] * 3, "nufft->_interp_w");
	for (int k = 0; k < 3; k++)
	{
		for (int i = 0; i < num_nodes[k]; i++)
		{
			_interp_nodes_norm[i + k * num_nodes[0]] = 1.0f / (num_nodes[k] - 1) * i;
		}

		for (int i = 0; i < num_nodes[k]; i++)
		{
			_interp_w[i + k * num_nodes[0]] = 1.0f;

			for (int j = 0; j < num_nodes[k]; j++)
			{
				if (i != j) _interp_w[i + k * num_nodes[0]] *= _interp_nodes_norm[i + k * num_nodes[0]] - _interp_nodes_norm[j + k * num_nodes[0]];
			}
		}
	}

	// Calculating interpolation coefficient
	int _box_idx_dim[3];
	int *_index_r = NULL;
	nbodyfast->memory->alloc_host<int>(&_index_r, num_nodes[0] * 3, "nufft->_index_r");
	FP_TYPE _r_coord_norm[3];
	FP_TYPE _interp_lr; 
	FP_TYPE *_interp_coeff_r_dim = NULL;
	nbodyfast->memory->alloc_host<FP_TYPE>(&_interp_coeff_r_dim, num_nodes[0] * 3, "nufft->_interp_coeff_r_dim");

	int _r_coinci_flag; 
	_r_coinci_flag = 0;

	for (int i = 0; i < total_num_boxes; i++)
	{
		_box_idx_dim[2] = i / (num_boxes[0] * num_boxes[1]);
		_box_idx_dim[1] = (i % (num_boxes[0] * num_boxes[1])) / num_boxes[0];
		_box_idx_dim[0] = i % num_boxes[0];


		// Generating projection coefficient for the source
		for (int m = src_box_map[i]; m < src_box_map[total_num_boxes + i]; m++)
		{
			for (int k = 0; k < 3; k++)
			{		
				for (int j = 0; j < num_nodes[k]; j++)
				{
					_index_r[num_nodes[0] * k + j] = _box_idx_dim[k] * (num_nodes[k] - 1) + j;		
				}
				_interp_lr = 1.0f;

				for (int j = 0; j < num_nodes[0]; j++) _interp_coeff_r_dim[j + k * num_nodes[0]] = 0.0f;

				_r_coord_norm[k] = (nbodyfast->field->src_coord[m + k * problem_size] - _index_r[num_nodes[0] * k] * cell_size[k] - nbodyfast->domain->src_domain_range[k]) / (cell_size[k] * (num_nodes[k] - 1));

				// If the observer / source point coincides with one of the interpolation nodes,
				// set the corresponding interp coeff to be 1.0f, and all others to 0.0f
				for (int j = 0; j < num_nodes[k]; j++)
				{
					if (_r_coord_norm[k] == _interp_nodes_norm[j + k * num_nodes[0]])
					{
						_interp_coeff_r_dim[j + k * num_nodes[0]] = 1.0f;
						_r_coinci_flag = 1;
					}
				}

				// If the observer / source point does not coincide with any of the interpolation nodes,
				// calculate the interp coeff

				if (_r_coinci_flag != 1)
				{
					for (int j =0; j < num_nodes[k]; j++)
					{
						_interp_lr *= _r_coord_norm[k] - _interp_nodes_norm[j + k * num_nodes[0]];
					}

					for (int j =0; j < num_nodes[k]; j++)
					{
						_interp_coeff_r_dim[j + k * num_nodes[0]] = _interp_lr / _interp_w[j + k * num_nodes[0]] / (_r_coord_norm[k] - _interp_nodes_norm[j + k * num_nodes[0]] );
					}

				}
				_r_coinci_flag = 0;		
			}

			// Assemble the interpolation coefficients matrix
			for (int kk = 0; kk < num_nodes[2]; kk++)
				for (int jj = 0; jj < num_nodes[1]; jj++)
					for (int ii = 0; ii < num_nodes[0]; ii++)
					{
						int _idx = ii + jj * num_nodes[0] + kk * num_nodes[0] * num_nodes[1];

						src_interp_coeff[_idx + m * interp_nodes_per_box] = _interp_coeff_r_dim[ii] * _interp_coeff_r_dim[jj + num_nodes[0]] * _interp_coeff_r_dim[kk + 2 * num_nodes[0]];
						src_interp_idx[_idx + m * interp_nodes_per_box] = _index_r[ii] + _index_r[jj + num_nodes[0]] * num_grid_pts[0] + _index_r[kk + 2 * num_nodes[0]] * num_grid_pts[0] * num_grid_pts[1];
					}	
		} // m loop (loop inside of a certain box)
	} // i loop (loop around boxes)

	if (nbodyfast->field->src_obs_overlap == false)
	{
		for (int i = 0; i < total_num_boxes; i++)
		{
			// Generating interpolation coefficient for the observer
			for (int m = obs_box_map[i]; m < obs_box_map[total_num_boxes + i]; m++)
			{
				for (int k = 0; k < 3; k++)
				{		
					for (int j = 0; j < num_nodes[k]; j++)
					{
						_index_r[num_nodes[0] * k + j] = _box_idx_dim[k] * (num_nodes[k] - 1) + j;		
					}
					_interp_lr = 1.0f;

					for (int j = 0; j < num_nodes[0]; j++) _interp_coeff_r_dim[j + k * num_nodes[0]] = 0.0f;

					_r_coord_norm[k] = (nbodyfast->field->obs_coord[m + k * problem_size] - _index_r[num_nodes[0] * k] * cell_size[k] - nbodyfast->domain->obs_domain_range[k]) / (cell_size[k] * (num_nodes[k] - 1));

					// If the observer / source point coincides with one of the interpolation nodes,
					// set the corresponding interp coeff to be 1.0f, and all others to 0.0f
					for (int j = 0; j < num_nodes[k]; j++)
					{
						if (_r_coord_norm[k] == _interp_nodes_norm[j + k * num_nodes[0]])
						{
							_interp_coeff_r_dim[j + k * num_nodes[0]] = 1.0f;
							_r_coinci_flag = 1;
						}
					}

					// If the observer / source point does not coincide with any of the interpolation nodes,
					// calculate the interp coeff

					if (_r_coinci_flag != 1)
					{
						for (int j =0; j < num_nodes[k]; j++)
						{
							_interp_lr *= _r_coord_norm[k] - _interp_nodes_norm[j + k * num_nodes[0]];
						}

						for (int j =0; j < num_nodes[k]; j++)
						{
							_interp_coeff_r_dim[j + k * num_nodes[0]] = _interp_lr / _interp_w[j + k * num_nodes[0]] / (_r_coord_norm[k] - _interp_nodes_norm[j + k * num_nodes[0]] );
						}

					}
					_r_coinci_flag = 0;		
				}

				// Assemble the interpolation coefficients matrix
				for (int kk = 0; kk < num_nodes[2]; kk++)
					for (int jj = 0; jj < num_nodes[1]; jj++)
						for (int ii = 0; ii < num_nodes[0]; ii++)
						{
							int _idx = ii + jj * num_nodes[0] + kk * num_nodes[0] * num_nodes[1];

							obs_interp_coeff[_idx + m * interp_nodes_per_box] = _interp_coeff_r_dim[ii] * _interp_coeff_r_dim[jj + num_nodes[0]] * _interp_coeff_r_dim[kk + 2 * num_nodes[0]];
							obs_interp_idx[_idx + m * interp_nodes_per_box] = _index_r[ii] + _index_r[jj + num_nodes[0]] * num_grid_pts[0] + _index_r[kk + 2 * num_nodes[0]] * num_grid_pts[0] * num_grid_pts[1];
						}	
			} // m loop (loop inside of a certain box)
		}
	}
	else
	{
		obs_interp_coeff = src_interp_coeff;
		obs_interp_idx = src_interp_idx;
	}

	nbodyfast->memory->free_host<FP_TYPE>(&_interp_coeff_r_dim);
	nbodyfast->memory->free_host<int>(&_index_r);
	nbodyfast->memory->free_host<FP_TYPE>(&_interp_w);
	nbodyfast->memory->free_host<FP_TYPE>(&_interp_nodes_norm);

	return 0;
}

// near grid generation, used by CPU NUFFT only //!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!NOT DONE YET!!!!!!!
inline int Nufft::near_grid_gen(FP_TYPE *_near_grid_coord, int _near_num_grid_pts[3], int _total_num_near_grid_pts)
{
	std::cout << "Generating near grid... \n";

	int _pp = 0;
	for (int i = 0; i < _near_num_grid_pts[2]; i++)
		for (int j = 0; j < _near_num_grid_pts[1]; j++)
			for (int k = 0; k < _near_num_grid_pts[0]; k++)
			{
				_near_grid_coord[_pp] =  nbodyfast->domain->src_domain_range[0] + k * cell_size[0];
				_near_grid_coord[_pp + _total_num_near_grid_pts] = nbodyfast->domain->src_domain_range[1] + j * cell_size[1];
				_near_grid_coord[_pp + _total_num_near_grid_pts * 2] = nbodyfast->domain->src_domain_range[2] + i * cell_size[2];

				_pp++;
			}
	return 0;
}
// source ordering
inline int Nufft::source_swap(FP_TYPE *_coord, int _coord_size, int* _map, int *_trans_idx, FP_TYPE *_domain_range, FP_TYPE *_box_size, int *_num_boxes)
{
	int _low_idx[3], _high_idx[3], _final_idx[3];
	FP_TYPE _pivot[3];
	int _total_num_boxes = _num_boxes[0] * _num_boxes[1] * _num_boxes[2]; 
	for (int k = 0; k < 3; k++)
	{
		_low_idx[k] = 0; 
		_high_idx[k] = problem_size;
		_final_idx[k] = 0;
		_pivot[k] = 0.0f;
	}

	for (int i = 1; i < _num_boxes[2] + 1; i++)
	{
		_pivot[2] = _domain_range[2] + i * _box_size[2];
		_final_idx[2] = swap(_coord, _coord_size, _trans_idx, _low_idx[2], _high_idx[2], _pivot[2], 2);
		
		_low_idx[1] = _low_idx[2];
		_high_idx[1] = _final_idx[2];			
		
		for (int j = 1; j < _num_boxes[1] + 1; j++)
		{
			_pivot[1] = _domain_range[1] + j * _box_size[1];		
			_final_idx[1] = swap(_coord, _coord_size, _trans_idx, _low_idx[1], _high_idx[1], _pivot[1], 1);
			
			_low_idx[0] = _low_idx[1];
			_high_idx[0] = _final_idx[1];				
			
			for (int m = 1; m < _num_boxes[0] + 1; m++)
			{
				_pivot[0] = _domain_range[0] + m * _box_size[0];		
				_final_idx[0] = swap(_coord, _coord_size, _trans_idx, _low_idx[0], _high_idx[0], _pivot[0], 0);

				_low_idx[0] = _final_idx[0];
				_map[(m - 1) + (j - 1) * _num_boxes[0] + (i - 1) * _num_boxes[0] * _num_boxes[1] + _total_num_boxes] = _final_idx[0];
			}

			_low_idx[1] = _final_idx[1];
		}

		_low_idx[2] = _final_idx[2];			
	}
	
	
	//_map[2 * NUFFTPlan->_total_num_boxes - 1] = _coord_size;
	for (int i = 1; i < _total_num_boxes; i++) _map[i] = _map[i - 1 + _total_num_boxes];
	return 0;
}


inline int Nufft::swap(FP_TYPE *Coord, int ProblemSize, int *TranIdx, int LowIdx, int HighIdx, FP_TYPE Pivot, int k)
{
	int LowIdxLockFlag = 0;
	int HighIdxLockFlag = 0;
	HighIdx--;
	while (LowIdx <= HighIdx)
	{
		LowIdxLockFlag = 1;
		HighIdxLockFlag = 1;		

		if (Coord[LowIdx + k * ProblemSize] < Pivot)
		{
			LowIdx++;
			LowIdxLockFlag = 0;
		}
		if (Coord[HighIdx + k * ProblemSize] > Pivot)
		{
			HighIdx--;
			HighIdxLockFlag = 0;	
		}

		if (LowIdxLockFlag == 1 && HighIdxLockFlag == 1)
		{
			for (int m = 0; m < 3; m++)
			{
				FP_TYPE temp;
				temp = Coord[LowIdx +  m * ProblemSize];
				Coord[LowIdx + m * ProblemSize] = Coord[HighIdx + m * ProblemSize];
				Coord[HighIdx + m * ProblemSize] = temp;
				
				int IdxTemp = TranIdx[LowIdx];
				TranIdx[LowIdx] = TranIdx[HighIdx];
				TranIdx[HighIdx] = IdxTemp;
			}
			LowIdx++;
			HighIdx--;
		}
	}
	return HighIdx + 1;
}



// additional preprocessing for multiGPU execution, including task division across multiple GPUs
// it divide the whole computational task as equal pieces for each GPU (this may have to be changed if GPUs on a node is not the same speed)
inline int Nufft::multithread_task_division()//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
{
	std::cout << "Multithread execution preprocessing... \n";
	
	int _average_src_per_device = ceil(FP_TYPE(problem_size) / FP_TYPE(nbodyfast->num_devices));

	int _box_cnt_glb = 0;

	nbodyfast->memory->alloc_host<int*>(&src_box_list_dev, nbodyfast->num_devices, "nufft->src_box_list_dev");
	nbodyfast->memory->alloc_host<int*>(&src_box_list_inv_dev, nbodyfast->num_devices, "nufft->src_box_list_inv_dev");
	nbodyfast->memory->alloc_host<int>(&src_start_dev, nbodyfast->num_devices, "nufft->src_start_dev");
	nbodyfast->memory->alloc_host<int>(&src_grid_start_dev, nbodyfast->num_devices, "nufft->src_grid_start_dev");
	nbodyfast->memory->alloc_host<int*>(&near_box_list_dev, nbodyfast->num_devices, "nufft->near_box_list_dev");

	src_start_dev[0] = 0;
	src_grid_start_dev[0] = 0;
	
	// for each device, find how many boxes it should handle
	for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id++)
	{
		std::stringstream _array_name;
		_array_name <<"nufft->src_box_list_dev[" << _thread_id << "]";
		src_box_list_dev[_thread_id] = NULL;

		nbodyfast->memory->alloc_host<int>(&src_box_list_dev[_thread_id], total_num_boxes + 1, _array_name.str());
		for (int i = 0; i < total_num_boxes + 1; i++)
		{
			src_box_list_dev[_thread_id][i] = -1;
		}
		////////////////////////////////////////////////////////////////////////////////////////////////
		// find number of boxes the each device need to process		
		int _acc_num_src_temp = 0;
		int _box_cnt = 0;
		while (_acc_num_src_temp < _average_src_per_device && _box_cnt_glb < total_num_boxes)
		{
			src_box_list_dev[_thread_id][_box_cnt + 1] = _box_cnt_glb;
			_acc_num_src_temp += src_box_map[total_num_boxes * 2 + _box_cnt_glb];
			_box_cnt++;
			_box_cnt_glb++;
		}
		src_box_list_dev[_thread_id][0] = _box_cnt;
		////////////////////////////////////////////////////////////////////////////////////////////////

		// Set the offset of each device pointed to the global source array
		if (_thread_id < nbodyfast->num_devices - 1)
		{
			src_start_dev[_thread_id + 1] = src_start_dev[_thread_id] + _acc_num_src_temp;
			src_grid_start_dev[_thread_id + 1] = src_grid_start_dev[_thread_id] + _box_cnt * interp_nodes_per_box;
		}
	}

	// for each device, build its src_box_list, near_box_list, etc.
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		std::stringstream _array_name;
		//for debug
		std::cout << "arrive multi-task division in thread " << _thread_id << std::endl;
		//for debug
		// build near-box list for each device from the global near-box list
		// near-box list contains ghost boxes
		_array_name <<"nufft->near_box_list_dev[" << _thread_id << "]";

		//BEN DEFINE THIS THREE VARIABLES
		int num_near_box_per_dim = 2*near_correct_layer+1;
		int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;
		int total_num_near_box_per_box_p1 = total_num_near_box_per_box+1;

		near_box_list_dev[_thread_id] = NULL;

		//nbodyfast->memory->alloc_host<int>(&near_box_list_dev[_thread_id], 28 * src_box_list_dev[_thread_id][0], _array_name.str());
		nbodyfast->memory->alloc_host<int>(&near_box_list_dev[_thread_id], total_num_near_box_per_box_p1 * src_box_list_dev[_thread_id][0], _array_name.str());
		
		/*for (int i = 0; i < 28 * src_box_list_dev[_thread_id][0]; i++)*/
		for (int i = 0; i < total_num_near_box_per_box_p1 * src_box_list_dev[_thread_id][0]; i++)
		{
			near_box_list_dev[_thread_id][i] = -1;
		}
		int _src_box_cnt = src_box_list_dev[_thread_id][0];
		int _percentage = 0;
		for (int i = 0; i < src_box_list_dev[_thread_id][0]; i++)
		{
			int _obs_box_idx = src_box_list_dev[_thread_id][i + 1];
			int _near_box_cnt = 0;
			while (near_box_list[_near_box_cnt + 1 + total_num_near_box_per_box_p1 * _obs_box_idx] >= 0 && _near_box_cnt < total_num_near_box_per_box)
			{
				near_box_list_dev[_thread_id][_near_box_cnt + i * total_num_near_box_per_box_p1 + 1] = near_box_list[_near_box_cnt + total_num_near_box_per_box_p1 * _obs_box_idx + 1];

				if (find_box(src_box_list_dev[_thread_id], near_box_list[_near_box_cnt + total_num_near_box_per_box_p1 * _obs_box_idx + 1], _src_box_cnt) == false)
				{
					_src_box_cnt++;
					src_box_list_dev[_thread_id][_src_box_cnt] = near_box_list[_near_box_cnt + total_num_near_box_per_box_p1 * _obs_box_idx + 1];///NEEDED BY MULTI-GPU///BEN
				}
				_near_box_cnt++;
			}
			near_box_list_dev[_thread_id][total_num_near_box_per_box_p1 * i] = near_box_list[total_num_near_box_per_box_p1 * _obs_box_idx];
		}
		////////////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////////////////////////
		// build src-box list inv for each device from the global near-box list
		_array_name.str("");
		_array_name <<"nufft->src_box_list_inv_dev[" << _thread_id << "]";
		src_box_list_inv_dev[_thread_id] = NULL;
		nbodyfast->memory->alloc_host<int>(&src_box_list_inv_dev[_thread_id], total_num_boxes + 1, _array_name.str());
		for (int i = 1; i < total_num_boxes + 1; i++)
		{
			src_box_list_inv_dev[_thread_id][i] = -1;
		}
		int _cnt = 1;
		while (src_box_list_dev[_thread_id][_cnt] >=0)
		{
			src_box_list_inv_dev[_thread_id][src_box_list_dev[_thread_id][_cnt] + 1] = _cnt - 1;
			_cnt++;
		}
		src_box_list_inv_dev[_thread_id][0] = _cnt - 1;
		////////////////////////////////////////////////////////////////////////////////////////////////

	} // _thread_id

	nbodyfast->memory->alloc_host<int*>(&src_box_map_dev, nbodyfast->num_devices, "nufft->src_box_map_dev");
	nbodyfast->memory->alloc_host<int>(&nbodyfast->src_size_dev, nbodyfast->num_devices, "src_size_dev");
	nbodyfast->memory->alloc_host<int>(&nbodyfast->src_size_act_dev, nbodyfast->num_devices, "src_size_act_dev");
	nbodyfast->memory->alloc_host<int*>(&src_trans_idx_dev, nbodyfast->num_devices, "nufft->src_trans_idx_dev");
	
	// obs_size_dev is different with src_size_dev even when the sources and observers are overlapped 

	////////////////////////////////////////////////////////////////////////////////////////////////
	// build src_box_map, src_trans_idx_dev, for each device, openMP acceleration to be implemented
	for (int _thread_id = 0; _thread_id < nbodyfast->num_devices; _thread_id++)
	{
		std::stringstream _array_name;

		////////////////////////////////////////////////////////////////////////////////////////////////
		_array_name <<"nufft->src_box_map_dev[" << _thread_id << "]";

		src_box_map_dev[_thread_id] = NULL;
		nbodyfast->memory->alloc_host<int>(&src_box_map_dev[_thread_id], total_num_boxes * 3, _array_name.str());

		for (int i = 0; i < total_num_boxes * 3; i++)
		{
			src_box_map_dev[_thread_id][i] = -1;
		}

		nbodyfast->src_size_dev[_thread_id] = 0;
		nbodyfast->src_size_act_dev[_thread_id] = 0;
		
		int _cur_box = 0;
		while (src_box_list_dev[_thread_id][_cur_box + 1] >= 0)
		{
			nbodyfast->src_size_dev[_thread_id] += src_box_map[src_box_list_dev[_thread_id][_cur_box + 1] + total_num_boxes * 2];
			if (_cur_box < src_box_list_dev[_thread_id][0]) nbodyfast->src_size_act_dev[_thread_id] += src_box_map[src_box_list_dev[_thread_id][_cur_box + 1] + total_num_boxes * 2];
			_cur_box++;
		}

		////////////////////////////////////////////////////////////////////////////////////////////////
		// this src_trans_idx_dev is not optimal, to be changed in the future
		// right now, it mapps a sorted global source array to those on multiple devices
		// but it should map an unsorted global source array direct to multiple devices
		src_trans_idx_dev[_thread_id] = NULL;

		_array_name.str("");
		_array_name <<"nufft->src_trans_idx_dev[" << _thread_id << "]";
		nbodyfast->memory->alloc_host<int>(&src_trans_idx_dev[_thread_id], nbodyfast->src_size_dev[_thread_id], _array_name.str());
		
		int _local_obs_pos[3] = {0};
		int _global_obs_pos[3] = {0};
		_cur_box = 0;

		while (src_box_list_dev[_thread_id][_cur_box + 1] >= 0)
		{
			_local_obs_pos[2] = src_box_map[src_box_list_dev[_thread_id][_cur_box + 1] + total_num_boxes * 2];

			_local_obs_pos[1] = _local_obs_pos[0] + _local_obs_pos[2];

			_global_obs_pos[0] = src_box_map[src_box_list_dev[_thread_id][_cur_box + 1]];	
			_global_obs_pos[1] = src_box_map[src_box_list_dev[_thread_id][_cur_box + 1] + total_num_boxes];
			_global_obs_pos[2] = src_box_map[src_box_list_dev[_thread_id][_cur_box + 1] + 2 * total_num_boxes];		
			
			for (int k = 0; k < 3; k++)
			{
				src_box_map_dev[_thread_id][src_box_list_dev[_thread_id][_cur_box + 1] + k * total_num_boxes] = _local_obs_pos[k];
			}
			for (int i = 0; i < _local_obs_pos[2]; i++)
			{
				src_trans_idx_dev[_thread_id][_local_obs_pos[0] + i] = _global_obs_pos[0] + i;
				//src_trans_idx_dev[_thread_id][_local_obs_pos[0] + i] = src_trans_idx[_global_obs_pos[0] + i];

			}
			_local_obs_pos[0] = _local_obs_pos[1];
			_cur_box++;
		} // _cur_box
		////////////////////////////////////////////////////////////////////////////////////////////////

	} // _cur_dev
	////////////////////////////////////////////////////////////////////////////////////////////////

	if (nbodyfast->field->src_obs_overlap == false)
	{
		std::cout << "src_obs_overlap == false not implemented yet. Exiting... at inline int Nufft::multithread_task_division()" << std::endl;
		exit(0);
		// to be done
	}
	else
	{
		nbodyfast->obs_size_act_dev = nbodyfast->src_size_act_dev;
		nbodyfast->obs_size_dev = nbodyfast->src_size_dev;
		obs_box_list_dev = src_box_list_dev; 
		obs_start_dev = src_start_dev;
		obs_grid_start_dev = src_grid_start_dev;
		obs_box_map_dev = src_box_map_dev;
		obs_trans_idx_dev = src_trans_idx_dev;
		obs_box_list_inv_dev = src_box_list_inv_dev; 

	}


//	std::cout << "multithread prepro: " << _average_src_per_device << std::endl;

	return 0;
}

// host array required for multiGPU calculation is allocated after all NUFFT preprocessing is done
inline int Nufft::multithread_host_array_alloc()
{
	std::cout << "Allocating additional data structure for multithread execution... \n";

	nbodyfast->field->array_alloc_multi_interface();
	return 0;
}
// find box
inline bool Nufft::find_box(int* arr, int idx, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (arr[i + 1] == idx) return true;
	}
	return false;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Nufft :: preprocessing()
{

	gpu_on = nbodyfast->gpu_on;
	cpu_near_onfly = false;

	// reading input file, parameters
	nbodyfast->error->last_error = input_file_reading();

	// domain resize according to source distribution and geometric distribution
	nbodyfast->error->last_error = domain_resize();

	// source and observer reordering and box assignment	
	total_num_boxes  = num_boxes[0] * num_boxes[1] * num_boxes[2];

	nbodyfast->memory->alloc_host<int>(&src_trans_idx, problem_size, "nufft->src_trans_idx");
	nbodyfast->memory->alloc_host<int>(&src_box_map, total_num_boxes * 3, "nufft->src_box_map");

	// generating interpolation grid
	total_num_grid_pts = num_grid_pts[0] * num_grid_pts[1] * num_grid_pts[2];
	nbodyfast->memory->alloc_host<FP_TYPE>(&src_grid_coord, total_num_grid_pts * 3, "nufft->src_grid_coord");

	// generating source box list
	nbodyfast->memory->alloc_host<int>(&src_box_list, total_num_boxes + 1, "nufft->src_box_list");

	// generating near box list
	int num_near_box_per_dim = 2*near_correct_layer+1;
	int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;
	nbodyfast->memory->alloc_host<int>(&near_box_list, total_num_boxes * (total_num_near_box_per_box+1), "nufft->near_box_list");//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
	
	//BEN ADDED
	nbodyfast->memory->alloc_host<int>(&near_bound_list, 6*total_num_boxes, "nufft->near_bound_list");
	// if sources and observers are not overlapping, allocate a separate set of arrays for observers
	if (nbodyfast->field->src_obs_overlap == false)
	{
		nbodyfast->memory->alloc_host<int>(&obs_box_list, problem_size, "nufft->obs_box_list");
		nbodyfast->memory->alloc_host<int>(&obs_trans_idx, problem_size, "nufft->obs_trans_idx");
		nbodyfast->memory->alloc_host<int>(&obs_box_map, total_num_boxes * 3, "nufft->obs_box_map");
		nbodyfast->memory->alloc_host<FP_TYPE>(&obs_grid_coord, total_num_grid_pts * 3, "nufft->obs_grid_coord");
	}
	else // otherwise, just set them to be the same as source arrays
	{
		obs_box_list = src_box_list;
		obs_trans_idx = src_trans_idx;
		obs_box_map = src_box_map;
		obs_grid_coord = src_grid_coord;
	}
	
	// identify the sources with its box and put them together
	nbodyfast->error->last_error = source_ordering();
	// generate source box list
	nbodyfast->error->last_error = source_list_gen();
	// generate near-field box list
	nbodyfast->error->last_error = near_list_gen();
	// generate near_bound box list
	nbodyfast->error->last_error = near_bound_gen();
	// generate interpolation grid (FFT grid)
	nbodyfast->error->last_error = interp_grid_gen();

	// FFT grid padding
	for (int m = 0; m < 3; m++)
	{
		int _temp = num_grid_pts[m] * 2;
		find_opt_fft_size(fft_size[m], _temp);
		green_size[m] = fft_size[m]/2+1;
	}
	std::cout << "grid_size = (" << num_grid_pts[0] << ", " << num_grid_pts[1] << ", " << num_grid_pts[2] << ")" << std::endl;
	std::cout << "fft_size = (" << fft_size[0] << ", " << fft_size[1] << ", " << fft_size[2] << ")" << std::endl;

	fft_r2c_size[0] = green_size[0]; fft_r2c_size[1] = fft_size[1]; fft_r2c_size[2] = fft_size[2];
	total_num_fft_pts = fft_size[0] * fft_size[1] * fft_size[2];
	total_num_green_pts = green_size[0] * green_size[1] * green_size[2];
	total_num_fft_r2c_pts = fft_r2c_size[0] * fft_r2c_size[1] * fft_r2c_size[2];
	
	// calculating interpolation coefficients, only in CPU version of code
	if (gpu_on == false)//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!WORK DONE
	{

		nbodyfast->memory->alloc_host<FP_TYPE>(&src_interp_coeff, problem_size * interp_nodes_per_box, "nufft->src_interp_coeff");
		nbodyfast->memory->alloc_host<int>(&src_interp_idx, problem_size * interp_nodes_per_box, "nufft->src_interp_idx");

		if (nbodyfast->field->src_obs_overlap == false)
		{
			nbodyfast->memory->alloc_host<FP_TYPE>(&obs_interp_coeff, problem_size * interp_nodes_per_box, "nufft->obs_interp_coeff");
			nbodyfast->memory->alloc_host<int>(&obs_interp_idx, problem_size * interp_nodes_per_box, "nufft->obs_interp_idx");
		}
		else
		{
			obs_interp_coeff = src_interp_coeff;
			obs_interp_idx = src_interp_idx;
		}

		nbodyfast->error->last_error = interp_coeff_gen();

		// Generating FFT plan for FFT on impedance matrix, CPU only
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_in_f, FIELD_DIM*total_num_fft_pts, "nufft->fft_in_f");
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_out_f, FIELD_DIM*total_num_fft_pts, "nufft->fft_out_f");
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_in_b, FIELD_DIM*total_num_fft_pts, "nufft->fft_in_b");
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_out_b, FIELD_DIM*total_num_fft_pts, "nufft->fft_out_b");

		nbodyfast->memory->alloc_host<fftw_plan>(&plan_grid, 2*FIELD_DIM, "nufft->plan_grid");

		plan_grid[0] = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], fft_in_f, fft_out_f, FFTW_FORWARD, FFTW_ESTIMATE);
		plan_grid[1] = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], fft_in_f+total_num_fft_pts, fft_out_f+total_num_fft_pts, FFTW_FORWARD, FFTW_ESTIMATE);
		plan_grid[2] = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], fft_in_f+2*total_num_fft_pts, fft_out_f+2*total_num_fft_pts, FFTW_FORWARD, FFTW_ESTIMATE);
		plan_grid[3] = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], fft_in_b, fft_out_b, FFTW_BACKWARD, FFTW_ESTIMATE);
		plan_grid[4] = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], fft_in_b+total_num_fft_pts, fft_out_b+total_num_fft_pts, FFTW_BACKWARD, FFTW_ESTIMATE);
		plan_grid[5] = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], fft_in_b+2*total_num_fft_pts, fft_out_b+2*total_num_fft_pts, FFTW_BACKWARD, FFTW_ESTIMATE);

	}
	
	// Generating impedance matrix for FFT convolution
	// calling of the pure virutal function, g_mat_alloc and g_mat_set actually leads to the body of respective function in Nufft's derived classes. I am not sure if this is a good programming style or not.
	nbodyfast->error->last_error = g_mat_alloc(&imp_mat_ptr, num_grid_pts);
	nbodyfast->error->last_error = g_mat_set(imp_mat_ptr, src_grid_coord, obs_grid_coord, num_grid_pts, fft_size);

	// Generating impedance matrix for near-field subtraction, only in CPU version AND on-fly calculation is OFF
	for (int i = 0; i < 3; i++)//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!WORK DONE
	{
		near_num_grid_pts[i] = (num_nodes[i] - 1) * (2*near_correct_layer+1) + 1;
		fft_size_near[i] = 2 * near_num_grid_pts[i];
	}

	total_num_near_grid_pts = near_num_grid_pts[0] * near_num_grid_pts[1] * near_num_grid_pts[2];
	
	if (gpu_on == false)//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!FIELD DIM NOT DONE YET
	{
		
		FP_TYPE *_near_grid_coord = NULL;
		nbodyfast->memory->alloc_host<FP_TYPE>(&_near_grid_coord, total_num_near_grid_pts * 3, "nufft->_near_grid_coord");
	
		near_grid_gen(_near_grid_coord, near_num_grid_pts, total_num_near_grid_pts);

		// Generating FFT plan for FFT on near field impedance matrix
	
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_in_f_near, 8 * FIELD_DIM*total_num_near_grid_pts, "nufft->fft_in_f_near");
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_out_f_near, 8 * FIELD_DIM*total_num_near_grid_pts, "nufft->fft_out_f_near");
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_in_b_near, 8 * FIELD_DIM*total_num_near_grid_pts, "nufft->fft_in_b_near");
		nbodyfast->memory->alloc_host<fftw_complex>(&fft_out_b_near, 8 * FIELD_DIM*total_num_near_grid_pts, "nufft->fft_out_b_near");
	
		nbodyfast->memory->alloc_host<fftw_plan>(&plan_grid_near, FIELD_DIM*2, "nufft->plan_grid_near");// FIELD_DIM CHANGE!!!!!!!!!!!!!!!!!!NOT DONE YET

		plan_grid_near[0] = fftw_plan_dft_3d(fft_size_near[2], fft_size_near[1], fft_size_near[0], fft_in_f_near, fft_out_f_near, FFTW_FORWARD, FFTW_ESTIMATE);
		plan_grid_near[1] = fftw_plan_dft_3d(fft_size_near[2], fft_size_near[1], fft_size_near[0], fft_in_f_near+8*total_num_near_grid_pts, fft_out_f_near+8*total_num_near_grid_pts, FFTW_FORWARD, FFTW_ESTIMATE);
		plan_grid_near[2] = fftw_plan_dft_3d(fft_size_near[2], fft_size_near[1], fft_size_near[0], fft_in_f_near+16*total_num_near_grid_pts, fft_out_f_near+16*total_num_near_grid_pts, FFTW_FORWARD, FFTW_ESTIMATE);
		plan_grid_near[3] = fftw_plan_dft_3d(fft_size_near[2], fft_size_near[1], fft_size_near[0], fft_in_b_near, fft_out_b_near, FFTW_BACKWARD, FFTW_ESTIMATE);
		plan_grid_near[4] = fftw_plan_dft_3d(fft_size_near[2], fft_size_near[1], fft_size_near[0], fft_in_b_near+8*total_num_near_grid_pts, fft_out_b_near+8*total_num_near_grid_pts, FFTW_BACKWARD, FFTW_ESTIMATE);
		plan_grid_near[5] = fftw_plan_dft_3d(fft_size_near[2], fft_size_near[1], fft_size_near[0], fft_in_b_near+16*total_num_near_grid_pts, fft_out_b_near+16*total_num_near_grid_pts, FFTW_BACKWARD, FFTW_ESTIMATE);

		// calling of the pure virutal function, g_mat_alloc and g_mat_set actually leads to the body of respective function in Nufft's derived classes. I am not sure if this is a good programming style or not.
		nbodyfast->error->last_error = g_mat_alloc(&near_imp_mat_ptr, near_num_grid_pts);
		nbodyfast->error->last_error = g_mat_set(near_imp_mat_ptr, _near_grid_coord, _near_grid_coord, near_num_grid_pts, fft_size_near);
		total_num_fft_pts_near = fft_size_near[0] * fft_size_near[1] * fft_size_near[2];
		
		nbodyfast->memory->free_host<FP_TYPE>(&_near_grid_coord);
	}


	if (nbodyfast->multi_device == true)
	{
		nbodyfast->error->last_error = multithread_task_division();
		nbodyfast->error->last_error = multithread_host_array_alloc();
 	}

	return 0;
}
int Nufft :: execution()
{
	
	// Five stages of execution
	nbodyfast->timer->time_stamp("PROJ");
	nbodyfast->error->last_error = projection();
	nbodyfast->timer->time_stamp("PROJ");

	nbodyfast->timer->time_stamp("FFT");
	nbodyfast->error->last_error = fft_real_to_k();
	nbodyfast->timer->time_stamp("FFT");

	nbodyfast->timer->time_stamp("CONV");
	nbodyfast->error->last_error = convolution_k();
	nbodyfast->timer->time_stamp("CONV");

	nbodyfast->timer->time_stamp("FFT");
	nbodyfast->error->last_error = fft_k_to_real();
	nbodyfast->timer->time_stamp("FFT");

	nbodyfast->timer->time_stamp("INTE");
	nbodyfast->error->last_error = correction_interpolation();
	nbodyfast->timer->time_stamp("INTE");

	return 0;
}


}


