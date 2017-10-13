/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
*nufft_gpu.h: class declaration of Class NufftGpu, Class NufftParamGpu, Class NufftArrayGpu
*Class NufftParamGpu and Class NufftArrayGpu should later be put inside NufftGpu
*/
#ifndef _NUFFT_GPU
#define _NUFFT_GPU
#include "field_gpu.h"
#include "nufft.h"
#include "cufft.h"
namespace NBODYFAST_NS{
// Parameters of NUFFT that might be used by GPU kernels
// They are transferred to GPU during preprocessing and all kernels will be given a pointer to access it
// They might also be stored in constant memory, but I didn't see the performance improvement...
class NufftParamGpu
{
public:
	NufftParamGpu()
	{
		
	}
	virtual ~NufftParamGpu()
	{
	}

	int problem_size;
	int src_size_dev;
	int obs_size_dev;
	int interp_order[3];
	int num_nodes[3];
	int interp_nodes_per_box;
	int num_boxes[3];
	int total_num_boxes;
	int total_num_boxes_dev;
	int src_num_nonempty_boxes;
	int obs_num_nonempty_boxes;
	int num_grid_pts[3];
	int total_num_grid_pts;
	int near_num_grid_pts[3];
	int total_num_near_grid_pts;
	int near_box_cnt;
	int max_near_src_cnt;
	int near_correct_layer;

	int fft_size[3];
	int total_num_fft_pts;
	int green_size[3];
	int total_num_green_pts;
	int fft_r2c_size[3];
	int total_num_fft_r2c_pts;

	FP_TYPE cell_size[3];
	FP_TYPE box_size[3];

	FP_TYPE src_domain_range[9];
	FP_TYPE obs_domain_range[9];
	FP_TYPE src_domain_center[3];
	FP_TYPE obs_domain_center[3];
	FP_TYPE unified_domain_size[3];

	FP_TYPE epsilon;
	bool linear_interp; // not in use

	virtual void set_value(class NufftGpu *nufft_ptr);

	class NBODYFAST *nbodyfast;

protected:
};
// Intermediate data structures that are required by NUFFT kernels
// They are allocated on GPUs during preprocessing and all kernels will be given a pointer to access it
// NufftArrayGpu will be inherited by NufftGpu's derived class to added field related arrays to it
// I am not very happy with this design....
class NufftArrayGpu
{
public:
	NufftArrayGpu()
	{
		d_src_coord = NULL;
		d_obs_coord = NULL;
		
		d_src_box_map = NULL;
		d_obs_box_map = NULL;

		d_src_trans_idx = NULL;
		d_obs_trans_idx = NULL;

		//d_near_box_list = NULL;
		d_near_bound_list = NULL;

		d_src_box_list = NULL;
		d_obs_box_list = NULL;

		d_src_box_list_inv = NULL;
		d_obs_box_list_inv = NULL;

		d_src_grid_coord = NULL;
		d_obs_grid_coord = NULL;

		d_fft_inplace_r2c = NULL;//BEN ADDED
		d_fft_inplace_r2c_FP = NULL;
		//d_fft_inplace_b = NULL;

	}
	virtual ~NufftArrayGpu()
	{
	}

	virtual void set_value(class NufftGpu *nufft_ptr);
	
	FP_TYPE *d_src_coord;
	FP_TYPE *d_obs_coord;

	int *d_src_box_map;
	int *d_obs_box_map;

	int *d_src_trans_idx;
	int *d_obs_trans_idx;

	//int *d_near_box_list;
	int *d_near_bound_list;

	int *d_src_box_list;
	int *d_obs_box_list;

	int *d_src_box_list_inv;
	int *d_obs_box_list_inv;

	FP_TYPE *d_src_grid_coord;
	FP_TYPE *d_obs_grid_coord;

	CUFFT_COMPLEX_TYPE *d_fft_inplace_r2c;
	FP_TYPE *d_fft_inplace_r2c_FP;
	
	cufftHandle cufftPlan_r2c;
	cufftHandle cufftPlan_c2r;

	class NBODYFAST *nbodyfast;

};
class NufftGpu : public virtual Nufft
{
public:
	NufftGpu() : Nufft(){}
	NufftGpu(class NBODYFAST *n_ptr);
	virtual ~NufftGpu() = 0;

	virtual int preprocessing();

	class NufftParamGpu **nufft_const_gpu; 
	class NufftParamGpu **d_nufft_const_gpu; 

	class NufftArrayGpu **nufft_array_gpu;
	class NufftArrayGpu **d_nufft_array_gpu; // we actually never need this. an inherited version of d_nufft_array_gpu is what we need to transfer to GPU. 

	FieldGpu *field_gpu; // points to its field sibling
private:
	int preprocessing_single();
	int preprocessing_multi();


};
}
#endif