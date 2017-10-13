/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft.h: class decalaration for Class Nufft
*/
#ifndef _NUFFT
#define _NUFFT
#include "fftw3.h"
#include "fp_precision.h"
namespace NBODYFAST_NS{
// optimal FFT sizes for CUFFT library
const int OPT_FFT_SIZE[83] = {5, 8, 16, 32, 49, 64, 81, 84, 90, 96, 100, 108, 112, 125, 128, 135, 140, 144, 147, 150, 160, 168, 175, 180, 189, 192, 196, 200, 216, 224, 225, 243, 245, 250, 256, 270, 280, 288, 294, 300, 315, 320, 324, /*43*/ 336, 343, 350, 360, 375, 378,  384, 392, 400, 405, 420, 432, 441, 448, 480, 486, 490, 500, 504, 512, 525, 540, 560, 567, 576, 588, 600, 625, 640, 648, 672, 675, 686, 700, 720, 729, 735, 750, 756, 768 /*83*/};
inline void find_opt_fft_size(int &out, int in)
{
	if (in > 768)
	{
		std::cout << "the grid  size is greater than 768, so it is not padded" << std::endl;
		out = in;
		//std::cout << "grid size is " << in << "\n" << std::endl;
		//std::cout << "fft size is " << out << "\n" << std::endl;
		return;
	}
	int i = 1;
	while (i < 83 && OPT_FFT_SIZE[i] < in)
	{
		i++;
	}
	out = OPT_FFT_SIZE[i];
	//for debug
	out = 1;
	while( out < in )
		out = out*2;
	//for debug
	//std::cout << "grid size is " << in << "\n" << std::endl;
	//std::cout << "fft size is " << out << "\n" << std::endl;
	return;
}


// class decalaration
class Nufft
{
friend class NufftParamGpu; // make NufftParamGpu friend as it will need to access many protected member of Nufft
public:
	Nufft(){}
	Nufft(class NBODYFAST *n_ptr);
	virtual ~Nufft() = 0;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// These virtual functions will all be implemented separately by its derived classes
	// However, they have body in Nufft class to do common taskes that all derived classes will do
	virtual int preprocessing();
	virtual int projection() = 0;
	virtual int fft_real_to_k() = 0;
	virtual int convolution_k() = 0;
	virtual int fft_k_to_real() = 0;
	virtual int correction_interpolation() = 0;
	virtual int execution();

	// stupid impedance matrix generation method... I can't remember why I need separate methods to generate impedance matrix...
	virtual int g_mat_alloc(class ImpMatrix **g_mat, int n_grid_ptr[3]) = 0;
	virtual int g_mat_set(class ImpMatrix *g_mat, FP_TYPE *src_grid_coord, FP_TYPE *obs_grid_coord, int n_grid_ptr[3], int fft_size[3]) = 0;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// two interface inlined function to fetch trans_idx
	// will be used by copying source amps and field amps from outside
	inline int* get_src_trans_idx()
	{
		return src_trans_idx;
	}
	inline int* get_obs_trans_idx()
	{
		return obs_trans_idx;
	}

	// two interface inlined function to fetch trans_idx_dev
	// will be used by setting source amps and field amps from outside
	// used while multiGPU calculation is active
	inline int** get_src_trans_idx_dev()
	{
		return src_trans_idx_dev;
	}
	inline int** get_obs_trans_idx_dev()
	{
		return obs_trans_idx_dev;
	}


	class NBODYFAST *nbodyfast;

protected:
	class ImpMatrix *imp_mat_ptr; // pointer to the impedance matrix
	class ImpMatrix *near_imp_mat_ptr; // pointer to the impedance matrix for pre-compute near field correction, only for CPU NUFFT

	bool gpu_on;
	bool cpu_near_onfly; // is the near field correction done on-the-fly or pre-computed, only valid for CPU NUFFT

	int problem_size; // a copy from nbodyfast->problem_size, want to save some time from writing "nbodyfast->" every time we need problem_size
	int interp_order[3]; // interpolation order
	int near_correct_layer;
	int num_nodes[3]; // number nodes for interpolation. is interp_order + 1
	int interp_nodes_per_box; // num_nodes[0] * num_nodes[1] *num_nodes[2] 
	int num_boxes[3]; // number of boxes
	int total_num_boxes; //  num_boxes[0] * num_boxes[1] *num_boxes[2] 
	int src_num_nonempty_boxes; // number of non-empty boxes...right now only used to calculate average number of sources per box...
	int obs_num_nonempty_boxes; //
	int num_grid_pts[3]; // number of points on FFT grid, along each dimension
	int total_num_grid_pts; // num_grid_pts[0] * num_grid_pts[1] *num_grid_pts[2] 
	int near_num_grid_pts[3]; // number of points on NEAR FFT grid, along each dimension
	int total_num_near_grid_pts; // near_num_grid_pts[0] * near_num_grid_pts[1] *near_num_grid_pts[2] 
	
	// temporary variables used while generating nearbox-list and near correction coefficient table
	int near_box_cnt; 
	int max_near_src_cnt; 

	// interplation cell size
	FP_TYPE cell_size[3];

	// box size
	FP_TYPE box_size[3];
	
	// map box number to src number
	int *src_box_map;
	int *obs_box_map;

	// reordering sources belongs to the same box together in memory
	int *src_trans_idx;
	int *obs_trans_idx;

	// mapping local box number to global box number, a dummy array in singleGPU calculation
	int *src_box_list;
	int *obs_box_list;

	// mapping global box number to local box number, a dummy array in singleGPU calculation
	int *src_box_list_inv;
	int *obs_box_list_inv;

	// near-box list
	int *near_box_list;
	int *near_bound_list;
	
	// coordinates of FFT grid built upon sources
	FP_TYPE *src_grid_coord;
	FP_TYPE *obs_grid_coord;

	// interpolation coefficients for interpolating field amps from FFT grids to observers, only in use while doing CPU NUFFT
	FP_TYPE *src_interp_coeff;
	FP_TYPE *obs_interp_coeff;

	// interpolation idx for interpolating field amps from FFT grids to observers, only in use while doing CPU NUFFT
	int *src_interp_idx;
	int *obs_interp_idx;
	
	// mapping local box number to global box number, in multiGPU calculation
	int **src_box_list_dev;
	int **obs_box_list_dev;

	// mapping global box number to local box number, in multiGPU calculation
	int **src_box_list_inv_dev;
	int **obs_box_list_inv_dev;

	// the offset in the global src amp. array for each GPU device
	int *src_start_dev;
	int *obs_start_dev;

	// the offset in the global FFT grid array for each GPU device
	int *src_grid_start_dev;
	int *obs_grid_start_dev;

	// source reordering array for multiGPU
	int **src_trans_idx_dev;
	int **obs_trans_idx_dev;

	// near-box list array for multiGPU
	int **near_box_list_dev;////NEEDED FOR MULTIPLE GPUS IMPLEMENTATION/////////////BEN

	// box-to-source mapping array for multiGPU
	int **src_box_map_dev;
	int **obs_box_map_dev;
	
	int fft_size[3]; 	// FFT size
	int total_num_fft_pts; // total FFT points
	int green_size[3];
	int total_num_green_pts;
	int fft_r2c_size[3];
	int total_num_fft_r2c_pts;

	// input, output array for CPU FFT
	// we could save half the memory by using in-place transfer... this has been implemented on GPU FFT but I'm just too lazy to change the CPU version... we don't use CPU NUFFT anyway...
	fftw_complex *fft_in_f;
	fftw_complex *fft_out_f;
	fftw_complex *fft_in_b;
	fftw_complex *fft_out_b;

	// near FFT parameters
	int fft_size_near[3]; 
	int total_num_fft_pts_near;

	fftw_complex *fft_in_f_near;
	fftw_complex *fft_out_f_near;
	fftw_complex *fft_in_b_near;
	fftw_complex *fft_out_b_near;

	fftw_plan *plan_grid, *plan_grid_near;


private:
	// temporary variables
	int num_box_inp;
	FP_TYPE box_size_inp;
	int interp_order_inp;
	bool adaptive_decomp;

	// internal functions just to make the preprocessing subroutine looks better
	inline int input_file_reading();
	inline int domain_resize();
	inline int source_ordering();
	inline int source_list_gen();
	inline int near_list_gen();
	inline int near_bound_gen();
	inline int interp_grid_gen();
	inline int interp_coeff_gen();
	inline int multithread_task_division();
	inline int multithread_host_array_alloc();
	
	// find if a box is already in a list of boxes
	inline bool find_box(int*, int, int);

	// generating near field grid, only used by CPU NUFFT
	inline int near_grid_gen(FP_TYPE *_near_grid_coord, int _near_num_grid_pts[3], int _total_num_near_grid_pts);

	// these two subroutines is for reordering sources into boxes
	inline int source_swap(FP_TYPE *_coord, int _coord_size, int* _map, int *_trans_idx, FP_TYPE *_domain_range, FP_TYPE *_box_size, int *_num_boxes);
	inline int swap(FP_TYPE *Coord, int ProblemSize, int *TranIdx, int LowIdx, int HighIdx, FP_TYPE Pivot, int k);
	//

};
}
#endif
