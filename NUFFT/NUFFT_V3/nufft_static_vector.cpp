/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft_static_vector.cpp: class definition of Class NufftStaticVector
*/
#include "error.h"
#include "field_static_vector.h"
#include "imp_matrix_static_vector.h"
#include "memory.h"
#include "nbodyfast.h"
#include "nufft.h"
#include "nufft_static_vector.h"

//#include <cstdlib>

namespace NBODYFAST_NS{
NufftStaticVector :: NufftStaticVector(class NBODYFAST *n_ptr) : Nufft(n_ptr)
{
	k_u_src_grid = NULL;
	u_src_grid = NULL;
	k_u_obs_grid = NULL;
	u_obs_grid = NULL;
	g_direct_near = NULL;
	u_near_src_grid = NULL	;
	u_near_obs_grid = NULL;
	k_u_near_src_grid = NULL;
	
	g_grid = NULL;
	g_grid_near = NULL;
	
	field_static_vector = dynamic_cast<FieldStaticVector*>(n_ptr->field);
};
NufftStaticVector :: ~NufftStaticVector()
{

	nbodyfast->memory->free_host<FP_TYPE>(&g_direct_near);
	nbodyfast->memory->free_host<ImpMatrixStaticVector>(&g_grid_near);
	nbodyfast->memory->free_host<ImpMatrixStaticVector>(&g_grid);

	nbodyfast->memory->free_host<std::complex<FP_TYPE> >(&k_u_near_src_grid);
	nbodyfast->memory->free_host<FP_TYPE>(&u_near_obs_grid);
	nbodyfast->memory->free_host<FP_TYPE>(&u_near_src_grid);
	nbodyfast->memory->free_host<FP_TYPE>(&k_u_src_grid);
	nbodyfast->memory->free_host<FP_TYPE>(&u_src_grid);
}
////NEAR CHANGE//////////////////!!!!!!!!!!!!!!!
int NufftStaticVector :: preprocessing()
{

	std::cout << "NufftStaticVector :: preprocessing()" << std::endl;
	
	Nufft::preprocessing(); // above everything, call preprocessing() of its parent class
	
	// pointers to the impedance matrix
	g_grid = dynamic_cast<ImpMatrixStaticVector*>(imp_mat_ptr);
	g_grid_near = dynamic_cast<ImpMatrixStaticVector*>(near_imp_mat_ptr);

	// if we are using CPU, then build the following arrays, including a smaller FFT grid for near-field correction
	if (gpu_on == false)
	{
		nbodyfast->memory->alloc_host<FP_TYPE>(&u_src_grid, FIELD_DIM*total_num_grid_pts, "nufft->u_src_grid");

		// k_u_src_grid is actually not in use
		//	nbodyfast->memory->alloc_host<FP_TYPE>(&k_u_src_grid, total_num_grid_pts, "nufft->k_u_src_grid");

		// These two are usually not in use. Data can be stored in the place of "u_src_grid" and "k_u_src_grid"
		u_obs_grid = u_src_grid;
		//k_u_obs_grid = k_u_src_grid;

		//nbodyfast->memory->alloc_host<FP_TYPE>(&u_obs_grid, total_num_grid_pts, "nufft->u_obs_grid");
		//nbodyfast->memory->alloc_host<FP_TYPE>(&k_u_obs_grid, total_num_grid_pts, "nufft->k_u_obs_grid");

		nbodyfast->memory->alloc_host<FP_TYPE>(&u_near_src_grid, FIELD_DIM*total_num_near_grid_pts, "nufft->u_near_src_grid");
		nbodyfast->memory->alloc_host<FP_TYPE>(&u_near_obs_grid, FIELD_DIM*interp_nodes_per_box, "nufft->u_near_obs_grid");
		nbodyfast->memory->alloc_host<std::complex<FP_TYPE> >(&k_u_near_src_grid, FIELD_DIM*total_num_near_grid_pts * 8, "nufft->k_u_near_src_grid");//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!

		if (cpu_near_onfly == false)//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!NOT DONE YET!!!!!!!!!!!!!
		{
			int num_near_box_per_dim = 2*near_correct_layer+1;
			int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;
			int green_num = FIELD_DIM*(FIELD_DIM+1)/2;
			nbodyfast->memory->alloc_host<FP_TYPE>(&g_direct_near, green_num*problem_size*max_near_src_cnt, "nufft->g_direct_near");

			for (int i = 0; i < green_num*problem_size * max_near_src_cnt; i++)
			{
				g_direct_near[i] = 0.0f;
			}

			for (int i = 0; i < total_num_boxes; i++)
			{
				for (int m = obs_box_map[i]; m < obs_box_map[total_num_boxes + i]; m++)
				{
					int _cnt_src = 0;
					for (int j = 0; j < near_box_list[i * (total_num_near_box_per_box+1)]; j++)
					{
						int _near_box_idx = near_box_list[i * (total_num_near_box_per_box+1) + j + 1];
						for (int l = src_box_map[_near_box_idx]; l < src_box_map[total_num_boxes + _near_box_idx]; l++, _cnt_src++)
						{
							FP_TYPE r[3];

							for (int k = 0; k < 3; k++) 
							{
								r[k] = (field_static_vector->src_coord[l + k * problem_size] - field_static_vector->obs_coord[m + k * problem_size]);
							}
							int _idx_near_temp = _cnt_src + m * max_near_src_cnt;

							FP_TYPE _G_temp[6];
							FP_TYPE _r_temp[3];

							for (int k = 0; k < 3; k++) _r_temp[k] = r[k];
							FieldStaticVector::get_G(_G_temp, _r_temp, Field::epsilon());
							
							for(unsigned int l = 0; l < green_num; l++)
								g_direct_near[_idx_near_temp+l*problem_size*max_near_src_cnt] = _G_temp[l]; 
						} // l
					} // j
				} // m
			} // i
		}
	}
	return 0;
}

int NufftStaticVector :: g_mat_alloc(class ImpMatrix **_g_mat, int n_grid_ptr[3])
{
	class ImpMatrixStaticVector *_imp_matrix = NULL;
	nbodyfast->memory->alloc_host<ImpMatrixStaticVector>(&_imp_matrix, 1, "nufft->_g_mat");
	*_g_mat = _imp_matrix;
	return 0;
}

int NufftStaticVector :: g_mat_set(class ImpMatrix *_g_mat, FP_TYPE *_src_grid_coord, FP_TYPE *_obs_grid_coord, int n_grid_ptr[3], int fft_size[3])
{
	// I believe this subroutine are not yet working when _src_grid_coord and _obs_grid_coord are not overlapped. So I put this line here
	 if (field_static_vector->src_obs_overlap == false)
	 {
		std::cout << "The code is not yet working for src_obs_overalp = false at int NufftStaticVector :: g_mat_set" << std::endl;
		exit(0);
	 }

	 class ImpMatrixStaticVector *_imp_matrix = dynamic_cast<ImpMatrixStaticVector*>(_g_mat);
	
	_imp_matrix->set_parent(nbodyfast);
	
	// set the size of impedance matrix
	for (int i = 0; i < 3; i++){
		_imp_matrix->imp_mat_size[i] = n_grid_ptr[i];
		_imp_matrix->_padded_green_dim[i] = fft_size[i]/2+1;
	}
	
	// _num_padded_grid_pts should be able to be referenced during iteration....!!!!!!!!!!!!!!!!!!
	int _num_padded_grid_pts = fft_size[0]*fft_size[1]*fft_size[2]; // number of points in the padded NUFFT grid
	int _total_num_grid_pts = n_grid_ptr[0] * n_grid_ptr[1] * n_grid_ptr[2]; // total number of grid points, pre-padding
	// total number of points for scalar green's function
	int _total_num_green_pts = _imp_matrix->_padded_green_dim[0] * _imp_matrix->_padded_green_dim[1] *_imp_matrix->_padded_green_dim[2];
	
	// allocate impedance matrices in both real space and k-space
	int num_greens = FIELD_DIM * (FIELD_DIM + 1) / 2; // number of sub-matrices in the Green's func tensor, taking advantage of symmetry
	nbodyfast->memory->alloc_host<FP_TYPE>(&_imp_matrix->imp_mat_data, num_greens*_num_padded_grid_pts, "field->_imp_matrix->imp_mat_data");
	nbodyfast->memory->alloc_host<std::complex<FP_TYPE> >(&_imp_matrix->k_imp_mat_data, num_greens*_num_padded_grid_pts, "field->_imp_matrix->k_imp_mat_data");
	nbodyfast->memory->alloc_host<FP_TYPE>(&_imp_matrix->k_imp_mat_data_gpu, num_greens*_total_num_green_pts, "field->_imp_matrix->k_imp_mat_data_gpu");
	
	// initialize
	for (int i = 0; i < num_greens*_num_padded_grid_pts; i++)
	{
		_imp_matrix->imp_mat_data[i] = 0.0f;
	}

	// Calculating Green's function between grid points and filling the symmetric part
	FP_TYPE *_g_temp = NULL;
	nbodyfast->memory->alloc_host<FP_TYPE>(&_g_temp, num_greens, "_g_temp_tensor");

	//FP_TYPE	_g_temp[6];
	for (int k = 0; k < n_grid_ptr[2]; k++)
		for (int j = 0; j < n_grid_ptr[1]; j++)
			for (int i = 0; i < n_grid_ptr[0]; i++)
			{
				int _idx_grid_coord = i + j * n_grid_ptr[0] + k * n_grid_ptr[0] * n_grid_ptr[1];
//				int _idx_g_grid = i + j * fft_size[0] + k * fft_size[0] * fft_size[1];

				FP_TYPE r[3];
				
				// shift between observer grid and source grid
				for (int l = 0; l < 3; l++) 
				{
					r[l] = (_obs_grid_coord[0 + _total_num_grid_pts * l] - _src_grid_coord[_idx_grid_coord + _total_num_grid_pts * l]);
				}

				FP_TYPE _r_temp[3];
				
				// do Greens function calculation for 8 times for each combination of (+/- r_x, +/- r_y, +/- r_z) 
				// and fill the impedance matrix
				for (int l = 0; l < 3; l++) _r_temp[l] = r[l];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				for (unsigned int l = 0; l < num_greens; l++ )
					_imp_matrix->imp_mat_data[i + j * fft_size[0] + k * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] = _g_temp[l];
				
				for (int l = 0; l < 3; l++) _r_temp[l] = r[l];
				_r_temp[0] = -_r_temp[0];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if (i > 0){
					for (unsigned int l = 0; l < num_greens; l++ )
						_imp_matrix->imp_mat_data[(fft_size[0] - i) + j * fft_size[0] + k * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] = _g_temp[l];
				}

				for (int l = 0; l < 3; l++) _r_temp[l] = r[l];
				_r_temp[1] = -_r_temp[1];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if (j > 0){
					for (unsigned int l = 0; l < num_greens; l++ )
						_imp_matrix->imp_mat_data[i + (fft_size[1] - j) * fft_size[0] + k * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] = _g_temp[l];
				}
				
				for (int l = 0; l < 3; l++) _r_temp[l] = r[l];
				_r_temp[2] = -_r_temp[2];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if (k > 0){
					for (unsigned int l = 0; l < num_greens; l++ )
					_imp_matrix->imp_mat_data[i + j * fft_size[0] + (fft_size[2] - k) * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] =  _g_temp[l];
				}

				for (int l = 0; l < 3; l++) _r_temp[l] = -r[l];
				_r_temp[2] = -_r_temp[2];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if ((i > 0) && (j > 0)){
					for (unsigned int l = 0; l < num_greens; l++ )
						_imp_matrix->imp_mat_data[(fft_size[0] - i) + (fft_size[1] - j) * fft_size[0] + k * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] =  _g_temp[l];
				}

				for (int l = 0; l < 3; l++) _r_temp[l] = -r[l];
				_r_temp[0] = -_r_temp[0];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if ((j > 0) && (k > 0)){
					for (unsigned int l = 0; l < num_greens; l++ )
						_imp_matrix->imp_mat_data[i + (fft_size[1] - j) * fft_size[0] + (fft_size[2] - k) * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] =  _g_temp[l];
				}

				for (int l = 0; l < 3; l++) _r_temp[l] = -r[l];
				_r_temp[1] = -_r_temp[1];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if ((k > 0) && (i > 0)){
					for (unsigned int l = 0; l < num_greens; l++ )
						_imp_matrix->imp_mat_data[(fft_size[0] - i) + j * fft_size[0] + (fft_size[2] - k) * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] =  _g_temp[l];
				}

				for (int l = 0; l < 3; l++) _r_temp[l] = -r[l];
				FieldStaticVector::get_G(_g_temp, _r_temp, Field::epsilon());
				if ((i > 0) && (j > 0) && (k > 0)){
					for (unsigned int l = 0; l < num_greens; l++ )
						_imp_matrix->imp_mat_data[(fft_size[0] - i) + (fft_size[1] - j) * fft_size[0] + (fft_size[2] - k) * fft_size[0] * fft_size[1] + l*_num_padded_grid_pts] =  _g_temp[l];
				}

			}
	
	fftw_complex *_fft_in_f = NULL;

	nbodyfast->memory->alloc_host<fftw_complex>(&_fft_in_f, _num_padded_grid_pts, "field->_fft_in_f");
	//nbodyfast->memory->alloc_host<fftw_complex>(&_fft_out_f, _num_padded_grid_pts, "field->_fft_out_f");

	//fftw_plan _plan1 = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], _fft_in_f, _fft_out_f, FFTW_FORWARD, FFTW_ESTIMATE);	
	fftw_plan _plan1 = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], _fft_in_f, _fft_in_f, FFTW_FORWARD, FFTW_ESTIMATE);	
	
	for(unsigned int l = 0; l < num_greens; l++){
		// Get the impedance matrix in k_space
		for (int i = 0; i < _num_padded_grid_pts; i++)
		{
			_fft_in_f[i][0] = _imp_matrix->imp_mat_data[i + l*_num_padded_grid_pts];
			_fft_in_f[i][1] = 0.0f;
		}

		fftw_execute(_plan1);

		for (int i = 0; i < _num_padded_grid_pts; i++)
		{
			_imp_matrix->k_imp_mat_data[i+l*_num_padded_grid_pts] = std::complex<FP_TYPE>(_fft_in_f[i][0], _fft_in_f[i][1]);
		}

		//BEN ADDED
		int green_id = -1;
		int fft_id = -1;
		for (int k = 0; k < _imp_matrix->_padded_green_dim[2]; k++)
			for (int j = 0; j < _imp_matrix->_padded_green_dim[1]; j++)
				for (int i = 0; i < _imp_matrix->_padded_green_dim[0]; i++){
					green_id = i + j*_imp_matrix->_padded_green_dim[0] + k*_imp_matrix->_padded_green_dim[0]*_imp_matrix->_padded_green_dim[1];
					fft_id  = i + j*fft_size[0] + k*fft_size[0]*fft_size[1];
					_imp_matrix->k_imp_mat_data_gpu[green_id + l*_total_num_green_pts] = _fft_in_f[fft_id][0];
				}
				//BEN ADDED ABOVE
	}
	// free the temporay fftw matrix and the impedance matrix in real space.
	
	//nbodyfast->memory->free_host<FP_TYPE>(&_imp_matrix->imp_mat_data);
	//nbodyfast->memory->output_allocated_list_file();
	nbodyfast->memory->free_host<fftw_complex>(&_fft_in_f);
	nbodyfast->memory->free_host<FP_TYPE>(&_g_temp);
	//nbodyfast->memory->free_host<fftw_complex>(&_fft_out_f);
	fftw_destroy_plan(_plan1);  
	return 0;
}

//The following subroutines are all not implemented yet, Ben, 02/02/14
int NufftStaticVector :: projection()
{
#pragma omp parallel for
	for (int i = 0; i < total_num_grid_pts; i++)
	{
		for(unsigned int j = 0; j < FIELD_DIM; j++) u_src_grid[i+j*total_num_grid_pts] = 0.0f;
	}
#pragma omp parallel for
	for (int i = 0; i < total_num_boxes; i++)
	{
		for (int m = src_box_map[i]; m < src_box_map[total_num_boxes + i]; m++)
		{
			for (int j = 0; j < interp_nodes_per_box; j++)
			{
				for(unsigned int l = 0; l < FIELD_DIM; l++)
					u_src_grid[src_interp_idx[j + m * interp_nodes_per_box] + l*total_num_grid_pts] += 
						src_interp_coeff[j + m * interp_nodes_per_box] * field_static_vector->src_amp[m+l*problem_size];
			} // loop j
		} // loop m
	} // loop i

	return 0;
}
int NufftStaticVector :: fft_real_to_k()
{
#pragma omp parallel for
	for (int kk = 0; kk < fft_size[2]; kk++)
		for (int jj = 0; jj < fft_size[1]; jj++)
			for (int ii = 0; ii < fft_size[0]; ii++)
			{
				int _idx1 = ii + jj * fft_size[0] + kk * fft_size[0] * fft_size[1];
				int _idx2 = ii + jj * num_grid_pts[0] + kk * num_grid_pts[0] * num_grid_pts[1];

				for(unsigned int l = 0; l < FIELD_DIM; l++ )
				{	fft_in_f[_idx1+l*total_num_fft_pts][0] = 0.0f;	fft_in_f[_idx1+l*total_num_fft_pts][1] = 0.0f;	}

				if (ii < num_grid_pts[0] && jj < num_grid_pts[1] && kk < num_grid_pts[2])
				{
					for(unsigned int l = 0; l < FIELD_DIM; l++ )
						fft_in_f[_idx1+l*total_num_fft_pts][0] = u_src_grid[_idx2+l*total_num_grid_pts];
				}
			}
	for(unsigned int l = 0; l < FIELD_DIM; l++ )
		fftw_execute(plan_grid[l]);

	return 0;
}
int NufftStaticVector :: convolution_k()
{
	std::complex<FP_TYPE> *_impedance_matrix = g_grid->k_imp_mat_data;

#pragma omp parallel for
	for (int kk = 0; kk < fft_size[2]; kk++)
		for (int jj = 0; jj < fft_size[1]; jj++)
			for (int ii = 0; ii < fft_size[0]; ii++)
			{
				int _idx1 = ii + jj * fft_size[0] + kk * fft_size[0] * fft_size[1];
				fft_in_b[_idx1					  ][0] = 
						  fft_out_f[_idx1					 ][0] * _impedance_matrix[_idx1].real() 
						- fft_out_f[_idx1					 ][1] * _impedance_matrix[_idx1].imag()

						+ fft_out_f[_idx1+total_num_fft_pts  ][0] * _impedance_matrix[_idx1+total_num_fft_pts].real() 
						- fft_out_f[_idx1+total_num_fft_pts  ][1] * _impedance_matrix[_idx1+total_num_fft_pts].imag()

						+ fft_out_f[_idx1+2*total_num_fft_pts][0] * _impedance_matrix[_idx1+2*total_num_fft_pts].real() 
						- fft_out_f[_idx1+2*total_num_fft_pts][1] * _impedance_matrix[_idx1+2*total_num_fft_pts].imag();
				fft_in_b[_idx1					  ][1] = 
						  fft_out_f[_idx1					 ][0] * _impedance_matrix[_idx1].imag() 
						+ fft_out_f[_idx1					 ][1] * _impedance_matrix[_idx1].real()

						+ fft_out_f[_idx1+total_num_fft_pts  ][0] * _impedance_matrix[_idx1+total_num_fft_pts].imag() 
						+ fft_out_f[_idx1+total_num_fft_pts  ][1] * _impedance_matrix[_idx1+total_num_fft_pts].real()

						+ fft_out_f[_idx1+2*total_num_fft_pts][0] * _impedance_matrix[_idx1+2*total_num_fft_pts].imag() 
						+ fft_out_f[_idx1+2*total_num_fft_pts][1] * _impedance_matrix[_idx1+2*total_num_fft_pts].real();


				fft_in_b[_idx1+total_num_fft_pts  ][0] = 
						  fft_out_f[_idx1][0] * _impedance_matrix[_idx1+total_num_fft_pts].real() 
						- fft_out_f[_idx1][1] * _impedance_matrix[_idx1+total_num_fft_pts].imag()

						+ fft_out_f[_idx1+total_num_fft_pts][0] * _impedance_matrix[_idx1+3*total_num_fft_pts].real() 
						- fft_out_f[_idx1+total_num_fft_pts][1] * _impedance_matrix[_idx1+3*total_num_fft_pts].imag()

						+ fft_out_f[_idx1+2*total_num_fft_pts][0] * _impedance_matrix[_idx1+4*total_num_fft_pts].real() 
						- fft_out_f[_idx1+2*total_num_fft_pts][1] * _impedance_matrix[_idx1+4*total_num_fft_pts].imag();
				fft_in_b[_idx1+total_num_fft_pts][1] = 
						  fft_out_f[_idx1					 ][0] * _impedance_matrix[_idx1+total_num_fft_pts].imag() 
						+ fft_out_f[_idx1					 ][1] * _impedance_matrix[_idx1+total_num_fft_pts].real()

						+ fft_out_f[_idx1+total_num_fft_pts  ][0] * _impedance_matrix[_idx1+3*total_num_fft_pts].imag() 
						+ fft_out_f[_idx1+total_num_fft_pts  ][1] * _impedance_matrix[_idx1+3*total_num_fft_pts].real()

						+ fft_out_f[_idx1+2*total_num_fft_pts][0] * _impedance_matrix[_idx1+4*total_num_fft_pts].imag() 
						+ fft_out_f[_idx1+2*total_num_fft_pts][1] * _impedance_matrix[_idx1+4*total_num_fft_pts].real();


				fft_in_b[_idx1+2*total_num_fft_pts][0] = 
						  fft_out_f[_idx1					][0] * _impedance_matrix[_idx1+2*total_num_fft_pts].real() 
						- fft_out_f[_idx1					][1] * _impedance_matrix[_idx1+2*total_num_fft_pts].imag()

						+ fft_out_f[_idx1+total_num_fft_pts	][0] * _impedance_matrix[_idx1+4*total_num_fft_pts].real() 
						- fft_out_f[_idx1+total_num_fft_pts	][1] * _impedance_matrix[_idx1+4*total_num_fft_pts].imag()

						+ fft_out_f[_idx1+2*total_num_fft_pts][0] * _impedance_matrix[_idx1+5*total_num_fft_pts].real() 
						- fft_out_f[_idx1+2*total_num_fft_pts][1] * _impedance_matrix[_idx1+5*total_num_fft_pts].imag();
				fft_in_b[_idx1+2*total_num_fft_pts][1] = 
						  fft_out_f[_idx1					 ][0] * _impedance_matrix[_idx1+2*total_num_fft_pts].imag() 
						+ fft_out_f[_idx1					 ][1] * _impedance_matrix[_idx1+2*total_num_fft_pts].real()

						+ fft_out_f[_idx1+total_num_fft_pts  ][0] * _impedance_matrix[_idx1+4*total_num_fft_pts].imag() 
						+ fft_out_f[_idx1+total_num_fft_pts  ][1] * _impedance_matrix[_idx1+4*total_num_fft_pts].real()

						+ fft_out_f[_idx1+2*total_num_fft_pts][0] * _impedance_matrix[_idx1+5*total_num_fft_pts].imag() 
						+ fft_out_f[_idx1+2*total_num_fft_pts][1] * _impedance_matrix[_idx1+5*total_num_fft_pts].real();

			}
	return 0;
}
int NufftStaticVector :: fft_k_to_real()
{
	for(unsigned int l = 0; l < FIELD_DIM; l++ )
		fftw_execute(plan_grid[3+l]);
#pragma omp parallel for
	for (int kk = 0; kk < num_grid_pts[2]; kk++)
		for (int jj = 0; jj < num_grid_pts[1]; jj++)
			for (int ii = 0; ii < num_grid_pts[0]; ii++)
			{
				int _idx1 = ii + jj * fft_size[0] + kk * fft_size[0] * fft_size[1];
				int _idx2 = ii + jj * num_grid_pts[0] + kk * num_grid_pts[0] * num_grid_pts[1];
				for(unsigned int l = 0; l < FIELD_DIM; l++ )
					u_obs_grid[_idx2+l*total_num_grid_pts] = fft_out_b[_idx1+l*total_num_fft_pts][0] / FP_TYPE(total_num_fft_pts);
			}
	return 0;
}
//need to to be implemented
int NufftStaticVector :: correction_interpolation()//!!!!!!!!!!!!!NEAR CHANGE!!!!!!!!!!!!!
{
	int num_near_box_per_dim = 2*near_correct_layer+1;
	int total_num_near_box_per_box = num_near_box_per_dim*num_near_box_per_dim*num_near_box_per_dim;
	std::complex<FP_TYPE> *_impedance_matrix_near = g_grid_near->k_imp_mat_data;

#pragma omp parallel for num_threads( omp_get_num_procs() )
	for (int i = 0; i < total_num_boxes; i++)
	{

		for (int j = 0; j < total_num_near_grid_pts*FIELD_DIM; j++)
		{
				u_near_src_grid[j] = 0.0f;
		}

		int _box_idx_dim[3];
		_box_idx_dim[2] = i / (num_boxes[0] * num_boxes[1]);
		_box_idx_dim[1] = (i - _box_idx_dim[2] * (num_boxes[0] * num_boxes[1])) / num_boxes[0];
		_box_idx_dim[0] = i % num_boxes[0];

		int _current_near_idx = 0;
//projection
		for (int j = 0; j < near_box_list[i * (total_num_near_box_per_box+1)]; j++)
		{
			int _current_near_box_idx = near_box_list[i * (total_num_near_box_per_box+1) + j + 1];
			
			for (int m = src_box_map[_current_near_box_idx]; m < src_box_map[total_num_boxes + _current_near_box_idx]; m++, _current_near_idx++)
			{
				for (int n = 0; n < interp_nodes_per_box; n++)
				{
					int _node_idx, _node_idx_dim[3];

					_node_idx = src_interp_idx[n + m * interp_nodes_per_box];

					_node_idx_dim[2] = _node_idx / (num_grid_pts[1] * num_grid_pts[0]);
					_node_idx_dim[1] = (_node_idx % (num_grid_pts[1] * num_grid_pts[0])) / num_grid_pts[0];
					_node_idx_dim[0] = _node_idx % num_grid_pts[0];

					for (int k = 0; k < 3; k++)
					{
						_node_idx_dim[k] -= (_box_idx_dim[k] - near_correct_layer) * (num_nodes[k] - 1);
					}
					_node_idx = _node_idx_dim[0] + _node_idx_dim[1] * near_num_grid_pts[0] + _node_idx_dim[2] * near_num_grid_pts[0] * near_num_grid_pts[1];
					for(unsigned int l = 0; l < FIELD_DIM; l++)
						u_near_src_grid[_node_idx+l*total_num_near_grid_pts] += src_interp_coeff[n + m * interp_nodes_per_box] * field_static_vector->src_amp[m+l*problem_size];
			
				} // loop n

			} // loop m
		} // loop j

//preparation for forward fft
		for (int kk = 0; kk < 2 * near_num_grid_pts[2]; kk++)
			for (int jj = 0; jj < 2 * near_num_grid_pts[1]; jj++)
				for (int ii = 0; ii < 2 * near_num_grid_pts[0]; ii++)
				{
					int _idx1 = ii + jj * 2 * near_num_grid_pts[0] + kk * 2 * near_num_grid_pts[0] * 2 * near_num_grid_pts[1];
					int _idx2 = ii + jj * near_num_grid_pts[0] + kk * near_num_grid_pts[0] * near_num_grid_pts[1];

					for(unsigned int l = 0; l < FIELD_DIM; l++){
						fft_in_f_near[_idx1+l*total_num_fft_pts_near][0] = 0.0f;
						fft_in_f_near[_idx1+l*total_num_fft_pts_near][1] = 0.0f;
					
						if (ii < near_num_grid_pts[0] && jj < near_num_grid_pts[1] && kk < near_num_grid_pts[2])
						{
							fft_in_f_near[_idx1+l*total_num_fft_pts_near][0] = u_near_src_grid[_idx2+l*total_num_near_grid_pts];
						}
					}
				}
//forward fft
		for(unsigned int l = 0; l < FIELD_DIM; l++)
			fftw_execute(plan_grid_near[l]);

//k-space convolution
		for (int kk = 0; kk < 2 * near_num_grid_pts[2]; kk++)
			for (int jj = 0; jj < 2 * near_num_grid_pts[1]; jj++)
				for (int ii = 0; ii < 2 * near_num_grid_pts[0]; ii++)
				{
					int _idx1 = ii + jj * 2 * near_num_grid_pts[2] + kk * 2 * near_num_grid_pts[2] * 2 * near_num_grid_pts[1];
					//real0
					fft_in_b_near[_idx1][0] = 
						  fft_out_f_near[_idx1							  ][0] * _impedance_matrix_near[_idx1].real() 
						- fft_out_f_near[_idx1							  ][1] * _impedance_matrix_near[_idx1].imag()
							
						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][0] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1+total_num_fft_pts_near  ][1] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].imag()
							
						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][0] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1+2*total_num_fft_pts_near][1] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].imag();
					
					//img0
					fft_in_b_near[_idx1][1] = 
						  fft_out_f_near[_idx1							  ][0] * _impedance_matrix_near[_idx1].imag() 
						+ fft_out_f_near[_idx1							  ][1] * _impedance_matrix_near[_idx1].real()

						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][0] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][1] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].real()

						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][0] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][1] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].real();

					//real1
					fft_in_b_near[_idx1+total_num_fft_pts_near][0] = 
						  fft_out_f_near[_idx1							  ][0] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1							  ][1] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].imag()
							
						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][0] * _impedance_matrix_near[_idx1+3*total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1+total_num_fft_pts_near  ][1] * _impedance_matrix_near[_idx1+3*total_num_fft_pts_near].imag()
							
						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][0] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1+2*total_num_fft_pts_near][1] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].imag();
					
					//img1
					fft_in_b_near[_idx1+total_num_fft_pts_near][1] = 
						  fft_out_f_near[_idx1							  ][0] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1							  ][1] * _impedance_matrix_near[_idx1+total_num_fft_pts_near].real()

						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][0] * _impedance_matrix_near[_idx1+3*total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][1] * _impedance_matrix_near[_idx1+3*total_num_fft_pts_near].real()

						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][0] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][1] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].real();

					//real2
					fft_in_b_near[_idx1+2*total_num_fft_pts_near][0] = 
						  fft_out_f_near[_idx1							  ][0] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1							  ][1] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].imag()
							
						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][0] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1+total_num_fft_pts_near  ][1] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].imag()
							
						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][0] * _impedance_matrix_near[_idx1+5*total_num_fft_pts_near].real() 
						- fft_out_f_near[_idx1+2*total_num_fft_pts_near][1] * _impedance_matrix_near[_idx1+5*total_num_fft_pts_near].imag();
					
					//img2
					fft_in_b_near[_idx1+2*total_num_fft_pts_near][1] = 
						  fft_out_f_near[_idx1							  ][0] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1							  ][1] * _impedance_matrix_near[_idx1+2*total_num_fft_pts_near].real()

						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][0] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1+total_num_fft_pts_near  ][1] * _impedance_matrix_near[_idx1+4*total_num_fft_pts_near].real()

						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][0] * _impedance_matrix_near[_idx1+5*total_num_fft_pts_near].imag() 
						+ fft_out_f_near[_idx1+2*total_num_fft_pts_near][1] * _impedance_matrix_near[_idx1+5*total_num_fft_pts_near].real();
				}
//backward fft
		for(unsigned int l = 0; l < FIELD_DIM; l++)
			fftw_execute(plan_grid_near[FIELD_DIM+l]);
//post fft
		for (int kk = 0; kk < num_nodes[2]; kk++)
			for (int jj = 0; jj < num_nodes[1]; jj++)
				for (int ii = 0; ii < num_nodes[0]; ii++)
				{
					int _idx1 = ii + jj * num_nodes[0] + kk * num_nodes[0] * num_nodes[1];

					int _idx2 = near_correct_layer * (num_nodes[0] - 1) + ii + (near_correct_layer * (num_nodes[1] - 1) + jj) * 2 * near_num_grid_pts[0] + (near_correct_layer * (num_nodes[2] - 1) + kk) * 2 * near_num_grid_pts[0] * 2 * near_num_grid_pts[1];
					for(unsigned int l = 0; l < FIELD_DIM; l++)
						u_near_obs_grid[_idx1+l*interp_nodes_per_box] = fft_out_b_near[_idx2+l*total_num_fft_pts_near][0] / 8.0f / total_num_near_grid_pts;
				}
//subtraction and interpolation
		for (int m = obs_box_map[i]; m < obs_box_map[total_num_boxes + i]; m++)
		{
			for (int j = 0; j < interp_nodes_per_box; j++)
			{
				int _idx1 = obs_interp_idx[j + m * interp_nodes_per_box];

				int _node_idx_dim[3];
				
				_node_idx_dim[2] = _idx1 / (num_grid_pts[1] * num_grid_pts[0]);
				_node_idx_dim[1] = (_idx1 % (num_grid_pts[1] * num_grid_pts[0])) / num_grid_pts[0];
				_node_idx_dim[0] = _idx1 % num_grid_pts[0];

				int _idx2 = (_node_idx_dim[0] - _box_idx_dim[0] * (num_nodes[0] - 1)) +  (_node_idx_dim[1] - _box_idx_dim[1] * (num_nodes[1] - 1)) * num_nodes[0] + (_node_idx_dim[2] - _box_idx_dim[2] * (num_nodes[2] - 1)) * num_nodes[0] * num_nodes[1];	
				for(unsigned int l = 0; l < FIELD_DIM; l++){
					field_static_vector->field_amp[m+l*problem_size] += 
						( u_obs_grid[_idx1+l*total_num_grid_pts]- u_near_obs_grid[_idx2+l*interp_nodes_per_box]) 
						* obs_interp_coeff[j + m * interp_nodes_per_box];//0.f
				}
			}
		}

	} // loop i


	// Doing direct calculation
	unsigned int total_g_direct_near_size = problem_size * max_near_src_cnt;
#pragma omp parallel for num_threads( omp_get_num_procs() )
	for (int i = 0; i < total_num_boxes; i++)
	{
		for (int l = obs_box_map[i]; l < obs_box_map[total_num_boxes + i]; l++)
		{
			int _current_near_idx = 0;

			for (int j = 0; j < near_box_list[i * (total_num_near_box_per_box+1)]; j++)
			{
				int _current_near_box_idx = near_box_list[i * (total_num_near_box_per_box+1) + j + 1];
			
				for (int m = src_box_map[_current_near_box_idx]; m < src_box_map[total_num_boxes + _current_near_box_idx]; m++, _current_near_idx++)
				{
					if (cpu_near_onfly == false)
					{
						//not_implemented yet...
						//1*problem_size*max_near_src_cnt
						FP_TYPE *G_temp = g_direct_near + l *max_near_src_cnt + _current_near_idx;
						
						field_static_vector->field_amp[l]				+= G_temp[0] * field_static_vector->src_amp[m] 
							+ G_temp[1*total_g_direct_near_size] * field_static_vector->src_amp[m+problem_size]
							+ G_temp[2*total_g_direct_near_size] * field_static_vector->src_amp[m+2*problem_size];
						field_static_vector->field_amp[l+problem_size]	+= G_temp[total_g_direct_near_size] * field_static_vector->src_amp[m] 
							+ G_temp[3*total_g_direct_near_size] * field_static_vector->src_amp[m+problem_size]
							+ G_temp[4*total_g_direct_near_size] * field_static_vector->src_amp[m+2*problem_size];
						field_static_vector->field_amp[l+2*problem_size]+= G_temp[2*total_g_direct_near_size] * field_static_vector->src_amp[m] 
							+ G_temp[4*total_g_direct_near_size] * field_static_vector->src_amp[m+problem_size]
							+ G_temp[5*total_g_direct_near_size] * field_static_vector->src_amp[m+2*problem_size];
					}	

					if (cpu_near_onfly == true)
					{	

						/* This is another way of implementing onfly computation, but it seems to be slower
						FP_TYPE _temp_G[FIELD_DIM*(FIELD_DIM+1)/2]; 
						FP_TYPE _temp_R[3];

						for (int k = 0; k < 3; k++)
						{
							_temp_R[k] = field_static_vector->obs_coord[l + problem_size * k] - field_static_vector->src_coord[m + problem_size * k];
						}
						FieldStaticVector::get_G(_temp_G, _temp_R, Field::epsilon());
						field_static_vector->field_amp[l] += _temp_G[0] * field_static_vector->src_amp[m] 
							+ _temp_G[1] * field_static_vector->src_amp[m+problem_size]
							+ _temp_G[2] * field_static_vector->src_amp[m+2*problem_size];
						field_static_vector->field_amp[l+problem_size] += _temp_G[1] * field_static_vector->src_amp[m] 
							+ _temp_G[3] * field_static_vector->src_amp[m+problem_size]
							+ _temp_G[4] * field_static_vector->src_amp[m+2*problem_size];
						field_static_vector->field_amp[l+2*problem_size] += _temp_G[2] * field_static_vector->src_amp[m] 
							+ _temp_G[4] * field_static_vector->src_amp[m+problem_size]
							+ _temp_G[5] * field_static_vector->src_amp[m+2*problem_size];*/
						
						FP_TYPE _temp_R[3];
						for (int k = 0; k < 3; k++)
							_temp_R[k] = field_static_vector->obs_coord[l + problem_size * k] - field_static_vector->src_coord[m + problem_size * k];

						FP_TYPE mdotr = FP_TYPE(0.f);
						for(unsigned int p = 0; p < FIELD_DIM; p++)	mdotr += _temp_R[p]*field_static_vector->src_amp[m+p*problem_size];
						FP_TYPE r_amp2 = _temp_R[0] * _temp_R[0] + _temp_R[1] * _temp_R[1] + _temp_R[2] * _temp_R[2];

						if (r_amp2 > Field::epsilon())	
						{
							FP_TYPE inv_r_amp1 = 1.f/sqrtf(r_amp2);//  rsqrtf(r_amp2);
							FP_TYPE inv_r_amp3 = inv_r_amp1 * inv_r_amp1 * inv_r_amp1;
							FP_TYPE inv_r_amp5 = inv_r_amp3 * inv_r_amp1 * inv_r_amp1;
							for(unsigned int p =0; p< FIELD_DIM; p++){	
								field_static_vector->field_amp[l+p*problem_size] += 
									3 * mdotr * _temp_R[p] * inv_r_amp5 - field_static_vector->src_amp[m+p*problem_size] * inv_r_amp3;
							}
						}

					} // outputFlags.nearFlag
				} // m
			} // l
		} // j
	} // i



	return 0;
}


}
