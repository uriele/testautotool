/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
*nufft_static_vector_gpu.h: class declaration of Class NufftStaticVectorGpu, Class NufftArrayGpuStaticVector
*Class NufftParamGpu and Class NufftArrayGpu should later be put inside NufftGpu
*/
#ifndef _NUFFT_STATIC_VECTOR_GPU
#define _NUFFT_STATIC_VECTOR_GPU
#include <complex>
#include "fftw3.h"
#include "nufft_gpu.h"
#include "nufft_static_vector.h"

#include "cuComplex.h"

namespace NBODYFAST_NS{
class NufftArrayGpuStaticVector : public NufftArrayGpu
{
public:
	NufftArrayGpuStaticVector()
	{
		d_u_src_grid = NULL;
		d_k_u_src_grid = NULL;

		d_u_obs_grid = NULL;
		d_k_u_obs_grid = NULL;

		d_u_near_src_grid = NULL;
		d_u_near_obs_grid = NULL;

		d_k_u_near_src_grid = NULL;
		//d_k_imp_mat_data = NULL;//IMP MAT CHANGE!!!!!!!!!!!!!!!
		d_k_imp_mat_data_gpu = NULL;
	}
	virtual ~NufftArrayGpuStaticVector()
	{
	}

	
	virtual void set_value(class NufftStaticVectorGpu *nufft_ptr);
	virtual void set_value_base(class NufftArrayGpu *nufft_array_gpu);
	
	FP_TYPE *d_u_src_grid; 
	FP_TYPE *d_u_obs_grid;
	FP_TYPE *d_u_src_grid_dev; // for multiGPU calculation
	FP_TYPE *d_u_obs_grid_dev; // for multiGPU calculation


	//CUFFT_COMPLEX_TYPE *d_k_imp_mat_data;
	FP_TYPE *d_k_imp_mat_data_gpu;

	// Copies of data with the same name in the Field class
	FP_TYPE *d_src_amp;
	FP_TYPE *d_field_amp;
	
	// Not in use for now
	FP_TYPE *d_k_u_src_grid;
	FP_TYPE *d_k_u_obs_grid;
	FP_TYPE *d_u_near_src_grid;
	FP_TYPE *d_u_near_obs_grid;
	CUFFT_COMPLEX_TYPE *d_k_u_near_src_grid;


};
// Class NufftStaticVectorGpu 
class NufftStaticVectorGpu : public NufftStaticVector, public NufftGpu
{
public:
	NufftStaticVectorGpu() : NufftStaticVector(){}
	NufftStaticVectorGpu(class NBODYFAST *n_ptr);
	~NufftStaticVectorGpu();

	virtual int preprocessing();

	virtual int projection();
	virtual int fft_real_to_k();
	virtual int convolution_k();
	virtual int fft_k_to_real();
	virtual int correction_interpolation();

	//class NBODYFAST* _nbodyfast_test_ptr;

	class FieldStaticVectorGpu* field_static_vector_gpu;
	class NufftArrayGpuStaticVector **nufft_array_gpu_static_vector;
	class NufftArrayGpuStaticVector **d_nufft_array_gpu_static_vector;

	CUFFT_COMPLEX_TYPE *imp_matrix_temp;
private:

	int preprocessing_single();
	int preprocessing_multi();

	int projection_single();
	int projection_multi();

	int fft_real_to_k_single();
	int fft_real_to_k_multi();

	int convolution_k_single();
	int convolution_k_multi();

	int fft_k_to_real_single();
	int fft_k_to_real_multi();

	int correction_interpolation_single();
	int correction_interpolation_multi();

	//inline int copy
	



};
}
#endif