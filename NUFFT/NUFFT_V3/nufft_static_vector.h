/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nufft_static_vector.h: class decalaration for Class NufftStaticVector
*/
#ifndef _NUFFT_STATIC_VECTOR
#define _NUFFT_STATIC_VECTOR
#include "fftw3.h"
#include "nufft.h"
#include <complex>
namespace NBODYFAST_NS{
class NufftStaticVector : virtual public Nufft
{
public:
	NufftStaticVector() : Nufft(){}
	NufftStaticVector(class NBODYFAST *n_ptr);
	~NufftStaticVector();

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// implementation of all the pure virtual functions decalared in the base class
	virtual int preprocessing();

	virtual int g_mat_alloc(class ImpMatrix **g_mat, int n_grid_ptr[3]);
	virtual int g_mat_set(class ImpMatrix *g_mat, FP_TYPE *src_grid_coord, FP_TYPE *obs_grid_coord, int n_grid_ptr[3], int fft_size[3]);

	virtual int projection();
	virtual int fft_real_to_k();
	virtual int convolution_k();
	virtual int fft_k_to_real();
	virtual int correction_interpolation();
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class FieldStaticVector* field_static_vector;	// a pointer to the corresponding field object
protected:
	FP_TYPE *u_src_grid; // source amplitudes on the FFT grid, projected from sources
	FP_TYPE *k_u_src_grid; // amplitudes in k-space on the FFT grid after FFT transformation
	FP_TYPE *u_obs_grid; // field amplitudes on the FFT grid, ready to be interpolated to observers
	FP_TYPE *k_u_obs_grid; // not in use, write here just for symmetry *_* ...

	FP_TYPE *u_near_src_grid; // source amplitudes on the NEAR FFT grid, used only by CPU NUFFT
	FP_TYPE *u_near_obs_grid; // field amplitudes on the NEAR FFT grid, used only by CPU NUFFT
	std::complex<FP_TYPE> *k_u_near_src_grid; // not in use...but it seems it is allocated somewhere in the program..., used only by CPU NUFFT
	FP_TYPE *g_direct_near; // impedance matrix between near sources and observers

	class ImpMatrixStaticVector *g_grid; 	// stupid near impedance matrix
	class ImpMatrixStaticVector *g_grid_near; // stupid near impedance matrix...used only by CPU NUFFT
};
}
#endif