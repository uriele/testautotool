/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* imp_matrix_static_matrix.h: class declaration of Class ImpMatrix
* data structure to store impedance matrix for NUFFT
* this is dumb stuff...please change it in the future, at least put it inside the class NUFFT
*/
#ifndef _IMP_MATRIX_STATIC_SCALAR
#define _IMP_MATRIX_STATIC_SCALAR
#include <complex>
#include "imp_matrix.h"
#include "memory.h"
#include "nufft.h"
#include "nbodyfast.h"
namespace NBODYFAST_NS{
class ImpMatrixStaticScalar : public ImpMatrix
{
public:
	ImpMatrixStaticScalar()
	{
		imp_mat_data = NULL;
		k_imp_mat_data = NULL;
		k_imp_mat_data_gpu = NULL;
	}
	ImpMatrixStaticScalar(class NBODYFAST *n_ptr) : ImpMatrix(n_ptr)
	{
		imp_mat_data = NULL;
		k_imp_mat_data = NULL;
		k_imp_mat_data_gpu = NULL;
	}
	virtual ~ImpMatrixStaticScalar()
	{
		nbodyfast->memory->free_host<FP_TYPE>(&imp_mat_data);
		nbodyfast->memory->free_host<std::complex<FP_TYPE> >(&k_imp_mat_data);
		nbodyfast->memory->free_host<FP_TYPE>(&k_imp_mat_data_gpu);
	}

	FP_TYPE *imp_mat_data; // impedance matrix in real space
	std::complex<FP_TYPE> *k_imp_mat_data; // impedance matrix in the k-space
	FP_TYPE *k_imp_mat_data_gpu; // impedance matrix in k-space, for device memory saving
	
	int imp_mat_size[3]; // size of impedance matrix
	int _padded_green_dim[3];
private:

};
}
#endif
