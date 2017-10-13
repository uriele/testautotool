/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* imp_matrix.h: class declaration of Class ImpMatrix
* data structure to store impedance matrix for NUFFT
* this is dumb stuff...please change it in the future, at least put it inside the class NUFFT
*/
#ifndef _IMP_MATRIX
#define _IMP_MATRIX
#include <complex>
#include "fp_precision.h"
namespace NBODYFAST_NS{
class ImpMatrix
{
public:
	ImpMatrix()
	{
		nbodyfast = NULL;
	}
	ImpMatrix(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
	}
	void set_parent(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
	}
	virtual ~ImpMatrix(){}

protected:
	class NBODYFAST* nbodyfast;

};

}
#endif