/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_static_scalar.cpp: class definition of Class DirectStaticScalar
*/
#include "nbodyfast.h"
#include "direct_static_scalar.h"
#include "field_static_scalar.h"
namespace NBODYFAST_NS{
DirectStaticScalar :: DirectStaticScalar(class NBODYFAST *n_ptr) : Direct(n_ptr)
{
	// if DirectStaticScalar object is created, we are sure the field being processed in of type FieldStaticScalar
	field_static_scalar = dynamic_cast<FieldStaticScalar*>(n_ptr->field);
}
DirectStaticScalar :: ~DirectStaticScalar()
{
}
int DirectStaticScalar :: execution()
{
	Direct::execution(); // call its base common execution method
	
	// Straightforward direct superposition
	FP_TYPE _r[3], _temp_G[1]; 
#pragma omp parallel for num_threads(omp_get_num_procs())
	for (int i = 0; i < test_size; i++)
		for (int j = 0; j < problem_size; j++)
		{
			for (int k = 0; k < 3; k++) 
			{
				_r[k] = field_static_scalar->obs_coord[i + problem_size * k] - field_static_scalar->src_coord[j + problem_size * k];
			}
			FieldStaticScalar::get_G(_temp_G, _r, Field::epsilon());
			field_static_scalar->field_amp[i] += _temp_G[0] * field_static_scalar->src_amp[j];
		}

	return 0;
}
}