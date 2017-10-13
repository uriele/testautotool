/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_static_vector.cpp: class definition of Class DirectStaticVector
*/
#include "nbodyfast.h"
#include "direct_static_vector.h"
#include "field_static_vector.h"
namespace NBODYFAST_NS{
DirectStaticVector :: DirectStaticVector(class NBODYFAST *n_ptr) : Direct(n_ptr)
{
	// if DirectStaticVector object is created, we are sure the field being processed in of type FieldStaticVector
	field_static_vector = dynamic_cast<FieldStaticVector*>(n_ptr->field);
}
DirectStaticVector :: ~DirectStaticVector()
{
}
int DirectStaticVector :: execution()
{
	Direct::execution(); // call its base common execution method
	
	// Straightforward direct superposition
	FP_TYPE _r[3], _temp_G[FIELD_DIM*(FIELD_DIM+1)/2]; 
#pragma omp parallel for num_threads(omp_get_num_procs())
	for (int i = 0; i < test_size; i++)
		for (int j = 0; j < problem_size; j++)
		{
			for (int k = 0; k < 3; k++) 
			{
				_r[k] = field_static_vector->obs_coord[i + problem_size * k] - field_static_vector->src_coord[j + problem_size * k];
			}
			FieldStaticVector::get_G(_temp_G, _r, Field::epsilon());
			field_static_vector->field_amp[i] += _temp_G[0] * field_static_vector->src_amp[j] 
				+ _temp_G[1] * field_static_vector->src_amp[j+problem_size]
				+ _temp_G[2] * field_static_vector->src_amp[j+2*problem_size];
			field_static_vector->field_amp[i+problem_size] += _temp_G[1] * field_static_vector->src_amp[j] 
				+ _temp_G[3] * field_static_vector->src_amp[j+problem_size]
				+ _temp_G[4] * field_static_vector->src_amp[j+2*problem_size];
			field_static_vector->field_amp[i+2*problem_size] += _temp_G[2] * field_static_vector->src_amp[j] 
				+ _temp_G[4] * field_static_vector->src_amp[j+problem_size]
				+ _temp_G[5] * field_static_vector->src_amp[j+2*problem_size];
		}

	return 0;
}
}