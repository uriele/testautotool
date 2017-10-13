/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_static_scalar.h: class declaration of Class FieldStaticScalar
*/
#ifndef _FIELD_STATIC_SCALAR
#define _FIELD_STATIC_SCALAR
#include "field.h"
#include <cmath>
namespace NBODYFAST_NS{
class FieldStaticScalar : virtual public Field
{
public:
	FieldStaticScalar(){}
	FieldStaticScalar(class NBODYFAST *n_ptr);
	virtual ~FieldStaticScalar();

	// source and field amplitudes
	FP_TYPE *src_amp;
	FP_TYPE *field_amp;

	// source and field amplitudes for multiGPU
	FP_TYPE **src_amp_dev;
	FP_TYPE **field_amp_dev;

	virtual int amp_field_alloc(); 	// allocating src and field amp.
	virtual int array_alloc_multi_interface(); 	// allocating src and field amp. for multiGPU

	virtual int set_src_amp(double *);
//	virtual int execute(); // don't need to implement its own execute(), its parents' execute() is good enough
	virtual int set_fld_amp(double *);

	// Green's function of this type of field
	static inline int get_G(FP_TYPE G[1], FP_TYPE r[3], FP_TYPE epsilon)
	{
		FP_TYPE r_amp2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		FP_TYPE r_amp1 = sqrtf(r_amp2);

		G[0] = 0.0f;

		if (r_amp2 > epsilon)
		{
			FP_TYPE inv_r_amp1 = 1.0f / r_amp1;
			G[0] = inv_r_amp1; 
		}
		return 0;
	}
private:
	int amp_field_alloc_single(); // not in use
	int amp_field_alloc_multi();

};
}

#endif