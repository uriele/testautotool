/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_static_vector.h: class declaration of Class FieldStaticVector
*/
#ifndef _FIELD_STATIC_VECTOR
#define _FIELD_STATIC_VECTOR
#include "field.h"
#include <cmath>
namespace NBODYFAST_NS{
class FieldStaticVector : virtual public Field
{
public:
	FieldStaticVector(){}
	FieldStaticVector(class NBODYFAST *n_ptr);
	virtual ~FieldStaticVector();

	// source and field amplitudes
	FP_TYPE *src_amp;
	FP_TYPE *field_amp;

	// source and field amplitudes for multiGPU
	FP_TYPE **src_amp_dev;
	FP_TYPE **field_amp_dev;

	//NEED TO BE IMPLEMENTED!!!!!!!!!!!!!!!!!!
	virtual int amp_field_alloc(); 	// allocating src and field amp.
	virtual int array_alloc_multi_interface(); 	// allocating src and field amp. for multiGPU

	virtual int set_src_amp(double *);
//	virtual int execute(); // don't need to implement its own execute(), its parents' execute() is good enough
	virtual int set_fld_amp(double *);

	// Green's function of this type of field
	// precision can be improved by OOMMF's method
	static inline int get_G(FP_TYPE* G, FP_TYPE r[3], FP_TYPE epsilon)
	{
		for (int l = 0; l < 6; l++) G[l] = 0.0f;
		FP_TYPE r_amp2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		FP_TYPE r_amp1 = sqrtf(r_amp2);
		if (r_amp2 > 1e-7f)
		{
			FP_TYPE inv_r_amp2 = 1.0f / r_amp2;
			FP_TYPE inv_r_amp5 = inv_r_amp2 * inv_r_amp2 * inv_r_amp2 * r_amp1;

			G[0] = (3.0f * r[0] * r[0] - r_amp2) * inv_r_amp5; // tensor component: Gxx
			G[1] = (3.0f * r[0] * r[1]) * inv_r_amp5; // tensor component: Gxy
			G[2] = (3.0f * r[0] * r[2]) * inv_r_amp5; // tensor component: Gxz

			//G[3] = (3.0f * r[1] * r[0]) * inv_r_amp5; // tensor component: Gyz
			G[3] = (3.0f * r[1] * r[1] - r_amp2) * inv_r_amp5; // tensor component: Gyy
			G[4] = (3.0f * r[1] * r[2]) * inv_r_amp5; // tensor component: Gyz

			//G[6] = (3.0f * r[2] * r[0]) * inv_r_amp5; // tensor component: Gzx
			//G[7] = (3.0f * r[2] * r[1]) * inv_r_amp5; // tensor component: Gzy
			G[5] = (3.0f * r[2] * r[2] - r_amp2) * inv_r_amp5; // tensor component: Gzz
		}
		return 0;
	}
private:
	int amp_field_alloc_single(); // not in use
	int amp_field_alloc_multi();

};
}

#endif
