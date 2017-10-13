/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_static_vector_gpu.h: class declaration of Class FieldStaticVectorGpu
*/
#ifndef _FIELD_STATIC_VECTOR_GPU
#define _FIELD_STATIC_VECTOR_GPU
#include "field_static_vector.h"
#include "field_gpu.h"
#include <cmath>
namespace NBODYFAST_NS{
class FieldStaticVectorGpu : public FieldStaticVector, public FieldGpu
{
public:
	FieldStaticVectorGpu(){}
	FieldStaticVectorGpu(class NBODYFAST *n_ptr);
	virtual ~FieldStaticVectorGpu();

	// device copies of source and field amplitudes
	FP_TYPE **d_src_amp;
	FP_TYPE **d_field_amp;
	
	// allocation of amplitudes and fields
	virtual int amp_field_alloc();

	virtual int array_alloc_multi_interface();
	
	virtual int set_src_amp(double *); 	// transferring source amplitudes before GPU calculation
	virtual int set_fld_amp(double *); // transferring field amplitudes after GPU calculation
	
	//// Green's function for static scalar field
	//inline static int get_G(FP_TYPE G[1], FP_TYPE r[3], FP_TYPE epsilon)
	//{
	//	FP_TYPE r_amp2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
	//	FP_TYPE r_amp1 = sqrtf(r_amp2);

	//	G[0] = 0.0f;

	//	if (r_amp2 > epsilon)
	//	{
	//		FP_TYPE inv_r_amp1 = 1.0f / r_amp1;
	//		G[0] = inv_r_amp1; 
	//	}
	//	return 0;
	//}
private:
	int amp_field_alloc_single();
	int amp_field_alloc_multi();

	int set_src_amp_single(double *_charge);
	int set_src_amp_multi(double *_charge);

	int set_fld_amp_single(double *_field);
	int set_fld_amp_multi(double *_field);

	int execute_single();
	int execute_multi();
};
}

#endif