/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field_gpu.h: class declaration of Class FieldGpu
*/
#ifndef _FIELD_GPU
#define _FIELD_GPU
#include "field.h"
namespace NBODYFAST_NS{
class FieldGpu : virtual public Field
{
public:
	FieldGpu(){}
	FieldGpu(class NBODYFAST *n_ptr);
	virtual ~FieldGpu();
	
	// device copies of source and observer coordinates
	FP_TYPE **d_src_coord;
	FP_TYPE **d_obs_coord;
	
	virtual int coord_alloc(double *, double *); // allocate coordinates
	virtual int array_alloc_multi_interface(); // allocate coordinate related arrays related to multiGPU calculation
	
private:
	
	int coord_alloc_single(); // dummy function..
	int coord_alloc_multi(); // receivce calls from array_alloc_multi_interface()
};
}
#endif