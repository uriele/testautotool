/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* field.h: class declaration of Class Field
*/
#ifndef _FIELD
#define _FIELD
#include <iostream>
#include "fp_precision.h"
namespace NBODYFAST_NS{
class Field
{
public:
	Field(){}
	Field(class NBODYFAST *n_ptr);
	virtual ~Field() = 0;
	
	static FP_TYPE max_domain_size; // max domain size along a certain dimension

	bool src_obs_overlap; // if the source and observer overlaps each other

	FP_TYPE *src_coord; // source coordinates
	FP_TYPE *obs_coord; // observer coordinates

	FP_TYPE **src_coord_dev; // source coordinates for multi GPU 
	FP_TYPE **obs_coord_dev; // souce coordinates for multi GPU
	
	static inline FP_TYPE epsilon() // source/observer overlap tolerance
	{
		return 1e-7f * max_domain_size;
	}
	
	// All these virtual functions are implemented in derived classes
	virtual int coord_alloc(double *, double *);
	virtual int amp_field_alloc() = 0;
	virtual int set_src_amp(double *) = 0;
	virtual int execute();
	virtual int set_fld_amp(double *) = 0;

	virtual int array_alloc_multi_interface(); // not in use, a dummy entry for derived class to implement its own method


	class NBODYFAST *nbodyfast;
protected:
private:
	int coord_alloc_single(); //  not in use
	int coord_alloc_multi(); // allocating necessary arrays for multi GPU calcuation

};
}
#endif