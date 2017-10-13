/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* domain.h: class declaration of Class Domain
*/
#ifndef _CAL_DOMAIN
#define _CAL_DOMAIN
#include "fp_precision.h"
namespace NBODYFAST_NS{
class Domain
{
public:
	Domain(){}
	Domain(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
	}
	virtual ~Domain(){}
	
	// domain setup method
	int setup();

	// stores the domain size information
	// src_domain_range[0~2] stores the lower limit
	// src_domain_range[3~5] stores the higher limit
	// src_domain_range[6~8] stores the domain size (higher - lower)
	FP_TYPE src_domain_range[9]; 
	FP_TYPE obs_domain_range[9];
	FP_TYPE src_domain_center[3];
	FP_TYPE obs_domain_center[3];

	// unified_domain_size[0~3] stores the modified domain size (domain size has to be slightly larger than the actual source spaces to tolerate numerical error, and it has to be integer times of box size too)
	FP_TYPE unified_domain_size[3];
	
	// max size among three dimensions
	FP_TYPE max_size;
	// which dimension has the max size
	// this is used to calculate appropriate box size
	int max_size_dim;

	// a pointer to its parent nbodyfast
	class NBODYFAST *nbodyfast;

protected:
};

}
#endif