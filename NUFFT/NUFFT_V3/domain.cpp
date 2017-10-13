/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* domain.cpp: class definition of Class Domain
*/
#include "field.h"
#include "domain.h"
#include "nbodyfast.h"
namespace NBODYFAST_NS{
int Domain :: setup()
{
	int _problem_size = nbodyfast->problem_size;
	max_size = 0.0f;
	max_size_dim = 0;

	// Determining the upper and lower boundary of the computation domain
	for (int k = 0; k < 3; k++)
	{
		src_domain_range[k] = 1.0e30f;
		src_domain_range[k + 3] = -1.0e30f;

		obs_domain_range[k] = 1.0e30f;
		obs_domain_range[k + 3] = -1.0e30f;

		for (int i = 0; i < _problem_size; i++)
		{	
			if (nbodyfast->field->src_coord[_problem_size * k + i] < src_domain_range[k]) src_domain_range[k] = nbodyfast->field->src_coord[_problem_size * k + i];
			if (nbodyfast->field->src_coord[_problem_size * k + i] > src_domain_range[k + 3]) src_domain_range[k + 3] = nbodyfast->field->src_coord[_problem_size * k + i];

			if (nbodyfast->field->obs_coord[_problem_size * k + i] < obs_domain_range[k]) obs_domain_range[k] = nbodyfast->field->obs_coord[_problem_size * k + i];
			if (nbodyfast->field->obs_coord[_problem_size * k + i] > obs_domain_range[k + 3]) obs_domain_range[k + 3] = nbodyfast->field->obs_coord[_problem_size * k + i];
		}

		src_domain_range[k + 6] = src_domain_range[k + 3] -  src_domain_range[k];
		src_domain_center[k] = (src_domain_range[k + 3] + src_domain_range[k]) / 2.0f;

		obs_domain_range[k + 6] = obs_domain_range[k + 3] -  obs_domain_range[k];
		obs_domain_center[k] = (obs_domain_range[k + 3] + obs_domain_range[k]) / 2.0f;
	}
	
	// making the interpolation region slightly larger than actual to tolerate possible numerical inaccuracies
	// src domain and obs domain should have the same size (required for FFT grid)
	for (int k = 0; k < 3; k++)
	{
		unified_domain_size[k] = src_domain_range[k + 6] > obs_domain_range[k + 6] ? src_domain_range[k + 6] * 1.001f : obs_domain_range[k + 6] * 1.001f;

		if (unified_domain_size[k] > max_size)
		{
			max_size = unified_domain_size[k];
			max_size_dim = k;
		}

	src_domain_range[k] = src_domain_center[k] -  unified_domain_size[k] * 0.5f;
	src_domain_range[k + 3] = src_domain_center[k] +  unified_domain_size[k] * 0.5f;

	obs_domain_range[k] = obs_domain_center[k] -  unified_domain_size[k] * 0.5f;
	obs_domain_range[k + 3] = obs_domain_center[k] +  unified_domain_size[k] * 0.5f;
	}

	Field::max_domain_size = max_size;
	return 0;
}
}