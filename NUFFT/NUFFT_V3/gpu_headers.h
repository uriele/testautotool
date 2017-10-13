/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* gpu_headers.h: GPU related headers that should be included to nbodyfast.h in order to run on GPUs
*/
#include "gpu.h"
#include "field_static_scalar_gpu.h"
#include "direct_static_scalar_gpu.h"
#include "nufft_static_scalar_gpu.h"

#include "field_static_vector_gpu.h"
#include "direct_static_vector_gpu.h"
#include "nufft_static_vector_gpu.h"
