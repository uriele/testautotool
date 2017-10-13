/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_static_scalar_gpu.h: class declaration of Class DirectStaticScalarGpu
*/
#ifndef _DIRECT_STATIC_SCALAR_GPU
#define _DIRECT_STATIC_SCALAR_GPU
#include "direct_static_scalar.h"
namespace NBODYFAST_NS{
class DirectStaticScalarGpu : public DirectStaticScalar
{
public:
	DirectStaticScalarGpu() : DirectStaticScalar(){}
	DirectStaticScalarGpu(class NBODYFAST *n_ptr);
	~DirectStaticScalarGpu();
	
	virtual int execution(); 
private:
	class FieldStaticScalarGpu* field_static_scalar_gpu;
	int execution_single();
	int execution_multi();
};

}
#endif