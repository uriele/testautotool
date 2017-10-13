/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct_static_vector_gpu.h: class declaration of Class DirectStaticVectorGpu
*/
#ifndef _DIRECT_STATIC_VECTOR_GPU
#define _DIRECT_STATIC_VECTOR_GPU
#include "direct_static_vector.h"
namespace NBODYFAST_NS{
class DirectStaticVectorGpu : public DirectStaticVector
{
public:
	DirectStaticVectorGpu() : DirectStaticVector(){}
	DirectStaticVectorGpu(class NBODYFAST *n_ptr);
	~DirectStaticVectorGpu();
	
	virtual int execution(); 
private:
	class FieldStaticVectorGpu* field_static_vector_gpu;
	int execution_single();
	int execution_multi();
};

}
#endif