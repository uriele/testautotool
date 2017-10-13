/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* gpu.h: class declaration of Class Gpu
*/
#ifndef _GPU
#define _GPU
#include <string>
#include "fp_precision.h"


namespace NBODYFAST_NS{
// constants that controls number of threads per block for each GPU calculation kernel
const int BLOCK_SIZE_DIRECT = 128;
const int BLOCK_SIZE_PROJ_INTERP = 64;
const int BLOCK_SIZE_CORRECT = 128;//64
const int BLOCK_SIZE_CONV = 256;
const int BLOCK_SIZE_NEAR = 96;//CHANGED BY BEN

// Not sure why I defined this class...but not using cudaDeviceProp class...
class DeviceProp
{
public:
	int num;
	int index;
	std::string dev_name;
	size_t dev_mem;
};

class Gpu
{
public:
	Gpu(){}
	Gpu(class NBODYFAST *n_ptr, int *_num_gpus, int *_gpu_list);
	virtual ~Gpu();
	class MemoryGpu *memory_gpu; // MemoryGpu class object is similar to Memory class object

	bool multi_gpu;
	int num_gpus;
	class DeviceProp *dev_list; // a list of device

	class NBODYFAST *nbodyfast; // pointer to its parent

protected:
	inline void cuda_env_early_test(DeviceProp); // early GPU test to see if it runs well
private:
	int **_d_early_test; // temporary array to do the early GPU test before all other GPU operations
};
// parameters that are transferred by value to each GPU kernel
class GpuExecParam
{
public:
	GpuExecParam(){}
	virtual ~GpuExecParam(){}
	int num_blk;
	int num_blk_per_box;
	int num_box_per_blk;

};
}



#endif