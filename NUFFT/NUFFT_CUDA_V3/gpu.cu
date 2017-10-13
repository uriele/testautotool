/* 
* Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* gpu.cu: class definition of Class Gpu
*/
#include "nbodyfast.h"
#include "gpu.h"
#include "memory.h"
#include "memory_gpu.h"
namespace NBODYFAST_NS{
Gpu :: Gpu(class NBODYFAST *n_ptr, int *_num_gpus, int *_gpu_list)
{
	// Initialization of Gpu data-structure and do early device test to see if the selected devices are valid
	nbodyfast = n_ptr;
	multi_gpu = false;
	dev_list = NULL;
	_d_early_test = NULL;
	memory_gpu = new MemoryGpu(nbodyfast);
	num_gpus = *_num_gpus;

	if (num_gpus > 1)
	{
		multi_gpu = true;
	}

	dev_list = new DeviceProp[num_gpus];
	nbodyfast->memory->alloc_host<int*>(&_d_early_test, num_gpus, "gpu->_d_early_test[]");

	for (int i = 0; i < num_gpus; i++)
	{
		cudaDeviceProp _dev_prop;
		dev_list[i].num = i;
		dev_list[i].index = _gpu_list[i];
		nbodyfast->device_name[i] = _gpu_list[i];
		cudaGetDeviceProperties(&_dev_prop, _gpu_list[i]);
		dev_list[i].dev_name.assign(_dev_prop.name);
		dev_list[i].dev_mem = _dev_prop.totalGlobalMem;
	}

#pragma omp barrier
#pragma omp parallel
	{
		int _thread_id = omp_get_thread_num();
		{
			cuda_env_early_test(dev_list[_thread_id]); // early test of GPU devices
		}
	}
#pragma omp barrier

	nbodyfast->memory->free_host<int*>(&_d_early_test);
}

Gpu :: ~Gpu()
{ 
	memory_gpu->output_allocated_list_scr(); // during the destructor, output a list of any remaining allocated arrays (it should be empty under normal circumstances)
	delete memory_gpu;
	delete [] dev_list; dev_list = NULL;
}

inline void Gpu::cuda_env_early_test(DeviceProp _dev)
{
	/* device early test 
	 * try to check if the device index supplied is valid
	 * try to allocate a small memory array on the specified device
	*/
	cudaError_t _cuda_error;
	_cuda_error = cudaSetDevice(_dev.index);
	
	if (_cuda_error != cudaSuccess)
	{
		std::cout << "Cannot select device: " << _dev.index << std::endl;
		std::cout << "Please check if there is/are " <<  _dev.index + 1 << " CUDA capable device(s) in this system and the specified device is working properly...exiting..." << std::endl;

		exit(0);
	}
	_d_early_test[_dev.num] = NULL;
	std::cout << "Testing device: " << _dev.index << " " << _dev.dev_name << std::endl;

	memory_gpu->alloc_device<int>(_cuda_error, &_d_early_test[_dev.num], 1, "gpu->_d_early_test", _dev.index);

	if (_cuda_error != cudaSuccess)
	{
		std::cout << "Device: " << _dev.index << " initialization failed...exiting..." << std::endl;		
		std::cout << cudaGetErrorString(_cuda_error) << std::endl;
		exit(0);
	}

	std::cout << "Device: " << _dev.index << " Initial test passed..." << std::endl;
	memory_gpu->free_device<int>(_cuda_error, &_d_early_test[_dev.num], _dev.index);

}
}