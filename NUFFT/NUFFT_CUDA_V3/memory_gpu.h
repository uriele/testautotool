/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* memory_gpu.h: class declaration of Class MemoryGpu
*/
#ifndef _MEMORY_GPU
#define _MEMORY_GPU
#include "nbodyfast.h"
#include "memory.h"
#include "cufft.h"
namespace NBODYFAST_NS{
class MemoryObjGpu
{
public:
	MemoryObjGpu(){}
	MemoryObjGpu(size_t arg1, size_t arg2, int _dev, std::string arg3 = "")
	{
		obj_addr = arg1;
		obj_size = arg2;
		dev = _dev;
		array_name = arg3;
	}
	size_t obj_addr;
	size_t obj_size;
	int dev;
	std::string array_name;
	virtual ~MemoryObjGpu()
	{
	}
};
class MemoryGpu
{
public:
	MemoryGpu(){}
	MemoryGpu(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
		_mem_used = 0;
		_peak_mem_used = 0;
	}
	virtual ~MemoryGpu(){}
	template<class T>inline int alloc_device(cudaError_t& _cuda_error_code, T ** array_for_alloc, int num_elements, std::string array_name = "", int dev = 0)
	{

		// do the normal allocation
		if (*array_for_alloc != NULL)
		{
			std::cout << "array already allocated..." << std::endl;
			std::cout << "array name: " << array_name << std::endl;
			return -1;
		}

		// normal allocation
		cudaSetDevice(dev);
		//cudaDeviceSynchronize();	
		_cuda_error_code = cudaMalloc((void**)array_for_alloc, sizeof(T) * num_elements);
		//cudaDeviceSynchronize();	

		if (*array_for_alloc == NULL)
		{
			std::cout << "device global memory allocation failed..." << std::endl;
			std::cout << cudaGetErrorString(_cuda_error_code) << std::endl;
			exit(0);
		}

		size_t rel_addr = size_t(*array_for_alloc) + size_t(dev * dev_address_space_bias);
		// create a memory object and inserted to map		
#pragma omp critical
		{
			class MemoryObjGpu _mem_obj(rel_addr, size_t(num_elements * sizeof(T)), dev, array_name);
			ret = allocated_list.insert(std::make_pair(_mem_obj.obj_addr, _mem_obj));	
			//if (ret.second == 0)
			//{
			//	std::cout << "insertion failure!!" << std::endl;
			//	std::cout << "array name: " << _mem_obj.array_name << std::endl;
			//}
			_mem_used += num_elements * sizeof(T);
			if (_mem_used > _peak_mem_used){
				_peak_mem_used = _mem_used;
				output_allocated_list_file();
			}
		}
		return 0;
	}

	template<class T> inline int free_device(cudaError_t& _cuda_error_code, T ** array_for_dealloc, int dev = 0)
	{

		if (* array_for_dealloc == NULL)
		{
			return -1;
		}
		std::map<size_t, class MemoryObjGpu>::iterator _it;
		size_t rel_addr = size_t(*array_for_dealloc) + size_t(dev * dev_address_space_bias);
		_it = allocated_list.find(rel_addr);
		if (_it == allocated_list.end())
		{
			//std::cout << "GPU memory address: " <<  size_t(*array_for_dealloc) << " has already been deallocated..." << std::endl;
			return -1;
		}
		if (dev != _it->second.dev)
		{
			std::cout << "Device argument is wrong..." << std::endl;
			return -1;
		}
		cudaSetDevice(_it->second.dev);


		// do the standard deallocation
		//cudaDeviceSynchronize();	
		_cuda_error_code = cudaFree(*array_for_dealloc);
		//cudaDeviceSynchronize();	
		if (_cuda_error_code != cudaSuccess)
		{
				std::cout << "Free device memory failed... array name: " << _it->second.array_name << std::endl;
				std::cout << cudaGetErrorString(_cuda_error_code) << std::endl;
				exit(0);
				return -1;
		}
#pragma omp critical
		{
			_mem_used -= _it->second.obj_size; 
			allocated_list.erase(_it);
		}

		*array_for_dealloc = NULL;
		return 0;
	}

	template<class T> inline int memcpy_host_to_device(T *_d_des, T *_h_src, int _num_elements, int dev_index)
	{
		cudaSetDevice(dev_index);
		//cudaDeviceSynchronize();	
		cudaMemcpy(_d_des, _h_src, sizeof(T) * _num_elements, cudaMemcpyHostToDevice);
		//cudaDeviceSynchronize();	
		return 0;
	}
	template<class T> inline int memcpy_device_to_host(T *_h_des, T *_d_src, int _num_elements, int dev_index)
	{
#pragma omp critical
		{
		cudaSetDevice(dev_index);
		//cudaDeviceSynchronize();	
		cudaMemcpy(_h_des, _d_src, sizeof(T) * _num_elements, cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();	
		}
		return 0;
	}
	template<class T> inline int memset_device(T *_d_des, int _val, int _num_elements, int dev_index)
	{
		cudaSetDevice(dev_index);
		//cudaDeviceSynchronize();	
		cudaMemset(_d_des, _val, sizeof(T) * _num_elements);
		//cudaDeviceSynchronize();	
		return 0;
	}
	
	template <class T> inline void device_array_to_disk(T* _output, unsigned const int _n, std::string filename = "")
	{
		if (filename == "") filename = std::string("..\\Temp\\TestArrayDevice.out");
		std::ofstream out(filename.c_str());
		if(!out) 
		{
			std::cout << "Cannot open file. \n";
			return;
		}

		T* _host_image = new T[_n];
		//cudaDeviceSynchronize();	
		cudaMemcpy(_host_image, _output, sizeof(T) * _n, cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();	
		for (unsigned int i = 0; i < _n; i++ ) 
		{
				out << _host_image[i] << std::endl;
		}
		out.close();
		delete [] _host_image;
	}

	template <class T> inline void device_array_to_disk_complex(T* _output, unsigned const int _n, std::string filename = "")
	{
		if (filename == "") filename = std::string("..\\Temp\\TestArrayDevice.out");
		std::ofstream out(filename.c_str());
		if(!out) 
		{
			std::cout << "Cannot open file. \n";
			return;
		}

		T* _host_image = new T[_n];
		//cudaDeviceSynchronize();	
		//cudaSetDevice( 3 );
		cudaMemcpy(_host_image, _output, sizeof(T) * _n, cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();	
		for (unsigned int i = 0; i < _n; i++ ) 
		{
				out << _host_image[i].x << std::endl;
		}
		out.close();
		delete [] _host_image;
	}




	void output_allocated_list()
	{
		std::map<size_t, class MemoryObjGpu>::iterator _it;
		std::ofstream out("..\\Temp\\array_allocated_device.out");
		if(!out) 
		{
			std::cout << "Cannot open \"array_allocated_device.out\". \n";
			return;
		}

		out << "Total memory used: " << _mem_used << " bytes; ";
		out << "Peak memory used: " << _peak_mem_used << " bytes; \n";
		if (!allocated_list.empty())
		{
			for (_it = allocated_list.begin(); _it != allocated_list.end(); _it++)
			{
				out << _it->second.array_name << ": " << _it->second.obj_size << " bytes on device: " << _it->second.dev << "\n";
			}
		}
	}
	void output_allocated_list_file(std::string filename = "..\\Temp\\GPUmemProf.out")
	{
		//if (filename == "") filename = std::string("..\\Temp\\GPUmemProf.out");
		std::ofstream out(filename.c_str());

		if(!out) {
			//std::cout << "Cannot open file for GPU mem profiling. \n";
			return;
		}

		std::map<size_t, class MemoryObjGpu>::iterator _it;
		out << "-----------------------------------------------------------" << std::endl;
		out << "GPU memory Profiling " << std::endl;
		out << "Current GPU memory used: " << _mem_used << " bytes;" << std::endl;
		out << "Peak GPU memory used: " << _peak_mem_used << " bytes;" << std::endl;
		out << "-----------------------------------------------------------" << std::endl;
		if (!allocated_list.empty())
		{
			for (_it = allocated_list.begin(); _it != allocated_list.end(); _it++)
			{
				out << _it->second.array_name << "	" << _it->second.obj_size << "	" << _it->second.dev << "\n";
			}
		}
		out.close();
	}
	void output_allocated_list_scr()
	{
		std::map<size_t, class MemoryObjGpu>::iterator _it;
		std::cout << "-----------------------------------------------------------" << std::endl;
		std::cout << "GPU memory report " << std::endl;
		std::cout << "Total memory used: " << _mem_used << " bytes;" << std::endl;
		std::cout << "Peak memory used: " << _peak_mem_used << " bytes;" << std::endl;
		std::cout << "-----------------------------------------------------------" << std::endl;
		if (!allocated_list.empty())
		{
			for (_it = allocated_list.begin(); _it != allocated_list.end(); _it++)
			{
				std::cout << _it->second.array_name << ": " << _it->second.obj_size << " bytes on device: " << _it->second.dev << "\n";
			}
		}
	}	
	class NBODYFAST *nbodyfast;
protected:
private:
	size_t _mem_used;
	size_t _peak_mem_used;
	static const long long int dev_address_space_bias = 1073732608ll;
	std::map<size_t, class MemoryObjGpu> allocated_list;
	std::pair<std::map<size_t, class MemoryObjGpu>::iterator,bool> ret;
};
template <> inline void MemoryGpu::device_array_to_disk<CUFFT_COMPLEX_TYPE>(CUFFT_COMPLEX_TYPE * _output, unsigned const int _n, std::string filename)
{
	if (filename == "") filename = std::string("..\\Temp\\TestArrayDevice.out");
	std::ofstream out(filename.c_str());

	if(!out) 
	{
		std::cout << "Cannot open file. \n";
		return;
	}

	FP_TYPE* _host_image = new FP_TYPE[_n * 2];
	cudaMemcpy(_host_image, _output, sizeof(FP_TYPE) * _n * 2, cudaMemcpyDeviceToHost);
	for (unsigned int i = 0; i < _n; i++ ) 
	{
			out << _host_image[2 * i] << ", " << _host_image[2 * i + 1] << std::endl;
	}
	out.close();
	delete [] _host_image;
}
}

#endif
