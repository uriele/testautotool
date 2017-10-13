/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* memory.h: class declaration of Class Memory.
* This class is intended for storing various memory allocation information inside NBODYFAST library for eaiser debugging
*/
#ifndef _MEMORY
#define _MEMORY
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include "fp_precision.h"
namespace NBODYFAST_NS{
class MemoryObj
{
public:
	MemoryObj(){}
	MemoryObj(size_t arg1, size_t arg2, std::string arg3 = "")
	{
		obj_addr = arg1;
		obj_size = arg2;
		array_name = arg3;
	}

	size_t obj_addr;
	size_t obj_size;
	std::string array_name;
	virtual ~MemoryObj(){}
};

class Memory
{
public:
	Memory(){}
	Memory(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
		_mem_used = 0;
		_peak_mem_used = 0;
	}

	virtual ~Memory()
	{
		output_allocated_list_scr();
	}
	// allocate memory on host and put it to the list
	template<class T>inline int alloc_host(T ** array_for_alloc, int num_elements, std::string array_name = "")
	{

		// do the normal allocation
		if (*array_for_alloc != NULL)
		{
			std::cout << "array: " <<  array_name << "has already been allocated..." << std::endl;
			return -1;
		}

		// normal allocation
		*array_for_alloc = new T[num_elements];

		if (*array_for_alloc == NULL)
		{
			std::cout << "host memory allocation failed..." << std::endl;
			return -1;
		}
		// create a memory object and inserted to map		
#pragma omp critical
		{
			class MemoryObj _mem_obj(size_t(*array_for_alloc), size_t(num_elements * sizeof(T)), array_name);
			ret = allocated_list.insert(std::pair<size_t, class MemoryObj>(size_t(*array_for_alloc), _mem_obj));	
			_mem_used += num_elements * sizeof(T);
			if (_mem_used > _peak_mem_used){
				_peak_mem_used = _mem_used;
				output_allocated_list_file();
			}
		}
		return 0;
	}

	// deallocated memory on the host and remove it from the list
	template<class T> inline int free_host(T ** array_for_dealloc)
	{

		if (* array_for_dealloc == NULL)
		{
			return -1;
		}

		std::map<size_t, class MemoryObj>::iterator _it;
		_it = allocated_list.find(size_t(*array_for_dealloc));
		if (_it == allocated_list.end())
		{
			//std::cout << "CPU memory address: " <<  size_t(*array_for_dealloc) << " has already been deallocated..." << std::endl;
			return -1;
		}
#pragma omp critical
		{
			_mem_used -= _it->second.obj_size; 
			allocated_list.erase(_it);
		}
		// do the standard deallocation
		delete [] *array_for_dealloc;
		*array_for_dealloc = NULL;

		return 0;
	}


	// add an already allocated array to the list
	template<class T>inline int add_to_list_host(T * array_for_alloc, int num_elements, std::string array_name = "")
	{
		if (array_for_alloc == NULL)
		{
			std::cout << "Cannot add an unallocated array to list..." << std::endl;
			return -1;
		}

		// create a memory object and inserted to map		
		class MemoryObj _mem_obj(size_t(array_for_alloc), size_t(num_elements * sizeof(T)), array_name);
		ret = allocated_list.insert(std::pair<size_t, class MemoryObj>(size_t(array_for_alloc), _mem_obj));	
		_mem_used += num_elements * sizeof(T);
		if (_mem_used > _peak_mem_used) _peak_mem_used = _mem_used;
		return 0;

	}

	template <class T> inline void host_array_to_disk(T* _output, unsigned const int _n, std::string filename = "")
	{
		if (filename == "") filename = std::string("..\\Temp\\TestArrayDevice.out");
		std::ofstream out(filename.c_str());
		if(!out) 
		{
			std::cout << "Cannot open file. \n";
			return;
		}

		for (unsigned int i = 0; i < _n; i++ ) 
		{
				out << _output[i] << std::endl;
		}
		out.close();
	}






	void output_allocated_list()
	{
		std::map<size_t, class MemoryObj>::iterator _it;
		std::ofstream out("..\\Temp\\array_allocated_host.out");
		if(!out) 
		{
			std::cout << "Cannot open \"array_allocated_host.out\". \n";
			return;
		}

		out << "Total memory used: " << _mem_used << " bytes; ";
		out << "Peak memory used: " << _peak_mem_used << " bytes; \n";
		if (!allocated_list.empty())
		{
			for (_it = allocated_list.begin(); _it != allocated_list.end(); _it++)
			{
				out << _it->second.array_name << ": " << _it->second.obj_size << " bytes \n";
			}
		}
	}
	void output_allocated_list_file(std::string filename = "..\\Temp\\CPUmemProf.out")
	{
		//if (filename == "") filename = std::string("..\\Temp\\GPUmemProf.out");
		std::ofstream out(filename.c_str());

		if(!out) {
			//std::cout << "Cannot open file for CPU mem profiling. \n";
			return;
		}

		std::map<size_t, class MemoryObj>::iterator _it;
		out << "-----------------------------------------------------------" << std::endl;
		out << "CPU memory Profiling " << std::endl;
		out << "Current CPU memory used: " << _mem_used << " bytes;" << std::endl;
		out << "Peak CPU memory used: " << _peak_mem_used << " bytes;" << std::endl;
		out << "-----------------------------------------------------------" << std::endl;
		if (!allocated_list.empty())
		{
			for (_it = allocated_list.begin(); _it != allocated_list.end(); _it++)
			{
				out << _it->second.array_name << "	" << _it->second.obj_size << "	" << "\n";
			}
		}
		out.close();
	}
	void output_allocated_list_scr()
	{
		std::map<size_t, class MemoryObj>::iterator _it;
		std::cout << "-----------------------------------------------------------" << std::endl;
		std::cout << "CPU memory report " << std::endl;
		std::cout << "Total memory used: " << _mem_used << " bytes; " << std::endl;
		std::cout << "Peak memory used: " << _peak_mem_used << " bytes; " << std::endl;
		std::cout << "-----------------------------------------------------------" << std::endl;
		if (!allocated_list.empty())
		{
			for (_it = allocated_list.begin(); _it != allocated_list.end(); _it++)
			{
				std::cout << _it->second.array_name << ": " << _it->second.obj_size << " bytes \n";
			}
		}
	}
protected:
	NBODYFAST *nbodyfast;
private:
	size_t _mem_used;
	size_t _peak_mem_used;
	std::map<size_t, class MemoryObj> allocated_list;
	std::pair<std::map<size_t, class MemoryObj>::iterator,bool> ret;



};
}
#endif
