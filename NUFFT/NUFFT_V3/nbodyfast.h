/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* nbodyfast.h: class declaration of Class NBODYFAST
*/
#ifndef _NBODYFAST
#define _NBODYFAST
#define _OPEN_MP
#include <string>
#include <sstream>
#include <cstdlib>
#ifdef _OPEN_MP
#include <omp.h>
#endif
#include "fp_precision.h"
namespace NBODYFAST_NS{
class NBODYFAST
{
public:
	NBODYFAST();
	virtual ~NBODYFAST();
	
	class Domain *domain;          // simulation box
	class Memory *memory;          // memory allocation functions
	class Error *error;            // accuracy checking
	class Field *field;            // inter-particle forces
	class Input *input;            // input script processing
	class Output *output;          // thermo/dump/restart
	class Timer *timer;            // CPU timing info
	class Direct *direct;		// direct method related data and methods
	class Nufft *nufft;		// Nufft related data-structures and methods
	class Gpu *gpu;              // CUDA accelerator class
		
	bool gpu_on; // if gpu is being used
	bool multi_device; // if multiple threads or multiple GPUs are used
	int num_devices; // number of threads/GPUs used
	int *device_name; // stores the GPUs in use (the index number of GPU devices)
	std::string algo_name; // NUFFT or DIRECT
	int problem_size; // problem size

	int *src_size_dev; // source size used by multiGPU calculation (ghost sources included)
	int *obs_size_dev; // observer size used by multiGPU calculation (ghost observer included). there is no ghost observer actually... :-|
	int *src_size_act_dev; // source size used by multiGPU calculation (ghost sources not included)
	int *obs_size_act_dev; // source size used by multiGPU calculation (ghost sources not included)


	// These three functions are only interfaces for outside call
	void global_init(double *src_coord, double *obs_coord, int *problem_size, double *wavenumber, int *field_type, int *gpu_on, int *use_nufft, int *num_gpus, int *gpu_list);
	void execute(double *src_amp, double *fld_amp);
	void global_deinit();
	int get_field_type() const {return field_type;}


private:
	// Is source and observer identical?
	bool _identical;
	// Compute mode, master or slave? for MPI-Nufft
	bool _mode;
	int _num_threads_sys;
	int field_type;
};
}
#endif

