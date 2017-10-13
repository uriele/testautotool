/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* error.h: class declaration of Class Timer.
* This class is intended for providing various timing utilities to help profile the code
* This class is very rudimentary
*/
#ifndef _TIMER
#define _TIMER
#define _OPEN_MP
#include <ctime>
#include <string>
#include <iostream>
#include <fstream>

#ifdef _OPEN_MP
#include <omp.h>
#endif
#include "fp_precision.h"
namespace NBODYFAST_NS{
class Timer
{
public:
	inline double get_time()
	{
#ifdef _OPEN_MP
		return omp_get_wtime();
#endif
		return 0.0;
	}
	Timer()
	{
	}
	Timer(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
		for (int i = 0; i < 4; i++)
		{
			_exec_time[i] = 0.0;
			_prep_time[i] = 0.0;
			_file_time[i] = 0.0;
			_memory_time[i] = 0.0;
			_fft_time[i] = 0.0;
			_interp_time[i] = 0.0;
			_proj_time[i] = 0.0;
			_conv_time[i] = 0.0;
			_debug_time1[i] = 0.0;
			_debug_time2[i] = 0.0;
			_debug_time3[i] = 0.0;
			_debug_time4[i] = 0.0;
		}

		for (int i =0; i < 5; i++)
		{
			_stage_time[i] = 0.0;
		}

		_last_time = 0.0;
		_this_time = 0.0;
		_elapsed_time = 0.0;
		_timing_started = true;

		time( &_program_start_time);
		struct tm* timeinfo;
		timeinfo = localtime ( &_program_start_time);
		std::cout << std::endl << std::endl << std::endl << "*****************************************************" << std::endl;
		std::cout << "NUFFT Library ver 3.8 dbl/sgl prec, 020714, by Shaojing Li and Sidi Fu" << std::endl;
		std::cout << "*****************************************************" << std::endl;
		std::cout << "-----------------------------------------------------------" << std::endl;
		std::cout << "Program start time and date: " << asctime (timeinfo) << std::endl;
		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	virtual ~Timer()
	{
		time( &_program_end_time);
		struct tm* timeinfo;
		timeinfo = localtime ( &_program_end_time);
		std::cout << "-----------------------------------------------------------" << std::endl;
		std::cout << "Program end time and date: " << asctime (timeinfo) << std::endl;
		_elapsed_time = difftime(_program_end_time, _program_start_time);
		std::cout << "Total elapsed time: " << _elapsed_time << " s" << std::endl;

		std::cout << "Preprocessing time: " << _prep_time[2] * 1e3 << " ms" << std::endl;
		std::cout << "Execution time: " << _exec_time[2] * 1e3<< " ms" << std::endl;
		std::cout << "Projection time: " << _proj_time[2] * 1e3<< " ms" << std::endl;
		std::cout << "FFT and iFFT time: " << _fft_time[2] * 1e3<< " ms" << std::endl;
		std::cout << "Convolution time: " << _conv_time[2] * 1e3<< " ms" << std::endl;
		std::cout << "Interpolation time: " << _interp_time[2] * 1e3<< " ms" << std::endl;
		std::cout << "Memory copy time: " << _memory_time[2] * 1e3 << " ms" << std::endl;
		std::cout << "post_fft time: " << _debug_time1[2] * 1e3 << " ms" << std::endl;
		std::cout << "Correction time: " << _debug_time2[2] * 1e3 << " ms" << std::endl;
		std::cout << "Interpolation time: " << _debug_time3[2] * 1e3 << " ms" << std::endl;
		std::cout << "Exact near field time: " << _debug_time4[2] * 1e3 << " ms" << std::endl;
		std::cout << "-----------------------------------------------------------" << std::endl;

		//std::cout << "File reading/writing time: " << _file_time * 1e3 << " ms" << std::endl;

	}

	virtual void inline time_stamp(std::string _entry = "")
	{
		_this_time = get_time();
		_elapsed_time = _this_time - _last_time;
		if (_entry == "EXEC")
		{
			if (_exec_time[3] == 0.0) 
			{
				_exec_time[0] = _this_time;
				_exec_time[3] = 1.0;
			}
			else
			{
				_exec_time[1] = _this_time;
				_exec_time[2] += _exec_time[1] - _exec_time[0];
				_exec_time[0] = _this_time;
				_exec_time[3] = 0.0;
			}
		}
		if (_entry == "PREP")
		{
			if (_prep_time[3] == 0.0) 
			{
				_prep_time[0] = _this_time;
				_prep_time[3] = 1.0;
			}
			else
			{
				_prep_time[1] = _this_time;
				_prep_time[2] += _prep_time[1] - _prep_time[0];
				_prep_time[0] = _this_time;
				_prep_time[3] = 0.0;
			}
		}
		if (_entry == "FFT")
		{
			if (_fft_time[3] == 0.0) 
			{
				_fft_time[0] = _this_time;
				_fft_time[3] = 1.0;
			}
			else
			{
				_fft_time[1] = _this_time;
				_fft_time[2] += _fft_time[1] - _fft_time[0];
				_fft_time[0] = _this_time;
				_fft_time[3] = 0.0;
			}
		}

		if (_entry == "PROJ")
		{
			if (int(_proj_time[3]) == 0) 
			{
				_proj_time[0] = _this_time;
				_proj_time[3] = 1.0;
			}
			else
			{
				_proj_time[1] = _this_time;
				_proj_time[2] += _proj_time[1] - _proj_time[0];
				_proj_time[0] = _this_time;
				_proj_time[3] = 0.0;
			}
		}

		if (_entry == "CONV")
		{
			if (int(_conv_time[3]) == 0) 
			{
				_conv_time[0] = _this_time;
				_conv_time[3] = 1.0;
			}
			else
			{
				_conv_time[1] = _this_time;
				_conv_time[2] += _conv_time[1] - _conv_time[0];
				_conv_time[0] = _this_time;
				_conv_time[3] = 0.0;
			}
		}

		if (_entry == "INTE")
		{
			if (int(_interp_time[3]) == 0) 
			{
				_interp_time[0] = _this_time;
				_interp_time[3] = 1.0;
			}
			else
			{
				_interp_time[1] = _this_time;
				_interp_time[2] += _interp_time[1] - _interp_time[0];
				_interp_time[0] = _this_time;
				_interp_time[3] = 0.0;
			}

		}

		if (_entry == "MEMO")
		{
			if (_memory_time[3] == 0.0) 
			{
				_memory_time[0] = _this_time;
				_memory_time[3] = 1.0;
			}
			else
			{
				_memory_time[1] = _this_time;
				_memory_time[2] += _memory_time[1] - _memory_time[0];
				_memory_time[0] = _this_time;
				_memory_time[3] = 0.0;
			}
		}
		if (_entry == "DEBUG1")
		{
			if (_debug_time1[3] == 0.0) 
			{
				_debug_time1[0] = _this_time;
				_debug_time1[3] = 1.0;
			}
			else
			{
				_debug_time1[1] = _this_time;
				_debug_time1[2] += _debug_time1[1] - _debug_time1[0];
				_debug_time1[0] = _this_time;
				_debug_time1[3] = 0.0;
			}
		}

		if (_entry == "DEBUG2")
		{
			if (_debug_time2[3] == 0.0) 
			{
				_debug_time2[0] = _this_time;
				_debug_time2[3] = 1.0;
			}
			else
			{
				_debug_time2[1] = _this_time;
				_debug_time2[2] += _debug_time2[1] - _debug_time2[0];
				_debug_time2[0] = _this_time;
				_debug_time2[3] = 0.0;
			}
		}

		if (_entry == "DEBUG3")
		{
			if (_debug_time3[3] == 0.0) 
			{
				_debug_time3[0] = _this_time;
				_debug_time3[3] = 1.0;
			}
			else
			{
				_debug_time3[1] = _this_time;
				_debug_time3[2] += _debug_time3[1] - _debug_time3[0];
				_debug_time3[0] = _this_time;
				_debug_time3[3] = 0.0;
			}
		}

		if (_entry == "DEBUG4")
		{
			if (_debug_time4[3] == 0.0) 
			{
				_debug_time4[0] = _this_time;
				_debug_time4[3] = 1.0;
			}
			else
			{
				_debug_time4[1] = _this_time;
				_debug_time4[2] += _debug_time4[1] - _debug_time4[0];
				_debug_time4[0] = _this_time;
				_debug_time4[3] = 0.0;
			}
		}
		_last_time = _this_time;

	}
	virtual void inline time_show()
	{
		std::cout << "Elapsed time: " << _elapsed_time * 1e3 << " ms" << std::endl;
	}
	class NBODYFAST *nbodyfast;
protected:
	time_t _program_start_time;
	time_t _program_end_time;

	double _stage_time[5];

	double _exec_time[4];
	double _prep_time[4];
	double _memory_time[4];
	double _file_time[4];
	double _fft_time[4];
	double _interp_time[4];
	double _proj_time[4];
	double _conv_time[4];
	double _debug_time1[4];
	double _debug_time2[4];
	double _debug_time3[4];
	double _debug_time4[4];

	double _last_time;
	double _this_time;
	double _elapsed_time;

	bool _timing_started;


};
}
#endif
