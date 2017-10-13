/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* fp_precision.h: precision definition
*/
#ifndef _FP_PRECISION
#define _FP_PRECISION
namespace NBODYFAST_NS{
#ifndef S_V_IMPLEMENT
#define S_V_IMPLEMENT 1
#endif
#ifndef FIELD_DIM
#define FIELD_DIM 3
#endif

#ifdef _DOUBLE_PREC
	#define FP_TYPE double
	#define CUFFT_COMPLEX_TYPE cufftDoubleComplex
	/*#define CUFFT_TYPE CUFFT_Z2Z
	#define CUFFT_EXEC cufftExecZ2Z
	#define CUFFT_EXEC_TEMP cufftExecR2Z*/
	#define CUFFT_EXEC_F cufftExecD2Z
	#define CUFFT_EXEC_B cufftExecZ2D
	#define CUFFT_TYPE_F CUFFT_D2Z
	#define CUFFT_TYPE_B CUFFT_Z2D

#else
	#define FP_TYPE float
	#define CUFFT_COMPLEX_TYPE cufftComplex
	//#define CUFFT_TYPE CUFFT_C2C//R2C CHANGE!!!!
	//#define CUFFT_EXEC cufftExecC2C//R2C CHANGE!!!!
	#define CUFFT_EXEC_F cufftExecR2C
	#define CUFFT_EXEC_B cufftExecC2R
	#define CUFFT_TYPE_F CUFFT_R2C
	#define CUFFT_TYPE_B CUFFT_C2R

#endif
}
#endif