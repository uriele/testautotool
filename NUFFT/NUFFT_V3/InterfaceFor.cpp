/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* InterfaceFor.cpp: Fortran calling interface
*/
#include "nbodyfast.h"
#include <iostream>
using namespace NBODYFAST_NS;

static NBODYFAST *nbodyfast;
extern "C" void cuInit(double *SRC_COORD, double *OBS_COORD, int *PROBLEMSIZE, double *WAVENUMBER, int *FIELDTYPE, int *GPUON, int *ALGO, int *NUMDEVS, int *GPULIST)
{
	nbodyfast = new NBODYFAST;
	nbodyfast->global_init(SRC_COORD, OBS_COORD, PROBLEMSIZE, WAVENUMBER, FIELDTYPE, GPUON, ALGO, NUMDEVS, GPULIST);
}
extern "C" void cuExec(double *CHARGES, double *POTENTIAL)
{
	nbodyfast->execute(CHARGES, POTENTIAL);
}
extern "C" void cuDeInit()
{
	nbodyfast->global_deinit();

	delete nbodyfast;
	nbodyfast = NULL;

}

extern "C" void cuinit_(double *SRC_COORD, double *OBS_COORD, int *PROBLEMSIZE, double *WAVENUMBER, int *FIELDTYPE, int *GPUON, int *ALGO, int *NUMDEVS, int *GPULIST)
{
	nbodyfast = new NBODYFAST;
	nbodyfast->global_init(SRC_COORD, OBS_COORD, PROBLEMSIZE, WAVENUMBER, FIELDTYPE, GPUON, ALGO, NUMDEVS, GPULIST);
}
extern "C" void cuexec_(double *CHARGES, double *POTENTIAL)
{
	nbodyfast->execute(CHARGES, POTENTIAL);
}
extern "C" void cudeinit_()
{
	nbodyfast->global_deinit();

	delete nbodyfast;
	nbodyfast = NULL;

}
