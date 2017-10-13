/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct.cpp: class definition of Class Domain
*/
#include "nbodyfast.h"
#include "direct.h"
#include "error.h"
namespace NBODYFAST_NS{
Direct :: Direct(class NBODYFAST *n_ptr)
{
	nbodyfast = n_ptr;
	problem_size = nbodyfast->problem_size;
	test_size = problem_size;
}
Direct :: ~Direct()
{
}
int Direct :: execution() //  the execution function is implemented in various derived classes of Direct
{
	return 0;
}
}