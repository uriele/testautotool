/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* direct.h: class decalaration for Class Direct
*/
#ifndef _DIRECT
#define _DIRECT
#include "fp_precision.h"
namespace NBODYFAST_NS{
class Direct
{
public:
	Direct(){}
	Direct(class NBODYFAST *n_ptr);
	virtual ~Direct() = 0;

	virtual int execution(); //  the execution function is implemented in various derived classes of Direct
	class NBODYFAST *nbodyfast;

protected:
	bool gpu_on;
	int problem_size;
	int test_size;
};
}
#endif