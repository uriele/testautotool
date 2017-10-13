/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* output.h: class declaration of Class Output.
* This class is intended for controlling screen and file output that records the execution status of the program
* This class is in not use right now
*/
#ifndef _OUTPUT
#define _OUTPUT
#include "fp_precision.h"

namespace NBODYFAST_NS{
class Output
{
public:
	Output(){}
	Output(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
	}
	virtual ~Output(){}
	class NBODYFAST *nbodyfast;
protected:
};
}
#endif