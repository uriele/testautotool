/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* error.h: class declaration of Class Error.
* This class is intended for storing various internal debugging and diagnosing information, e.g. error codes
* This class is in not use right now
*/
#ifndef _ERROR
#define _ERROR
#include "fp_precision.h"
namespace NBODYFAST_NS{
class Error
{
public:
	Error(){}
	Error(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
	}
	virtual ~Error(){}

	int last_error;
protected:
	NBODYFAST *nbodyfast;
};
}
#endif