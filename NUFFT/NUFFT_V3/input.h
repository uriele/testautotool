/*
 * Copyright 2007-2012 Computational Electromagnetic Group (CEM), Dept. ECE, UC San Diego.  All rights reserved.
 * Author: Shaojing Li, March 2012
 */
/*
* error.h: class declaration of Class Input.
* This class is intended for reading commands from execution scripts
* This class is not in use right now
*/
#ifndef _INPUT
#define _INPUT
#include "fp_precision.h"
namespace NBODYFAST_NS{
class Input
{
public:
	Input(){}
	Input(class NBODYFAST *n_ptr)
	{
		nbodyfast = n_ptr;
	}
	virtual ~Input(){}

	class NBODYFAST *nbodyfast;

protected:
};
}
#endif