#ifndef _DIRECT_STATIC_SCALAR
#define _DIRECT_STATIC_SCALAR
#include "direct.h"
namespace NBODYFAST_NS{
class DirectStaticScalar : public Direct
{
public:
	DirectStaticScalar() : Direct(){}
	DirectStaticScalar(class NBODYFAST *n_ptr);
	~DirectStaticScalar();

	virtual int execution();
private:
	class FieldStaticScalar* field_static_scalar; // a pointer to the field object that this class apply on
};
}
#endif