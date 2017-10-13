#ifndef _DIRECT_STATIC_VECTOR
#define _DIRECT_STATIC_VECTOR
#include "direct.h"
namespace NBODYFAST_NS{
class DirectStaticVector : public Direct
{
public:
	DirectStaticVector() : Direct(){}
	DirectStaticVector(class NBODYFAST *n_ptr);
	~DirectStaticVector();

	virtual int execution();
private:
	class FieldStaticVector* field_static_vector; // a pointer to the field object that this class apply on
};
}
#endif