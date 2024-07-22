#include "makros.h"

void DEBUG_PRINTING(std::string out)
{
#ifdef DEBUG_PRINT
	std::cout << out;
#endif // DEBUG_PRINT
}
