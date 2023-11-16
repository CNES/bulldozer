#ifndef C_REGULAR_H
#define C_REGULAR_H

#include <iostream>
#include <cmath>
#include <map>
#include <limits>
#include <fstream>
#include <algorithm>

namespace bulldoproto
{
    void buildRegularMask(float * dsm,
                            unsigned char * noisyMask,
                            unsigned char * regularMask,
                            unsigned int nb_rows,
                            unsigned int nb_cols,
                            float thresh,
                            float nodata_value);

} // end of namespace bulldozer

#endif