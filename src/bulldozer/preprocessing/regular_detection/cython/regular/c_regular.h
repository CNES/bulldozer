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
                            unsigned char * regularMask,
                            unsigned int nb_rows,
                            unsigned int nb_cols,
                            float thresh,
                            float nodata_value);
    
    void predict_anchorage_mask(unsigned char * regular_mask,
								unsigned char * anchorage_mask,
								unsigned int nb_rows,
                          		unsigned int nb_cols,
								unsigned int max_object_size);

} // end of namespace bulldozer

#endif