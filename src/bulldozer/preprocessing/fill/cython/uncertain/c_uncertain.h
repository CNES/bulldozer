#ifndef C_UNCERTAIN_H
#define C_UNCERTAIN_H

#include <iostream>
#include <algorithm>
#include <cmath>

namespace bulldoproto
{
    float distance(long int r1, 
				   long int c1, 
				   long int r2,
				   long int c2);

	void prefill_uncertain(float * dsm,
						   unsigned char * uncertain_mask,
						   unsigned char * regular_mask,
						   float * prefilled_dsm,
						   float * uncertain_map,
						   unsigned int rows,
						   unsigned int cols,
						   unsigned int search_radius,
						   float max_slope_percent,
						   float dsm_resolution);
                      
} // end of namespace bulldozer

#endif