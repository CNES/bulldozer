#ifndef C_HIST_H
#define C_HIST_H

#include <iostream>

namespace bulldoproto
{
    void compute_hist(float * dsm,
					  unsigned int * hist,
					  float min_z,
					  float bin_width,
					  unsigned int nb_bins,
                      unsigned int nb_rows,
                      unsigned int nb_cols,
                      float nodata);
                      
} // end of namespace bulldozer

#endif