#ifndef C_STATS_H
#define C_STATS_H

#include <iostream>
#include <algorithm>

namespace bulldoproto
{
    void compute_stats(float * dsm,
                       unsigned char * invalid_mask,
                       float * stats,
                       unsigned int nb_rows,
                       unsigned int nb_cols,
                       float nodata);
} // end of namespace bulldozer

#endif