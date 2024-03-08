#ifndef C_ANCHORAGE_H
#define C_ANCHORAGE_H

#include <cmath>

namespace bulldoproto {

    void predict_anchorage(float * dsm,
                           float nodata,
                           unsigned char * regular_mask,
                           unsigned char * anchorage_mask,
                           unsigned int nb_rows,
                           unsigned int nb_cols,
                           float max_object_size);

} // end of namespace bulldoproto

#endif