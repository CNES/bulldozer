/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.
*/
#include "c_bordernodata.h"

namespace bulldozer 
{
        BorderNodata::BorderNodata()
    {
    }

    void BorderNodata::buildBorderNodataMask(float * dsm,
                                            unsigned char * border_nodata_mask,
                                            unsigned int nb_rows,
                                            unsigned int nb_cols,
                                            float nodata_value) 
    {
        unsigned int c;
        for(unsigned int r = 0; r < nb_rows; r++) {
            // extracts border nodata for the left side of the input DSM
            c = r * nb_cols;
            while(c < ((r * nb_cols)-1 + nb_cols) && dsm[c] == nodata_value){
                border_nodata_mask[c] = true;

                c++;
            }
            // extracts border nodata for the right side of the input DSM
            c = r * nb_cols + nb_cols - 1;
            while(c > r * nb_cols && dsm[c] == nodata_value){
                border_nodata_mask[c] = true;
                c--;
            }
        }
    }
}