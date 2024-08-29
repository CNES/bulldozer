/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.
*/
#include "c_bordernodata.h"

namespace bulldoproto 
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
        

        unsigned int c, l;
        for(unsigned int r = 0; r < nb_rows; r++) {
            // extracts border nodata for the left side of the input DSM
            c = r * nb_cols;
            while(c < ((r * nb_cols)-1 + nb_cols) && dsm[c] == nodata_value){
                border_nodata_mask[c] = true;
                // std::cout << "Left side, index: " << c << std::endl;
                c++;
            }
            // extracts border nodata for the right side of the input DSM
            c = r * nb_cols + nb_cols - 1;
            while(c > r * nb_cols && dsm[c] == nodata_value){
                border_nodata_mask[c] = true;
                // std::cout << "Right side, index: " << c << std::endl;
                c--;
            }
        }


        for(unsigned int k = 0; k < nb_cols; k++) {
            // Extracts border nodata for the top side of the input DSM
            l = k;
            while(l < (nb_rows * nb_cols) && dsm[l] == nodata_value) {
                border_nodata_mask[l] = true;
                // std::cout << "Top side, index: " << l << std::endl;
                l += nb_cols;
            }

            // Extracts border nodata for the bottom side of the input DSM
            l = (nb_rows - 1) * nb_cols + k;
            while(l >= 0 && l < nb_rows * nb_cols && dsm[l] == nodata_value) {
                border_nodata_mask[l] = true;
                // std::cout << "Bottom side, index: " << l << std::endl;
                l -= nb_cols;
            }
        }
    }
}