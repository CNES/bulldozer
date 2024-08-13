/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.

    author:     Pierre Lassalle (DNO/OT/IS)
    contact:    pierre.lassalle@cnes.fr
*/
#include "c_springforce.h"

namespace bulldozer {

    BulldozerFilters::BulldozerFilters()
    {
    }

    void BulldozerFilters::applyUniformFilter(float * input_data,
                                              float * output_data,
                                              unsigned int rows,
                                              unsigned int cols,
                                              float nodata,
                                              unsigned int filter_size)
    {
        // output_data is a buffer already allocated.

        // input_data is the input raster from which the uniform filter is applied
        // values are flattened, ie row = value / cols and col = value % cols
        for(unsigned int c = 0; c < rows * cols; c++)
        {
            if(input_data[c] != nodata)
            {
                output_data[c] = getUniformValue(c, input_data, rows, cols, filter_size, nodata);
            }
            else
            {
                output_data[c] = nodata;
            }
        }
    }

    inline float BulldozerFilters::getUniformValue(const unsigned int central_coord,
                                                   float * input_data,
                                                   const unsigned int rows,
                                                   const unsigned int cols,
                                                   const unsigned int filter_size,
                                                   const float nodata)
    {
        unsigned int row = central_coord / cols;
        unsigned int col = central_coord % cols;
        unsigned int min_row = (row >= filter_size) ? row - filter_size : 0;
        unsigned int min_col = (col >= filter_size) ? col - filter_size : 0;
        unsigned int max_row = ( row + filter_size < rows ) ? row + filter_size : rows - 1;
        unsigned int max_col = (col + filter_size < cols ) ? col + filter_size : cols - 1;

        float num_valid = 0.f;
        
        float mean = 0.f;

        for(unsigned int r = min_row; r <= max_row; r++)
        {
            for(unsigned int c = min_col; c <= max_col; c++)
            {
                unsigned int coord = r * cols + c;
                
                if(input_data[coord] != nodata)
                {
                    num_valid++;
                    mean += input_data[coord];
                }
            }
        }
        // num_valid must always be greater than 0 since we check before calling this method if
        // the center value is valid.
        return mean / num_valid;
    }


} // end of namespace bulldozer
