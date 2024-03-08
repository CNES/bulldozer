#ifndef C_SPRING_FORCE_H
#define C_SPRING_FORCE_H

/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.

    author:     Pierre Lassalle (DNO/OT/IS)
    contact:    pierre.lassalle@cnes.fr
*/
#include <algorithm>

/*
    This class contains a list of useful filters for applying spring tension
    in the drap cloth algorithm while taking into account the no data value
*/
namespace bulldozer 
{

    class BulldozerFilters
    {
        public:

            BulldozerFilters();

            void applyUniformFilter(float * input_data,
                                    float * output_data,
                                    unsigned int rows,
                                    unsigned int cols,
                                    float nodata,
                                    unsigned int filter_size);
        
        private:

            float getUniformValue(const unsigned int central_coord,
                                float * input_data,
                                const unsigned int rows,
                                const unsigned int cols,
                                const unsigned int filter_size,
                                const float nodata);


    };

} // end of namespace bulldozer

#endif
