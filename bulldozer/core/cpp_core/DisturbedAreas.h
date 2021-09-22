#ifndef DISTURBEDAREAS_H
#define DISTURBEDAREAS_H

/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.
*/

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>

/*
    This class is used to build a mask that flags the pixels associated to a disturbed area.
    Most of those areas correspond to water or correlation issues during the DSM generation (obstruction, etc.).
*/
namespace bulldozer 
{

    class DisturbedAreas
    {
        public:

            DisturbedAreas();

            void build_disturbance_mask(float * dsm_strip,
                                        bool * disturbance_mask,
                                        unsigned int nb_rows,
                                        unsigned int nb_cols,
                                        float slope_treshold,
                                        unsigned int disturbed_treshold,
                                        float disturbed_influence_distance,
                                        float dsm_resolution);
        
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