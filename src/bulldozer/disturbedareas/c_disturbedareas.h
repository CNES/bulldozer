#ifndef C_DISTURBEDAREAS_H
#define C_DISTURBEDAREAS_H

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
            DisturbedAreas(bool is_four_connexity);

            void build_disturbance_mask(float * dsm_strip,
                                        unsigned char * disturbance_mask,
                                        unsigned int nb_rows,
                                        unsigned int nb_cols,
                                        float slope_treshold,
                                        float nodata_value);

        private:

            bool m_IsFourConnexity;

            bool isVerticalDisturbed(float * dsm, 
                                    const unsigned int coords,
                                    const unsigned int nb_cols,
                                    const float thresh,
                                    const float nodata_value);

            bool isHorizontalDisturbed(float * dsm, 
                                        const unsigned int coords,
                                        const float thresh,
                                        const float nodata_value);

            bool isDiag1Disturbed(float * dsm, 
                                    const unsigned int coords,
                                    const unsigned int nb_cols,
                                    const float thresh,
                                    const float nodata_value);

            bool isDiag2Disturbed(float * dsm, 
                                    const unsigned int coords,
                                    const unsigned int nb_cols,
                                    const float thresh,
                                    const float nodata_value);
        
    };

} // end of namespace bulldozer

#endif