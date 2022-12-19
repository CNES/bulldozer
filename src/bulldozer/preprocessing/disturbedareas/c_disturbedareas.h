#ifndef C_DISTURBEDAREAS_H
#define C_DISTURBEDAREAS_H

/*Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).

This file is part of Bulldozer
(see https://github.com/CNES/bulldozer).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>
#include <cmath>

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
