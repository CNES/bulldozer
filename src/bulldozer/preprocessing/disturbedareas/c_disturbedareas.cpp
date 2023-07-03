
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

#include "c_disturbedareas.h"

namespace bulldozer 
{


    DisturbedAreas::DisturbedAreas(): m_IsFourConnexity(true){}
    DisturbedAreas::DisturbedAreas(bool is_four_connexity): m_IsFourConnexity(is_four_connexity){}

    inline bool DisturbedAreas::isVerticalDisturbed(float * dsm, 
                                                    const unsigned int coords,
                                                    const unsigned int nb_cols,
                                                    const float thresh,
                                                    const float nodata_value) {
        if (dsm[coords-nb_cols] != nodata_value && dsm[coords+nb_cols] != nodata_value){
            return abs(dsm[coords-nb_cols] - dsm[coords]) > thresh || abs(dsm[coords] - dsm[coords+nb_cols]) > thresh;
        } else if (dsm[coords-nb_cols] == nodata_value){
            return abs(dsm[coords] - dsm[coords+nb_cols]) > thresh;
        } else if (dsm[coords+nb_cols] == nodata_value){
            return abs(dsm[coords-nb_cols] - dsm[coords]) > thresh;
        } else {
            return true;
        }
    }

    inline bool DisturbedAreas::isHorizontalDisturbed(float * dsm, 
                                                    const unsigned int coords,
                                                    const float thresh,
                                                    const float nodata_value) {
        if (dsm[coords-1] != nodata_value && dsm[coords+1] != nodata_value){
            return abs(dsm[coords-1] - dsm[coords]) > thresh || abs(dsm[coords] - dsm[coords+1]) > thresh;
        } else if (dsm[coords-1] == nodata_value){
            return abs(dsm[coords] - dsm[coords+1]) > thresh;
        } else if (dsm[coords+1] == nodata_value){
            return abs(dsm[coords-1] - dsm[coords]) > thresh;
        } else {
            return true;
        }
    }

    inline bool DisturbedAreas::isDiag1Disturbed(float * dsm, 
                                                const unsigned int coords,
                                                const unsigned int nb_cols,
                                                const float thresh,
                                                const float nodata_value) {
        if (dsm[coords-nb_cols-1] != nodata_value && dsm[coords+nb_cols+1] != nodata_value){
            return abs(dsm[coords-nb_cols-1] - dsm[coords]) > std::sqrt(2) * thresh || abs(dsm[coords] - dsm[coords+nb_cols+1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords-nb_cols-1] == nodata_value){
            return abs(dsm[coords] - dsm[coords+nb_cols+1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords+nb_cols+1] == nodata_value){
            return abs(dsm[coords-nb_cols-1] - dsm[coords]) > std::sqrt(2) * thresh;
        } else {
            return true;
        }
    }

    inline bool DisturbedAreas::isDiag2Disturbed(float * dsm, 
                                                const unsigned int coords,
                                                const unsigned int nb_cols,
                                                const float thresh,
                                                const float nodata_value) {
        if (dsm[coords-nb_cols+1] != nodata_value && dsm[coords+nb_cols-1] != nodata_value){
            return abs(dsm[coords-nb_cols+1] - dsm[coords]) > std::sqrt(2) * thresh || abs(dsm[coords] - dsm[coords+nb_cols-1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords-nb_cols+1] == nodata_value){
            return abs(dsm[coords] - dsm[coords+nb_cols-1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords+nb_cols-1] == nodata_value){
            return abs(dsm[coords-nb_cols+1] - dsm[coords]) > std::sqrt(2) * thresh;
        } else {
            return true;
        }
    }

    void DisturbedAreas::build_disturbance_mask(float * dsm,
                                                unsigned char * disturbance_mask,
                                                unsigned int nb_rows,
                                                unsigned int nb_cols,
                                                float thresh,
                                                float nodata_value) {

        if(m_IsFourConnexity) {
            for(unsigned int r = 1; r < nb_rows - 1; r++) {
                for(unsigned int c = 1; c < nb_cols - 1; c++) {
                    const unsigned int coords = r * nb_cols + c;
                    if(dsm[coords] != nodata_value){
                        disturbance_mask[coords] = isVerticalDisturbed(dsm, coords, nb_cols, thresh, nodata_value) 
                                                    && isHorizontalDisturbed(dsm, coords, thresh, nodata_value);
                    }
                }
            }
        }
        else {
            for(unsigned int r = 1; r < nb_rows - 1; r++) {
                for(unsigned int c = 1; c < nb_cols - 1; c++) {
                    const unsigned int coords = r * nb_cols + c;
                    if(dsm[coords] != nodata_value){
                        disturbance_mask[coords] = isVerticalDisturbed(dsm, coords, nb_cols, thresh, nodata_value) && 
                                                isHorizontalDisturbed(dsm, coords, thresh, nodata_value) &&
                                                isDiag1Disturbed(dsm, coords, nb_cols, thresh, nodata_value) &&
                                                isDiag2Disturbed(dsm, coords, nb_cols, thresh, nodata_value);
                    }
                }
            }
        }                                        
    }
}

