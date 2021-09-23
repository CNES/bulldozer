#include "DisturbedAreas.h"
#include <iostream>

namespace bulldozer 
{


    DisturbedAreas::DisturbedAreas(): m_IsFourConnexity(true){}
    DisturbedAreas::DisturbedAreas(bool is_four_connexity): m_IsFourConnexity(is_four_connexity){}

    inline bool DisturbedAreas::isVerticalDisturbed(float * dsm, 
                                                    const unsigned int coords,
                                                    const unsigned int nb_cols,
                                                    const float thresh) {
        if (dsm[coords-nb_cols] != -32768 && dsm[coords+nb_cols] != -32768){
            return abs(dsm[coords-nb_cols] - dsm[coords]) > thresh || abs(dsm[coords] - dsm[coords+nb_cols]) > thresh;
        } else if (dsm[coords-nb_cols] == -32768){
            return abs(dsm[coords] - dsm[coords+nb_cols]) > thresh;
        } else if (dsm[coords+nb_cols] == -32768){
            return abs(dsm[coords-nb_cols] - dsm[coords]) > thresh;
        } else {
            return true;
        }

    inline bool DisturbedAreas::isHorizontalDisturbed(float * dsm, 
                                                    const unsigned int coords,
                                                    const float thresh) {
        if (dsm[coords-1] != -32768 && dsm[coords+1] != -32768){
            return abs(dsm[coords-1] - dsm[coords]) > thresh || abs(dsm[coords] - dsm[coords+1]) > thresh;
        } else if (dsm[coords-1] == -32768){
            return abs(dsm[coords] - dsm[coords+1]) > thresh;
        } else if (dsm[coords+1] == -32768){
            return abs(dsm[coords-1] - dsm[coords]) > thresh;
        } else {
            return true;
        }
    }

    inline bool DisturbedAreas::isDiag1Disturbed(float * dsm, 
                                                const unsigned int coords,
                                                const unsigned int nb_cols,
                                                const float thresh) {
        if (dsm[coords-nb_cols-1] != -32768 && dsm[coords+nb_cols+1] != -32768){
            return abs(dsm[coords-nb_cols-1] - dsm[coords]) > std::sqrt(2) * thresh || abs(dsm[coords] - dsm[coords+nb_cols+1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords-nb_cols-1] == -32768){
            return abs(dsm[coords] - dsm[coords+nb_cols+1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords+nb_cols+1] == -32768){
            return abs(dsm[coords-nb_cols-1] - dsm[coords]) > std::sqrt(2) * thresh;
        } else {
            return true;
        }
    }

    inline bool DisturbedAreas::isDiag2Disturbed(float * dsm, 
                                                const unsigned int coords,
                                                const unsigned int nb_cols,
                                                const float thresh) {
        if (dsm[coords-nb_cols+1] != -32768 && dsm[coords+nb_cols-1] != -32768){
            return abs(dsm[coords-nb_cols+1] - dsm[coords]) > std::sqrt(2) * thresh || abs(dsm[coords] - dsm[coords+nb_cols-1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords-nb_cols+1] == -32768){
            return abs(dsm[coords] - dsm[coords+nb_cols-1]) > std::sqrt(2) * thresh;
        } else if (dsm[coords+nb_cols-1] == -32768){
            return abs(dsm[coords-nb_cols+1] - dsm[coords]) > std::sqrt(2) * thresh;
        } else {
            return true;
        }
    }

    void DisturbedAreas::build_disturbance_mask(float * dsm,
                                                bool * disturbance_mask,
                                                unsigned int nb_rows,
                                                unsigned int nb_cols,
                                                float thresh) {

        if(m_IsFourConnexity) {

            std::cout << "ok " << nb_rows << " " << nb_cols << std::endl;

            for(unsigned int r = 1; r < nb_rows -2; r++) {
                for(unsigned int c = 1; c < nb_cols -2; c++) {
                    const unsigned int coords = r * nb_cols + c;
                    if(dsm[coords] != -32768){
                        disturbance_mask[coords] = isVerticalDisturbed(dsm, coords, nb_cols, thresh) 
                                                    && isHorizontalDisturbed(dsm, coords, thresh);
                    }
                }
            }
        }
        else {
            for(unsigned int r = 1; r < nb_rows -2; r++) {
                for(unsigned int c = 1; c < nb_cols -2; c++) {
                    const unsigned int coords = r * nb_cols + c;
                    if(dsm[coords] != -32768){
                        disturbance_mask[coords] = isVerticalDisturbed(dsm, coords, nb_cols, thresh) && 
                                                isHorizontalDisturbed(dsm, coords, thresh) &&
                                                isDiag1Disturbed(dsm, coords, nb_cols, thresh) &&
                                                isDiag2Disturbed(dsm, coords, nb_cols, thresh);
                    }
                }
            }
        }
                                                    
    }

}

