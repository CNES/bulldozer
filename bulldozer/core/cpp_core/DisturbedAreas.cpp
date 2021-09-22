/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.
*/
#include "DisturbedAreas.h"

namespace bulldozer {

    DisturbedAreas::DisturbedAreas()
    {
    }

    void DisturbedAreas::build_disturbance_mask(float * dsm_strip,
                                        bool * disturbance_mask,
                                        unsigned int nb_rows,
                                        unsigned int nb_cols,
                                        float slope_treshold,
                                        unsigned int disturbed_treshold,
                                        float disturbed_influence_distance,
                                        float dsm_resolution)
    {
        // disturbed_areas will contain nb_rows vectors and each vector will contains k_row vector of disturbed columns
        std::vector<std::vector<std::vector<int>>> disturbed_areas;
        //TODO CHECK
        disturbed_areas.reserve(nb_rows);
        // number of disturbed areas for a given row
        unsigned int nb_disturbed_areas;
        // column iterator
        unsigned int col;
        // slope between two consecutive pixels
        float slope;
                        
        // loop over each row of the input DSM strip
        // /!\ input DSM elevation array is flattened
        for (unsigned int row = 0; row < nb_rows; row++)
        {
            //TODO CHECK TYPE std::vector<unsigned int> => std::vector<std::vector<unsigned int>>
            // disturbed areas for a given row
            std::vector<unsigned int> da_row;
            //TODO CHECK
            // each disturbed area required at least 3 pixels : 2 for the slope and 1 for a space
            // it's not necessary to reserve more than nb_cols/3 memory blocks 
            disturbed_areas.reserve(nb_cols/3 + 1);
            disturbed_areas.push_back(da_row);

            col = 1;
            // loop over each column of the current DSM strip row
            while(col < nb_cols)
            {
                // if the slope between two consecutive pixel is greater than the treshold => disturbed area detected and stored
                slope = abs(dsm_strip[(row * nb_cols) + col] - dsm_strip[(row * nb_cols) + col - 1]);
                if(slope >= slope_treshold)
                {
                    nb_disturbed_areas++;
                    da_row.push_back(std::vector<unsigned int>());
                    da_row.back().reserve(nb_cols - col + 1);
                    da_row.back().push_back(col);
                    da_row.back().push_back(col - 1);
                    col++;
                    // case of disturbed area bigger than just two consecutive diturbed pixels
                    // while the disturbed area is not finished, keep storing the disturbed pixels
                    while(col < nb_cols)
                    {
                       slope = abs(dsm_strip[(row * nb_cols) + col] - dsm_strip[(row * nb_cols) + col - 1]); 
                       if(slope >= slope_treshold)
                        {
                            row_lists.back().push_back(col);
                            col++;
                        }
                        else
                        {
                            col++;
                            break;
                        }
                    }
                    row_lists.back().resize(row_lists.back().size());
                } 
                else
                {
                    col++;
                }
            }
        }
        // merged_disturbed_areas will contain row lists which will contain p_row lists with p_row <= k_row. The p_row lists
        // contain one or more disturbed cols from the k_row lists 
        std::vector<std::vector<unsigned int>> merged_disturbed_areas;
        //TODO CHECK
        disturbed_areas.reserve(nb_rows);

        // retrieve the distance influence in pixel unit
        float distance_treshold = disturbed_influence_distance / dsm_resolution;
        
        // keep track of the maximum distance of a merged disturbed area
        unsigned int max_distance = 0;
        unsigned int nb_merged_areas;
        // merged areas iterator
        unsigned int merged_areas_it;

        /*
        TODO : Remove
        for (auto const& row : input_data) {
            std::cout << i.name;
        */        
        for(unsigned int da_row = 0; da_row < nb_rows; da_row++)
        {
            nb_merged_areas = 0;
            // merged areas iterator
            merged_areas_it = 0;
            // merged disturbed areas list for a giving row
            std::vector<unsigned int> merged_area_row;
            while(merged_areas_it < da_row.size())
            {
                // if the number of successive disturbed pixels along a row is lower than the threshold
                // then this sequence of pixels is considered as a disturbed area 
                if(da_row[merged_areas_it].size() >= disturbed_treshold)
                {
                    
                }
                else
                {
                    merged_areas_it++;
                }
            }
        }

        // merging disturbed areas
        for(unsigned int row = 0; row < disturbance_mask.size(); row++)
        {
            if(merged_disturbed_areas[row].size() > 0)
            {
                for (unsigned int area = 0; area < merged_disturbed_areas[row].size(); area++)
                {
                    for(unsigned int col; col < merged_disturbed_areas[row][area].size(); col++)
                    {
                        //TODO Check
                        disturbance_mask[row * nb_cols + col] = true;
                    }
                }
            }
        }
    }