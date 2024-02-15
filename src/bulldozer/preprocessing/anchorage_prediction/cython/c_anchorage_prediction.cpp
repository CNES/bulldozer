#include "c_anchorage_prediction.h"

namespace bulldoproto {

    void predict_anchorage(float * dsm,
                           float nodata,
                           unsigned char * regular_mask,
                           unsigned char * anchorage_mask,
                           unsigned int nb_rows,
                           unsigned int nb_cols,
                           float max_object_size) {

        // Compute the tiles
        // sqrt(2) factor because tiles are square and also factor 2 because of Shannon
        // 64 mètres
        //const unsigned int tile_size = static_cast<unsigned int>(max_object_size * sqrt(2));
        const unsigned int tile_size = static_cast<unsigned int>(128 * sqrt(2));
        
        unsigned int nb_tiles_x = nb_cols / tile_size;
        unsigned int remainder_x = nb_cols % tile_size;
        // Automatic floor operation when division of unsigned integers
        unsigned int offset_x = remainder_x / 2;
        unsigned int nb_tiles_y = nb_rows / tile_size;
        unsigned int remainder_y = nb_rows % tile_size;
        unsigned int offset_y = remainder_y / 2;

        unsigned int coords;

        for(unsigned int ty = 0; ty < nb_tiles_y; ty++){
            for(unsigned int tx = 0; tx < nb_tiles_x; tx++){

                // Image coordinates of the tile frame
                const unsigned int start_x = tx * tile_size + offset_x;
                const unsigned int start_y = ty * tile_size + offset_y;
                const unsigned int end_x = start_x + tile_size - 1;
                const unsigned int end_y = start_y + tile_size - 1;

                // Cache the min z
                float min_z = 9999.0;
                long int min_coords = -1;

                for(unsigned int r = start_y; r <= end_y; r++){
                    for(unsigned int c = start_x; c <= end_x; c++){
                        coords = r * nb_cols + c;
                        if(dsm[coords] < min_z &&
                           regular_mask[coords] > 0 &&
                           dsm[coords] != nodata){
                            min_z = dsm[coords];
                            min_coords = coords;
                        }
                    }
                }

                if (min_coords > - 1) {
                    anchorage_mask[min_coords] = 1;
                } 

            }
        }
    }


} // end of namespace bulldoproto