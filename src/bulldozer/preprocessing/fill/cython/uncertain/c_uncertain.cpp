#include "c_uncertain.h"

namespace bulldoproto {

	// max_slope_percent => 100m
	// z => distance
	// z = max_slope_percent * distance / 100.0

	float distance(long int r1, 
				   long int c1, 
				   long int r2,
				   long int c2) {
		return static_cast<float>( sqrt( (r1-r2) * (r1-r2) + (c1-c2)*(c1-c2) ) );
	}

	void prefill_uncertain(float * dsm,
						   unsigned char * uncertain_mask,
						   unsigned char * regular_mask,
						   float * prefilled_dsm,
						   float * uncertain_map,
						   unsigned int rows,
						   unsigned int cols,
						   unsigned int search_radius,
						   float max_slope_percent,
						   float dsm_resolution) {

		long int start_row, start_col, end_row, end_col;
		unsigned int coords, neigh_coords;
		float min_z;
		long int min_r, min_c;

		for(long int r = 0; r < static_cast<long int>(rows); r++){
			for(long int c = 0; c < static_cast<long int>(cols); c++ ) {

				coords = r * cols + c;

				if(uncertain_mask[coords] > 0){

					// It is an uncertain pixel

					start_row = std::max<long int>(0, r - search_radius);
					start_col = std::max<long int>(0, c - search_radius);
					end_row = std::min<long int>(rows - 1, r + search_radius);
					end_col = std::min<long int>(cols - 1, c + search_radius);

					min_z = 8849.0; // guess what is this height ;)
					min_r = -1;
					min_c = -1;

					for(long int neigh_r = start_row; neigh_r <= end_row; neigh_r++) {
						for(long int neigh_c = start_col; neigh_c <= end_col; neigh_c++){
							neigh_coords = neigh_r * cols + neigh_c;
							if(uncertain_mask[neigh_coords] < 1 &&
							   regular_mask[neigh_coords] > 0 && 
							   min_z > dsm[neigh_coords]){
								// It is certain and has a min z
								min_z = dsm[neigh_coords];
								min_r  = neigh_r;
								min_c = neigh_c;
							}
						}
					}

					if(min_r > -1){
						prefilled_dsm[coords] = min_z;
						uncertain_map[coords] = distance(r, c, min_r, min_c) * dsm_resolution * max_slope_percent / 100.0;
					}
				} else {
					prefilled_dsm[coords] = dsm[coords];
				}
			}
		}

	}

} // end of namespace bulldoproto