#include "c_regular.h"

namespace bulldoproto {


	
	void buildRegularMask(float * dsm,
						  unsigned char * regularMask,
                          unsigned int nb_rows,
                          unsigned int nb_cols,
                          float thresh,
                          float nodata_dsm) {

		const long int x_size = nb_cols;
		const long int y_size = nb_rows;


		const int nb_neigbhors=8;
		std::ptrdiff_t v8_off[nb_neigbhors] = {-x_size-1, -x_size,  -x_size+1, -1, +1, x_size-1, x_size,  x_size+1 };
		// const int nb_neigbhors=24;
		// std::ptrdiff_t v8_off[nb_neigbhors] = {-2*x_size-2, -2*x_size-1, -2*x_size, -2*x_size+1, -2*x_size+2, -x_size-2, -x_size-1, -x_size, -x_size+1, -x_size+2, -2, -1, +1, +2, x_size-2, x_size-1, x_size, x_size+1, x_size+2, 2*x_size-2, 2*x_size-1, 2*x_size, 2*x_size+1, 2*x_size+2};
		//std::ptrdiff_t v9_off[9] = {-x_size-1, -x_size,  -x_size+1, -1, 0, +1, x_size-1, x_size,  x_size+1 };
		//std::ptrdiff_t vHG_off[4] = {-x_size-1, -x_size,  -x_size+1, -1};

		float sum;
		float used;
		std::ptrdiff_t pos;


		for (long int y=1; y<y_size-1; y++) {
			for (long int x=1; x<x_size-1; x++) {

				pos = x_size*y + x;

				// compute slope on dsm
				sum = 0;
				used = 0;

				if (dsm[pos] > nodata_dsm) {
					for(int v=0; v<nb_neigbhors; v++) {
						if(dsm[pos+v8_off[v]] >  nodata_dsm) {
							sum += std::fabs(dsm[pos+v8_off[v]] - dsm[pos]);
							used++;
						}
					}
				}

				if (sum < used * thresh) {
					regularMask[pos] = 1;
				}
				else {
					regularMask[pos] = 0;
				}
			}
		}
	}

	void predict_anchorage_mask(unsigned char * regular_mask,
								unsigned char * anchorage_mask,
								unsigned int nb_rows,
                          		unsigned int nb_cols,
								unsigned int max_object_size) {
		
		const long int n_rows = nb_rows;
		const long int n_cols = nb_cols;


		long int min_r, max_r, min_c, max_c;
		long int pos, centered_pos;
		bool is_anchor = false;

		for(long int r = 0; r < n_rows; r++){
			for(long int c = 0; c < n_cols; c++){

				centered_pos = r * n_cols + c;

				if(regular_mask[centered_pos] > 0){
				
					min_r = std::max((long int)0, r - max_object_size);
					max_r = std::min(n_rows - 1, r + max_object_size);
					min_c = std::max((long int)0, c - max_object_size);
					max_c = std::min(n_cols - 1, c + max_object_size);

					is_anchor = true;
					for(long int nr = min_r; nr <= max_r; nr++){
						for(long int nc = min_c; nc <= max_c; nc++){
							pos = nr * n_cols + nc;
							if(regular_mask[pos] < 1){
								is_anchor = false;
								break;
							}
						}
						if(!is_anchor){
							break;
						}
					}

					if (is_anchor) {
						anchorage_mask[centered_pos] = 1;
					}
				}
			}
		}

	}

} // end of namespace bulldoproto