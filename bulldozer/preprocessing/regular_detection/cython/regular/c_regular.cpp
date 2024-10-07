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


} // end of namespace bulldoproto