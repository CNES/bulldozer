#include "c_stats.h"

namespace bulldoproto {

	void compute_stats(float * dsm,
                       unsigned char * invalid_mask,
                       float * stats,
                       unsigned int nb_rows,
                       unsigned int nb_cols,
                       float nodata) {
		
		// Min value to the Everest height
		stats[0] = 8849.0;
		// Max value to the Dead Sea height
		stats[1] = -430.0;

		const unsigned int nb_pixels = nb_rows * nb_cols;
		for(unsigned int p = 0; p < nb_pixels; p++){
			if(dsm[p] > nodata && invalid_mask[p] > 0){
				stats[0] = std::min(stats[0], dsm[p]);
				stats[1] = std::max(stats[1], dsm[p]);
			}
		}
	}

} // end of namespace bulldoproto