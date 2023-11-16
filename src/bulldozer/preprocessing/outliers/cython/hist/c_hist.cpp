#include "c_hist.h"

namespace bulldoproto {

	void compute_hist(float * dsm,
					  unsigned char * invalid_mask,
					  unsigned int * hist,
					  float min_z,
					  float bin_width,
					  unsigned int nb_bins,
                      unsigned int nb_rows,
                      unsigned int nb_cols,
                      float nodata){
		
		const unsigned int nb_pixels = nb_rows * nb_cols;

		for(unsigned int p = 0; p < nb_pixels; p++){
			if(dsm[p] > nodata && invalid_mask[p] > 0){
				unsigned int hist_idx = static_cast<unsigned int>((dsm[p] - min_z) / bin_width);
				hist[hist_idx]++; 
			}
		}
	}

} // end of namespace bulldoproto