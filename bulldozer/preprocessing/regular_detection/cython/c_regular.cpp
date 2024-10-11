 /*Copyright (c) 2022-2025 Centre National d'Etudes Spatiales (CNES).

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
 limitations under the License.*/
 
 #include "c_regular.h"

namespace bulldozer {

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

		float sum;
		float used;
		std::ptrdiff_t pos;


		for (long int y=1; y<y_size-1; y++) {
			for (long int x=1; x<x_size-1; x++) {

				pos = x_size*y + x;

				// compute slope on dsm
				sum = 0;
				used = 0;

				if (dsm[pos] != nodata_dsm) {
					for(int v=0; v<nb_neigbhors; v++) {
						if(dsm[pos+v8_off[v]]  != nodata_dsm) {
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


} // end of namespace bulldozer