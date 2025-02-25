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
                          unsigned int nbRows,
                          unsigned int nbCols,
                          float thresh,
                          float nodataValue) {

		const long int x_size = nbCols;
		const long int y_size = nbRows;


		const int nbNeigbhors=8;
		std::ptrdiff_t v8_off[nbNeigbhors] = {-x_size-1, -x_size,  -x_size+1, -1, +1, x_size-1, x_size,  x_size+1 };

		float sum;
		float used;
		std::ptrdiff_t pos;
        std::ptrdiff_t pos_off;


		for (long int y=0; y<y_size; y++) {
			for (long int x=0; x<x_size; x++) {

				pos = x_size*y + x;

				// compute slope on dsm
				sum = 0;
				used = 0;

				if (dsm[pos] != nodataValue) {
					for(int v=0; v<nbNeigbhors; v++) {
                        
                        pos_off = pos+v8_off[v];
                            
                        if(pos_off>=0 && pos_off<x_size*y_size) {
                        
                            if(dsm[pos_off] != nodataValue) {
                                sum += std::fabs(dsm[pos_off] - dsm[pos]);
                                used++;
                            }
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
