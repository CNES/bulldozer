/*Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).

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

		const long int xSize = nbCols;
		const long int ySize = nbRows;


		const int nbNeigbhors=8;
		std::ptrdiff_t v8Off[nbNeigbhors] = {-xSize-1, -xSize,  -xSize+1, -1, +1, xSize-1, xSize,  xSize+1 };

		float sum;
		float used;
		std::ptrdiff_t pos;
        std::ptrdiff_t posOff;


		for (long int y=0; y<ySize; y++) {
			for (long int x=0; x<xSize; x++) {

				pos = xSize*y + x;

				// compute slope on dsm
				sum = 0;
				used = 0;

				if (dsm[pos] != nodataValue) {
					for(int v=0; v<nbNeigbhors; v++) {
                        
                        posOff = pos+v8Off[v];
                            
                        if(posOff>=0 && posOff<xSize*ySize) {
                        
                            if(dsm[posOff] != nodataValue) {
                                sum += std::fabs(dsm[posOff] - dsm[pos]);
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
