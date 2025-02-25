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
 
#include "c_border.h"

namespace bulldozer {

    void buildBorderNodataMask(float * dsm,
                               unsigned char * borderNodataMask,
                               unsigned int nbRows,
                               unsigned int nbCols,
                               float nodataValue) {

        unsigned int col;
        for(unsigned int row = 0; row < nbRows; row++) {
            // extracts border nodata for the left side of the input DSM
            col = row * nbCols;
            while(col < ((row * nbCols)-1 + nbCols) && dsm[col] == nodataValue){
                borderNodataMask[col] = true;
                col++;
            }
            // extracts border nodata for the right side of the input DSM
            col = row * nbCols + nbCols - 1;
            while(col > row * nbCols && dsm[col] == nodataValue){
                borderNodataMask[col] = true;
                col--;
            }
        }
    }
}