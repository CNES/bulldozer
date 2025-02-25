#ifndef C_FILL_H
#define C_FILL_H

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

#include <cmath>
#include <memory>  
#include <iostream>

/*
    This function is used to fill the inner nodata of the input DSM.
*/
namespace bulldozer
{
    void iterativeFilling(float * dsm,
                          unsigned char * borderNodataMask,
                          int dsmHeight,
                          int dsmWidth,
                          float nodataVal,
                          int numIterations);

} // end of namespace bulldozer

#endif