#ifndef C_REGULAR_H
#define C_REGULAR_H

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

#include <iostream>
#include <cmath>
#include <map>
#include <limits>
#include <fstream>
#include <algorithm>

/*
    This function is used to build a mask that flags the pixels considered as "regular" meaning not disturbed from an altimetric perspective.
*/
namespace bulldozer
{
    void buildRegularMask(float * dsm,
                          unsigned char * regularMask,
                          unsigned int nbRows,
                          unsigned int nbCols,
                          float thresh,
                          float nodataValue);

} // end of namespace bulldozer

#endif