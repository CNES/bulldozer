#ifndef BORDER_H
#define BORDER_H

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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>

/*
    This function is used to build a mask that flags the pixels associated to the border nodata of the input DSM.
    Those no data areas appears on the edges if the DSM is skewed or detoured (e.g. when part of the sea has been removed with a water mask).
*/
namespace bulldozer 
{
    void buildBorderNodataMask(float * dsm,
                                unsigned char * borderNodataMask,
                                unsigned int nbRows,
                                unsigned int nbCols,
                                float nodataValue);
} // end of namespace bulldozer

#endif