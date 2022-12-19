#ifndef BORDERNODATA_H
#define BORDERNODATA_H

/*
    Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
    This file is part of Bulldozer library
    All rights reserved.
*/

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>

/*
    This class is used to build a mask that flags the pixels associated to the border nodata of the input DSM.
    Those no data areas appears on the edges if the DSM is skewed or detoured (e.g. when part of the sea has been removed with a water mask).
*/
namespace bulldozer 
{

    class BorderNodata
    {
        public:

            BorderNodata();

            void buildBorderNodataMask(float * dsm_strip,
                                       unsigned char * disturbance_mask,
                                       unsigned int nb_rows,
                                       unsigned int nb_cols,
                                       float nodata_value);
    };

} // end of namespace bulldozer

#endif