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

 #include "c_fill.h"

namespace bulldozer {

    void iterativeFilling(float * dsm,
                          unsigned char * disturbance_mask,
                          unsigned char * border_nodata_mask,
                          int dsm_h,
                          int dsm_w,
                          float nodata_val,
                          int num_iterations) {

        double diag_weight = 1 / sqrt(2);
        int corrected = 0, toCorrect = 0, nbPass = 0;
        bool hasNoData = true;

        std::unique_ptr<int[]> goods(new int[8]);
        std::unique_ptr<double[]> weights(new double[8]);
        std::unique_ptr<unsigned char[]> inv_msk(new unsigned char[dsm_h * dsm_w]);

        for (int i = 1; i < dsm_h - 1; ++i) {
            for (int j = 1; j < dsm_w - 1; ++j) {
                int idx = i * dsm_w + j;

                // Inversion logique : inv_msk = !(regular_mask)
                inv_msk[idx] = (disturbance_mask[idx] == 0) ? 1 : 0;

                // Si border_mask[i] == 1, alors inv_msk[i] = 0
                if (border_nodata_mask[idx] == 1) {
                    inv_msk[idx] = 0;
                }
            }
        }

        for (int k = 0; k < num_iterations; ++k) {
            toCorrect = 0;
            corrected = 0;
            hasNoData = false;

            for (int i = 1; i < dsm_h - 1; ++i) {
                for (int j = 1; j < dsm_w - 1; ++j) {
                    int idx = i * dsm_w + j;

                    if (inv_msk[idx] == 1) {
                        hasNoData = true;
                        ++toCorrect;

                        // Neighborhood checks
                        goods[0] = inv_msk[idx - dsm_w - 1] == 0 && dsm[idx - dsm_w - 1] != nodata_val;
                        goods[1] = inv_msk[idx - dsm_w] == 0 && dsm[idx - dsm_w] != nodata_val;
                        goods[2] = inv_msk[idx - dsm_w + 1] == 0 && dsm[idx - dsm_w + 1] != nodata_val;

                        goods[3] = inv_msk[idx - 1] == 0 && dsm[idx - 1] != nodata_val;
                        goods[4] = inv_msk[idx + 1] == 0 && dsm[idx + 1] != nodata_val;

                        goods[5] = inv_msk[idx + dsm_w - 1] == 0 && dsm[idx + dsm_w - 1] != nodata_val;
                        goods[6] = inv_msk[idx + dsm_w] == 0 && dsm[idx + dsm_w] != nodata_val;
                        goods[7] = inv_msk[idx + dsm_w + 1] == 0 && dsm[idx + dsm_w + 1] != nodata_val;

                        int goodNeighbor = goods[0] + goods[1] + goods[2] + goods[3] + goods[4] + goods[5] + goods[6] + goods[7];

                        if (goodNeighbor >= 3) {
                            ++corrected;

                            weights[0] = goods[0] * diag_weight;
                            weights[1] = goods[1];
                            weights[2] = goods[2] * diag_weight;
                            weights[3] = goods[3];
                            weights[4] = goods[4];
                            weights[5] = goods[5] * diag_weight;
                            weights[6] = goods[6];
                            weights[7] = goods[7] * diag_weight;

                            double totalWeight = 0.0;
                            double newValue = 0.0;

                            newValue += weights[0] * dsm[idx - dsm_w - 1];
                            newValue += weights[1] * dsm[idx - dsm_w];
                            newValue += weights[2] * dsm[idx - dsm_w + 1];
                            newValue += weights[3] * dsm[idx - 1];
                            newValue += weights[4] * dsm[idx + 1];
                            newValue += weights[5] * dsm[idx + dsm_w - 1];
                            newValue += weights[6] * dsm[idx + dsm_w];
                            newValue += weights[7] * dsm[idx + dsm_w + 1];

                            totalWeight = weights[0] + weights[1] + weights[2] + weights[3] + weights[4] + weights[5] + weights[6] + weights[7];
                            dsm[idx] = newValue / totalWeight;

                            inv_msk[idx] = 2;
                        }
                    }
                }
            }

            for (int i = 1; i < dsm_h - 1; ++i) {
                for (int j = 1; j < dsm_w - 1; ++j) {
                    int idx = i * dsm_w + j;
                    if (inv_msk[idx] == 2) {
                        inv_msk[idx] = 0;
                    }
                }
            }

            if (!hasNoData || corrected == 0) {
                break;
            }

            ++nbPass;
        }

    }
} // end of namespace bulldozer
