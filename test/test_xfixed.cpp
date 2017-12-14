/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    using xfixed3x3 = xfixed<double, xt::fixed_shape<3, 3>>; 

    TEST(xfixed, basic)
    {
        xfixed3x3 a({{1,2,3}, {4,5,6}, {7,8,9}});
        xfixed3x3 b({{1,2,3}, {4,5,6}, {7,8,9}});

        xfixed3x3 res1 = a + b;
        xfixed3x3 res2 = 2.0 * a;

        EXPECT_EQ(res1, res2);
    }
}