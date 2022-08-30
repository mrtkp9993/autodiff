#include "pch.h"
#include "../autodiff_lib/autodiff_lib.cpp"

TEST(TestCaseName, TestName) {
	Dual x(5, 1);
	Dual y(6);
	Dual f = pow(x, 2) * y;
	EXPECT_EQ(60, f.dual);
}
