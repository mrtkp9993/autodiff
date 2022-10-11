#include "../autodiff.h"
#include "gtest/gtest.h"

namespace tests {
TEST(TestArith, TestArith1) {
  Dual x(3, 1);
  Dual y(5);
  Dual f = x * (x / y) + y - 1;
  EXPECT_FLOAT_EQ(1.2, f.dual);
}

TEST(TestPow, TestPow1) {
  Dual x(7);
  Dual y(1, 1);
  Dual f = pow(exp(root(log(x * y), 2)), 2);
  EXPECT_FLOAT_EQ(11.6703654, f.dual);
}

TEST(TestTrig, TestN) {
  Dual x(1, 1);
  Dual y(2);
  Dual f = (sin(x * y) + cos(y)) / tan(x * y);
  EXPECT_FLOAT_EQ(-0.8119769, f.dual);
}

TEST(TestTrig, TestArc) {
  Dual x(1, 1);
  Dual y(2);
  Dual f = (sinh(x * y) + cosh(y)) / tanh(x * y);
  EXPECT_FLOAT_EQ(6.68170281, f.dual);
}

TEST(TestAbs, TestAbs1) {
  Dual x(5);
  Dual y(2, 1);
  Dual f = pow(abs(x * y) - y, 2);
  EXPECT_FLOAT_EQ(64, f.dual);
}

}  // namespace tests