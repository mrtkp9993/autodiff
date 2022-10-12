#include "../autodiff.h"
#include "gtest/gtest.h"

namespace tests {
TEST(TestArith, TestArithDual) {
  Dual x(3, 1);
  Dual y(5);
  Dual f = x * (x / y) + y - 1;
  EXPECT_FLOAT_EQ(1.2, f.dual);
}

TEST(TestPow, TestPowDual) {
  Dual x(7);
  Dual y(1, 1);
  Dual f = pow(exp(root(log(x * y), 2)), 2);
  EXPECT_FLOAT_EQ(11.6703654, f.dual);
}

TEST(TestTrig, TestDualN) {
  Dual x(1, 1);
  Dual y(2);
  Dual f = (sin(x * y) + cos(y)) / tan(x * y);
  EXPECT_FLOAT_EQ(-0.8119769, f.dual);
}

TEST(TestTrig, TestDualArc) {
  Dual x(1, 1);
  Dual y(2);
  Dual f = (sinh(x * y) + cosh(y)) / tanh(x * y);
  EXPECT_FLOAT_EQ(6.68170281, f.dual);
}

TEST(TestAbs, TestAbsDual) {
  Dual x(5);
  Dual y(2, 1);
  Dual f = pow(abs(x * y) - y, 2);
  EXPECT_FLOAT_EQ(64, f.dual);
}

TEST(TestArith, TestHyperDual) {
  HyperDual x(7, 1, 0, 0);
  HyperDual y(3);
  HyperDual z(-2, 0, 1, 0);
  HyperDual f = ((x * y) / z) + (x * z) - 1;
  EXPECT_FLOAT_EQ(0.25, f.dual1dual2);
  EXPECT_FLOAT_EQ(-3.5, f.dual1);
  EXPECT_FLOAT_EQ(1.75, f.dual2);

  HyperDual x2(7, 1, 0, 0);
  HyperDual y2(3, 0, 1, 0);
  HyperDual z2(-2);
  HyperDual f2 = ((x2 * y2) / z2) + (x2 * z2) - 1;
  EXPECT_FLOAT_EQ(-0.5, f2.dual1dual2);
  EXPECT_FLOAT_EQ(-3.5, f2.dual1);
  EXPECT_FLOAT_EQ(-3.5, f2.dual2);

  HyperDual x3(7, 1, 1, 0);
  HyperDual y3(3);
  HyperDual z3(-2);
  HyperDual f3 = ((x3 * y3) / z3) + (x3 * z3) - 1;
  EXPECT_FLOAT_EQ(0, f3.dual1dual2);
  EXPECT_FLOAT_EQ(-3.5, f3.dual1);
}

}  // namespace tests