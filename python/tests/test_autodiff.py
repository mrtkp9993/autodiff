from numpy.testing import assert_almost_equal

from autodiff.autodiff import *


class TestClass:
    def test_sanity(self):
        x1 = Variable(2)
        x2 = Variable(5)

        f = x1.log() + x1 * x2 - sin(x2)
        f.backward()

        assert_almost_equal(f.f, 11.652, decimal=3)
        assert_almost_equal(x1.d, 5.5, decimal=1)
        assert_almost_equal(x2.d, 1.716, decimal=3)

    def test_arith(self):
        x = Variable(5)
        y = Variable(-1)
        z = Variable(7)

        f = (y + (x * y) - z) / (y * z)
        f.backward()

        assert_almost_equal(f.f, 1.85714, decimal=5)
        assert_almost_equal(x.d, 0.142857, decimal=6)
        assert_almost_equal(y.d, 1, decimal=1)
        assert_almost_equal(z.d, -0.122449, decimal=6)

    def test_exp_log(self):
        x = Variable(1.1)
        y = Variable(1.3)

        f = ((x * y).exp() - x) / y.log()
        f.backward()

        assert_almost_equal(x.d, 16.8937, decimal=4)
        assert_almost_equal(y.d, -16.8846, decimal=4)

    def test_log_b(self):
        x = Variable(5)

        f = logb(x, 11)
        f.backward()

        assert_almost_equal(x.d, -0.185145, decimal=6)

    def test_abs(self):
        x = Variable(1)
        y = Variable(1)

        f = abs(x) * cos(x + y)
        f.backward()

        assert_almost_equal(x.d, -1.32544, decimal=5)

    def test_trig(self):
        x = Variable(1.1)
        y = Variable(1.7)

        f = (sin(x + y) - cos(x - y)) / (tan(x) * tan(y))
        f.backward()

        assert_almost_equal(x.d, 0.0194339, decimal=7)
        assert_almost_equal(y.d, 0.278753, decimal=6)

    def test_inv_trig(self):
        x = Variable(0.97)
        y = Variable(-0.02)

        f = (asin(x + y) - acos(x - y)) / (atan(x) - sin(x / y))
        f.backward()

        assert_almost_equal(x.d, 180.016, decimal=3)
        assert_almost_equal(y.d, 11739.8, decimal=1)

    def test_hyperbolic_trig(self):
        x = Variable(1)
        y = Variable(-1)

        f = (sinh(x + y) - cosh(x - y)) / tanh(x / y)
        f.backward()

        assert_almost_equal(x.d, 0.725099, decimal=6)
        assert_almost_equal(y.d, -8.79929, decimal=5)

    def test_hyperbolic_inv_trig(self):
        x = Variable(1.6)
        y = Variable(-80)

        f = (asinh(x + y) - acosh(x - y)) / atanh(x / y)
        f.backward()

        assert_almost_equal(x.d, -317.253, decimal=3)
        assert_almost_equal(y.d, -7.59489, decimal=5)

    def test_pow_root(self):
        x = Variable(8)
        y = Variable(2)

        f = (x + y) ** 3 - (x - y).root(2)
        f.backward()

        assert_almost_equal(x.d, 299.796, decimal=3)
        assert_almost_equal(y.d, 300.204, decimal=3)

    def test_erf(self):
        x = Variable(0.1)

        f = (x.erf() - x.erfc() * x.erfinv()) / x.erfcinv()
        f.backward()

        assert_almost_equal(x.d, 0.449368, decimal=6)
