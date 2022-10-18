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

    def test_trig(self):
        x = Variable(1.1)
        y = Variable(1.7)

        f = (sin(x + y) - cos(x - y)) / (tan(x) * tan(y))
        f.backward()

        assert_almost_equal(x.d, 0.0194339, decimal=7)
        assert_almost_equal(y.d, 0.278753, decimal=6)
