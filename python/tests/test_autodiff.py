from numpy.testing import assert_almost_equal

from autodiff.autodiff import Variable


class TestClass:
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
