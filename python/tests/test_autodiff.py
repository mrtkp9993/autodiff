from numpy.testing import assert_almost_equal

from autodiff.autodiff import Variable


class TestClass:
    def test_arith(self):
        x = Variable(5)
        y = Variable(-1)
        z = Variable(7)

        f = ((x * y) - z) / (y * z)
        f.backward()
        assert_almost_equal(f.f, 1.71429, decimal=5)
        assert_almost_equal(x.d, 0.142857, decimal=6)
        assert_almost_equal(y.d, 1, decimal=1)
        assert_almost_equal(z.d, -0.102041, decimal=6)
