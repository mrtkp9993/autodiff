from autodiff.autodiff import Variable
from numpy.testing import assert_almost_equal


class TestClass:
    def test_arith(self):
        x = Variable(5)
        y = Variable(-1)
        z = Variable(7)

        f = ((x * y) - z) / (y * z)
        f.backward()
        assert_almost_equal(f.f, 1.71429, decimal=5)
