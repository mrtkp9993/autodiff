from autodiff.autodiff import Variable
from numpy.testing import assert_almost_equal


class TestClass:
    def test_one(self):
        x1 = Variable(2)
        x2 = Variable(5)

        y = x1.log() + x1 * x2 - x2.sin()
        y.backward()
        assert_almost_equal(y.f, 11.652071, decimal=6)
