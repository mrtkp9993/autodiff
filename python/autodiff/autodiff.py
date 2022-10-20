import sys

sys.setrecursionlimit(10_000)
print("recursion limit:", sys.getrecursionlimit())

import math
import scipy.special

# Some codes taken from https://github.com/karpathy/micrograd
identifier = 1


class Variable:
    def __init__(self, f, children=(), op=""):
        global identifier
        self.__f = f
        self.__d = 0
        self.__op = op
        self.__backwardfn = lambda: None
        self.__parents = set(children)
        self.__name = "v" + str(identifier)
        identifier += 1

    def get_data(self):
        return self.__f

    def get_grad(self):
        return self.__d

    def get_op(self):
        return self.__op

    def get_name(self):
        return self.__name

    def __repr__(self):
        return f"Variable(data={self.get_data()}, grad={self.get_grad()}, op={self.get_op()}, id={self.get_name()})"

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.get_data() + other.get_data(), (self, other), "+")

        def __backwardfn():
            self.__d += out.get_grad()
            other.__d += out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.get_data() - other.get_data(), (self, other), "-")

        def __backwardfn():
            self.__d += out.get_grad()
            other.__d -= out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.get_data() * other.get_data(), (self, other), "*")

        def __backwardfn():
            self.__d += other.get_data() * out.get_grad()
            other.__d += self.get_data() * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers"
        out = Variable(self.get_data() ** other, (self,), f"**{other}")

        def __backwardfn():
            self.__d += (
                other * self.get_data() ** (other - 1)
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def log(self):
        out = Variable(math.log(self.get_data()), (self,), "log")

        def __backwardfn():
            self.__d += 1 / self.get_data() * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def logb(self, base):
        out = Variable(math.log(self.get_data(), base), (self,), f"log{base}")

        def __backwardfn():
            self.__d += (
                -math.log(base)
                / (self.get_data() * math.log(self.get_data()) ** 2)
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def exp(self):
        out = Variable(math.exp(self.get_data()), (self,), "exp")

        def __backwardfn():
            self.__d += math.exp(self.get_data()) * out.get_grad()

        out.__backwardfn = __backwardfn
        return out

    def root(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float roots"
        out = Variable(self.get_data() ** (1 / other), (self,), f"root {other}")

        def __backwardfn():
            self.__d += (
                (1 / other)
                * (self.get_data() ** (1 / other - 1))
                * out.get_grad()
            )

        out.__backwardfn = __backwardfn

        return out

    def abs(self):
        out = Variable(math.fabs(self.get_data()), (self,), "abs")

        def __backwardfn():
            self.__d += (
                math.fabs(self.get_data()) / self.get_data()
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def sin(self):
        out = Variable(math.sin(self.get_data()), (self,), "sin")

        def __backwardfn():
            self.__d += math.cos(self.get_data()) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def asin(self):
        out = Variable(math.asin(self.get_data()), (self,), "arcsin")

        def __backwardfn():
            self.__d += (
                1 / math.sqrt(1 - self.get_data() ** 2)
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def cos(self):
        out = Variable(math.cos(self.get_data()), (self,), "cos")

        def __backwardfn():
            self.__d += -math.sin(self.get_data()) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def acos(self):
        out = Variable(math.acos(self.get_data()), (self,), "arccos")

        def __backwardfn():
            self.__d += (
                -1 / math.sqrt(1 - self.get_data() ** 2)
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def tan(self):
        out = Variable(math.tan(self.get_data()), (self,), "tan")

        def __backwardfn():
            self.__d += (1 / math.cos(self.get_data()) ** 2) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def atan(self):
        out = Variable(math.atan(self.get_data()), (self,), "arctan")

        def __backwardfn():
            self.__d += (1 / (1 + self.get_data() ** 2)) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def sinh(self):
        out = Variable(math.sinh(self.get_data()), (self,), "sinh")

        def __backwardfn():
            self.__d += math.cosh(self.get_data()) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def asinh(self):
        out = Variable(math.asinh(self.get_data()), (self,), "arcsinh")

        def __backwardfn():
            self.__d += (
                1 / math.sqrt(1 + self.get_data() ** 2)
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def cosh(self):
        out = Variable(math.cosh(self.get_data()), (self,), "cosh")

        def __backwardfn():
            self.__d += math.sinh(self.get_data()) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def acosh(self):
        out = Variable(math.acosh(self.get_data()), (self,), "arccosh")

        def __backwardfn():
            self.__d += (
                1
                / (
                    math.sqrt(1 + self.get_data())
                    * math.sqrt(-1 + self.get_data())
                )
            ) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def tanh(self):
        out = Variable(math.tanh(self.get_data()), (self,), "tanh")

        def __backwardfn():
            self.__d += (1 / math.cosh(self.get_data()) ** 2) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def atanh(self):
        out = Variable(math.atanh(self.get_data()), (self,), "arctanh")

        def __backwardfn():
            self.__d += (1 / (1 - self.get_data() ** 2)) * out.get_grad()

        out.__backwardfn = __backwardfn

        return out

    def erf(self):
        out = Variable(math.erf(self.get_data()), (self,), "erf")

        def __backwardfn():
            self.__d += (
                2
                * math.exp(-math.pow(self.get_data(), 2))
                * (1 / math.sqrt(math.pi))
                * out.get_grad()
            )

        out.__backwardfn = __backwardfn

        return out

    def erfinv(self):
        out = Variable(scipy.special.erfinv(self.get_data()), (self,), "erfinv")

        def __backwardfn():
            self.__d += (
                0.5
                * math.exp(scipy.special.erfinv(self.get_data()) ** 2)
                * math.sqrt(math.pi)
                * out.get_grad()
            )

        out.__backwardfn = __backwardfn

        return out

    def erfc(self):
        out = Variable(math.erfc(self.get_data()), (self,), "erfc")

        def __backwardfn():
            self.__d -= (
                2
                * math.exp(-math.pow(self.get_data(), 2))
                * (1 / math.sqrt(math.pi))
                * out.get_grad()
            )

        out.__backwardfn = __backwardfn

        return out

    def erfcinv(self):
        out = Variable(
            scipy.special.erfcinv(self.get_data()), (self,), "erfcinv"
        )

        def __backwardfn():
            self.__d -= (
                0.5
                * math.exp(scipy.special.erfcinv(self.get_data()) ** 2)
                * math.sqrt(math.pi)
                * out.get_grad()
            )

        out.__backwardfn = __backwardfn

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.__parents:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.__d = 1
        for v in reversed(topo):
            print(v)
            v.__backwardfn()

    # https://mdrk.io/introduction-to-automatic-differentiation-part2/
    def draw_dag(self):
        from graphviz import Digraph

        def trace(self):
            nodes, edges = set(), set()

            def build(v):
                if v not in nodes:
                    nodes.add(v)
                    for parent in v.parents:
                        edges.add((parent, v))
                        build(parent)

            build(self)
            return nodes, edges

        nodes, edges = trace(self)
        dot = Digraph(format="png", graph_attr={"rankdir": "TB"})
        for n in nodes:
            if n.op == "":
                dot.node(
                    name=str(id(n)),
                    label="{Value %.3f} | {Grad %.3f} | {Id %s} | Input"
                    % (n.f, n.d, n.name),
                    shape="record",
                    _attributes={"color": "blue"},
                )
            else:
                dot.node(
                    name=str(id(n)),
                    label="{Value %.3f} | {Grad %.3f} | {Id %s}"
                    % (n.f, n.d, n.name),
                    shape="record",
                )
            if n.op:
                dot.node(name=str(id(n)) + n.op, label=n.op)
                dot.edge(str(id(n)), str(id(n)) + n.op)
        for n1, n2 in edges:
            dot.edge(str(id(n2)) + n2.op, str(id(n1)))
        return dot


def sin(x):
    return x.sin()


def asin(x):
    return x.asin()


def cos(x):
    return x.cos()


def acos(x):
    return x.acos()


def tan(x):
    return x.tan()


def atan(x):
    return x.atan()


def sinh(x):
    return x.sinh()


def asinh(x):
    return x.asinh()


def cosh(x):
    return x.cosh()


def acosh(x):
    return x.acosh()


def tanh(x):
    return x.tanh()


def atanh(x):
    return x.atanh()


def power(x, p):
    return x**p


def root(x, n):
    return x.root(n)


def exp(x):
    return x.exp()


def log(x):
    return x.log()


def logb(x, b):
    return x.logb(b)


def abs(x):
    return x.abs()
