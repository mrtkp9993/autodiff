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
        self.f = f
        self.d = 0
        self.op = op
        self.backwardfn = lambda: None
        self.parents = set(children)
        self.name = "v" + str(identifier)
        identifier += 1

    def __repr__(self):
        return f"Variable(data={self.f}, grad={self.d}, op={self.op}, id={self.name})"

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.f + other.f, (self, other), "+")

        def backwardfn():
            self.d += out.d
            other.d += out.d

        out.backwardfn = backwardfn

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.f - other.f, (self, other), "-")

        def backwardfn():
            self.d += out.d
            other.d -= out.d

        out.backwardfn = backwardfn

        return out

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.f * other.f, (self, other), "*")

        def backwardfn():
            self.d += other.f * out.d
            other.d += self.f * out.d

        out.backwardfn = backwardfn

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers"
        out = Variable(self.f ** other, (self,), f"**{other}")

        def backwardfn():
            self.d += (other * self.f ** (other - 1)) * out.d

        out.backwardfn = backwardfn

        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def log(self):
        out = Variable(math.log(self.f), (self,), "log")

        def backwardfn():
            self.d += 1 / self.f * out.d

        out.backwardfn = backwardfn

        return out

    def logb(self, base):
        out = Variable(math.log(self.f, base), (self,), f"log{base}")

        def backwardfn():
            self.d += (- math.log(base) / (self.f * math.log(self.f) ** 2)) * out.d

        out.backwardfn = backwardfn

        return out

    def exp(self):
        out = Variable(math.exp(self.f), (self,), "exp")

        def backwardfn():
            self.d += math.exp(self.f) * out.d

        out.backwardfn = backwardfn
        return out

    def root(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float roots"
        out = Variable(self.f ** (1 / other), (self,), f"root {other}")

        def backwardfn():
            self.d += (1 / other) * (self.f ** (1 / other - 1)) * out.d

        out.backwardfn = backwardfn

        return out

    def abs(self):
        out = Variable(math.fabs(self.f), (self,), "abs")

        def backwardfn():
            self.d += (math.fabs(self.f) / self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def sin(self):
        out = Variable(math.sin(self.f), (self,), "sin")

        def backwardfn():
            self.d += math.cos(self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def asin(self):
        out = Variable(math.asin(self.f), (self,), "arcsin")

        def backwardfn():
            self.d += (1 / math.sqrt(1 - self.f ** 2)) * out.d

        out.backwardfn = backwardfn

        return out

    def cos(self):
        out = Variable(math.cos(self.f), (self,), "cos")

        def backwardfn():
            self.d += -math.sin(self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def acos(self):
        out = Variable(math.acos(self.f), (self,), "arccos")

        def backwardfn():
            self.d += (-1 / math.sqrt(1 - self.f ** 2)) * out.d

        out.backwardfn = backwardfn

        return out

    def tan(self):
        out = Variable(math.tan(self.f), (self,), "tan")

        def backwardfn():
            self.d += (1 / math.cos(self.f) ** 2) * out.d

        out.backwardfn = backwardfn

        return out

    def atan(self):
        out = Variable(math.atan(self.f), (self,), "arctan")

        def backwardfn():
            self.d += (1 / (1 + self.f ** 2)) * out.d

        out.backwardfn = backwardfn

        return out

    def sinh(self):
        out = Variable(math.sinh(self.f), (self,), "sinh")

        def backwardfn():
            self.d += math.cosh(self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def asinh(self):
        out = Variable(math.asinh(self.f), (self,), "arcsinh")

        def backwardfn():
            self.d += (1 / math.sqrt(1 + self.f ** 2)) * out.d

        out.backwardfn = backwardfn

        return out

    def cosh(self):
        out = Variable(math.cosh(self.f), (self,), "cosh")

        def backwardfn():
            self.d += math.sinh(self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def acosh(self):
        out = Variable(math.acosh(self.f), (self,), "arccosh")

        def backwardfn():
            self.d += (1 / (math.sqrt(1 + self.f) * math.sqrt(-1 + self.f))) * out.d

        out.backwardfn = backwardfn

        return out

    def tanh(self):
        out = Variable(math.tanh(self.f), (self,), "tanh")

        def backwardfn():
            self.d += (1 / math.cosh(self.f) ** 2) * out.d

        out.backwardfn = backwardfn

        return out

    def atanh(self):
        out = Variable(math.atanh(self.f), (self,), "arctanh")

        def backwardfn():
            self.d += (1 / (1 - self.f ** 2)) * out.d

        out.backwardfn = backwardfn

        return out

    def erf(self):
        out = Variable(math.erf(self.f), (self,), "erf")

        def backwardfn():
            self.d += 2 * math.exp(-math.pow(self.f, 2)) * (1 / math.sqrt(math.pi)) * out.d

        out.backwardfn = backwardfn

        return out

    def erfinv(self):
        out = Variable(scipy.special.erfinv(self.f), (self,), "erfinv")

        def backwardfn():
            self.d += 0.5 * math.exp(scipy.special.erfinv(self.f) ** 2) * math.sqrt(math.pi) * out.d

        out.backwardfn = backwardfn

        return out

    def erfc(self):
        out = Variable(math.erfc(self.f), (self,), "erfc")

        def backwardfn():
            self.d -= 2 * math.exp(-math.pow(self.f, 2)) * (1 / math.sqrt(math.pi)) * out.d

        out.backwardfn = backwardfn

        return out

    def erfcinv(self):
        out = Variable(scipy.special.erfcinv(self.f), (self,), "erfcinv")

        def backwardfn():
            self.d -= 0.5 * math.exp(scipy.special.erfcinv(self.f) ** 2) * math.sqrt(math.pi) * out.d

        out.backwardfn = backwardfn

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.d = 1
        for v in reversed(topo):
            print(v)
            v.backwardfn()

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
    return x ** p


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
