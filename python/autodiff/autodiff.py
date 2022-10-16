from math import log, sin, cos, tan

# Some codes taken from https://github.com/karpathy/micrograd
class Variable:
    def __init__(self, f, children=(), op=""):
        self.f = f
        self.d = 0
        self.op = op
        self.backwardfn = lambda: None
        self.parents = set(children)

    def __repr__(self):
        return f"Variable(data={self.f}, grad={self.d}, op={self.op})"

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
            self.d -= out.d
            other.d -= out.d

        out.backwardfn = backwardfn

        return out

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.f * other.f, (self, other), "*")

        def backwardfn():
            self.d += other.f * out.d
            other.d += self.f * out.d

        out.backwardfn = backwardfn

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Variable(self.f**other, (self,), f"**{other}")

        def backwardfn():
            self.d += (other * self.f ** (other - 1)) * out.d

        out.backwardfn = backwardfn

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __rmul__(self, other):
        return self * other

    def log(self):
        out = Variable(log(self.f), (self,), "log")

        def backwardfn():
            self.d += 1 / self.f * out.d

        out.backwardfn = backwardfn
        return out

    def sin(self):
        out = Variable(sin(self.f), (self,), "sin")

        def backwardfn():
            self.d += cos(self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def cos(self):
        out = Variable(cos(self.f), (self,), "cos")

        def backwardfn():
            self.d += -sin(self.f) * out.d

        out.backwardfn = backwardfn

        return out

    def tan(self):
        out = Variable(tan(self.f), (self,), "tan")

        def backwardfn():
            self.d += 2 / (1 + cos(2 * self.f))

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
                label="{f %.3f} | {d %.3f}" % (n.f, n.d),
                shape="record",
            )
            if n.op:
                dot.node(name=str(id(n)) + n.op, label=n.op)
                dot.edge(str(id(n)), str(id(n)) + n.op)
        for n1, n2 in edges:
            dot.edge(str(id(n2)) + n2.op, str(id(n1)))
        return dot
