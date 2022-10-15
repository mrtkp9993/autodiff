#ifndef AUTODIFF_AUTODIFF_H
#define AUTODIFF_AUTODIFF_H

#include <cmath>
#include <iostream>

struct Dual {
  long double real;
  long double dual;

  Dual(const long double &a) {
    this->real = a;
    this->dual = 0;
  }

  Dual(const long double &a, const long double &b) {
    this->real = a;
    this->dual = b;
  }
};

Dual operator+(const Dual &u, const Dual &v) {
  return {u.real + v.real, u.dual + v.dual};
}

Dual operator-(const Dual &u, const Dual &v) {
  return {u.real - v.real, u.dual - v.dual};
}

Dual operator*(const Dual &u, const Dual &v) {
  return {u.real * v.real, u.dual * v.real + u.real * v.dual};
}

Dual operator/(const Dual &u, const Dual &v) {
  return {u.real / v.real,
          (u.dual * v.real - u.real * v.dual) / (v.real * v.real)};
}

bool operator>(const Dual &u, const Dual &v) { return (u.real > v.real); }

bool operator>=(const Dual &u, const Dual &v) { return (u.real >= v.real); }

bool operator<(const Dual &u, const Dual &v) { return (u.real < v.real); }

bool operator==(const Dual &u, const Dual &v) { return (u.real == v.real); }

bool operator!=(const Dual &u, const Dual &v) { return (u.real != v.real); }

std::ostream &operator<<(std::ostream &os, const Dual &d) {
  os << "" << d.real << " + " << d.dual << "ε" << std::endl;
  return os;
}

Dual pow(const Dual &d, const long double &p) {
  return {std::pow(d.real, p), p * d.dual * std::pow(d.real, p - 1)};
}

Dual root(const Dual &d, const long double &p) {
  return {std::pow(d.real, 1.0 / p),
          d.dual * 1 / (p * std::pow(std::pow(d.real, 1.0 / p), p - 1))};
}

Dual exp(const Dual &d) {
  return {std::exp(d.real), d.dual * std::exp(d.real)};
}

Dual log(const Dual &d) { return {std::log(d.real), d.dual / d.real}; }

Dual sin(const Dual &d) {
  return {std::sin(d.real), d.dual * std::cos(d.real)};
}

Dual cos(const Dual &d) {
  return {std::cos(d.real), -d.dual * std::sin(d.real)};
}

Dual tan(const Dual &d) {
  return {std::tan(d.real), d.dual / (std::pow(std::cos(d.real), 2))};
}

Dual sinh(const Dual &d) {
  return {std::sinh(d.real), d.dual * std::cosh(d.real)};
}

Dual cosh(const Dual &d) {
  return {std::cosh(d.real), d.dual * std::sinh(d.real)};
}

Dual tanh(const Dual &d) {
  return {std::tanh(d.real), d.dual / std::pow(std::cosh(d.real), 2)};
}

Dual abs(const Dual &d) {
  if (d.real > 0) {
    return d;
  } else {
    return Dual(-d.real, -d.dual);
  }
}

Dual ceil(const Dual &d) { return {std::ceil(d.real)}; }

Dual floor(const Dual &d) { return {std::floor(d.real)}; }

// Hyperdual implementation is not completed
struct HyperDual {
  long double real;
  long double dual1;
  long double dual2;
  long double dual1dual2;

  HyperDual(const long double &a) {
    this->real = a;
    this->dual1 = 0;
    this->dual2 = 0;
    this->dual1dual2 = 0;
  }

  HyperDual(const long double &a, const long double &b, const long double &c,
            const long double &d) {
    this->real = a;
    this->dual1 = b;
    this->dual2 = c;
    this->dual1dual2 = d;
  }
};

HyperDual operator+(const HyperDual &u, const HyperDual &v) {
  return {u.real + v.real, u.dual1 + v.dual1, u.dual2 + v.dual2,
          u.dual1dual2 + v.dual1dual2};
}

HyperDual operator-(const HyperDual &u, const HyperDual &v) {
  return {u.real - v.real, u.dual1 - v.dual1, u.dual2 - v.dual2,
          u.dual1dual2 - v.dual1dual2};
}

HyperDual operator*(const HyperDual &u, const HyperDual &v) {
  return {u.real * v.real, u.real * v.dual1 + u.dual1 * v.real,
          u.real * v.dual2 + u.dual2 * v.real,
          u.real * v.dual1dual2 + u.dual1 * v.dual2 + u.dual2 * v.dual1 +
              u.dual1dual2 * v.real};
}

HyperDual pow(const HyperDual &d, const long double &p) {
  return {std::pow(d.real, p), d.dual1 * p * std::pow(d.real, p - 1),
          d.dual2 * p * std::pow(d.real, p - 1),
          d.dual1dual2 * p * std::pow(d.real, p - 1) +
              p * (p - 1) * d.dual1 * d.dual2 * std::pow(d.real, p - 2)};
}

HyperDual operator/(const HyperDual &u, const HyperDual &v) {
  return (u * pow(v, -1.0));
}

bool operator>(const HyperDual &u, const HyperDual &v) {
  return (u.real > v.real);
}

bool operator>=(const HyperDual &u, const HyperDual &v) {
  return (u.real >= v.real);
}

bool operator<(const HyperDual &u, const HyperDual &v) {
  return (u.real < v.real);
}

bool operator==(const HyperDual &u, const HyperDual &v) {
  return (u.real == v.real);
}

bool operator!=(const HyperDual &u, const HyperDual &v) {
  return (u.real != v.real);
}

std::ostream &operator<<(std::ostream &os, const HyperDual &d) {
  os << "" << d.real << " + " << d.dual1 << "ε₁ + " << d.dual2 << "ε₂ + "
     << d.dual1dual2 << "ε₁ε₂" << std::endl;
  return os;
}

#endif  // AUTODIFF_AUTODIFF_H
