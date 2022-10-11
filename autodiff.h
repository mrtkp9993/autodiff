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

std::ostream &operator<<(std::ostream &os, const Dual &d) {
  os << "Derivative: " << d.dual;
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

#endif  // AUTODIFF_AUTODIFF_H
