```@meta
CurrentModule = Riemann1D
```

# Riemann1D.jl

Exact and approximate Riemann solvers for the one-dimensional compressible
Euler equations, written in Julia.

## Overview

The Riemann problem is the initial-value problem for the Euler equations
with piecewise-constant initial data separated by a single discontinuity:

```math
\mathbf{W}(x, 0) = \begin{cases}
\mathbf{W}_L & x < 0 \\
\mathbf{W}_R & x > 0
\end{cases}
```

The exact solution consists of three waves (left nonlinear, contact, right
nonlinear) separating four constant states. This package implements the
exact iterative solver described in Toro (2009).

## Quick start

```@repl
using Riemann1D

# Sod shock tube
W_L = PrimitiveState(ρ=1.0, u=0.0, p=1.0)
W_R = PrimitiveState(ρ=0.125, u=0.0, p=0.1)
eos = PerfectGasEOS(γ=1.4)

sol = solve_Riemann_problem(W_L, W_R, eos)
sol.p★, sol.u★

# Sample the solution at (x, t)
sample_solution(0.2, 0.1, sol)
```

## References

- Toro, E. F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd ed., Springer, 2009.

## Index

```@index
```
