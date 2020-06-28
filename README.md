# DIRTREL.jl

This repository provides a basic implementation of [DIRTREL: Robust Trajectory Optimization with Ellipsoidal Disturbances and LQR Feedback](https://agile.seas.harvard.edu/files/agile/files/dirtrel.pdf) written in Julia.

The following robust trajectory optimization problem is solved,
```
minimize        lw(X,U) + l(X,U)
  X,U,H
subject to      x+ = f(x,u,h)
                u_l < u < u_u
                x_l < x < x_u
                u_l < uw < u_u
                x_l < xw < x_u
                h_l < h < h_u             
```
where for simplicity, all constraints (apart from dynamics) are linear bounds.
