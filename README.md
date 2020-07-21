## DIRTREL.jl

This repository provides a basic implementation of [DIRTREL: Robust Trajectory Optimization with Ellipsoidal Disturbances and LQR Feedback](https://rexlab.stanford.edu/papers/dirtrel-auro.pdf) written in Julia.

DIRTREL finds locally optimal solutions to the robust trajectory optimization problem:
```
minimize        l(X,U) + lw(X,U)
  X,U,H
subject to      f(x+,x,u+,u,h) = 0
                h+ = h
                x1 = x(0)
                xT = x(tf)
                ul <= u <= uu
                xl <= x <= xu
                ul <= uw <= uu
                xl <= xw <= xu
                hl <= h <= hu.            
```
For simplicity of the implementation,
robust constraints are only implemented for state and control bounds.
Additionally, first-order-hold is implemented for controls.

## Installation
```code
git clone https://github.com/thowell/DIRTREL.jl
```

## Examples
Examples similar to the [pendulum](https://github.com/thowell/DIRTREL.jl/blob/master/examples/pendulum_robust.jl) and [cartpole](https://github.com/thowell/DIRTREL.jl/blob/master/examples/cartpole_robust.jl) from the paper are provided.

### Pendulum
![](examples/results/pendulum_state.png)
![](examples/results/pendulum_control.png)

### Cartpole
![](examples/results/cartpole_state.png)
![](examples/results/cartpole_control.png)

## TODO
- [X] add linear robust state bounds
- [X] first-order-hold controls
- [ ] add general robust constraints
- [ ] replace ForwardDiff with analytical derivatives
- [ ] compare results with SNOPT
