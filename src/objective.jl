# l(X,U) = (x-xT)'QT(x-xT) + h Σ {(x-xt)'Qt(x-xt) + (u-ut)'Rt(u-ut) + c}

abstract type Objective end

mutable struct QuadraticTrackingObjective <: Objective
    Q
    R
    c
    x_ref
    u_ref
end

function objective(Z,l::QuadraticTrackingObjective,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c

    s = 0
    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]

        s += h*((x-x_ref[t])'*Q[t]*(x-x_ref[t]) + (u-u_ref[t])'*R[t]*(u-u_ref[t]) + c)
    end
    x = view(Z,idx.x[T])
    s += (x-x_ref[T])'*Q[T]*(x-x_ref[T])

    return s
end

function objective_gradient!(∇l,Z,l::QuadraticTrackingObjective,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]

        ∇l[idx.x[t]] = 2.0*h*Q[t]*(x-x_ref[t])
        ∇l[idx.u[t]] = 2.0*h*R[t]*(u-u_ref[t])
        ∇l[idx.h[t]] = (x-x_ref[t])'*Q[t]*(x-x_ref[t]) + (u-u_ref[t])'*R[t]*(u-u_ref[t]) + c

    end
    x = view(Z,idx.x[T])
    ∇l[idx.x[T]] = 2.0*Q[T]*(x-x_ref[T])

    return nothing
end
