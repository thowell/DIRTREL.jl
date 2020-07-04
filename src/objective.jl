# l(X,U) = (x-xT)'QT(x-xT) + h Σ {(x-xt)'Qt(x-xt) + (u-ut)'Rt(u-ut) + c}

abstract type Objective end

mutable struct QuadraticTrackingObjective <: Objective
    Q
    R
    c
    x_ref
    u_ref
end

function quadratic_cost(x,u,Q,R,x_ref,u_ref)
    (x-x_ref)'*Q*(x-x_ref) + (u-u_ref)'*R*(u-u_ref)
end

function stage_cost(model,x⁺,x,u,Q,R,x_ref,u_ref,h,c,w)
    xm = xm_rk3_implicit(model,x⁺,x,u,w,h)
    ℓ1 = quadratic_cost(x,u,Q,R,x_ref,u_ref)
    ℓ2 = quadratic_cost(xm,u,Q,R,x_ref,u_ref)
    ℓ3 = quadratic_cost(x⁺,u,Q,R,x_ref,u_ref)

    return h[1]/6.0*ℓ1 + 4.0*h[1]/6.0*ℓ2 + h[1]/6.0*ℓ3 + c*h[1]
end

function terminal_cost(x,Q,x_ref)
    return (x-x_ref)'*Q*(x-x_ref)
end

function objective(Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c
    w = zeros(model.nw)

    s = 0
    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        s += stage_cost(model,x⁺,x,u,Q[t],R[t],x_ref[t],u_ref[t],h,c,w)
    end
    x = view(Z,idx.x[T])
    s += terminal_cost(x,Q[T],x_ref[T])

    return s
end

function objective_gradient!(∇l,Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c
    w = zeros(model.nw)

    ∇l .= 0.0
    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        stage_cost_x(z) = stage_cost(model,x⁺,z,u,Q[t],R[t],x_ref[t],u_ref[t],h,c,w)
        stage_cost_u(z) = stage_cost(model,x⁺,x,z,Q[t],R[t],x_ref[t],u_ref[t],h,c,w)
        stage_cost_h(z) = stage_cost(model,x⁺,x,u,Q[t],R[t],x_ref[t],u_ref[t],z,c,w)
        stage_cost_x⁺(z) = stage_cost(model,z,x,u,Q[t],R[t],x_ref[t],u_ref[t],h,c,w)

        ∇l[idx.x[t]] += ForwardDiff.gradient(stage_cost_x,x)
        ∇l[idx.u[t]] += ForwardDiff.gradient(stage_cost_u,u)
        ∇l[idx.h[t]:idx.h[t]] += ForwardDiff.gradient(stage_cost_h,view(Z,prob.idx.h[t]:prob.idx.h[t]))
        ∇l[idx.x[t+1]] += ForwardDiff.gradient(stage_cost_x⁺,x⁺)
    end
    x = view(Z,idx.x[T])
    ∇l[idx.x[T]] += 2.0*Q[T]*(x-x_ref[T])

    return nothing
end
