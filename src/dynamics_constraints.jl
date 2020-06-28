function dynamics_constraints!(c,Z,idx,n,m,T,model,integration)
    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    p_dyn = n*(T-1) # number of dynamics constraints
    p_h = (T-2)     # number of time-step constraints

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        c[(t-1)*n .+ (1:n)] = x⁺ - integration(model,x,u,h)

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            c[p_dyn + t] = h⁺ - h
        end
    end

    return nothing
end

function dynamics_constraints_jacobian!(∇c,Z,idx,n,m,T,model,integration)
    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    p_dyn = n*(T-1) # number of dynamics constraints
    p_h = (T-2)     # number of time-step constraints

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        dyn_x(z) = x⁺ - integration(model,z,u,h)
        dyn_u(z) = x⁺ - integration(model,x,z,h)
        dyn_h(z) = x⁺ - integration(model,x,u,z)
        # dyn_x⁺(z) = z

        ∇c[(t-1)*n .+ (1:n),idx.x[t]] = ForwardDiff.jacobian(dyn_x,x)
        ∇c[(t-1)*n .+ (1:n),idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        ∇c[(t-1)*n .+ (1:n),idx.h[t]] = ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t]))
        ∇c[CartesianIndex.((t-1)*n .+ (1:n),idx.x[t+1])] .= 1.0

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            ∇c[p_dyn + t,idx.h[t]] = -1.0
            ∇c[p_dyn + t,idx.h[t+1]] = 1.0
        end
    end

    return nothing
end
