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

    shift = 0

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        dyn_x(z) = x⁺ - integration(model,z,u,h)
        dyn_u(z) = x⁺ - integration(model,x,z,h)
        dyn_h(z) = x⁺ - integration(model,x,u,z)
        # dyn_x⁺(z) = z
        r_idx = (t-1)*n .+ (1:n)
        ∇c[r_idx,idx.x[t]] = ForwardDiff.jacobian(dyn_x,x)
        ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        ∇c[r_idx,idx.h[t]] = ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t]))
        ∇c[CartesianIndex.(r_idx,idx.x[t+1])] .= 1.0

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            r_idx = p_dyn + t
            ∇c[r_idx,idx.h[t]] = -1.0
            ∇c[r_idx,idx.h[t+1]] = 1.0
        end
    end

    return nothing
end

function sparse_dynamics_constraints_jacobian!(∇c,Z,idx,n,m,T,model,integration)
    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    p_dyn = n*(T-1) # number of dynamics constraints
    p_h = (T-2)     # number of time-step constraints

    shift = 0

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        dyn_x(z) = x⁺ - integration(model,z,u,h)
        dyn_u(z) = x⁺ - integration(model,x,z,h)
        dyn_h(z) = x⁺ - integration(model,x,u,z)
        # dyn_x⁺(z) = z
        # r_idx = (t-1)*n .+ (1:n)
        # ∇c[r_idx,idx.x[t]] = ForwardDiff.jacobian(dyn_x,x)
        s = n*n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x,x))
        shift += s

        # ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        s = n*m
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u,u))
        shift += s

        # ∇c[r_idx,idx.h[t]] = ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t]))
        s = n*1
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t])))
        shift += s

        # ∇c[CartesianIndex.(r_idx,idx.x[t+1])] .= 1.0
        s = n
        ∇c[shift .+ (1:s)] .= 1.0
        shift += s

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            # r_idx = p_dyn + t
            # ∇c[r_idx,idx.h[t]] = -1.0
            s = 1
            ∇c[shift + s] = -1.0
            shift += s

            # ∇c[r_idx,idx.h[t+1]] = 1.0
            s = 1
            ∇c[shift + s] = 1.0
            shift += s
        end
    end

    return nothing
end

function sparsity_dynamics_jacobian(idx,n,m,T)
    row = []
    col = []

    # r = 1:prob.m
    # c = 1:prob.n
    #
    # row_col!(row,col,r,c)


    p_dyn = n*(T-1) # number of dynamics constraints
    p_h = (T-2)     # number of time-step constraints

    for t = 1:T-1
        r_idx = (t-1)*n .+ (1:n)
        # ∇c[r_idx,idx.x[t]] = ForwardDiff.jacobian(dyn_x,x)
        row_col!(row,col,r_idx,idx.x[t])


        # ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        row_col!(row,col,r_idx,idx.u[t])


        # ∇c[r_idx,idx.h[t]] = ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t]))
        row_col!(row,col,r_idx,idx.h[t])


        # ∇c[CartesianIndex.(r_idx,idx.x[t+1])] .= 1.0
        row_col_cartesian!(row,col,r_idx,idx.x[t+1])

        if t < T-1
            r_idx = p_dyn + t
            # ∇c[r_idx,idx.h[t]] = -1.0
            row_col!(row,col,r_idx,idx.h[t])


            # ∇c[r_idx,idx.h[t+1]] = 1.0
            row_col!(row,col,r_idx,idx.h[t+1])

        end
    end

    return collect(zip(row,col))
end
