function compute_KEK(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    KEK = [K[t]*E[t]*K[t]' for t = 1:T-1]
end

function compute_uw(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    KEK = compute_KEK(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    uw = []
    for t = 1:T-1
        u = view(Z,idx.u[t])
        uwt = []
        cols = matrix_sqrt(KEK[t])
        for j = 1:m
            push!(uwt,u + cols[:,j])
            push!(uwt,u - cols[:,j])
        end
        push!(uw,uwt)
    end
    return uw
end

# Robust linear control bounds
function uw_bounds!(c,Z,ul,uu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    # uw - uu <= 0, ul - uw <= 0
    # M = 2*(2*m*m*(T-1))
    # c = zeros(M)
    uw = compute_uw(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    for t = 1:T-1
        for j = 1:2*m
            c[(t-1)*(2*m*m) + (j-1)*m .+ (1:m)] = uw[t][j] - uu[t] # upper bounds
            c[(2*m*m)*(T-1) + (t-1)*(2*m*m) + (j-1)*m .+ (1:m)] = ul[t] - uw[t][j] # lower bounds
        end
    end
    return nothing
end

function num_robust_control_bounds(m,T)
    return 2*(2*m*m*(T-1))
end
