# robust linear state bounds
function compute_E(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    return E
end

function compute_E_vec(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    E_vec = zeros(eltype(Z),n*n*T)

    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    for t = 1:T
        E_vec[(t-1)*(n*n) .+ (1:n*n)] = vec(E[t])
    end
    return E_vec
end

function compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    E = compute_E_vec(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δx = zeros(eltype(Z),length(E))
    for t = 1:T
        e = reshape(E[(t-1)*n*n .+ (1:n*n)],n,n)
        cols = fastsqrt(e)
        for j = 1:n
            δx[(t-1)*n*n + (j-1)*n .+ (1:n)] = cols[:,j]
        end
    end
    return δx
end

function compute_∇δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    E = compute_E_vec(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    gen_E(z) = compute_E_vec(z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    ∇E = ForwardDiff.jacobian(gen_E,Z)

    N = length(Z)
    ∇δx = zeros(eltype(Z),n*n*T,N)
    In = Diagonal(ones(n))
    for t = 1:T
       r_idx = (t-1)*n*n .+ (1:n*n)
       e_sqrt = fastsqrt(reshape(E[r_idx],n,n))
       ∇δx[r_idx,1:N] = inv(kron(In,e_sqrt) + kron(e_sqrt,In))*∇E[r_idx,1:N]
    end

    return ∇δx
end

function xw_bounds!(c,Z,xl,xu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    # note: robust bounds on state do not include t = 1 or t = T
    shift = 0
    for t = 2:T-1
        for j = 1:m
            _δx = δx[(t-1)*n*n + (j-1)*n .+ (1:n)]
            xw⁺ = Z[idx.x[t]] + _δx
            xw⁻ = Z[idx.x[t]] - _δx
            c[shift .+ (1:n)] = -xw⁺ + xu[t] # upper bounds
            shift += n
            c[shift .+ (1:n)] = -xw⁻ + xu[t] # upper bounds
            shift += n
            c[shift .+ (1:n)] = -xl[t] + xw⁺ # lower bounds
            shift += n
            c[shift .+ (1:n)] = -xl[t] + xw⁻ # lower bounds
            shift += n
        end
    end
    return nothing
end

function ∇xw_bounds!(∇c,Z,xl,xu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    ∇δx = compute_∇δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    N = length(Z)
    shift = 0
    for t = 2:T-1
        for j = 1:n

            ∇c[shift .+ (1:n),1:N] = -∇δx[(t-1)*n*n + (j-1)*n .+ (1:n),1:N]
            ∇c[CartesianIndex.(shift .+ (1:n),idx.x[t])] .= -1.0
            shift += n

            ∇c[shift .+ (1:n),1:N] = 1.0*∇δx[(t-1)*n*n + (j-1)*n .+ (1:n),1:N]
            ∇c[CartesianIndex.(shift .+ (1:n),idx.x[t])] .= -1.0
            shift += n

            ∇c[shift .+ (1:n),1:N] = 1.0*∇δx[(t-1)*n*n + (j-1)*n .+ (1:n),1:N]
            ∇c[CartesianIndex.(shift .+ (1:n),idx.x[t])] .= 1.0
            shift += n

            ∇c[shift .+ (1:n),1:N] = -∇δx[(t-1)*n*n + (j-1)*n .+ (1:n),1:N]
            ∇c[CartesianIndex.(shift .+ (1:n),idx.x[t])] .= 1.0
            shift += n
        end
    end
end

function num_robust_state_bounds(n,T)
    return 2*(2*n*n*(T-2))
end
