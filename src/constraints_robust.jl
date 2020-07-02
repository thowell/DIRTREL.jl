function compute_E(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    return E
end

function compute_KEK(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    KEK = [K[t]*E[t]*K[t]' for t = 1:T-1]
end

function compute_KEK_vec(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    KEK_vec = zeros(eltype(Z),m*m*(T-1))
    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    for t = 1:T-1
        KEK_vec[(t-1)*(m*m) .+ (1:m*m)] = vec(K[t]*E[t]*K[t]')
    end

    return KEK_vec
end

function compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    KEK = compute_KEK_vec(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δu = zeros(eltype(Z),length(KEK))
    for t = 1:T-1
        kek = reshape(KEK[(t-1)*m*m .+ (1:m*m)],m,m)
        cols = fastsqrt(kek)
        for j = 1:m
            δu[(t-1)*m*m + (j-1)*m .+ (1:m)] = cols[:,j]
        end
    end
    return δu
end

function compute_∇δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    KEK = compute_KEK_vec(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    gen_KEK(z) = compute_KEK_vec(z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    ∇KEK = ForwardDiff.jacobian(gen_KEK,Z)

    N = length(Z)
    ∇δu = zeros(eltype(Z),m*m*(T-1),N)
    Im = Diagonal(ones(m))
    for t = 1:T-1
       r_idx = (t-1)*m*m .+ (1:m*m)
       kek_sqrt = fastsqrt(reshape(KEK[r_idx],m,m))
       ∇δu[r_idx,1:N] = inv(kron(Im,kek_sqrt) + kron(kek_sqrt,Im))*∇KEK[r_idx,1:N]
    end

    return ∇δu
end


# Robust linear control bounds
function uw_bounds!(c,Z,ul,uu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    shift = 0
    for t = 1:T-1
        for j = 1:m
            _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
            uw⁺ = Z[idx.u[t]] + _δu
            uw⁻ = Z[idx.u[t]] - _δu
            c[shift .+ (1:m)] = uw⁺ - uu[t] # upper bounds
            shift += m
            c[shift .+ (1:m)] = uw⁻ - uu[t] # upper bounds
            shift += m
            c[shift .+ (1:m)] = ul[t] - uw⁺ # lower bounds
            shift += m
            c[shift .+ (1:m)] = ul[t] - uw⁻ # lower bounds
            shift += m
        end
    end
    return nothing
end

function ∇uw_bounds!(∇c,Z,ul,uu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    ∇δu = compute_∇δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    N = length(Z)
    shift = 0
    for t = 1:T-1
        for j = 1:m
            # _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
            # uw⁺ = Z[idx.u[t]] + _δu
            # uw⁻ = Z[idx.u[t]] - _δu
            # c[shift .+ (1:m)] = uw⁺ - uu[t] # upper bounds
            ∇c[shift .+ (1:m),1:N] = ∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
            ∇c[CartesianIndex.(shift .+ (1:m),idx.u[t])] .= 1.0
            shift += m
            # c[shift .+ (1:m)] = uw⁻ - uu[t] # upper bounds
            ∇c[shift .+ (1:m),1:N] = -1.0*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
            ∇c[CartesianIndex.(shift .+ (1:m),idx.u[t])] .= 1.0
            shift += m
            # c[shift .+ (1:m)] = ul[t] - uw⁺ # lower bounds
            ∇c[shift .+ (1:m),1:N] = -1.0*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
            ∇c[CartesianIndex.(shift .+ (1:m),idx.u[t])] .= -1.0
            shift += m
            # c[shift .+ (1:m)] = ul[t] - uw⁻ # lower bounds
            ∇c[shift .+ (1:m),1:N] = ∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
            ∇c[CartesianIndex.(shift .+ (1:m),idx.u[t])] .= -1.0
            shift += m
        end
    end
end

function num_robust_control_bounds(m,T)
    return 2*(2*m*m*(T-1))
end
#
# A = rand(3,3)
# B = A'*A
# B = Diagonal([0.0;2.0;2.0])
# B_sqrt = fastsqrt(B)
# B_sqrt*B_sqrt
# Matrix(I,size(A))
#
#
#
#
# get_mat(x) = [x[1] 0.0 0.0; 0.0 x[2] 0.0; 0.0 0.0 x[3]]
# get_mat_sqrt(x) = fastsqrt(get_mat(x))
#
# ForwardDiff.jacobian(get_mat_sqrt,rand(3))
# function fastsqrt(A)
#     #FASTSQRT computes the square root of a matrix A with Denman-Beavers iteration
#
#     #S = sqrtm(A);
#     Ep = 1e-8*Matrix(I,size(A))
#
#     if count(diag(A) .> 0.0) != size(A,1)
#         S = diagm(sqrt.(diag(A)));
#         return S
#     end
#
#     In = Matrix(1.0*I,size(A));
#     S = A;
#     T = Matrix(1.0*I,size(A));
#
#     T = .5*(T + inv(S+Ep));
#     S = .5*(S+In);
#     for k = 1:4
#         Snew = .5*(S + inv(T+Ep));
#         T = .5*(T + inv(S+Ep));
#         S = Snew;
#     end
#     return S
# end
