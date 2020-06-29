function linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    # linearize about nominal trajectories
    A = [zeros(eltype(Z),n,n) for t = 1:T-1]
    B = [zeros(eltype(Z),n,m) for t = 1:T-1]
    G = [zeros(eltype(Z),n,nw) for t = 1:T-1]

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        dyn_x(z) = x⁺ - integration(model,z,u,w0,h)
        dyn_u(z) = x⁺ - integration(model,x,z,w0,h)
        dyn_w(z) = x⁺ - integration(model,x,u,z,h)

        A[t] = ForwardDiff.jacobian(dyn_x,x)
        B[t] = ForwardDiff.jacobian(dyn_u,u)
        G[t] = ForwardDiff.jacobian(dyn_w,w0)
    end

    return A, B, G
end

function tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    K = [zeros(eltype(Z),m,n) for t = 1:T-1]

    P = Q_lqr[T]
    for t = T-1:-1:1
        K[t] = (R_lqr[t] + B[t]'*P*B[t])\(B[t]'*P*A[t])
        P = Q_lqr[t] + A[t]'*P*A[t] - (A[t]'*P*B[t])*K[t]
    end

    return K
end

function disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)
    E = [zeros(eltype(Z),n,n) for t = 1:T]
    E[1] = copy(E1)
    H = copy(H1)

    for t = 1:T-1
        tmp = (A[t] - B[t]*K[t])
        E[t+1] = (tmp*E[t]*tmp'
            + tmp*H*G[t]'
            + G[t]*H'*tmp'
            + G[t]*D*G[t]')
        H = tmp*H + G[t]*D
    end

    return E
end
