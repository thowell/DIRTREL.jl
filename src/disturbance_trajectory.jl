function linearize_trajectories(Z,n,m,T,idx,nw,w,model,integration)
    # linearize about nominal trajectories
    A = [zeros(eltype(Z),n,n) for t = 1:T-1]
    B = [zeros(eltype(Z),n,2*m) for t = 1:T-1]
    G = [zeros(eltype(Z),n,nw) for t = 1:T-1]

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])
        u⁺ = view(Z,idx.u[t+1])

        dyn_x(z) = integration(model,x⁺,z,u⁺,u,w,h)
        dyn_u(z) = integration(model,x⁺,x,u⁺,z,w,h)
        dyn_w(z) = integration(model,x⁺,x,u⁺,u,z,h)
        dyn_x⁺(z) = integration(model,z,x,u⁺,u,w,h)
        dyn_u⁺(z) = integration(model,x⁺,x,z,u,w,h)

        # (implicit function theorem)
        A⁺ = ForwardDiff.jacobian(dyn_x⁺,x⁺)
        A[t] = -A⁺\ForwardDiff.jacobian(dyn_x,x)
        B[t] = -A⁺\[ForwardDiff.jacobian(dyn_u,u) ForwardDiff.jacobian(dyn_u⁺,u⁺)]
        G[t] = -A⁺\ForwardDiff.jacobian(dyn_w,w)
    end

    return A, B, G
end

function tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    K = [zeros(eltype(Z),2*m,n) for t = 1:T-1]

    P = Q_lqr[T]
    for t = T-1:-1:1
        R = cat(R_lqr[t],R_lqr[t+1],dims=(1,2))
        K[t] = (R + B[t]'*P*B[t])\(B[t]'*P*A[t])
        P = Q_lqr[t] + K[t]'*R*K[t] + (A[t] - B[t]*K[t])'*P*(A[t] - B[t]*K[t])
    end

    return K
end

function disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)
    E = [zeros(eltype(Z),n,n) for t = 1:T]
    E[1] = copy(E1)
    H = copy(H1)

    for t = 1:T-1
        Acl = A[t] - B[t]*K[t]

        E[t+1] = (Acl*E[t]*Acl'
                    + Acl*H*G[t]'
                    + G[t]*H'*Acl'
                    + G[t]*D*G[t]')

        H = Acl*H + G[t]*D
    end

    return E
end
