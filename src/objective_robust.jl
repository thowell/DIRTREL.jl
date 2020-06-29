function robust_cost(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    A, B, G = linearize_trajectories(Z,n,m,T,idx,nw,w0,model,integration)
    K = tvlqr(Z,A,B,Q_lqr,R_lqr,n,m,T)
    E = disturbance_trajectory(Z,A,B,G,K,Qw,Rw,E1,H1,D,n,T)

    l = 0
    for t = 1:T-1
        l += tr((Qw[t] + K[t]'*Rw[t]*K[t])*E[t])
    end
    l += tr(Qw[T]*E[T])

    return l
end
