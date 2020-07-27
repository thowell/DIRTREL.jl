function num_robust_general_bounds(m_con,n,m,T; mode=:both)
    if mode == :both
        m_con*(2*n)*(2*m)
    elseif mode == :state
        m_con*(2*n)
    elseif mode == :control
        m_con*(2*m)
    else
        error("specify: :both OR  :state OR :control")
    return
end

function con_robust!(c,Z,ul,uu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D,
        con,m_con;
        mode=:both)

    con(c,z) = con(c,z,idx,T)

    δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    shift = 0
    for t = 2:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]

        for i = 1:n
            _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
            xw⁺ = x + _δx
            xw⁻ = x - _δx
            for j = 1:m
                _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
                uw⁺ = u + _δu
                uw⁻ = u - _δu

                con(view(c,shift .+ (1:m_con)),xw⁺,uw⁺)
                shift += m_con

                con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
                shift += m_con

                con(view(c,shift .+ (1:m_con)),xw⁻,uw⁺)
                shift += m_con

                con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
                shift += m_con
            end
        end
    end
    return nothing
end

function con_robust!(c,Z,ul,uu,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D,
        con,m_con;
        mode=:both)

    δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    ∇δx = compute_∇δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    ∇δu = compute_∇δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    shift = 0
    for t = 2:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]



        for i = 1:n
            _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
            xw⁺ = x + _δx
            xw⁻ = x - _δx
            for j = 1:m
                _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
                uw⁺ = u + _δu
                uw⁻ = u - _δu

                con(view(c,shift .+ (1:m_con)),xw⁺,uw⁺)
                shift += m_con

                con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
                shift += m_con

                con(view(c,shift .+ (1:m_con)),xw⁻,uw⁺)
                shift += m_con

                con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
                shift += m_con
            end
        end
    end
    return nothing
end
