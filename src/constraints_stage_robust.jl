function num_robust_stage(m_stage,n,m,T; mode=:both)
    if mode == :both
        return sum(m_stage[2:end])*(2*n)*(2*m)
    elseif mode == :state
        return sum(m_stage[2:end])*(2*n)
    elseif mode == :control
        return sum(m_stage[2:end])*(2*m)
    else
        return 0
    end
end
function stage_constraints_robust!(c,Z,n,m,T,idx,nw,w0,model,integration,
        Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage;
        mode=:both)

    if mode == :both
        stage_constraints_robust_both!(c,Z,n,m,T,idx,nw,w0,model,integration,
            Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)
    elseif mode == :state
        stage_constraints_robust_state!(c,Z,n,m,T,idx,nw,w0,model,integration,
            Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)
    elseif mode == :control
        stage_constraints_robust_control!(c,Z,n,m,T,idx,nw,w0,model,integration,
            Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)
    else
        error("mode specified error")
    end
    return nothing
end

function ∇stage_constraints_robust!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,
        Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage;
        mode=:both)

    if mode == :both
        M = num_robust_stage(m_stage,n,m,T,mode=:both)
        tmp_both(c,z) = stage_constraints_robust_both!(c,z,n,m,T,idx,nw,w0,model,integration,
                Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)
        ForwardDiff.jacobian!(reshape(∇c,M,N),tmp_both,zeros(M),Z)
        # ∇stage_constraints_robust_both!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,
        #     Q_lqr,R_lqr,Qw,Rw,E1,H1,D,con,m_con)
    elseif mode == :state
        M = num_robust_stage(m_stage,n,m,T,mode=:state)
        tmp_state(c,z) = stage_constraints_robust_state!(c,z,n,m,T,idx,nw,w0,model,integration,
                Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)
        ForwardDiff.jacobian!(reshape(∇c,M,N),tmp_state,zeros(M),Z)
        # ∇stage_constraints_robust_state!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,
        #     Q_lqr,R_lqr,Qw,Rw,E1,H1,D,con,m_con)
    elseif mode == :control
        M = num_robust_stage(m_stage,n,m,T,mode=:control)
        tmp_control(c,z) = stage_constraints_robust_control!(c,z,n,m,T,idx,nw,w0,model,integration,
                Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)
        ForwardDiff.jacobian!(reshape(∇c,M,N),tmp_control,zeros(M),Z)
        # ∇stage_constraints_robust_control!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,
        #     Q_lqr,R_lqr,Qw,Rw,E1,H1,D,con,m_con)
    else
        error("mode specified error")
    end
    return nothing
end

function stage_constraints_robust_both!(c,Z,n,m,T,idx,nw,w0,model,integration,
        Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)

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

                c_stage!(view(c,shift .+ (1:m_stage[t])),xw⁺,uw⁺,t)
                shift += m_stage[t]

                c_stage!(view(c,shift .+ (1:m_stage[t])),xw⁺,uw⁻,t)
                shift += m_stage[t]

                c_stage!(view(c,shift .+ (1:m_stage[t])),xw⁻,uw⁺,t)
                shift += m_stage[t]

                c_stage!(view(c,shift .+ (1:m_stage[t])),xw⁻,uw⁻,t)
                shift += m_stage[t]
            end
        end
    end
    return nothing
end

# function ∇stage_constraints_robust_both!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D,
#         con,m_con)
#
#     δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     ∇δx = compute_∇δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     ∇δu = compute_∇δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#
#     ∇c_tmp = zeros(eltype(Z),m_con,N)
#     c_tmp = zeros(eltype(Z),m_con)
#     ∇cx = zeros(eltype(Z),m_con,n)
#     ∇cu = zeros(eltype(Z),m_con,m)
#
#     shift = 0
#     shift_r_idx = 0
#     for t = 2:T-1
#         x = Z[idx.x[t]]
#         u = Z[idx.u[t]]
#
#         cx(c,z) = con(c,z,u)
#         cu(c,z) = con(c,x,z)
#
#         ForwardDiff.jacobian!(∇cx,cx,c_tmp,x)
#         ForwardDiff.jacobian!(∇cu,cu,c_tmp,u)
#
#         for i = 1:n
#             # _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
#             # xw⁺ = x + _δx
#             # xw⁻ = x - _δx
#             for j = 1:m
#                 # _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
#                 # uw⁺ = u + _δu
#                 # uw⁻ = u - _δu
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁺)
#                 ∇c_tmp .= 0
#
#                 ∇c_tmp[:,idx.x[t]] += ∇cx
#                 ∇c_tmp[:,1:N] += ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#                 ∇c_tmp[:,idx.u[t]] += ∇cu
#                 ∇c_tmp[:,1:N] += ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#                 r_idx = shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 len = length(r_idx)*length(c_idx)
#                 ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#                 shift += len
#                 shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
#                 ∇c_tmp .= 0
#
#                 ∇c_tmp[:,idx.x[t]] += ∇cx
#                 ∇c_tmp[:,1:N] += ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#                 ∇c_tmp[:,idx.u[t]] += ∇cu
#                 ∇c_tmp[:,1:N] -= ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#                 r_idx = shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 len = length(r_idx)*length(c_idx)
#                 ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#                 shift += len
#                 shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁺)
#                 ∇c_tmp .= 0
#
#                 ∇c_tmp[:,idx.x[t]] += ∇cx
#                 ∇c_tmp[:,1:N] -= ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#                 ∇c_tmp[:,idx.u[t]] += ∇cu
#                 ∇c_tmp[:,1:N] += ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#                 r_idx = shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 len = length(r_idx)*length(c_idx)
#                 ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#                 shift += len
#                 shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
#                 ∇c_tmp .= 0
#
#                 ∇c_tmp[:,idx.x[t]] += ∇cx
#                 ∇c_tmp[:,1:N] -= ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#                 ∇c_tmp[:,idx.u[t]] += ∇cu
#                 ∇c_tmp[:,1:N] -= ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#                 r_idx = shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 len = length(r_idx)*length(c_idx)
#                 ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#                 shift += len
#                 shift_r_idx += m_con
#             end
#         end
#     end
#     return nothing
# end

function stage_constraints_robust_state!(c,Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D,
        m_stage)

    δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    # δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    shift = 0
    for t = 2:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]

        for i = 1:n
            _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
            xw⁺ = x + _δx
            xw⁻ = x - _δx
            # for j = 1:m
                # _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
                # uw⁺ = u + _δu
                # uw⁻ = u - _δu

            c_stage!(view(c,shift .+ (1:m_stage[t])),xw⁺,u,t)
            shift += m_stage[t]

                # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
                # shift += m_con

            c_stage!(view(c,shift .+ (1:m_stage[t])),xw⁻,u,t)
            shift += m_stage[t]

                # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
                # shift += m_con
            # end
        end
    end
    return nothing
end

# function ∇stage_constraints_robust_state!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,
#         Q_lqr,R_lqr,Qw,Rw,E1,H1,D,con,m_con)
#
#     δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     # δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     ∇δx = compute_∇δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     # ∇δu = compute_∇δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#
#     ∇c_tmp = zeros(eltype(Z),m_con,N)
#     c_tmp = zeros(eltype(Z),m_con)
#     ∇cx = zeros(eltype(Z),m_con,n)
#     ∇cu = zeros(eltype(Z),m_con,m)
#
#     shift = 0
#     shift_r_idx = 0
#     for t = 2:T-1
#         x = Z[idx.x[t]]
#         u = Z[idx.u[t]]
#
#         cx(c,z) = con(c,z,u)
#         cu(c,z) = con(c,x,z)
#
#         ForwardDiff.jacobian!(∇cx,cx,c_tmp,x)
#         ForwardDiff.jacobian!(∇cu,cu,c_tmp,u)
#
#         for i = 1:n
#             # _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
#             # xw⁺ = x + _δx
#             # xw⁻ = x - _δx
#             # for j = 1:m
#                 # _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
#                 # uw⁺ = u + _δu
#                 # uw⁻ = u - _δu
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁺,u)
#             ∇c_tmp .= 0
#
#             ∇c_tmp[:,idx.x[t]] += ∇cx
#             ∇c_tmp[:,1:N] += ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#             ∇c_tmp[:,idx.u[t]] += ∇cu
#             # ∇c_tmp[:,1:N] += ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#             r_idx = shift_r_idx .+ (1:m_con)
#             c_idx = 1:N
#             len = length(r_idx)*length(c_idx)
#             ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#             shift += len
#             shift_r_idx += m_con
#
#                 # # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
#                 # ∇c_tmp .= 0
#                 #
#                 # ∇c_tmp[:,idx.x[t]] += ∇cx
#                 # ∇c_tmp[:,1:N] += ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#                 # ∇c_tmp[:,idx.u[t]] += ∇cu
#                 # ∇c_tmp[:,1:N] -= ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#                 #
#                 # r_idx = shift_r_idx .+ (1:m_con)
#                 # c_idx = 1:N
#                 # len = length(r_idx)*length(c_idx)
#                 # ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#                 # shift += len
#                 # shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁻,u)
#             ∇c_tmp .= 0
#
#             ∇c_tmp[:,idx.x[t]] += ∇cx
#             ∇c_tmp[:,1:N] -= ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#             ∇c_tmp[:,idx.u[t]] += ∇cu
#             # ∇c_tmp[:,1:N] += ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#             r_idx = shift_r_idx .+ (1:m_con)
#             c_idx = 1:N
#             len = length(r_idx)*length(c_idx)
#             ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#             shift += len
#             shift_r_idx += m_con
#
#                 # # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
#                 # ∇c_tmp .= 0
#                 #
#                 # ∇c_tmp[:,idx.x[t]] += ∇cx
#                 # ∇c_tmp[:,1:N] -= ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#                 # ∇c_tmp[:,idx.u[t]] += ∇cu
#                 # ∇c_tmp[:,1:N] -= ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#                 #
#                 # r_idx = shift_r_idx .+ (1:m_con)
#                 # c_idx = 1:N
#                 # len = length(r_idx)*length(c_idx)
#                 # ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#                 # shift += len
#                 # shift_r_idx += m_con
#             # end
#         end
#     end
#     return nothing
# end

function stage_constraints_robust_control!(c,Z,n,m,T,idx,nw,w0,model,integration,
        Q_lqr,R_lqr,Qw,Rw,E1,H1,D,m_stage)

    # δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
    δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)

    shift = 0
    for t = 2:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]

        # for i = 1:n
        #     _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
        #     xw⁺ = x + _δx
        #     xw⁻ = x - _δx
        for j = 1:m
            _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
            uw⁺ = u + _δu
            uw⁻ = u - _δu

            c_stage!(view(c,shift .+ (1:m_stage[t])),x,uw⁺,t)
            shift += m_stage[t]

            c_stage!(view(c,shift .+ (1:m_stage[t])),x,uw⁻,t)
            shift += m_stage[t]

            # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁺)
            # shift += m_con
            #
            # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
            # shift += m_con
        end
        # end
    end
    return nothing
end

# function ∇stage_constraints_robust_control!(∇c,Z,n,m,N,T,idx,nw,w0,model,integration,
#         Q_lqr,R_lqr,Qw,Rw,E1,H1,D,con,m_con)
#
#     # δx = compute_δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     δu = compute_δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     # ∇δx = compute_∇δx(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#     ∇δu = compute_∇δu(Z,n,m,T,idx,nw,w0,model,integration,Q_lqr,R_lqr,Qw,Rw,E1,H1,D)
#
#     ∇c_tmp = zeros(eltype(Z),m_con,N)
#     c_tmp = zeros(eltype(Z),m_con)
#     ∇cx = zeros(eltype(Z),m_con,n)
#     ∇cu = zeros(eltype(Z),m_con,m)
#
#     shift = 0
#     shift_r_idx = 0
#     for t = 2:T-1
#         x = Z[idx.x[t]]
#         u = Z[idx.u[t]]
#
#         cx(c,z) = con(c,z,u)
#         cu(c,z) = con(c,x,z)
#
#         ForwardDiff.jacobian!(∇cx,cx,c_tmp,x)
#         ForwardDiff.jacobian!(∇cu,cu,c_tmp,u)
#
#         # for i = 1:n
#         #     # _δx = δx[(t-1)*n*n + (i-1)*n .+ (1:n)]
#         #     # xw⁺ = x + _δx
#         #     # xw⁻ = x - _δx
#         for j = 1:m
#             # _δu = δu[(t-1)*m*m + (j-1)*m .+ (1:m)]
#             # uw⁺ = u + _δu
#             # uw⁻ = u - _δu
#
#             # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁺)
#             ∇c_tmp .= 0
#
#             ∇c_tmp[:,idx.x[t]] += ∇cx
#             # ∇c_tmp[:,1:N] += ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#             ∇c_tmp[:,idx.u[t]] += ∇cu
#             ∇c_tmp[:,1:N] += ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#             r_idx = shift_r_idx .+ (1:m_con)
#             c_idx = 1:N
#             len = length(r_idx)*length(c_idx)
#             ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#             shift += len
#             shift_r_idx += m_con
#
#             # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
#             ∇c_tmp .= 0
#
#             ∇c_tmp[:,idx.x[t]] += ∇cx
#             # ∇c_tmp[:,1:N] += ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#             ∇c_tmp[:,idx.u[t]] += ∇cu
#             ∇c_tmp[:,1:N] -= ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#
#             r_idx = shift_r_idx .+ (1:m_con)
#             c_idx = 1:N
#             len = length(r_idx)*length(c_idx)
#             ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#             shift += len
#             shift_r_idx += m_con
#
#             # # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁺)
#             # ∇c_tmp .= 0
#             #
#             # ∇c_tmp[:,idx.x[t]] += ∇cx
#             # ∇c_tmp[:,1:N] -= ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#             # ∇c_tmp[:,idx.u[t]] += ∇cu
#             # ∇c_tmp[:,1:N] += ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#             #
#             # r_idx = shift_r_idx .+ (1:m_con)
#             # c_idx = 1:N
#             # len = length(r_idx)*length(c_idx)
#             # ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#             # shift += len
#             # shift_r_idx += m_con
#             #
#             # # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
#             # ∇c_tmp .= 0
#             #
#             # ∇c_tmp[:,idx.x[t]] += ∇cx
#             # ∇c_tmp[:,1:N] -= ∇cx*∇δx[(t-1)*n*n + (i-1)*n .+ (1:n),1:N]
#             # ∇c_tmp[:,idx.u[t]] += ∇cu
#             # ∇c_tmp[:,1:N] -= ∇cu*∇δu[(t-1)*m*m + (j-1)*m .+ (1:m),1:N]
#             #
#             # r_idx = shift_r_idx .+ (1:m_con)
#             # c_idx = 1:N
#             # len = length(r_idx)*length(c_idx)
#             # ∇c[shift .+ (1:len)] = vec(∇c_tmp)
#             # shift += len
#             # shift_r_idx += m_con
#         end
#         # end
#     end
#     return nothing
# end

# function stage_constraints_robust_sparsity(n,m,N,T,idx,m_con;
#         shift_r=0,mode=:both)
#
#     row = []
#     col = []
#
#     shift = 0
#     shift_r_idx = 0
#
#     for t = 2:T-1
#         for i = 1:n
#             for j = 1:m
#                 # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁺)
#                 r_idx = shift_r+shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 row_col!(row,col,r_idx,c_idx)
#                 shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁺,uw⁻)
#                 r_idx = shift_r+shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 row_col!(row,col,r_idx,c_idx)
#
#                 shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁺)
#                 r_idx = shift_r+shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 row_col!(row,col,r_idx,c_idx)
#
#                 shift_r_idx += m_con
#
#                 # con(view(c,shift .+ (1:m_con)),xw⁻,uw⁻)
#                 r_idx = shift_r+shift_r_idx .+ (1:m_con)
#                 c_idx = 1:N
#                 row_col!(row,col,r_idx,c_idx)
#
#                 shift_r_idx += m_con
#             end
#         end
#     end
#     return collect(zip(row,col))
# end
