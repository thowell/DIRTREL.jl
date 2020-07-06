mutable struct RobustProblem <: Problem
    prob::TrajectoryOptimizationProblem
    nw::Int
    w0::Vector
    Q_lqr
    R_lqr
    Qw
    Rw
    E1
    H1
    D
    M_robust
    M_robust_control_bnds
    M_robust_state_bnds
end

function robust_problem(prob,nw,Q_lqr,R_lqr,Qw,Rw,E1,H1,D;
        robust_control_bnds=false,
        robust_state_bnds=false)

        M_robust_control_bnds = robust_control_bnds*num_robust_control_bounds(prob.m,prob.T)
        M_robust_state_bnds = robust_state_bnds*num_robust_state_bounds(prob.n,prob.T)

        M_robust = M_robust_control_bnds + M_robust_state_bnds

        RobustProblem(prob,nw,zeros(nw),Q_lqr,R_lqr,Qw,Rw,E1,H1,D,
            M_robust,M_robust_control_bnds,M_robust_state_bnds)
end

function constraints_robust!(cw,Z,prob_robust::RobustProblem)
    prob = prob_robust.prob
    M_ctrl = prob_robust.M_robust_control_bnds
    M_state = prob_robust.M_robust_state_bnds

    if M_ctrl > 0
        uw_bounds!(view(cw,1:M_ctrl),Z,prob.ul,prob.uu,prob.n,prob.m,prob.T,prob.idx,
                    prob_robust.nw,prob_robust.w0,
                    prob.model,prob.integration,
                    prob_robust.Q_lqr,prob_robust.R_lqr,
                    prob_robust.Qw,prob_robust.Rw,
                    prob_robust.E1,prob_robust.H1,prob_robust.D)
    end

    if M_state > 0
        xw_bounds!(view(cw,M_ctrl .+ (1:M_state)),Z,prob.xl,prob.xu,prob.n,prob.m,prob.T,prob.idx,
                    prob_robust.nw,prob_robust.w0,
                    prob.model,prob.integration,
                    prob_robust.Q_lqr,prob_robust.R_lqr,
                    prob_robust.Qw,prob_robust.Rw,
                    prob_robust.E1,prob_robust.H1,prob_robust.D)
    end

    return nothing
end

function constraints_robust_jacobian!(∇cw,Z,prob_robust::RobustProblem)
    prob = prob_robust.prob

    M_ctrl = prob_robust.M_robust_control_bnds
    M_state = prob_robust.M_robust_state_bnds

    if M_ctrl > 0
        tmp_uw!(cw,z) = uw_bounds!(cw,z,prob.ul,prob.uu,prob.n,prob.m,prob.T,prob.idx,
                    prob_robust.nw,prob_robust.w0,
                    prob.model,prob.integration,
                    prob_robust.Q_lqr,prob_robust.R_lqr,
                    prob_robust.Qw,prob_robust.Rw,
                    prob_robust.E1,prob_robust.H1,prob_robust.D)

        cw_ctrl = zeros(M_ctrl)
        ForwardDiff.jacobian!(view(∇cw,1:M_ctrl,1:prob.N),tmp_uw!,cw_ctrl,Z)
    end

    if M_state > 0
        tmp_xw!(cw,z) = xw_bounds!(cw,z,prob.xl,prob.xu,prob.n,prob.m,prob.T,prob.idx,
                    prob_robust.nw,prob_robust.w0,
                    prob.model,prob.integration,
                    prob_robust.Q_lqr,prob_robust.R_lqr,
                    prob_robust.Qw,prob_robust.Rw,
                    prob_robust.E1,prob_robust.H1,prob_robust.D)

        cw_state = zeros(M_state)
        ForwardDiff.jacobian!(view(∇cw,M_ctrl .+ (1:M_state),1:prob.N),tmp_xw!,cw_state,Z)
     end
    return nothing
end

# MOI interface
function init_MOI_RobustProblem(prob_robust::RobustProblem)
    prob = prob_robust.prob
    n = prob.n
    m = prob.m
    T = prob.T

    N = prob.N
    M = prob.M
    M_robust = prob_robust.M_robust

    return MOIProblem(N,M+M_robust,prob_robust,false)
end

function primal_bounds(prob_robust::RobustProblem)
    primal_bounds(prob_robust.prob)
end

function constraint_bounds(prob_robust::RobustProblem)
    prob = prob_robust.prob

    n = prob.n
    m = prob.m
    T = prob.T
    idx = prob.idx

    M = prob.M
    M_robust = prob_robust.M_robust
    cl, cu = constraint_bounds(prob)
    cl_robust = -Inf*ones(M_robust)
    cu_robust = zeros(M_robust)

    return [cl; cl_robust], [cu; cu_robust]
end

function sparsity_robust_constraints(prob_robust::RobustProblem; shift=0)
    row = []
    col = []
    M_robust = prob_robust.M_robust

    r = shift .+ (1:M_robust)
    c = 1:prob_robust.prob.N

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function eval_objective(prob_robust::RobustProblem,Z)
    prob = prob_robust.prob
    return (objective(Z,prob.obj,prob.model,prob.idx,prob.T)
            + robust_cost(Z,
                prob.n,prob.m,prob.T,
                prob.idx,
                prob_robust.nw,
                prob_robust.w0,
                prob.model,
                prob.integration,
                prob_robust.Q_lqr,
                prob_robust.R_lqr,
                prob_robust.Qw,
                prob_robust.Rw,
                prob_robust.E1,
                prob_robust.H1,
                prob_robust.D))
end

function eval_objective_gradient!(∇l,Z,prob_robust::RobustProblem)
    prob = prob_robust.prob

    objective_gradient!(∇l,Z,prob.obj,prob.model,prob.idx,prob.T)

    tmp(z) = robust_cost(z,
                        prob.n,prob.m,prob.T,prob.idx,
                        prob_robust.nw,prob_robust.w0,
                        prob.model,prob.integration,
                        prob_robust.Q_lqr,prob_robust.R_lqr,
                        prob_robust.Qw,prob_robust.Rw,
                        prob_robust.E1,prob_robust.H1,prob_robust.D)
    ∇l .+= ForwardDiff.gradient(tmp,Z)
    return nothing
end

function eval_constraint!(c,Z,prob_robust::RobustProblem)
    prob = prob_robust.prob

    dynamics_constraints!(view(c,1:prob.M),Z,
        prob.idx,prob.n,prob.m,prob.T,prob.model,prob.integration)

    constraints_robust!(view(c,prob.M .+ (1:prob_robust.M_robust)),Z,prob_robust)

    return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob_robust::RobustProblem)
    prob = prob_robust.prob
    sparsity_dynamics = sparsity_jacobian(prob)
    L = length(sparsity_dynamics)

    sparse_dynamics_constraints_jacobian!(view(∇c,1:L),Z,
        prob.idx,prob.n,prob.m,prob.T,prob.model,prob.integration)

    constraints_robust_jacobian!(reshape(view(∇c,L .+ (1:prob_robust.M_robust*prob.N)),prob_robust.M_robust,prob.N),Z,prob_robust)
    return nothing
end

function sparsity_jacobian(prob_robust::RobustProblem)
    prob = prob_robust.prob
    sparsity_dynamics = sparsity_dynamics_jacobian(prob.idx,prob.n,prob.m,prob.T)
    sparsity_robust = sparsity_robust_constraints(prob_robust,shift=prob.M)
    return vcat(sparsity_dynamics...,sparsity_robust)
end
