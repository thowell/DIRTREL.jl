# z = (x,u,h)
# Z = [z1,...,zT-1,xT]
abstract type Problem end

mutable struct TrajectoryOptimizationProblem <: Problem
    n::Int # states
    m::Int # controls
    T::Int # horizon
    N::Int # number of decision variables
    M::Int # number of constraints
    x1     # initial state
    xT     # goal state
    ul     # control lower bound
    uu     # control upper bound
    xl     # state lower bound
    xu     # state upper bound
    hl     # time step lower bound
    hu     # time step upper bound
    idx    # indices
    model  # model
    integration # dynamics integration scheme
    obj    # objective
    goal_constraint::Bool
    stage_constraints::Bool
    m_stage
    c_stage_ineq
end

function init_problem(n,m,T,x1,xT,model,obj;
        ul=[-Inf*ones(m) for t = 1:T-1],
        uu=[Inf*ones(m) for t = 1:T-1],
        xl=[-Inf*ones(n) for t = 1:T],
        xu=[Inf*ones(n) for t = 1:T],
        hl=[-Inf for t = 1:T-1],
        hu=[Inf for t = 1:T-1],
        integration=rk3_implicit,
        goal_constraint::Bool=true,
        stage_constraints::Bool=false,
        m_stage=[0 for t=1:T-1],
        c_stage_ineq=[(1:m_stage[t]) for t=1:T-1])

    idx = init_indices(n,m,T)
    N = n*T + m*(T-1) + (T-1)
    M = n*(T-1) + (T-2) + sum(m_stage)

    return TrajectoryOptimizationProblem(n,m,T,N,M,
        x1,xT,
        ul,uu,
        xl,xu,
        hl,hu,
        idx,
        model,integration,
        obj,
        goal_constraint,
        stage_constraints,
        m_stage,
        c_stage_ineq)
end

function pack(X0,U0,h0,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    Z0 = zeros(prob.N)
    for t = 1:T-1
        Z0[(t-1)*(n+m+1) .+ (1:n)] = X0[t]
        Z0[(t-1)*(n+m+1)+n .+ (1:m)] = U0[t]
        Z0[(t-1)*(n+m+1)+n+m + 1] = h0
    end
    Z0[(T-1)*(n+m+1) .+ (1:n)] = X0[T]

    return Z0
end

function unpack(Z0,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    X = [Z0[(t-1)*(n+m+1) .+ (1:n)] for t = 1:T]
    U = [Z0[(t-1)*(n+m+1)+n .+ (1:m)] for t = 1:T-1]
    H = [Z0[(t-1)*(n+m+1)+n+m + 1] for t = 1:T-1]

    return X, U, H
end

function init_MOI_Problem(prob::TrajectoryOptimizationProblem)
    return MOIProblem(prob.N,prob.M,prob,false)
end


function primal_bounds(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T
    idx = prob.idx

    N = prob.N

    Zl = -Inf*ones(N)
    Zu = Inf*ones(N)

    for t = 1:T-1
        Zl[idx.x[t]] = (t==1 ? prob.x1 : prob.xl[t])
        Zl[idx.u[t]] = prob.ul[t]
        Zl[idx.h[t]] = prob.hl[t]

        Zu[idx.x[t]] = (t==1 ? prob.x1 : prob.xu[t])
        Zu[idx.u[t]] = prob.uu[t]
        Zu[idx.h[t]] = prob.hu[t]
    end

    Zl[idx.x[T]] = (prob.goal_constraint ? prob.xT : prob.xl[T])
    Zu[idx.x[T]] = (prob.goal_constraint ? prob.xT : prob.xu[T])

    return Zl, Zu
end

function constraint_bounds(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T
    idx = prob.idx
    M = prob.M
    m_stage = prob.m_stage
    c_stage_ineq = prob.c_stage_ineq

    cl = zeros(M)
    cu = zeros(M)

    shift = 0
    for t = 1:T-1
        cu[(n*(T-1) + (T-2) + shift .+ (1:m_stage[t]))[c_stage_ineq[t]]] .= Inf
        shift += m_stage[t]
    end

    return cl, cu
end

function eval_objective(prob::TrajectoryOptimizationProblem,Z)
    objective(Z,prob.obj,prob.model,prob.idx,prob.T)
end

function eval_objective_gradient!(∇l,Z,prob::TrajectoryOptimizationProblem)
    objective_gradient!(∇l,Z,prob.obj,prob.model,prob.idx,prob.T)
    return nothing
end

function eval_constraint!(c,Z,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    dynamics_constraints!(view(c,1:(n*(T-1) + (T-2))),Z,
        prob.idx,prob.n,prob.m,prob.T,prob.model,prob.integration)

    prob.stage_constraints > 0 && stage_constraints!(view(c,(n*(T-1) + (T-2)) .+ (1:sum(prob.m_stage))),
        Z,prob.idx,T,prob.m_stage)

    return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    len_dyn_jac = length(sparsity_dynamics_jacobian(prob.idx,prob.n,prob.m,prob.T))
    sparse_dynamics_constraints_jacobian!(view(∇c,1:len_dyn_jac),Z,
        prob.idx,prob.n,prob.m,prob.T,prob.model,prob.integration)
    len_stage_jac = length(stage_constraint_sparsity(prob.idx,prob.T,prob.m_stage))

    prob.stage_constraints > 0 && ∇stage_constraints!(view(∇c,len_dyn_jac .+ (1:len_stage_jac)),Z,prob.idx,prob.T,prob.m_stage)

    return nothing
end

function sparsity_jacobian(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T
    sparsity_dynamics = sparsity_dynamics_jacobian(prob.idx,prob.n,prob.m,prob.T)
    sparsity_stage = stage_constraint_sparsity(prob.idx,prob.T,prob.m_stage,shift_r=(n*(T-1) + (T-2)))
    collect([sparsity_dynamics...,sparsity_stage...])
end
