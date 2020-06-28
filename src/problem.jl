include("../dynamics/pendulum.jl")

# z = (x,u,h)
# Z = [z1,...,zT-1,xT]
abstract type Problem end

mutable struct TrajectoryOptimizationProblem <: Problem
    n::Int # states
    m::Int # controls
    T::Int # horizon
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
    goal_constraint
end

function init_problem(n,m,T,x1,xT,model,obj;
        ul=[-Inf*ones(m) for t = 1:T-1],
        uu=[Inf*ones(m) for t = 1:T-1],
        xl=[-Inf*ones(n) for t = 1:T],
        xu=[Inf*ones(n) for t = 1:T],
        hl=[-Inf for t = 1:T-1],
        hu=[Inf for t = 1:T-1],
        integration=midpoint,
        goal_constraint::Bool=true)

    idx = init_indices(n,m,T)

    return TrajectoryOptimizationProblem(n,m,T,
        x1,xT,
        ul,uu,
        xl,xu,
        hl,hu,
        idx,
        model,integration,
        obj,
        goal_constraint)
end

function pack(X0,U0,h0,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    Z0 = zeros(n*T + m*(T-1) + (T-1))
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

function MOIProblem(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    N = n*T + m*(T-1) + (T-1)
    M = n*(T-1) + (T-2)

    return MOIProblem(N,M,prob,false)
end


function primal_bounds(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T
    idx = prob.idx

    N = n*T + m*(T-1) + (T-1)

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

    M = n*(T-1) + (T-2)

    cl = zeros(M)
    cu = zeros(M)

    return cl, cu
end

function eval_objective(prob::TrajectoryOptimizationProblem,Z)
    objective(Z,prob.obj,prob.idx,prob.T)
end

function eval_objective_gradient!(∇l,Z,prob::TrajectoryOptimizationProblem)
    objective_gradient!(∇l,Z,prob.obj,prob.idx,prob.T)
    return nothing
end

function eval_constraint!(c,Z,prob::TrajectoryOptimizationProblem)
    dynamics_constraints!(c,Z,
        prob.idx,prob.n,prob.m,prob.T,prob.model,prob.integration)
    return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    sparse_dynamics_constraints_jacobian!(∇c,Z,
        prob.idx,prob.n,prob.m,prob.T,prob.model,prob.integration)
    return nothing
end

function sparsity_jacobian(prob::TrajectoryOptimizationProblem)
    sparsity_dynamics_jacobian(prob.idx,prob.n,prob.m,prob.T)
end
