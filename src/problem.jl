include("../dynamics/pendulum.jl")

# z = (x,u,h)
# Z = [z1,...,zT-1,xT]

mutable struct TrajectoryOptimizationProblem
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
end

function init_problem(n,m,T,x1,xT,model,obj;
        ul=-Inf*ones(m),
        uu=Inf*ones(m),
        xl=-Inf*ones(n),
        xu=Inf*ones(n),
        hl=-Inf*ones(1),
        hu=Inf*ones(1),
        integration=midpoint,
        goal_state::Bool=true)

    idx = init_indices(n,m,T)
    
    return TrajectoryOptimizationProblem(n,m,T,
        x1,xT,
        ul,uu,
        xl,xu,
        hl,hu,
        idx,
        model,integration,
        obj)
end
