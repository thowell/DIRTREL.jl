using LinearAlgebra, ForwardDiff, StaticArrays
using MathOptInterface, Ipopt
const MOI = MathOptInterface

include("utils.jl")
include("integration.jl")
include("indices.jl")
include("objective.jl")
include("dynamics_constraints.jl")
include("stage_constraints.jl")
include("problem.jl")
include("disturbance_trajectory.jl")
include("objective_robust.jl")
include("constraints_linear_control_robust.jl")
include("constraints_linear_state_robust.jl")
include("constraints_stage_robust.jl")
include("problem_robust.jl")
include("moi.jl")
