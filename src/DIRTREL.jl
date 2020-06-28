using LinearAlgebra, ForwardDiff, StaticArrays
using MathOptInterface, Ipopt
const MOI = MathOptInterface

include("integration.jl")
include("indices.jl")
include("objective.jl")
include("dynamics_constraints.jl")
include("problem.jl")
include("moi.jl")
