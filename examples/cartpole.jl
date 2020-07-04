include("../src/DIRTREL.jl")
include("../dynamics/cartpole.jl")

# Horizon
T = 101

# Bounds
uu = 10.0
ul = -10.0

tf0 = 5.0
h0 = tf0/(T-1)
hu = h0
hl = h0

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [0.0; π; 0.0; 0.0]

# Objective (minimum time)
Q = [Diagonal(ones(model.nx)) for t = 1:T]
R = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=rk3_implicit,
                    goal_constraint=true)

# Initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.01*rand(model.nu) for t = 1:T-1]
Z0 = pack(X0,U0,h0,prob)

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

primal_bounds(prob_moi)
constraint_bounds(prob_moi)
MOI.eval_objective(prob_moi,Z0)
MOI.eval_objective_gradient(prob_moi,zeros(prob_moi.n),Z0)
MOI.eval_constraint(prob_moi,zeros(prob_moi.m),Z0)
MOI.eval_constraint_jacobian(prob_moi,zeros(prob_moi.m*prob_moi.n),Z0)
sparsity_jacobian(prob_moi)

# Solve
@time Z_sol = solve(prob_moi,Z0)

# Unpack solution
X_sol, U_sol, H_sol = unpack(Z_sol,prob)

# Plot trajectories
using Plots
plot(Array(hcat(X_sol...))',width=2.0,xlabel="time step",ylabel="state",label="",title="Cartpole")
plot(Array(hcat(U_sol...))',width=2.0,xlabel="time step",ylabel="control",label="",title="Cartpole")
# plot(Array(hcat(H_sol...))',width=2.0,xlabel="time step",ylabel="h",label="",title="Cartpole")
