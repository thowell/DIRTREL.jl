include("../src/DIRTREL.jl")
include("../dynamics/pendulum.jl")

# Bounds
uu = 3.0
ul = -3.0

hu = Inf
hl = 0.0

# Initial and final states
x1 = [0.0; 0.0]
xT = [π; 0.0]

# Horizon
T = 51

# Objective (minimum time)
Q = [Diagonal(zeros(n)) for t = 1:T]
R = [Diagonal(zeros(m)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [zeros(n) for t=1:T],[zeros(m) for t=1:T])

# Problem
prob = init_problem(n,m,T,x1,xT,model,obj,
                    ul=[ul*ones(m) for t=1:T-1],
                    uu=[uu*ones(m) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=midpoint,
                    goal_constraint=true)

# Initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.01*rand(m) for t = 1:T-1]
tf0 = 2.0
h0 = tf0/(T-1)
Z0 = pack(X0,U0,h0,prob)

# MathOptInterface problem
prob_moi = MOIProblem(prob)

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
plot(Array(hcat(X_sol...))',width=2.0,xlabel="time step",ylabel="state",label="",title="Pendulum")
plot(Array(hcat(U_sol...))',width=2.0,xlabel="time step",ylabel="control",label="",title="Pendulum")
plot(Array(hcat(H_sol...))',width=2.0,xlabel="time step",ylabel="h",label="",title="Pendulum")
