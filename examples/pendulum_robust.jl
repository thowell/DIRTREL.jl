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

# disturbances
nw = 1
w0 = ones(nw)
E1 = Diagonal(1.0e-6*ones(n))
H1 = zeros(n,nw)
D = Diagonal([0.2^2])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;1.0]) : Diagonal([100.0; 100.0]) for t = 1:T]
R_lqr = [Diagonal(0.1*ones(m)) for t = 1:T-1]

# Robust cost
Qw = deepcopy(Q_lqr)
Rw = deepcopy(R_lqr)

# Problem
prob = init_problem(n,m,T,x1,xT,model,obj,
                    ul=[ul*ones(m) for t=1:T-1],
                    uu=[uu*ones(m) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=midpoint,
                    goal_constraint=true)

prob_robust = RobustProblem(prob,nw,w0,Q_lqr,R_lqr,Qw,Rw,E1,H1,D,2*(2*m*m*(T-1)))
prob_robust_moi = init_MOI_RobustProblem(prob_robust)

# Initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.01*rand(m) for t = 1:T-1]
tf0 = 2.0
h0 = tf0/(T-1)
Z0 = pack(X0,U0,h0,prob)

# MathOptInterface problem

primal_bounds(prob_robust_moi)
constraint_bounds(prob_robust_moi)
MOI.eval_objective(prob_robust_moi,Z0)
MOI.eval_objective_gradient(prob_robust_moi,zeros(prob_robust_moi.n),Z0)
MOI.eval_constraint(prob_robust_moi,zeros(prob_robust_moi.m),Z0)
# MOI.eval_constraint_jacobian(prob_robust_moi,zeros(),Z0)
sparsity_jacobian(prob_robust_moi)

# Solve
@time Z_sol = solve(prob_robust_moi,Z0)

# Unpack solution
X, U, H = unpack(Z_sol,prob)

sum(H)
# Plot trajectories
using Plots
plot(Array(hcat(X...))',width=2.0,xlabel="time step",ylabel="state",label="",title="Pendulum")
plot(Array(hcat(U...))',width=2.0,xlabel="time step",ylabel="control",label="",title="Pendulum")
plot(Array(hcat(H...))',width=2.0,xlabel="time step",ylabel="h",label="",title="Pendulum")


g(x) = [x[1] 0.0 0.0; 0.0 x[2] 0.0; 0.0 0.0 x[3]]

sqrt_g(x) = sqrt(g(x))
chol_g(x) = Array(cholesky(g(x)))

x0 = rand(3)
sqrt_g(x0)
chol_g(x0)

e = eigen(g(x0))
e_sqrt = sqrt.(e.values)

g(x0)
e.vectors*Diagonal(e.values)*e.vectors'
S = e.vectors*Diagonal(e_sqrt)*e.vectors



tmp(x) = matrix_sqrt(g(x))*x
tmp2(x) = sqrt_g(x)*x

ForwardDiff.jacobian(tmp2,x0)
matrix_sqrt(g(x0))

norm(vec(matrix_sqrt(g(x0))) - vec(sqrt_g(x0)))


A = [33.0 24.0; 48.0 57.0]
e = eigen(A)
A_sqrt = e.vectors*Diagonal(sqrt.(e.values))*inv(e.vectors)

A_sqrt*A_sqrt
