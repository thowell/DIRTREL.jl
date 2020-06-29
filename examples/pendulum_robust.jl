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

# Disturbances
nw = 1
w0 = zeros(nw)
E1 = Diagonal(1.0e-8*ones(n))
H1 = zeros(n,nw)
D = Diagonal([0.2^2]) # NOTE: this is modified from paper

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
prob_robust = RobustProblem(prob,nw,w0,
    Q_lqr,R_lqr,
    Qw,Rw,
    E1,H1,D,
    num_robust_control_bounds(m,T))

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)
prob_robust_moi = init_MOI_RobustProblem(prob_robust)

# Initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.01*rand(m) for t = 1:T-1]
tf0 = 2.0
h0 = tf0/(T-1)
Z0 = pack(X0,U0,h0,prob)

# Solve
@time Z_nominal = solve(prob_moi,copy(Z0))
@time Z_robust = solve(prob_robust_moi,copy(Z0))

# Unpack solution
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)
X_robust, U_robust, H_robust = unpack(Z_robust,prob)

display("time (nominal): $(sum(H_nominal))s")
display("time (robust): $(sum(H_robust))s")

# Plot
using Plots

# Time
t_nominal = zeros(T)
t_robust = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_robust[t] = t_robust[t-1] + H_robust[t-1]
end

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nominal...))',
    color=:purple,width=2.0,title="Pendulum",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:bottomright,
    linetype=:steppost)
plt = plot!(t_robust[1:T-1],Array(hcat(U_robust...))',
    color=:orange,width=2.0,label="robust",linetype=:steppost)
savefig(plt,joinpath(pwd(),"examples/results/pendulum_control.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],color=:purple,width=2.0,xlabel="time (s)",ylabel="state",label="θ (nominal)",title="Pendulum",legend=:bottomright)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="dθ (nominal)")
plt = plot!(t_robust,hcat(X_robust...)[1,:],color=:orange,width=2.0,label="θ (robust)")
plt = plot!(t_robust,hcat(X_robust...)[2,:],color=:orange,width=2.0,label="dθ (robust)")
savefig(plt,joinpath(pwd(),"examples/results/pendulum_state.png"))
