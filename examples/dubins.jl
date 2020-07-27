include("../src/DIRTREL.jl")
include("../dynamics/dubins.jl")
using Plots

# Horizon
T = 31

# Bounds

# ul <= u <= uu
uu = 3.0
ul = -3.0

# h = h0 (fixed timestep)
tf0 = 2.5
h0 = tf0/(T-1)
hu = 0.1
hl = 0.0

# Initial and final states
x1 = [0.0; 0.0; 0.0]
xT = [1.0; 1.0; 0.0]

# Circle obstacle
r = 0.1
xc = 0.5
yc = 0.5

# Constraints
function con_obstacles!(c,x,u)
    c[1] = circle_obs(x[1],x[2],xc,yc,r)
    nothing
end

m_con_obstacles = 1

# Objective
Q = [t < T ? Diagonal(zeros(model.nx)) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal(zeros(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# Initial disturbances
E1 = Diagonal(1.0e-8*ones(model.nx))
H1 = zeros(model.nx,model.nw)
D = Diagonal([1.0])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([1.0;1.0;1.0e-1]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]

# Robust cost
Qw = deepcopy(Q_lqr)
Rw = deepcopy(R_lqr)

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=rk3_implicit,
                    goal_constraint=true,
                    con=con_obstacles!,
                    m_con=m_con_obstacles)

# Robust problem
prob_robust = robust_problem(prob,model.nw,
    Q_lqr,R_lqr,
    Qw,Rw,
    E1,H1,D,
    robust_control_bnds=true)

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)
prob_robust_moi = init_MOI_RobustProblem(prob_robust)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

m_con_obstacles = 1
M_con = m_con_obstacles*(T-2)

c_obs = zeros(M_con)
stage_constraints!(c_obs,Z0,prob.idx,T,con_obstacles!,m_con_obstacles)
tmp(c,z) = stage_constraints!(c,z,prob.idx,T,con_obstacles!,m_con_obstacles)

sp = stage_constraint_sparsity(prob.idx,T,m_con_obstacles)
ln = length(sp)
∇c_vec = zeros(ln)
∇c = zeros(M_con,prob.N)
∇stage_constraints!(∇c_vec,Z0,prob.idx,T,con_obstacles!,m_con_obstacles)
ForwardDiff.jacobian(tmp,c_obs,Z0)[1,prob.idx.x[2]]
for (i,k) in enumerate(sp)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp,c_obs,Z0)))

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
x_nom_pos = [X_nom[t][1] for t = 1:T]
y_nom_pos = [X_nom[t][2] for t = 1:T]
pts = Plots.partialcircle(0,2π,100,r)
cx,cy = Plots.unzip(pts)
cx .+= xc
cy .+= yc
plot(Shape(cx,cy),color=:red,label="",linecolor=:red)
plot!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,width=2.0,label="nominal",color=:black,legend=:topleft)
# plot(hcat(U_nom...)')

# Solve robust problem
@time Z_robust = solve(prob_robust_moi,copy(Z0))

X_robust, U_robust, H_robust = unpack(Z_robust,prob)
x_robust_pos = [X_robust[t][1] for t = 1:T]
y_robust_pos = [X_robust[t][2] for t = 1:T]
plot!(x_robust_pos,y_robust_pos,aspect_ratio=:equal,width=2.0,label="robust (cost)",color=:cyan,legend=:topleft)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)
X_robust, U_robust, H_robust = unpack(Z_robust,prob)

# Time trajectories
t_nominal = zeros(T)
t_robust = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_robust[t] = t_robust[t-1] + H_robust[t-1]
end

display("time (nominal): $(sum(H_nominal))s")
display("time (robust): $(sum(H_robust))s")

# Plots results

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nominal...))',color=:purple,width=2.0,
    title="Cartpole",xlabel="time (s)",ylabel="control",label="nominal",
    legend=:topright,linetype=:steppost)
plt = plot!(t_robust[1:T-1],Array(hcat(U_robust...))',color=:orange,
    width=2.0,label="robust",linetype=:steppost)
savefig(plt,joinpath(pwd(),"examples/results/cartpole_control.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",
    ylabel="state",label="x (nominal)",title="Cartpole",legend=:topright)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],
    color=:purple,width=2.0,label="θ (nominal)")
plt = plot!(t_nominal,hcat(X_nominal...)[3,:],
    color=:purple,width=2.0,label="dx (nominal)")
plt = plot!(t_nominal,hcat(X_nominal...)[4,:],
    color=:purple,width=2.0,label="dθ (nominal)")
plt = plot!(t_robust,hcat(X_robust...)[1,:],
    color=:orange,width=2.0,label="x (robust)")
plt = plot!(t_robust,hcat(X_robust...)[2,:],
    color=:orange,width=2.0,label="θ (robust)")
plt = plot!(t_robust,hcat(X_robust...)[3,:],
    color=:orange,width=2.0,label="dx (robust)")
plt = plot!(t_robust,hcat(X_robust...)[4,:],
    color=:orange,width=2.0,label="dθ (robust)")
savefig(plt,joinpath(pwd(),"examples/results/cartpole_state.png"))
