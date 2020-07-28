include("../src/DIRTREL.jl")
include("../dynamics/dubins.jl")
using Plots

# Horizon
T = 30

# Bounds

# ul <= u <= uu
uu = 5.0
ul = -5.0

# h = h0 (fixed timestep)
tf0 = 1.0
h0 = tf0/(T-1)
hu = 0.05
hl = 0.0

# Initial and final states
x1 = [0.0; 0.0; 0.0]
xT = [1.0; 1.0; 0.0]

# Circle obstacle
r = 0.1
xc1 = 0.85
yc1 = 0.3
xc2 = 0.375
yc2 = 0.75
xc3 = 0.25
yc3 = 0.25
xc4 = 0.75
yc4 = 0.75

# Constraints
function con_obstacles!(c,x,u)
    c[1] = circle_obs(x[1],x[2],xc1,yc1,r)
    c[2] = circle_obs(x[1],x[2],xc2,yc2,r)
    c[3] = circle_obs(x[1],x[2],xc3,yc3,r)
    c[4] = circle_obs(x[1],x[2],xc4,yc4,r)
    nothing
end
m_con_obstacles = 4

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
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
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
                    m_con=m_con_obstacles
                    )

# Robust problem
prob_robust = robust_problem(prob,model.nw,
    Q_lqr,R_lqr,
    Qw,Rw,
    E1,H1,D,
    robust_control_bnds=true,
    robust_stage=:state)

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)
prob_robust_moi = init_MOI_RobustProblem(prob_robust)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.01*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# # test methods
# m_con_obstacles = 1
# M_con = m_con_obstacles*(T-2)
#
# c_obs = zeros(M_con)
# stage_constraints!(c_obs,Z0,prob.idx,T,con_obstacles!,m_con_obstacles)
# tmp(c,z) = stage_constraints!(c,z,prob.idx,T,con_obstacles!,m_con_obstacles)
#
# sp = stage_constraint_sparsity(prob.idx,T,m_con_obstacles)
# ln = length(sp)
# ∇c_vec = zeros(ln)
# ∇c = zeros(M_con,prob.N)
# ∇stage_constraints!(∇c_vec,Z0,prob.idx,T,con_obstacles!,m_con_obstacles)
# ForwardDiff.jacobian(tmp,c_obs,Z0)[1,prob.idx.x[2]]
# for (i,k) in enumerate(sp)
#     ∇c[k[1],k[2]] = ∇c_vec[i]
# end
# norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp,c_obs,Z0)))
#
# m_con_robust = num_robust_stage(m_con_obstacles,model.nx,model.nu,T)
# c_robust = zeros(m_con_robust)
# prob_robust.w0
# prob.integration
# stage_constraints_robust!(c_robust,Z0,model.nx,model.nu,T,prob.idx,prob_robust.nw,prob_robust.w0,model,prob.integration,Q_lqr,R_lqr,Qw,Rw,prob_robust.E1,prob_robust.H1,prob_robust.D,
#         con_obstacles!,m_con_obstacles)

# compute_δx(Z0,model.nx,model.nu,T,prob.idx,prob_robust.nw,prob_robust.w0,model,
#     prob.integration,Q_lqr,R_lqr,Qw,Rw,prob_robust.E1,prob_robust.H1,
#     prob_robust.D)
# tmp_δx(z) = compute_δx(z,model.nx,model.nu,T,prob.idx,prob_robust.nw,
#     prob_robust.w0,model,prob.integration,Q_lqr,R_lqr,Qw,Rw,prob_robust.E1,
#     prob_robust.H1,prob_robust.D)
# norm(vec(ForwardDiff.jacobian(tmp_δx,Z0))
#     - vec(compute_∇δx(Z0,model.nx,model.nu,T,prob.idx,prob_robust.nw,prob_robust.w0,
#     model,prob.integration,Q_lqr,R_lqr,Qw,Rw,prob_robust.E1,prob_robust.H1,
#     prob_robust.D)))
#
#
#
# tmp_scr(c,z) = stage_constraints_robust!(c,z,model.nx,model.nu,T,prob.idx,prob_robust.nw,prob_robust.w0,model,prob.integration,Q_lqr,R_lqr,Qw,Rw,prob_robust.E1,prob_robust.H1,prob_robust.D,
#         con_obstacles!,m_con_obstacles)
# ForwardDiff.jacobian(tmp_scr,zeros(m_con_robust),Z0)
# sc_sparsity = stage_constraints_robust_sparsity(model.nx,model.nu,prob.N,T,prob.idx,m_con_obstacles)
# l_sc = length(sc_sparsity)
# ∇c_robust_vec = zeros(l_sc)
# ∇stage_constraints_robust!(∇c_robust_vec,Z0,model.nx,model.nu,prob.N,T,prob.idx,prob_robust.nw,prob_robust.w0,model,prob.integration,Q_lqr,R_lqr,Qw,Rw,prob_robust.E1,prob_robust.H1,prob_robust.D,
#         con_obstacles!,m_con_obstacles)
# ∇c_robust = zeros(m_con_robust,prob.N)
#
# for (i,k) in enumerate(sc_sparsity)
#     ∇c_robust[k[1],k[2]] = ∇c_robust_vec[i]
# end
# norm(vec(∇c_robust))
# norm(vec(ForwardDiff.jacobian(tmp_scr,zeros(m_con_robust),Z0)))
#
# norm(vec(∇c_robust) - vec(ForwardDiff.jacobian(tmp_scr,zeros(m_con_robust),Z0)))
# sum(vec(∇c_robust))
# sum(vec(ForwardDiff.jacobian(tmp_scr,zeros(m_con_robust),Z0)))
#
#
# mat_tmp(x) = [x[1] 0.0; 0.0 x[2]]
# sqrt_mat(x) = sqrt(mat_tmp(x))
# fast_sqrt_mat(x) = fastsqrt(mat_tmp(x))
# fast_sqrt_mat_vec(x) = vec(fast_sqrt_mat(x))
# fast_sqrt_vec(A) = vec(fastsqrt(reshape(A,2,2)))
#
# A0 = mat_tmp(w0)
# w0 = rand(2)
# In = Diagonal(ones(2))
#
# norm(vec(sqrt_mat(w0)) - vec(fast_sqrt_mat(w0)))
#
# ForwardDiff.jacobian(fast_sqrt_vec,vec(A0))
#
# pinv(kron(In,sqrt(A0)) + kron(sqrt(A0)',In))

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

# Solve robust problem
@time Z_robust = solve(prob_robust_moi,copy(Z0))

X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
X_robust, U_robust, H_robust = unpack(Z_robust,prob)

# Time trajectories
t_nominal = zeros(T)
t_robust = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_robust[t] = t_robust[t-1] + H_robust[t-1]
end

display("time (nominal): $(sum(H_nom))s")
display("time (robust): $(sum(H_robust))s")

# Plots results

# Position trajectory
x_nom_pos = [X_nom[t][1] for t = 1:T]
y_nom_pos = [X_nom[t][2] for t = 1:T]
pts = Plots.partialcircle(0,2π,100,r)
cx,cy = Plots.unzip(pts)
cx1 = [_cx + xc1 for _cx in cx]
cy1 = [_cy + yc1 for _cy in cy]
cx2 = [_cx + xc2 for _cx in cx]
cy2 = [_cy + yc2 for _cy in cy]
cx3 = [_cx + xc3 for _cx in cx]
cy3 = [_cy + yc3 for _cy in cy]
cx4 = [_cx + xc4 for _cx in cx]
cy4 = [_cy + yc4 for _cy in cy]
plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
plt = plot!(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
plt = plot!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=2.0,label="nominal",color=:purple,legend=:topleft)

x_robust_pos = [X_robust[t][1] for t = 1:T]
y_robust_pos = [X_robust[t][2] for t = 1:T]
plt = plot!(x_robust_pos,y_robust_pos,aspect_ratio=:equal,width=2.0,label="robust (cost)",color=:orange,legend=:bottomright)
savefig(plt,joinpath(@__DIR__,"results/dubins_state.png"))

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',color=:purple,width=2.0,
    title="Dubins",xlabel="time (s)",ylabel="control",label=["v (nominal)" "ω (nominal)"],
    legend=:bottomright,linetype=:steppost)
plt = plot!(t_robust[1:T-1],Array(hcat(U_robust...))',color=:orange,
    width=2.0,label=["v (robust)" "ω (robust)"],linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/dubins_control.png"))
