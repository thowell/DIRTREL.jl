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
X, U, H = unpack(Z_sol,prob)

# Plot trajectories
using Plots
plot(Array(hcat(X...))',width=2.0,xlabel="time step",ylabel="state",label="",title="Pendulum")
plot(Array(hcat(U...))',width=2.0,xlabel="time step",ylabel="control",label="",title="Pendulum")
plot(Array(hcat(H...))',width=2.0,xlabel="time step",ylabel="h",label="",title="Pendulum")

# robust cost function
integration = prob.integration
idx = prob.idx

# TVLQR cost
Q_lqr = [Matrix(1.0*I,n,n) for t = 1:T]
R_lqr = [Matrix(1.0*I,m,m) for t = 1:T-1]

# robust cost
Ql = [Matrix(I,n,n) for t = 1:T]
Rl = [Matrix(I,m,m) for t = 1:T-1]

robust_cost(Z0)
ForwardDiff.gradient(robust_cost,Z0)

function robust_cost(Z)
    nw = 1
    w0 = zeros(nw)

    A = [zeros(eltype(Z),n,n) for t = 1:T-1]
    B = [zeros(eltype(Z),n,m) for t = 1:T-1]
    G = [zeros(eltype(Z),n,nw) for t = 1:T-1]

    for t = 1:T-1
        x = view(Z,idx.x[t])
        u = view(Z,idx.u[t])
        h = Z[idx.h[t]]
        x⁺ = view(Z,idx.x[t+1])

        dyn_x(z) = x⁺ - integration(model,z,u,w0,h)
        dyn_u(z) = x⁺ - integration(model,x,z,w0,h)
        dyn_w(z) = x⁺ - integration(model,x,u,z,h)

        A[t] = ForwardDiff.jacobian(dyn_x,x)
        B[t] = ForwardDiff.jacobian(dyn_u,u)
        G[t] = ForwardDiff.jacobian(dyn_w,w0)
    end


    # TVLQR
    K = [zeros(eltype(Z),m,n) for t = 1:T-1]
    P = [zeros(eltype(Z),n,n) for t = 1:T]

    P[T] = Q_lqr[T]
    for t = T-1:-1:1
        K[t] = (R_lqr[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
        P[t] = Q_lqr[t] + A[t]'*P[t+1]*A[t] - (A[t]'*P[t+1]*B[t])*K[t]
    end

    # robust cost function
    E1 = Matrix(I,n,n)

    l = 0

    E = E1
    H = zeros(eltype(Z),n,nw)
    D = Matrix(I,nw,nw)
    for t = 1:T-1
        l += tr(Ql[t] + K[t]'*Rl[t]*K[t]*E)
        E = ((A[t] - B[t]*K[t])*E*(A[t] - B[t]*K[t])'
            + (A[t] - B[t]*K[t])*H*G[t]'
            + G[t]*H'*(A[t] - B[t]*K[t])'
            + G[t]*D*G[t]')
        H = (A[t] - B[t]*K[t])*H + G[t]*D
    end

    l += tr(Ql[t]*E)

    return l
end
