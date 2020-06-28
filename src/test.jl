include("DIRTREL.jl")

T = 10
Q = [Matrix(1.0*I,n,n) for t = 1:T]
R = [Matrix(1.0*I,m,m) for t = 1:T-1]

x_ref = [zeros(n) for t = 1:T]
u_ref = [zeros(m) for t = 1:T-1]
obj = QuadraticTrackingObjective(Q,R,x_ref,u_ref)

x0 = rand(n)
xT = rand(n)
prob = init_problem(n,m,T,x0,xT,obj)

Z0 = rand(n*T + m*(T-1) + (T-1))
objective(Z0,obj,prob.idx,T)
obj_grad = zero(Z0)
objective_gradient(obj_grad,Z0,obj,prob.idx,T)

tmp(z) = objective(z,obj,prob.idx,T)
norm(ForwardDiff.gradient(tmp,Z0) - obj_grad)
