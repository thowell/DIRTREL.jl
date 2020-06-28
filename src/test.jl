include("DIRTREL.jl")

T = 10
Q = [Matrix(1.0*I,n,n) for t = 1:T]
R = [Matrix(1.0*I,m,m) for t = 1:T-1]

x_ref = [zeros(n) for t = 1:T]
u_ref = [zeros(m) for t = 1:T-1]
obj = QuadraticTrackingObjective(Q,R,x_ref,u_ref)

x0 = rand(n)
xT = rand(n)
prob = init_problem(n,m,T,x0,xT,model,obj)

Z0 = ones(n*T + m*(T-1) + (T-1))
objective(Z0,obj,prob.idx,T)
obj_grad = zero(Z0)
objective_gradient(obj_grad,Z0,obj,prob.idx,T)

tmp(z) = objective(z,obj,prob.idx,T)
norm(ForwardDiff.gradient(tmp,Z0) - obj_grad)

c0 = zeros(n*(T-1) + (T-2))

dynamics_constraints!(c0,Z0,prob.idx,n,m,T,prob.model,prob.integration)

c_jac = zeros(n*(T-1) + (T-2),n*T + m*(T-1) + (T-1))
dynamics_constraints_jacobian!(c_jac,Z0,prob.idx,n,m,T,prob.model,prob.integration)

tmp2!(c,z) = dynamics_constraints!(c,z,prob.idx,n,m,T,prob.model,prob.integration)

norm(vec(ForwardDiff.jacobian(tmp2!,c0,Z0)) - vec(c_jac))
