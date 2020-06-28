include("DIRTREL.jl")

T = 10
Q = [Matrix(1.0*I,n,n) for t = 1:T]
R = [Matrix(1.0*I,m,m) for t = 1:T-1]
c = 1.0
x_ref = [zeros(n) for t = 1:T]
u_ref = [zeros(m) for t = 1:T-1]
obj = QuadraticTrackingObjective(Q,R,c,x_ref,u_ref)

x1 = rand(n)
xT = rand(n)
prob = init_problem(n,m,T,x1,xT,model,obj)

Z0 = ones(n*T + m*(T-1) + (T-1))
objective(Z0,obj,prob.idx,T)
obj_grad = zero(Z0)
objective_gradient!(obj_grad,Z0,obj,prob.idx,T)

tmp(z) = objective(z,obj,prob.idx,T)
norm(ForwardDiff.gradient(tmp,Z0) - obj_grad)

c0 = zeros(n*(T-1) + (T-2))

dynamics_constraints!(c0,Z0,prob.idx,n,m,T,prob.model,prob.integration)

c_jac = zeros(n*(T-1) + (T-2),n*T + m*(T-1) + (T-1))
dynamics_constraints_jacobian!(c_jac,Z0,prob.idx,n,m,T,prob.model,prob.integration)

tmp2!(c,z) = dynamics_constraints!(c,z,prob.idx,n,m,T,prob.model,prob.integration)

norm(vec(ForwardDiff.jacobian(tmp2!,c0,Z0)) - vec(c_jac))

primal_bounds(prob)

constraint_bounds(prob)

eval_objective(prob,Z0)
obj_grad_2 = zero(obj_grad)
eval_objective_gradient!(obj_grad_2,Z0,prob)
norm(obj_grad - obj_grad_2)

c0_2 = zero(c0)
eval_constraint!(c0_2,Z0,prob)
norm(c0 - c0_2)

c_jac_2 = zero(c_jac)
eval_constraint_jacobian!(c_jac_2,Z0,prob)
norm(c_jac - c_jac_2)
