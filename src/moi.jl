struct MOIProblem <: MOI.AbstractNLPEvaluator
    n
    m
    prob::Problem
    enable_hessian::Bool
end

function primal_bounds(prob::MOI.AbstractNLPEvaluator)
    return primal_bounds(prob.prob)
end

function constraint_bounds(prob::MOI.AbstractNLPEvaluator)
    return constraint_bounds(prob.prob)
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    return eval_objective(prob.prob,x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    eval_objective_gradient!(grad_f,x,prob.prob)
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    eval_constraint!(g,x,prob.prob)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    eval_constraint_jacobian!(jac,x,prob.prob)
    return nothing
end

function sparsity_jacobian(prob::MOI.AbstractNLPEvaluator)
    sparsity_jacobian(prob.prob)
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob)
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = []
MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, μ) = nothing

function solve(prob::MOI.AbstractNLPEvaluator,x0)
    x_l, x_u = primal_bounds(prob)
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    solver = Ipopt.Optimizer()
    # # solver.options["nlp_scaling_method"] = "none"
    # # solver.options["max_iter"] = 5000
    solver.options["tol"] = 1.0e-3
    solver.options["constr_viol_tol"] = 1.0e-2

    # solver = SNOPT7.Optimizer()

    x = MOI.add_variables(solver,prob.n)

    for i = 1:prob.n
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    return MOI.get(solver, MOI.VariablePrimal(), x)
end
