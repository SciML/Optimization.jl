import .MathOptInterface
const MOI = MathOptInterface

struct MOIOptimizationProblem{T,F<:OptimizationFunction,uType,P} <: MOI.AbstractNLPEvaluator
    f::F
    u0::uType
	p::P
    J::Matrix{T}
    H::Matrix{T}
    cons_H::Vector{Matrix{T}}
end

MOI.eval_objective(moiproblem::MOIOptimizationProblem, x) = moiproblem.f(x, moiproblem.p)

MOI.eval_constraint(moiproblem::MOIOptimizationProblem, g, x) = g .= moiproblem.f.cons(x)

MOI.eval_objective_gradient(moiproblem::MOIOptimizationProblem, G, x) = moiproblem.f.grad(G, x)

function MOI.eval_hessian_lagrangian(moiproblem::MOIOptimizationProblem{T}, h, x, σ, μ) where {T}
    n = length(moiproblem.u0)
    a = zeros(n, n)
    moiproblem.f.hess(a, x)
    if iszero(σ)
        fill!(h, zero(T))
    else
        moiproblem.f.hess(moiproblem.H, x)
        k = 0
        for col in 1:n
            for row in 1:col
                k += 1
                h[k] = σ * moiproblem.H[row, col]
            end
        end
    end
    if !isempty(μ) && !all(iszero, μ)
        moiproblem.f.cons_h(moiproblem.cons_H, x)
        for (μi, Hi) in zip(μ, moiproblem.cons_H)
            k = 0
            for col in 1:n
                for row in 1:col
                    k += 1
                    h[k] += μi * Hi[row, col]
                end
            end
        end
    end
    return
end

function MOI.eval_constraint_jacobian(moiproblem::MOIOptimizationProblem, j, x)
    isempty(j) && return
    moiproblem.f.cons_j === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")
    n = length(moiproblem.u0)
    moiproblem.f.cons_j(moiproblem.J, x)
    for i in eachindex(j)
        j[i] = moiproblem.J[i]
    end
end

function MOI.jacobian_structure(moiproblem::MOIOptimizationProblem)
    return Tuple{Int,Int}[(con, var) for con in 1:size(moiproblem.J,1) for var in 1:size(moiproblem.J,2)]
end

function MOI.hessian_lagrangian_structure(moiproblem::MOIOptimizationProblem)
    return Tuple{Int,Int}[(row, col) for col in 1:length(moiproblem.u0) for row in 1:col]
end

function MOI.initialize(moiproblem::MOIOptimizationProblem, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(moiproblem))
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end

MOI.features_available(moiproblem::MOIOptimizationProblem) = [:Grad, :Hess, :Jac]

function make_moi_problem(prob::OptimizationProblem)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p,num_cons)
    T = eltype(prob.u0)
    n = length(prob.u0)
    moiproblem = MOIOptimizationProblem(f,prob.u0,prob.p,zeros(T,num_cons,n),zeros(T,n,n),Matrix{T}[zeros(T,n,n) for i in 1:num_cons])
    return moiproblem
end

function __map_optimizer_args(prob::OptimizationProblem, opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes};
    maxiters::Union{Number, Nothing}=nothing,
    maxtime::Union{Number, Nothing}=nothing,
    abstol::Union{Number, Nothing}=nothing,
    reltol::Union{Number, Nothing}=nothing,
    kwargs...)
    optimizer = MOI.instantiate(opt)
    for (key, value) in kwargs
        MOI.set(optimizer, MOI.RawOptimizerattribute("$(key)"), value)
    end
    if !isnothing(maxtime)
        MOI.set(optimizer, MOI.TimeLimitSec(), maxtime)
    end

    if !isnothing(reltol)
        @warn "common reltol argument is currently not used by $(optimizer). Set tolerances via optimizer specific keyword aguments."
    end

    if !isnothing(abstol)
        @warn "common abstol argument is currently not used by $(optimizer). Set tolerances via optimizer specific keyword aguments."
    end

    if !isnothing(maxiters)
        @warn "common maxiters argument is currently not used by $(optimizer). Set number of interations via optimizer specific keyword aguments."
    end

    return optimizer
end

function __solve(prob::OptimizationProblem, opt::Union{MOI.AbstractOptimizer, MOI.OptimizerWithAttributes};
    maxiters::Union{Number, Nothing}=nothing,
    maxtime::Union{Number, Nothing}=nothing,
    abstol::Union{Number, Nothing}=nothing,
    reltol::Union{Number, Nothing}=nothing,
    kwargs...)

    maxiters = _check_and_convert_maxiters(maxiters)
    maxtime = _check_and_convert_maxtime(maxtime)

    opt_setup = __map_optimizer_args(prob, opt; abstol=abstol, reltol=reltol, maxiters=maxiters, maxtime=maxtime, kwargs...)

    num_variables = length(prob.u0)
	θ = MOI.add_variables(opt_setup, num_variables)
	if prob.lb !== nothing
        @assert eachindex(prob.lb) == Base.OneTo(num_variables)
		for i in 1:num_variables
			MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(prob.lb[i]))
        end
    end
	if prob.ub !== nothing
        @assert eachindex(prob.ub) == Base.OneTo(num_variables)
		for i in 1:num_variables
			MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(prob.ub[i]))
        end
    end
    @assert eachindex(prob.u0) == Base.OneTo(num_variables)
    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], prob.u0[i])
        end
    end
    MOI.set(opt_setup, MOI.ObjectiveSense(), prob.sense === MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE)
    if prob.lcons === nothing
        @assert prob.ucons === nothing
        con_bounds = MOI.NLPBoundsPair[]
    else
        @assert prob.ucons !== nothing
        con_bounds = MOI.NLPBoundsPair.(prob.lcons, prob.ucons)
    end
	MOI.set(opt_setup, MOI.NLPBlock(), MOI.NLPBlockData(con_bounds, make_moi_problem(prob), true))

	MOI.optimize!(opt_setup)

    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = Symbol(string(MOI.get(opt_setup, MOI.TerminationStatus())))
    else
        minimizer = fill(NaN, num_variables)
        minimum = NaN
        opt_ret= :Default
    end
    SciMLBase.build_solution(prob, opt, minimizer, minimum; original=opt_setup, retcode=opt_ret)
end
