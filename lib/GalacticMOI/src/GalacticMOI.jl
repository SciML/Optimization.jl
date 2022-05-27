module GalacticMOI

using MathOptInterface, GalacticOptim, GalacticOptim.SciMLBase, GalacticOptim.SparseArrays
const MOI = MathOptInterface

struct MOIOptimizationProblem{T,F<:OptimizationFunction,uType,P} <: MOI.AbstractNLPEvaluator
    f::F
    u0::uType
    p::P
    J::Union{Matrix{T},SparseMatrixCSC{T}}
    H::Union{Matrix{T},SparseMatrixCSC{T}}
    cons_H::Vector{<:Union{Matrix{T},SparseMatrixCSC{T}}}
end

function MOIOptimizationProblem(prob::OptimizationProblem)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = GalacticOptim.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p, num_cons)
    T = eltype(prob.u0)
    n = length(prob.u0)
    return MOIOptimizationProblem(
        f,
        prob.u0,
        prob.p,
        isnothing(f.cons_jac_prototype) ? zeros(T, num_cons, n) : f.cons_jac_prototype,
        isnothing(f.hess_prototype) ? zeros(T, n, n) : f.hess_prototype,
        isnothing(f.cons_hess_prototype) ? Matrix{T}[zeros(T, n, n) for i in 1:num_cons] : f.cons_hess_prototype
    )
end

MOI.features_available(::MOIOptimizationProblem) = [:Grad, :Hess, :Jac]

function MOI.initialize(
    moiproblem::MOIOptimizationProblem,
    requested_features::Vector{Symbol},
)
    available_features = MOI.features_available(moiproblem)
    for feat in requested_features
        if !(feat in available_features)
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
    return
end

function MOI.eval_objective(moiproblem::MOIOptimizationProblem, x)
    return moiproblem.f(x, moiproblem.p)
end

function MOI.eval_constraint(moiproblem::MOIOptimizationProblem, g, x)
    g .= moiproblem.f.cons(x)
    return
end

function MOI.eval_objective_gradient(moiproblem::MOIOptimizationProblem, G, x)
    moiproblem.f.grad(G, x)
    return
end

# This structure assumes the calculation of moiproblem.J is dense.
function MOI.jacobian_structure(moiproblem::MOIOptimizationProblem)
    rows, cols = size(moiproblem.J)
    return Tuple{Int,Int}[(i, j) for j in 1:cols for i in 1:rows]
end

function MOI.eval_constraint_jacobian(moiproblem::MOIOptimizationProblem, j, x)
    if isempty(j)
        return
    elseif moiproblem.f.cons_j === nothing
        error(
            "Use OptimizationFunction to pass the derivatives or " *
            "automatically generate them with one of the autodiff backends",
        )
    end
    moiproblem.f.cons_j(moiproblem.J, x)
    for i in eachindex(j)
        j[i] = moiproblem.J[i]
    end
    return
end

# Because the Hessian is symmetrical, we choose to store the upper-triangular
# component. We also assume that it is dense.
function MOI.hessian_lagrangian_structure(moiproblem::MOIOptimizationProblem)
    num_vars = length(moiproblem.u0)
    return Tuple{Int,Int}[(row, col) for col in 1:num_vars for row in 1:col]
end

function MOI.eval_hessian_lagrangian(
    moiproblem::MOIOptimizationProblem{T},
    h,
    x,
    σ,
    μ,
) where {T}
    n = length(moiproblem.u0)
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

_create_new_optimizer(opt::MOI.AbstractOptimizer) = opt
_create_new_optimizer(opt::MOI.OptimizerWithAttributes) = MOI.instantiate(opt)

function __map_optimizer_args(
    prob::OptimizationProblem,
    opt::Union{MOI.AbstractOptimizer,MOI.OptimizerWithAttributes};
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    kwargs...
)
    optimizer = _create_new_optimizer(opt)
    for (key, value) in kwargs
        MOI.set(optimizer, MOI.RawOptimizerAttribute("$(key)"), value)
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

function SciMLBase.__solve(
    prob::OptimizationProblem,
    opt::Union{MOI.AbstractOptimizer,MOI.OptimizerWithAttributes};
    maxiters::Union{Number,Nothing}=nothing,
    maxtime::Union{Number,Nothing}=nothing,
    abstol::Union{Number,Nothing}=nothing,
    reltol::Union{Number,Nothing}=nothing,
    kwargs...
)
    maxiters = GalacticOptim._check_and_convert_maxiters(maxiters)
    maxtime = GalacticOptim._check_and_convert_maxtime(maxtime)
    opt_setup = __map_optimizer_args(
        prob,
        opt;
        abstol=abstol,
        reltol=reltol,
        maxiters=maxiters,
        maxtime=maxtime,
        kwargs...
    )
    num_variables = length(prob.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
    if prob.lb !== nothing
        @assert eachindex(prob.lb) == Base.OneTo(num_variables)
        for i in 1:num_variables
            if prob.lb[i] > -Inf
                MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(prob.lb[i]))
            end
        end
    end
    if prob.ub !== nothing
        @assert eachindex(prob.ub) == Base.OneTo(num_variables)
        for i in 1:num_variables
            if prob.ub[i] < Inf
                MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(prob.ub[i]))
            end
        end
    end
    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        @assert eachindex(prob.u0) == Base.OneTo(num_variables)
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], prob.u0[i])
        end
    end
    MOI.set(
        opt_setup,
        MOI.ObjectiveSense(),
        prob.sense === GalacticOptim.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE,
    )
    if prob.lcons === nothing
        @assert prob.ucons === nothing
        con_bounds = MOI.NLPBoundsPair[]
    else
        @assert prob.ucons !== nothing
        con_bounds = MOI.NLPBoundsPair.(prob.lcons, prob.ucons)
    end
    MOI.set(
        opt_setup,
        MOI.NLPBlock(),
        MOI.NLPBlockData(con_bounds, MOIOptimizationProblem(prob), true),
    )
    MOI.optimize!(opt_setup)
    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = Symbol(string(MOI.get(opt_setup, MOI.TerminationStatus())))
    else
        minimizer = fill(NaN, num_variables)
        minimum = NaN
        opt_ret = :Default
    end
    return SciMLBase.build_solution(
        prob,
        opt,
        minimizer,
        minimum;
        original=opt_setup,
        retcode=opt_ret
    )
end


end
