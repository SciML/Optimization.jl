# This file contains the top level solve interface functionality moved from SciMLBase.jl
# These functions provide the core optimization solving interface

struct IncompatibleOptimizerError <: Exception
    err::String
end

function Base.showerror(io::IO, e::IncompatibleOptimizerError)
    print(io, e.err)
end

const OPTIMIZER_MISSING_ERROR_MESSAGE = """
                                        Optimization algorithm not found. Either the chosen algorithm is not a valid solver
                                        choice for the `OptimizationProblem`, or the Optimization solver library is not loaded.
                                        Make sure that you have loaded an appropriate Optimization.jl solver library, for example,
                                        `solve(prob,Optim.BFGS())` requires `using OptimizationOptimJL` and
                                        `solve(prob,Adam())` requires `using OptimizationOptimisers`.

                                        For more information, see the Optimization.jl documentation: <https://docs.sciml.ai/Optimization/stable/>.
                                        """

struct OptimizerMissingError <: Exception
    alg::Any
end

function Base.showerror(io::IO, e::OptimizerMissingError)
    println(io, OPTIMIZER_MISSING_ERROR_MESSAGE)
    print(io, "Chosen Optimizer: ")
    print(e.alg)
end

# Algorithm compatibility checking function
function _check_opt_alg(prob::SciMLBase.OptimizationProblem, alg; kwargs...)
    !SciMLBase.allowsbounds(alg) && (!isnothing(prob.lb) || !isnothing(prob.ub)) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support box constraints. Either remove the `lb` or `ub` bounds passed to `OptimizationProblem` or use a different algorithm."))
    SciMLBase.requiresbounds(alg) && isnothing(prob.lb) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires box constraints. Either pass `lb` and `ub` bounds to `OptimizationProblem` or use a different algorithm."))
    !SciMLBase.allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support constraints. Either remove the `cons` function passed to `OptimizationFunction` or use a different algorithm."))
    SciMLBase.requiresconstraints(alg) && isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraints, pass them with the `cons` kwarg in `OptimizationFunction`."))
    # Check that if constraints are present and the algorithm supports constraints, both lcons and ucons are provided
    SciMLBase.allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        (isnothing(prob.lcons) || isnothing(prob.ucons)) &&
        throw(ArgumentError("Constrained optimization problem requires both `lcons` and `ucons` to be provided to OptimizationProblem. " *
                            "Example: OptimizationProblem(optf, u0, p; lcons=[-Inf], ucons=[0.0])"))
    !SciMLBase.allowscallback(alg) && haskey(kwargs, :callback) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support callbacks, remove the `callback` keyword argument from the `solve` call."))
    SciMLBase.requiresgradient(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires gradients, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoForwardDiff())` or pass it in with `grad` kwarg."))
    SciMLBase.requireshessian(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires hessians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(); kwargs...)` or pass them in with `hess` kwarg."))
    SciMLBase.requiresconsjac(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraint jacobians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(); kwargs...)` or pass them in with `cons` kwarg."))
    SciMLBase.requiresconshess(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraint hessians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(), AutoFiniteDiff(hess=true); kwargs...)` or pass them in with `cons` kwarg."))
    return
end

# Base solver dispatch functions (these will be extended by specific solver packages)
supports_opt_cache_interface(alg) = false

function __solve(cache::SciMLBase.AbstractOptimizationCache)::SciMLBase.AbstractOptimizationSolution
    throw(OptimizerMissingError(cache.opt))
end

function __init(prob::SciMLBase.OptimizationProblem, alg, args...;
        kwargs...)::SciMLBase.AbstractOptimizationCache
    throw(OptimizerMissingError(alg))
end

function __solve(prob::SciMLBase.OptimizationProblem, alg, args...; kwargs...)
    throw(OptimizerMissingError(alg))
end
