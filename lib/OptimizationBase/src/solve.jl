struct IncompatibleOptimizerError <: Exception
    err::String
end

function Base.showerror(io::IO, e::IncompatibleOptimizerError)
    print(io, e.err)
end

"""
```julia
solve(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm,
    args...; kwargs...)::OptimizationSolution
```

For information about the returned solution object, refer to the documentation for [`OptimizationSolution`](@ref)

## Keyword Arguments

The arguments to `solve` are common across all of the optimizers.
These common arguments are:

  - `maxiters`: the maximum number of iterations
  - `maxtime`: the maximum amount of time (typically in seconds) the optimization runs for
  - `abstol`: absolute tolerance in changes of the objective value
  - `reltol`: relative tolerance  in changes of the objective value
  - `callback`: a callback function

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `solve`. Similarly, the special
keyword arguments for the `local_method` of a global optimizer are passed as a
`NamedTuple` to `local_options`.

Over time, we hope to cover more of these keyword arguments under the common interface.

A warning will be shown if a common argument is not implemented for an optimizer.

## Callback Functions

The callback function `callback` is a function that is called after every optimizer
step. Its signature is:

```julia
callback = (state, loss_val) -> false
```

where `state` is an `OptimizationState` and stores information for the current
iteration of the solver and `loss_val` is loss/objective value. For more
information about the fields of the `state` look at the `OptimizationState`
documentation. The callback should return a Boolean value, and the default
should be `false`, so the optimization stops if it returns `true`.

### Callback Example

Here we show an example of a callback function that plots the prediction at the current value of the optimization variables.
For a visualization callback, we would need the prediction at the current parameters i.e. the solution of the `ODEProblem` `prob`.
So we call the `predict` function within the callback again.

```julia
function predict(u)
    Array(solve(prob, Tsit5(), p = u))
end

function loss(u, p)
    pred = predict(u)
    sum(abs2, batch .- pred)
end

callback = function (state, l; doplot = false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
        pred = predict(state.u)
        pl = scatter(t, ode_data[1, :], label = "data")
        scatter!(pl, t, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end
```

If the chosen method is a global optimizer that employs a local optimization
method, a similar set of common local optimizer arguments exists. Look at `MLSL` or `AUGLAG`
from NLopt for an example. The common local optimizer arguments are:

  - `local_method`: optimizer used for local optimization in global method
  - `local_maxiters`: the maximum number of iterations
  - `local_maxtime`: the maximum amount of time (in seconds) the optimization runs for
  - `local_abstol`: absolute tolerance in changes of the objective value
  - `local_reltol`: relative tolerance  in changes of the objective value
  - `local_options`: `NamedTuple` of keyword arguments for local optimizer
"""
function solve(prob::SciMLBase.OptimizationProblem, alg, args...;
        kwargs...)::SciMLBase.AbstractOptimizationSolution
    if SciMLBase.supports_opt_cache_interface(alg)
        solve!(init(prob, alg, args...; kwargs...))
    else
        if prob.u0 !== nothing && !isconcretetype(eltype(prob.u0))
            throw(SciMLBase.NonConcreteEltypeError(eltype(prob.u0)))
        end
        _check_opt_alg(prob, alg; kwargs...)
        __solve(prob, alg, args...; kwargs...)
    end
end

function solve(
        prob::SciMLBase.EnsembleProblem{T}, args...; kwargs...) where {T <:
                                                                       SciMLBase.OptimizationProblem}
    return __solve(prob, args...; kwargs...)
end

function _check_opt_alg(prob::SciMLBase.OptimizationProblem, alg; kwargs...)
    !allowsbounds(alg) && (!isnothing(prob.lb) || !isnothing(prob.ub)) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support box constraints. Either remove the `lb` or `ub` bounds passed to `OptimizationProblem` or use a different algorithm."))
    requiresbounds(alg) && isnothing(prob.lb) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires box constraints. Either pass `lb` and `ub` bounds to `OptimizationProblem` or use a different algorithm."))
    !allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support constraints. Either remove the `cons` function passed to `OptimizationFunction` or use a different algorithm."))
    requiresconstraints(alg) && isnothing(prob.f.cons) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraints, pass them with the `cons` kwarg in `OptimizationFunction`."))
    # Check that if constraints are present and the algorithm supports constraints, both lcons and ucons are provided
    allowsconstraints(alg) && !isnothing(prob.f.cons) &&
        (isnothing(prob.lcons) || isnothing(prob.ucons)) &&
        throw(ArgumentError("Constrained optimization problem requires both `lcons` and `ucons` to be provided to OptimizationProblem. " *
                            "Example: OptimizationProblem(optf, u0, p; lcons=[-Inf], ucons=[0.0])"))
    !allowscallback(alg) && haskey(kwargs, :callback) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) does not support callbacks, remove the `callback` keyword argument from the `solve` call."))
    requiresgradient(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires gradients, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoForwardDiff())` or pass it in with `grad` kwarg."))
    requireshessian(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires hessians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(); kwargs...)` or pass them in with `hess` kwarg."))
    requiresconsjac(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraint jacobians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(); kwargs...)` or pass them in with `cons` kwarg."))
    requiresconshess(alg) &&
        !(prob.f isa SciMLBase.AbstractOptimizationFunction) &&
        throw(IncompatibleOptimizerError("The algorithm $(typeof(alg)) requires constraint hessians, hence use `OptimizationFunction` to generate them with an automatic differentiation backend e.g. `OptimizationFunction(f, AutoFiniteDiff(), AutoFiniteDiff(hess=true); kwargs...)` or pass them in with `cons` kwarg."))
    return
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

"""
```julia
init(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm, args...; kwargs...)
```

## Keyword Arguments

The arguments to `init` are the same as to `solve` and common across all of the optimizers.
These common arguments are:

  - `maxiters` (the maximum number of iterations)
  - `maxtime` (the maximum of time the optimization runs for)
  - `abstol` (absolute tolerance in changes of the objective value)
  - `reltol` (relative tolerance  in changes of the objective value)
  - `callback` (a callback function)

Some optimizer algorithms have special keyword arguments documented in the
solver portion of the documentation and their respective documentation.
These arguments can be passed as `kwargs...` to `init`.

See also [`solve(prob::OptimizationProblem, alg, args...; kwargs...)`](@ref)
"""
function init(prob::SciMLBase.OptimizationProblem, alg, args...;
        kwargs...)::SciMLBase.AbstractOptimizationCache
    if prob.u0 !== nothing && !isconcretetype(eltype(prob.u0))
        throw(SciMLBase.NonConcreteEltypeError(eltype(prob.u0)))
    end
    _check_opt_alg(prob::SciMLBase.OptimizationProblem, alg; kwargs...)
    cache = __init(prob, alg, args...; prob.kwargs..., kwargs...)
    return cache
end

"""
```julia
solve!(cache::AbstractOptimizationCache)
```

Solves the given optimization cache.

See also [`init(prob::OptimizationProblem, alg, args...; kwargs...)`](@ref)
"""
function solve!(cache::SciMLBase.AbstractOptimizationCache)::SciMLBase.AbstractOptimizationSolution
    __solve(cache)
end

# needs to be defined for each cache
function __solve(cache::SciMLBase.AbstractOptimizationCache)::SciMLBase.AbstractOptimizationSolution end
function __init(prob::SciMLBase.OptimizationProblem, alg, args...;
        kwargs...)::SciMLBase.AbstractOptimizationCache
    throw(OptimizerMissingError(alg))
end

# if no cache interface is supported at least the following method has to be defined
function __solve(prob::SciMLBase.OptimizationProblem, alg, args...; kwargs...)
    throw(OptimizerMissingError(alg))
end
