"""
    IncompatibleOptimizerError(msg)

Error thrown when an optimizer cannot solve the supplied `OptimizationProblem`
because required features, such as bounds, constraints, callbacks, gradients, or
hessians, are unsupported or missing.
"""
struct IncompatibleOptimizerError <: Exception
    err::String
end

function Base.showerror(io::IO, e::IncompatibleOptimizerError)
    return print(io, e.err)
end

"""
    solve(prob::OptimizationProblem, alg, args...; kwargs...)

Solve an `OptimizationProblem` with `alg` and return an
`AbstractOptimizationSolution`.

`solve` validates the problem against the algorithm's capability traits and
dispatches to the solver package that implements `alg`. Solver-specific
keywords are forwarded unchanged.

# Arguments

- `prob`: the problem to optimize.
- `alg`: an algorithm provided by an Optimization.jl solver package.
- `args...`: positional arguments accepted by the solver implementation.

# Keyword Arguments

- `sensealg`: sensitivity algorithm used by differentiation integrations.
- `u0`: replacement initial value for the problem.
- `p`: replacement parameter value for the problem.
- `wrap`: whether to return the standard Optimization solution wrapper.
- `kwargs...`: common and solver-specific options.

# Returns

An `AbstractOptimizationSolution` containing the final variables, objective
value, return code, and `OptimizationStats`.

# Callbacks

When supported by `alg`, `callback` is called after an optimization step as
`callback(state, objective)`. `state` is an `OptimizationState`, and returning
`true` stops the optimization. The default callback returns `false`.

# Examples

```julia
using Optimization, OptimizationOptimJL

f(u, p) = sum(abs2, u)
prob = OptimizationProblem(OptimizationFunction(f), [1.0, -2.0])
sol = solve(prob, Optim.BFGS(); maxiters = 100)
```
"""
function solve(
        prob::SciMLBase.OptimizationProblem, args...; sensealg = nothing,
        u0 = nothing, p = nothing, wrap = Val(true), kwargs...
    )::SciMLBase.AbstractOptimizationSolution
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end

    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p
    return if wrap isa Val{true}
        wrap_sol(
            solve_up(
                prob,
                sensealg,
                u0,
                p,
                args...;
                originator = SciMLBase.ChainRulesOriginator(),
                kwargs...
            )
        )
    else
        solve_up(
            prob,
            sensealg,
            u0,
            p,
            args...;
            originator = SciMLBase.ChainRulesOriginator(),
            kwargs...
        )
    end
end

function solve(
        prob::SciMLBase.EnsembleProblem{T}, args...; kwargs...
    ) where {
        T <:
        SciMLBase.OptimizationProblem,
    }
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
        throw(
        ArgumentError(
            "Constrained optimization problem requires both `lcons` and `ucons` to be provided to OptimizationProblem. " *
                "Example: OptimizationProblem(optf, u0, p; lcons=[-Inf], ucons=[0.0])"
        )
    )
    !allowscallback(alg) && !(get(kwargs, :callback, DEFAULT_CALLBACK) isa NullCallback) &&
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

"""
    OptimizerMissingError(alg)

Error thrown when `solve` or `init` cannot find an Optimization.jl solver
implementation for `alg`. Load the package that provides the selected optimizer
before solving the problem.
"""
struct OptimizerMissingError <: Exception
    alg::Any
end

function Base.showerror(io::IO, e::OptimizerMissingError)
    println(io, OPTIMIZER_MISSING_ERROR_MESSAGE)
    print(io, "Chosen Optimizer: ")
    return print(e.alg)
end

"""
    init(prob::OptimizationProblem, alg, args...; kwargs...)

Prepare `prob` and `alg` for an incremental optimization run by constructing
an `AbstractOptimizationCache`.

# Arguments

- `prob`: the problem to optimize.
- `alg`: an algorithm provided by an Optimization.jl solver package.
- `args...`: positional arguments accepted by the solver implementation.

# Keyword Arguments

The common options are the same as for [`solve`](@ref), including `maxiters`,
`maxtime`, `abstol`, `reltol`, and `callback`. Solver-specific options are
forwarded to the implementation.

# Returns

An `AbstractOptimizationCache` ready for `solve!`.

# Interface

Solver packages implement `SciMLBase.__init(prob, alg; kwargs...)` when they
support the cache interface. The returned cache must implement
`SciMLBase.__solve(cache)`.

# Examples

```julia
cache = init(prob, alg; maxiters = 100)
sol = solve!(cache)
```
"""
function init(
        prob::SciMLBase.OptimizationProblem, alg, args...;
        kwargs...
    )::SciMLBase.AbstractOptimizationCache
    if prob.u0 !== nothing && !isconcretetype(eltype(prob.u0))
        throw(SciMLBase.NonConcreteEltypeError(eltype(prob.u0)))
    end
    _check_opt_alg(prob::SciMLBase.OptimizationProblem, alg; kwargs...)
    cache = __init(prob, alg, args...; prob.kwargs..., kwargs...)
    return cache
end

"""
    solve!(cache::AbstractOptimizationCache)

Continue an optimization represented by `cache` and return its solution.

# Arguments

- `cache`: a cache returned by [`init`](@ref).

# Returns

An `AbstractOptimizationSolution` containing the final optimization
state.

# Interface

Solver packages implement `SciMLBase.__solve(cache)` for their concrete cache
type. The cache is a developer-facing extension point; callers should use
documented constructors and methods rather than relying on concrete fields.

# Examples

```julia
cache = init(prob, alg)
sol = solve!(cache)
```
"""
function solve!(cache::SciMLBase.AbstractOptimizationCache)::SciMLBase.AbstractOptimizationSolution
    return __solve(cache)
end

# needs to be defined for each cache
function __solve(cache::SciMLBase.AbstractOptimizationCache)
    throw(ArgumentError("__solve not implemented for cache type $(typeof(cache))"))
end
function __init(
        prob::SciMLBase.OptimizationProblem, alg, args...;
        kwargs...
    )::SciMLBase.AbstractOptimizationCache
    throw(OptimizerMissingError(alg))
end

# if no cache interface is supported at least the following method has to be defined
function __solve(prob::SciMLBase.OptimizationProblem, alg, args...; kwargs...)
    throw(OptimizerMissingError(alg))
end

# Used for hooking up to AD rules / SciMLSensitivity
function solve_up(
        prob::SciMLBase.OptimizationProblem, sensealg, u0, p, args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...
    )
    alg = extract_opt_alg(args, kwargs, has_kwargs(prob) ? prob.kwargs : kwargs)
    _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)
    return if length(args) > 1
        solve_call(_prob, alg, Base.tail(args)...; kwargs...)
    else
        solve_call(_prob, alg; kwargs...)
    end
end

function solve_call(
        _prob, alg, args...; merge_callbacks = true, kwargshandle = nothing,
        kwargs...
    )
    kwargshandle = kwargshandle === nothing ? KeywordArgError : kwargshandle
    kwargshandle = has_kwargs(_prob) && haskey(_prob.kwargs, :kwargshandle) ?
        _prob.kwargs[:kwargshandle] : kwargshandle

    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    return if SciMLBase.has_init(alg)
        solve!(init(_prob, alg, args...; kwargs...))
    else
        if _prob.u0 !== nothing && !isconcretetype(eltype(_prob.u0))
            throw(SciMLBase.NonConcreteEltypeError(eltype(_prob.u0)))
        end
        _check_opt_alg(_prob, alg; kwargs...)
        __solve(_prob, alg, args...; kwargs...)
    end
end

function get_concrete_problem(prob::OptimizationProblem; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SymbolicIndexingInterface.state_values(prob), p = SymbolicIndexingInterface.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs)
    u0 = get_concrete_u0(prob, false, nothing, kwargs)
    u0 = promote_u0(u0, p, nothing)
    return remake(prob; u0 = u0, p = p)

end


@inline function extract_opt_alg(solve_args, solve_kwargs, prob_kwargs)
    return if isempty(solve_args) || isnothing(first(solve_args))
        if haskey(solve_kwargs, :alg)
            solve_kwargs[:alg]
        elseif haskey(prob_kwargs, :alg)
            prob_kwargs[:alg]
        else
            nothing
        end
    else
        first(solve_args)
    end
end


function _solve_forward(
        prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...
    )
    alg = extract_opt_alg(args, kwargs, prob.kwargs)
    _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)

    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    return if length(args) > 1
        _concrete_solve_forward(
            _prob, alg, sensealg, u0, p, originator,
            Base.tail(args)...; kwargs...
        )
    else
        _concrete_solve_forward(_prob, alg, sensealg, u0, p, originator; kwargs...)
    end
end

function _solve_adjoint(
        _prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...
    )
    alg = extract_opt_alg(args, kwargs, _prob.kwargs)

    _prob = get_concrete_problem(_prob; u0 = u0, p = p, kwargs...)

    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    return if length(args) > 1
        _concrete_solve_adjoint(
            _prob, alg, sensealg, u0, p, originator,
            Base.tail(args)...; kwargs...
        )
    else
        _concrete_solve_adjoint(_prob, alg, sensealg, u0, p, originator; kwargs...)
    end
end
