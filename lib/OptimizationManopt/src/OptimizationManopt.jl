module OptimizationManopt

using Reexport
@reexport using Manopt
using Optimization, Manopt, ManifoldsBase, ManifoldDiff, Optimization.SciMLBase

"""
    abstract type AbstractManoptOptimizer end

A Manopt solver without things specified by a call to `solve` (stopping criteria) and
internal state.
"""
abstract type AbstractManoptOptimizer end

SciMLBase.supports_opt_cache_interface(opt::AbstractManoptOptimizer) = true

function __map_optimizer_args!(cache::OptimizationCache,
        opt::AbstractManoptOptimizer;
        callback = nothing,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    solver_kwargs = (; kwargs...)

    if !isnothing(maxiters)
        solver_kwargs = (;
            solver_kwargs..., stopping_criterion = [Manopt.StopAfterIteration(maxiters)])
    end

    if !isnothing(maxtime)
        if haskey(solver_kwargs, :stopping_criterion)
            solver_kwargs = (; solver_kwargs...,
                stopping_criterion = push!(
                    solver_kwargs.stopping_criterion, Manopt.StopAfterTime(maxtime)))
        else
            solver_kwargs = (;
                solver_kwargs..., stopping_criterion = [Manopt.StopAfter(maxtime)])
        end
    end

    if !isnothing(abstol)
        if haskey(solver_kwargs, :stopping_criterion)
            solver_kwargs = (; solver_kwargs...,
                stopping_criterion = push!(
                    solver_kwargs.stopping_criterion, Manopt.StopWhenChangeLess(abstol)))
        else
            solver_kwargs = (;
                solver_kwargs..., stopping_criterion = [Manopt.StopWhenChangeLess(abstol)])
        end
    end

    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(typeof(opt).super)"
    end
    return solver_kwargs
end

## gradient descent
struct GradientDescentOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(
        M::ManifoldsBase.AbstractManifold, opt::GradientDescentOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = Manopt.AllocatingEvaluation(),
        stepsize::Stepsize = ArmijoLinesearch(M),
        kwargs...)
    opts = gradient_descent(M,
        loss,
        gradF,
        x0;
        return_state = true,
        evaluation,
        stepsize,
        stopping_criterion)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## Nelder-Mead
struct NelderMeadOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold, opt::NelderMeadOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        kwargs...)
    opts = NelderMead(M,
        loss;
        return_state = true,
        stopping_criterion)
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## conjugate gradient descent
struct ConjugateGradientDescentOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::ConjugateGradientDescentOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        stepsize::Stepsize = ArmijoLinesearch(M),
        kwargs...)
    opts = conjugate_gradient_descent(M,
        loss,
        gradF,
        x0;
        return_state = true,
        evaluation,
        stepsize,
        stopping_criterion)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## particle swarm
struct ParticleSwarmOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::ParticleSwarmOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        population_size::Int = 100,
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(M),
        kwargs...)
    initial_population = vcat([x0], [rand(M) for _ in 1:(population_size - 1)])
    opts = particle_swarm(M,
        loss;
        x0 = initial_population,
        n = population_size,
        return_state = true,
        retraction_method,
        inverse_retraction_method,
        vector_transport_method,
        stopping_criterion)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## quasi Newton

struct QuasiNewtonOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::Manopt.AbstractManifold,
        opt::QuasiNewtonOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(M),
        stepsize = WolfePowellLinesearch(M;
            retraction_method = retraction_method,
            vector_transport_method = vector_transport_method,
            linesearch_stopsize = 1e-12),
        kwargs...
)
    opts = quasi_Newton(M,
        loss,
        gradF,
        x0;
        return_state = true,
        evaluation,
        retraction_method,
        vector_transport_method,
        stepsize,
        stopping_criterion)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

## Optimization.jl stuff

function build_loss(f::OptimizationFunction, prob, cb)
    function (::AbstractManifold, θ)
        x = f.f(θ, prob.p)
        cb(x, θ)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
end

function build_gradF(f::OptimizationFunction{true}, cur)
    function g(M::AbstractManifold, G, θ)
        f.grad(G, θ, cur...)
        G .= riemannian_gradient(M, θ, G)
    end
    function g(M::AbstractManifold, θ)
        G = zero(θ)
        f.grad(G, θ, cur...)
        return riemannian_gradient(M, θ, G)
    end
end

# TODO:
# 1) convert tolerances and other stopping criteria
# 2) return convergence information
# 3) add callbacks to Manopt.jl

function SciMLBase.__solve(cache::OptimizationCache{
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O,
        D,
        P,
        C
}) where {
        F,
        RC,
        LB,
        UB,
        LC,
        UC,
        S,
        O <:
        AbstractManoptOptimizer,
        D,
        P,
        C
}
    local x, cur, state

    manifold = haskey(cache.solver_args, :manifold) ? cache.solver_args[:manifold] : nothing

    if manifold === nothing
        throw(ArgumentError("Manifold not specified in the problem for e.g. `OptimizationProblem(f, x, p; manifold = SymmetricPositiveDefinite(5))`."))
    end

    if cache.data !== Optimization.DEFAULT_DATA
        maxiters = length(cache.data)
    else
        maxiters = cache.solver_args.maxiters
    end

    cur, state = iterate(cache.data)

    function _cb(x, θ)
        opt_state = Optimization.OptimizationState(iter = 0,
            u = θ,
            objective = x[1])
        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        nx_itr = iterate(cache.data, state)
        if isnothing(nx_itr)
            true
        else
            cur, state = nx_itr
            cb_call
        end
    end
    solver_kwarg = __map_optimizer_args!(cache, cache.opt, callback = _cb,
        maxiters = maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
    )

    _loss = build_loss(cache.f, cache, _cb)

    gradF = build_gradF(cache.f, cur)

    if haskey(solver_kwarg, :stopping_criterion)
        stopping_criterion = Manopt.StopWhenAny(solver_kwarg.stopping_criterion...)
    else
        stopping_criterion = Manopt.StopAfterIteration(500)
    end

    opt_res = call_manopt_optimizer(manifold, cache.opt, _loss, gradF, cache.u0;
        solver_kwarg..., stopping_criterion = stopping_criterion)

    asc = get_active_stopping_criteria(opt_res.options.stop)

    opt_ret = any(Manopt.indicates_convergence, asc) ? ReturnCode.Success :
              ReturnCode.Failure

    return SciMLBase.build_solution(cache,
        cache.opt,
        opt_res.minimizer,
        cache.sense === Optimization.MaxSense ?
        -opt_res.minimum : opt_res.minimum;
        original = opt_res.options,
        retcode = opt_ret)
end

end # module OptimizationManopt
