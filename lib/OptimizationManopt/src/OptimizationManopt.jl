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
        stopping_criterion,
        kwargs...)
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
        stopping_criterion,
        kwargs...)
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
        stopping_criterion,
        kwargs...)
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
        stopping_criterion,
        kwargs...)
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
        stopping_criterion,
        kwargs...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opts)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opts)
end

struct CMAESOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::CMAESOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(M),
        basis = Manopt.DefaultOrthonormalBasis(),
        kwargs...)
    opt = cma_es(M,
        loss,
        x0;
        return_state = true,
        stopping_criterion,
        kwargs...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

struct ConvexBundleOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::ConvexBundleOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(M),
        kwargs...)
    opt = convex_bundle_method!(M,
        loss,
        gradF,
        x0;
        return_state = true,
        evaluation,
        retraction_method,
        vector_transport_method,
        stopping_criterion,
        kwargs...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

struct AdaptiveRegularizationCubicOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::AdaptiveRegularizationCubicOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing,
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        kwargs...)
    opt = adaptive_regularization_with_cubics(M,
        loss,
        gradF,
        hessF,
        x0;
        return_state = true,
        evaluation,
        retraction_method,
        stopping_criterion,
        kwargs...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

struct TrustRegionsOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::TrustRegionsOptimizer,
        loss,
        gradF,
        x0;
        hessF = nothing,
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        kwargs...)
    opt = trust_regions(M,
        loss,
        gradF,
        hessF,
        x0;
        return_state = true,
        evaluation,
        retraction = retraction_method,
        stopping_criterion,
        kwargs...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

struct FrankWolfeOptimizer <: AbstractManoptOptimizer end

function call_manopt_optimizer(M::ManifoldsBase.AbstractManifold,
        opt::FrankWolfeOptimizer,
        loss,
        gradF,
        x0;
        stopping_criterion::Union{Manopt.StoppingCriterion, Manopt.StoppingCriterionSet},
        evaluation::AbstractEvaluationType = InplaceEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        stepsize::Stepsize = DecreasingStepsize(; length = 2.0, shift = 2),
        kwargs...)
    opt = Frank_Wolfe_method(M,
        loss,
        gradF,
        x0;
        return_state = true,
        evaluation,
        retraction_method,
        stopping_criterion,
        stepsize,
        kwargs...)
    # we unwrap DebugOptions here
    minimizer = Manopt.get_solver_result(opt)
    return (; minimizer = minimizer, minimum = loss(M, minimizer), options = opt)
end

## Optimization.jl stuff
function SciMLBase.requiresgradient(opt::Union{
        GradientDescentOptimizer, ConjugateGradientDescentOptimizer,
        QuasiNewtonOptimizer, ConvexBundleOptimizer, FrankWolfeOptimizer,
        AdaptiveRegularizationCubicOptimizer, TrustRegionsOptimizer})
    true
end
function SciMLBase.requireshessian(opt::Union{
        AdaptiveRegularizationCubicOptimizer, TrustRegionsOptimizer})
    true
end

function build_loss(f::OptimizationFunction, prob, cb)
    function (::AbstractManifold, θ)
        x = f.f(θ, prob.p)
        cb(x, θ)
        __x = first(x)
        return prob.sense === Optimization.MaxSense ? -__x : __x
    end
end

function build_gradF(f::OptimizationFunction{true})
    function g(M::AbstractManifold, G, θ)
        f.grad(G, θ)
        G .= riemannian_gradient(M, θ, G)
    end
    function g(M::AbstractManifold, θ)
        G = zero(θ)
        f.grad(G, θ)
        return riemannian_gradient(M, θ, G)
    end
end

function build_hessF(f::OptimizationFunction{true})
    function h(M::AbstractManifold, H1, θ, X)
        H = zeros(eltype(θ), length(θ))
        f.hv(H, θ, X)
        G = zeros(eltype(θ), length(θ))
        f.grad(G, θ)
        riemannian_Hessian!(M, H1, θ, G, H, X)
    end
    function h(M::AbstractManifold, θ, X)
        H = zeros(eltype(θ), length(θ), length(θ))
        f.hess(H, θ)
        G = zeros(eltype(θ), length(θ))
        f.grad(G, θ)
        return riemannian_Hessian(M, θ, G, H, X)
    end
end

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

    manifold = cache.manifold
    gradF = haskey(cache.solver_args, :riemannian_grad) ?
            cache.solver_args[:riemannian_grad] : nothing
    hessF = haskey(cache.solver_args, :riemannian_hess) ?
            cache.solver_args[:riemannian_hess] : nothing

    if manifold === nothing
        throw(ArgumentError("Manifold not specified in the problem for e.g. `OptimizationProblem(f, x, p; manifold = SymmetricPositiveDefinite(5))`."))
    end

    function _cb(x, θ)
        opt_state = Optimization.OptimizationState(iter = 0,
            u = θ,
            objective = x[1])
        cb_call = cache.callback(opt_state, x...)
        if !(cb_call isa Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cb_call
    end
    solver_kwarg = __map_optimizer_args!(cache, cache.opt, callback = _cb,
        maxiters = cache.solver_args.maxiters,
        maxtime = cache.solver_args.maxtime,
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol;
        cache.solver_args...
    )

    _loss = build_loss(cache.f, cache, _cb)

    if gradF === nothing
        gradF = build_gradF(cache.f)
    end

    if hessF === nothing
        hessF = build_hessF(cache.f)
    end

    if haskey(solver_kwarg, :stopping_criterion)
        stopping_criterion = Manopt.StopWhenAny(solver_kwarg.stopping_criterion...)
    else
        stopping_criterion = Manopt.StopAfterIteration(500)
    end

    opt_res = call_manopt_optimizer(manifold, cache.opt, _loss, gradF, cache.u0;
        solver_kwarg..., stopping_criterion = stopping_criterion, hessF)

    asc = get_stopping_criterion(opt_res.options)
    opt_ret = Manopt.indicates_convergence(asc) ? ReturnCode.Success : ReturnCode.Failure

    return SciMLBase.build_solution(cache,
        cache.opt,
        opt_res.minimizer,
        cache.sense === Optimization.MaxSense ?
        -opt_res.minimum : opt_res.minimum;
        original = opt_res.options,
        retcode = opt_ret)
end

export GradientDescentOptimizer, NelderMeadOptimizer, ConjugateGradientDescentOptimizer,
       ParticleSwarmOptimizer, QuasiNewtonOptimizer, CMAESOptimizer, ConvexBundleOptimizer,
       FrankWolfeOptimizer

end # module OptimizationManopt
