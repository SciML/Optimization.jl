module OptimizationSimpleOptimization

using Reexport
@reexport using OptimizationBase
using SciMLBase
using SciMLBase: _unwrap_val
using SimpleNonlinearSolve
using ADTypes

abstract type SimpleOptimizationAlgorithm end

struct SimpleLBFGS{Threshold} <: SimpleOptimizationAlgorithm end

__get_threshold(::SimpleLBFGS{threshold}) where {threshold} = Val(threshold)
SimpleLBFGS(; threshold::Union{Val, Int} = Val(10)) = SimpleLBFGS{_unwrap_val(threshold)}()

struct SimpleBFGS <: SimpleOptimizationAlgorithm end

SciMLBase.has_init(::SimpleOptimizationAlgorithm) = true

export SimpleBFGS, SimpleLBFGS

# Source: https://github.com/SciML/Optimization.jl/blob/9c5070b3db838e05794ded348b8b17df0f9e38c1/src/function.jl#L104
function instantiate_gradient(f, adtype::ADTypes.AbstractADType)
    adtypestr = string(adtype)
    _strtind = findfirst('.', adtypestr)
    strtind = isnothing(_strtind) ? 5 : _strtind + 5
    open_nrmlbrkt_ind = findfirst('(', adtypestr)
    open_squigllybrkt_ind = findfirst('{', adtypestr)
    open_brkt_ind = isnothing(open_squigllybrkt_ind) ? open_nrmlbrkt_ind :
        min(open_nrmlbrkt_ind, open_squigllybrkt_ind)
    adpkg = adtypestr[strtind:(open_brkt_ind - 1)]
    throw(ArgumentError("The passed automatic differentiation backend choice is not available. Please load the corresponding AD package $adpkg."))
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleLBFGS}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 100
    end

    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol

    f = Base.Fix2(cache.f.f, cache.p)
    adtype = cache.f.adtype
    ∇f_inner = instantiate_gradient(f, adtype)
    # Wrap gradient to take (u, p) as NonlinearProblem expects
    ∇f = (u, _) -> ∇f_inner(u)

    nlprob = NonlinearProblem(∇f, cache.u0)
    nlsol = solve(
        nlprob,
        SimpleLimitedMemoryBroyden(;
            threshold = __get_threshold(cache.opt),
            linesearch = Val(false)
        );
        maxiters = maxiters,
        abstol = abstol,
        reltol = reltol
    )
    θ = nlsol.u

    stats = OptimizationBase.OptimizationStats(;
        iterations = maxiters,
        time = 0.0,
        fevals = 0
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        θ,
        cache.f(θ, cache.p);
        original = nlsol,
        retcode = nlsol.retcode,
        stats = stats
    )
end

function SciMLBase.__solve(cache::OptimizationCache{O}) where {O <: SimpleBFGS}
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    if maxiters === nothing
        maxiters = 100
    end

    abstol = cache.solver_args.abstol
    reltol = cache.solver_args.reltol

    f = Base.Fix2(cache.f.f, cache.p)
    adtype = cache.f.adtype
    ∇f_inner = instantiate_gradient(f, adtype)
    # Wrap gradient to take (u, p) as NonlinearProblem expects
    ∇f = (u, _) -> ∇f_inner(u)

    nlprob = NonlinearProblem(∇f, cache.u0)
    nlsol = solve(
        nlprob,
        SimpleBroyden(; linesearch = Val(false));
        maxiters = maxiters,
        abstol = abstol,
        reltol = reltol
    )
    θ = nlsol.u

    stats = OptimizationBase.OptimizationStats(;
        iterations = maxiters,
        time = 0.0,
        fevals = 0
    )
    return SciMLBase.build_solution(
        cache, cache.opt,
        θ,
        cache.f(θ, cache.p);
        original = nlsol,
        retcode = nlsol.retcode,
        stats = stats
    )
end

end
