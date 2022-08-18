module OptimizationNonconvex

using Nonconvex, Optimization, Optimization.SciMLBase, ChainRulesCore

include("nonconvex_bayesian.jl")
include("nonconvex_pavito.jl")
include("nonconvex_juniper.jl")
include("nonconvex_ipopt.jl")
include("nonconvex_nlopt.jl")
include("nonconvex_mma.jl")
# include("nonconvex_multistart.jl")
include("nonconvex_percival.jl")
include("nonconvex_search.jl")

struct NonconvexADWrapper{F, P}
    f::F
    prob::P
end

(f::NonconvexADWrapper)(θ) = begin
    x = f.f.f(θ, f.prob.p)
    return first(x)
end

## Create ChainRule
function ChainRulesCore.rrule(f::NonconvexADWrapper, θ::AbstractVector)
    val = f.f.f(θ, f.prob.p)
    G = similar(θ)
    f.f.grad(G, θ)
    val, Δ -> (ChainRulesCore.NoTangent(), Δ * G)
end

function separate_kwargs(orig_kwargs)
    sub_options = :sub_options .∈ Ref(keys(orig_kwargs)) ? orig_kwargs[:sub_options] :
                  nothing
    convergence_criteria = :convcriteria .∈ Ref(keys(orig_kwargs)) ?
                           orig_kwargs[:convcriteria] : nothing
    opt_kwargs_names = keys(orig_kwargs)[[
                                             (keys(orig_kwargs) .∉
                                              Ref([:sub_options, :convcriteria]))...,
                                         ]]
    opt_kwargs = (; zip(opt_kwargs_names, [orig_kwargs[j] for j in opt_kwargs_names])...)

    return opt_kwargs, sub_options, convergence_criteria
end

function __map_optimizer_args(prob::OptimizationProblem,
                              opt::Nonconvex.NonconvexCore.AbstractOptimizer;
                              callback = nothing,
                              maxiters::Union{Number, Nothing} = nothing,
                              maxtime::Union{Number, Nothing} = nothing,
                              abstol::Union{Number, Nothing} = nothing,
                              reltol::Union{Number, Nothing} = nothing,
                              integer = nothing,
                              kwargs...)
    opt_kwargs, sub_options, convergence_criteria = separate_kwargs(kwargs)
    opt_kwargs = convert_common_kwargs(opt, opt_kwargs, callback = callback,
                                       maxiters = maxiters, maxtime = maxtime,
                                       abstol = abstol, reltol = reltol)
    mapped_args = _create_options(opt, opt_kwargs = opt_kwargs, sub_options = sub_options,
                                  convergence_criteria = convergence_criteria)

    integer = isnothing(integer) ? fill(false, length(prob.u0)) : integer

    return mapped_args, integer
end

function SciMLBase.__solve(prob::OptimizationProblem,
                           opt::Nonconvex.NonconvexCore.AbstractOptimizer;
                           callback = nothing,
                           maxiters::Union{Number, Nothing} = nothing,
                           maxtime::Union{Number, Nothing} = nothing,
                           abstol::Union{Number, Nothing} = nothing,
                           reltol::Union{Number, Nothing} = nothing,
                           progress = false,
                           surrogate_objective::Union{Symbol, Nothing} = nothing,
                           integer = nothing,
                           kwargs...)

    # local x

    maxiters = Optimization._check_and_convert_maxiters(maxiters)
    maxtime = Optimization._check_and_convert_maxtime(maxtime)

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)

    _loss = NonconvexADWrapper(f, prob)

    opt_f(θ) = _loss(θ)

    opt_args, integer = __map_optimizer_args(prob, opt, callback = callback,
                                             maxiters = maxiters, maxtime = maxtime,
                                             abstol = abstol, reltol = reltol,
                                             integer = integer; kwargs...)

    opt_set = Nonconvex.Model()
    if isnothing(surrogate_objective)
        Nonconvex.set_objective!(opt_set, opt_f)
    else
        Nonconvex.set_objective!(opt_set, opt_f, flags = [surrogate_objective])
    end

    Nonconvex.addvar!(opt_set, prob.lb, prob.ub, integer = integer)

    if !isnothing(prob.f.cons)
        @warn "Equality constraints are current not passed on by Optimization"
        #add_ineq_constraint!(opt_set, f)
    end

    if !isnothing(prob.lcons)
        @warn "Inequality constraints are current not passed on by Optimization"
        #add_ineq_constraint!(opt_set, f)
    end

    if !isnothing(prob.ucons)
        @warn "Inequality constraints are current not passed on by Optimization"
        #add_ineq_constraint!(opt_set, f)
    end

    t0 = time()
    if check_optimizer_backend(opt)
        opt_res = Nonconvex.optimize(opt_set, opt; opt_args...)
    else
        opt_res = Nonconvex.optimize(opt_set, opt, prob.u0; opt_args...)
    end
    t1 = time()

    opt_ret = hasproperty(opt_res, :status) ? Symbol(string(opt_res.status)) : nothing

    SciMLBase.build_solution(prob, opt, opt_res.minimizer, opt_res.minimum;
                             (isnothing(opt_ret) ? (; original = opt_res) :
                              (; original = opt_res, retcode = opt_ret))...)
end

end
