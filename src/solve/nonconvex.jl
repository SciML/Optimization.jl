using ChainRulesCore


function __map_optimizer_args(prob::OptimizationProblem, opt::NonconvexCore.AbstractOptimizer;
    cb=nothing,
    maxiters::Union{Number, Nothing}=nothing,
    maxtime::Union{Number, Nothing}=nothing,
    abstol::Union{Number, Nothing}=nothing, 
    reltol::Union{Number, Nothing}=nothing, 
    kwargs...)

    mapped_args = (;)

    if !isnothing(cb)
        @warn "common callback argument is currently not used by $(opt)"
    end
  
    if !isnothing(maxiters)
        @warn "common maxiters argument is currently not used by $(opt)"
    end

    if !isnothing(maxtime)
        @warn "common maxtime argument is currently not used by $(opt)"
    end

    if !isnothing(abstol)
        @warn "common abstol argument is currently not used by $(opt)"
    end
    
    if !isnothing(reltol)
        @warn "common reltol argument is currently not used by $(opt)"
    end

    _mapped_args

    convcriteria

    ## makes this a function so sub_options of any solver can call this
    if isa(opt, Union{MMA02, MMA87})
        mapped_args = (options = MMAOptions(),convcriteria=)
    elseif isa(opt, IpoptAlg)
        mapped_args = (options = IpoptOptions())
    elseif isa(opt, NLoptAlg)
        mapped_args = (options = NLoptOptions())    
    elseif isa(opt, AugLag)
        mapped_args = (options = AugLagOptions())
    elseif isa(opt, JuniperIpoptAlg)
        mapped_args = (options = JuniperIpoptOptions())
    elseif isa(opt, PavitoIpoptCbcAlg)
        mapped_args = (options = PavitoIpoptCbcOptions())
    elseif isa(opt, HyperoptAlg)
        #separate by sampler
        mapped_args = (options = HyperoptOptions())
    elseif isa(opt, BayesOptAlg)
        mapped_args = (options = BayesOptOptions())
    elseif isa(opt, MTSAlg)
        mapped_args = (options = MTSOptions())
    end

    integer = :integer .∈ Ref(keys(kwargs)) ? kwargs[:integer] : fill(false, length(prob.u0))

    return mapped_args, integer
end

function __solve(prob::OptimizationProblem, opt::NonconvexCore.AbstractOptimizer;
                 cb = nothing, 
                 maxiters::Union{Number, Nothing} = nothing,
                 maxtime::Union{Number, Nothing} = nothing,
                 abstol::Union{Number, Nothing}=nothing,
                 reltol::Union{Number, Nothing}=nothing,
                 progress = false,
                 kwargs...)

    local x

    maxiters = _check_and_convert_maxiters(maxiters)
    maxtime = _check_and_convert_maxtime(maxtime)

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)


    _loss = function(θ)
        x = f.f(θ, prob.p)
        return first(x)
    end

    ## Create ChainRule
    function ChainRulesCore.rrule(::typeof(_loss), θ::AbstractVector)
        val =  f.f(θ, prob.p)
        G = similar(θ)
        f.grad(G, θ)
        val, Δ -> (NoTangent(), Δ * G)
    end

    opt_args, integer = _map_optimizer_args(prob, opt, cb=_cb, maxiters=maxiters, maxtime=maxtime,abstol=abstol, reltol=reltol; kwargs...)

    opt_set =  Model()
    set_objective!(opt_set, _loss)
    addvar!(opt_set, prob.lb, prob.ub, init = prob.u0, integer = integer)
    add_ineq_constraint!(opt_set, f)
    add_eq_constraint!(opt_set, f)


    t0 = time()
    opt_res = optimize(opt_set, opt, opt_args...)
    t1 = time()
    
    opt_ret = opt_res.status

    SciMLBase.build_solution(prob, opt, opt_res.minimizer, opt_res.minimum; original=opt_res, retcode=opt_ret)
end