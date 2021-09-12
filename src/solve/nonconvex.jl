export NC

struct NC
    method::Symbol
    NC(method) = new(method)
end


function __solve(prob::OptimizationProblem, opt::NC;
                 maxiters = nothing,
                 local_method = nothing,
                 local_maxiters = nothing,
                 local_options= Union{Nothing,NamedTuple},
                 integer=nothing, kwargs...)
    local x

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    _loss = function(θ)
        x = f.f(θ, prob.p)
        return first(x)
    end

    # fg! = function (θ,G)
    #     if length(G) > 0
    #         f.grad(G, θ)
    #     end

    #     return _loss(θ)
    # end

    function ChainRulesCore.rrule(::typeof(_loss), x::AbstractVector)
        val = _loss(x)
        grad = similar(x) 
        f.grad(grad, θ) #ForwardDiff.gradient(f, x)
        val, Δ -> (NoTangent(), Δ * grad)
    end


    model = Nonconvex.Model(_loss)

    if prob.lb !== nothing && prob.ub !== nothing
        Nonconvex.addvar!(opt, prob.lb, prob.ub, init=prob.u0)
    elseif prob.lb !== nothing && prob.ub !== nothing && !isnothing(integer)
        Nonconvex.addvar!(opt, prob.lb, prob.ub, init=prob.u0, integer=integer)
    else
        error("Lower and upper bounds have to be defined for Nonconvex.jl.")
    end

    # if !(isnothing(maxiters))
    #     NLopt.maxeval!(opt, maxiters)
    # end
    
    # if nstart > 1 && local_method !== nothing
    #     NLopt.local_optimizer!(opt, local_method)
    #     if !(isnothing(maxiters))
    #         NLopt.maxeval!(opt, nstart * maxiters)
    #     end
    # end
    # set default options struct
    if typeof(alg) <: Union{MMA, MMA87, MMA02, GCMMA}
        options = MMAOptions(outer_maxiter=maxiters, kwargs...)
    elseif typeof(alg) <: Union{IpoptAlg}
        options = IpoptOptions(max_iter=maxiters, kwargs...)
    elseif typeof(alg) <: Union{NLoptAlg}
        options = NLoptOptions(maxeval=maxiters, kwargs...)
    elseif typeof(alg) <: Union{AugLag}
        options = AugLagOptions(max_iter=maxiters, kwargs...)
    elseif typeof(alg) <: Union{JuniperIpoptAlg}
        options = JuniperIpoptOptions(outer_maxiter=maxiters, kwargs...)
    elseif typeof(alg) <: Union{PavitoIpoptCbcAlg}
        options = PavitoIpoptCbcOptions(outer_maxiter=maxiters, kwargs...)
    elseif typeof(alg) <: Union{HyperoptAlg(sampler = Hyperband())}
        options = HyperoptOptions(iters=maxiters, kwargs...)
    elseif typeof(alg) <: Union{HyperoptAlg(sampler = RandomSampler()), HyperoptAlg(sampler = LHSampler()), HyperoptAlg(sampler = CLHSampler()), HyperoptAlg(sampler = GPSampler())}
        options = HyperoptOptions(iters=maxiters, kwargs...)
    elseif typeof(alg) <: Union{BayesOptAlg}
        options = BayesOptOptions(maxiter=maxiters, kwargs...)
    elseif typeof(alg) <: Union{MTSAlg}
        options = MMAOptions(outer_maxiter=maxiters, kwargs...)
    end

    # set maxiters

    t0 = time()
    nc_res = Nonconvex.optimize(model, opt.method, options=options)
    _time = time()

    SciMLBase.build_solution(prob, opt, nc_res.minimizer, nc_res.minimum; original=nc_res)
end
