struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i=1) = nothing
Base.length(::NullData) = 0

get_maxiters(data) = Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
                     Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
                     typemax(Int) : length(data)

#=
function update!(x::AbstractArray, x̄::AbstractArray{<:ForwardDiff.Dual})
  x .-= x̄
end

function update!(x::AbstractArray, x̄)
  x .-= getindex.(ForwardDiff.partials.(x̄),1)
end

function update!(opt, x, x̄)
  x .-= Flux.Optimise.apply!(opt, x, x̄)
end

function update!(opt, x, x̄::AbstractArray{<:ForwardDiff.Dual})
  x .-= Flux.Optimise.apply!(opt, x, getindex.(ForwardDiff.partials.(x̄),1))
end

function update!(opt, xs::Flux.Zygote.Params, gs)
    update!(opt, xs[1], gs)
end
=#

maybe_with_logger(f, logger) = logger === nothing ? f() : Logging.with_logger(f, logger)

function default_logger(logger)
    Logging.min_enabled_level(logger) ≤ ProgressLogging.ProgressLevel && return nothing
    if Sys.iswindows() || (isdefined(Main, :IJulia) && Main.IJulia.inited)
          progresslogger = ConsoleProgressMonitor.ProgressLogger()
    else
          progresslogger = TerminalLoggers.TerminalLogger()
    end
    logger1 = LoggingExtras.EarlyFilteredLogger(progresslogger) do log
        log.level == ProgressLogging.ProgressLevel
    end
    logger2 = LoggingExtras.EarlyFilteredLogger(logger) do log
        log.level != ProgressLogging.ProgressLevel
    end
    LoggingExtras.TeeLogger(logger1, logger2)
end

macro withprogress(progress, exprs...)
    quote
      if $progress
          $maybe_with_logger($default_logger($Logging.current_logger())) do
              $ProgressLogging.@withprogress $(exprs...)
            end
      else
        $(exprs[end])
      end
    end |> esc
end

decompose_trace(trace) = trace

function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("solve_flux.jl")

    @require Optim="429524aa-4258-5aef-a3af-852621145aeb" include("solve_optim.jl")

    @require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209" begin
        decompose_trace(opt::BlackBoxOptim.OptRunController) = BlackBoxOptim.best_candidate(opt)

        struct BBO
            method::Symbol
            BBO(method) = new(method)
        end

        BBO() = BBO(:adaptive_de_rand_1_bin_radiuslimited) # the recommended optimizer as default


        function __solve(prob::OptimizationProblem, opt::BBO, data = DEFAULT_DATA;
                         cb = (args...) -> (false), maxiters = nothing,
                         progress = false, kwargs...)

            local x, cur, state

            if data != DEFAULT_DATA
                maxiters = length(data)
            end

            cur, state = iterate(data)

            function _cb(trace)
              cb_call = cb(decompose_trace(trace),x...)
              if !(typeof(cb_call) <: Bool)
                error("The callback should return a boolean `halt` for whether to stop the optimization process.")
              end
              if cb_call == true
                BlackBoxOptim.shutdown_optimizer!(trace) #doesn't work
              end
              cur, state = iterate(data, state)
              cb_call
            end

            if !(isnothing(maxiters)) && maxiters <= 0.0
                error("The number of maxiters has to be a non-negative and non-zero number.")
            elseif !(isnothing(maxiters))
                maxiters = convert(Int, maxiters)
            end

            _loss = function(θ)
                x = prob.f(θ, prob.p, cur...)
                return first(x)
            end

            bboptre = !(isnothing(maxiters)) ? BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], MaxFuncEvals = maxiters, CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...) : BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...)

            SciMLBase.build_solution(prob, opt, BlackBoxOptim.best_candidate(bboptre),
                                     BlackBoxOptim.best_fitness(bboptre); original=bboptre)
        end
    end

    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" begin
        function __solve(prob::OptimizationProblem, opt::NLopt.Opt;
                         maxiters = nothing, nstart = 1,
                         local_method = nothing,
                         progress = false, kwargs...)
            local x

            if !(isnothing(maxiters)) && maxiters <= 0.0
                error("The number of maxiters has to be a non-negative and non-zero number.")
            elseif !(isnothing(maxiters))
                maxiters = convert(Int, maxiters)
            end

            f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

            _loss = function(θ)
                x = f.f(θ, prob.p)
                return x[1]
            end

            fg! = function (θ,G)
                if length(G) > 0
                    f.grad(G, θ)
                end

                return _loss(θ)
            end

            NLopt.min_objective!(opt, fg!)

            if prob.ub !== nothing
                NLopt.upper_bounds!(opt, prob.ub)
            end
            if prob.lb !== nothing
                NLopt.lower_bounds!(opt, prob.lb)
            end
            if !(isnothing(maxiters))
                NLopt.maxeval!(opt, maxiters)
            end
            if nstart > 1 && local_method !== nothing
                NLopt.local_optimizer!(opt, local_method)
                if !(isnothing(maxiters))
                    NLopt.maxeval!(opt, nstart * maxiters)
                end
            end

            t0 = time()
            (minf,minx,ret) = NLopt.optimize(opt, prob.u0)
            _time = time()

            SciMLBase.build_solution(prob, opt, minx, minf; original=nothing)
        end
    end

    @require MultistartOptimization = "3933049c-43be-478e-a8bb-6e0f7fd53575" begin
        function __solve(prob::OptimizationProblem, opt::MultistartOptimization.TikTak;
                         local_method, local_maxiters = nothing,
                         progress = false, kwargs...)
            local x, _loss

            if !(isnothing(local_maxiters)) && local_maxiters <= 0.0
                error("The number of local_maxiters has to be a non-negative and non-zero number.")
            else !(isnothing(local_maxiters))
                local_maxiters = convert(Int, local_maxiters)
            end

            _loss = function(θ)
                x = prob.f(θ, prob.p)
                return first(x)
            end

            t0 = time()

            P = MultistartOptimization.MinimizationProblem(_loss, prob.lb, prob.ub)
            multistart_method = opt
            if !(isnothing(local_maxiters))
                local_method = MultistartOptimization.NLoptLocalMethod(local_method, maxeval = local_maxiters)
            else
                local_method = MultistartOptimization.NLoptLocalMethod(local_method)
            end
            p = MultistartOptimization.multistart_minimization(multistart_method, local_method, P)

            t1 = time()

            SciMLBase.build_solution(prob, opt, p.location, p.value; original=p)
        end
    end

    @require QuadDIRECT = "dae52e8d-d666-5120-a592-9e15c33b8d7a" begin
        export QuadDirect

        struct QuadDirect
        end

        function __solve(prob::OptimizationProblem, opt::QuadDirect; splits = nothing, maxiters = nothing, kwargs...)

            local x, _loss

            if !(isnothing(maxiters)) && maxiters <= 0.0
                error("The number of maxiters has to be a non-negative and non-zero number.")
            elseif !(isnothing(maxiters))
                maxiters = convert(Int, maxiters)
            end

            if splits === nothing
                error("You must provide the initial locations at which to evaluate the function in `splits` (a list of 3-vectors with values in strictly increasing order and within the specified bounds).")
            end

            _loss = function(θ)
                x = prob.f(θ, prob.p)
                return first(x)
            end

            t0 = time()

            root, x0 = !(isnothing(maxiters)) ? QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; maxevals = maxiters, kwargs...) : QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; kwargs...)
            box = minimum(root)
            t1 = time()

            SciMLBase.build_solution(prob, opt, QuadDIRECT.position(box, x0), QuadDIRECT.value(box); original=root)
        end
    end

    @require Evolutionary="86b6b26d-c046-49b6-aa0b-5f0f74682bd6" begin
        decompose_trace(trace::Evolutionary.OptimizationTrace) = last(trace)

        function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::Evolutionary.AbstractOptimizer, options)
            record["x"] = population
        end

        function __solve(prob::OptimizationProblem, opt::Evolutionary.AbstractOptimizer, data = DEFAULT_DATA;
                         cb = (args...) -> (false), maxiters = nothing,
                         progress = false, kwargs...)
            local x, cur, state

            if data != DEFAULT_DATA
                maxiters = length(data)
            end

            cur, state = iterate(data)

            function _cb(trace)
                cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
                if !(typeof(cb_call) <: Bool)
                    error("The callback should return a boolean `halt` for whether to stop the optimization process.")
                end
                cur, state = iterate(data, state)
                cb_call
            end

            if !(isnothing(maxiters)) && maxiters <= 0.0
                error("The number of maxiters has to be a non-negative and non-zero number.")
            elseif !(isnothing(maxiters))
                maxiters = convert(Int, maxiters)
            end

            _loss = function(θ)
                x = prob.f(θ, prob.p, cur...)
                return first(x)
            end

            t0 = time()

            result = Evolutionary.optimize(_loss, prob.u0, opt, !isnothing(maxiters) ? Evolutionary.Options(;iterations = maxiters, callback = _cb, kwargs...)
                                                                                : Evolutionary.Options(;callback = _cb, kwargs...))
            t1 = time()

            SciMLBase.build_solution(prob, opt, Evolutionary.minimizer(result), Evolutionary.minimum(result); original=result)
        end
    end
    @require CMAEvolutionStrategy="8d3b24bd-414e-49e0-94fb-163cc3a3e411" begin

        struct CMAEvolutionStrategyOpt end

        function __solve(prob::OptimizationProblem, opt::CMAEvolutionStrategyOpt, data = DEFAULT_DATA;
                         cb = (args...) -> (false), maxiters = nothing,
                         progress = false, kwargs...)
            local x, cur, state

            if data != DEFAULT_DATA
                maxiters = length(data)
            end

            cur, state = iterate(data)

            function _cb(trace)
                cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
                if !(typeof(cb_call) <: Bool)
                    error("The callback should return a boolean `halt` for whether to stop the optimization process.")
                end
                cur, state = iterate(data, state)
                cb_call
            end

            _loss = function(θ)
                x = prob.f(θ, prob.p, cur...)
                return first(x)
            end


            if !(isnothing(maxiters)) && maxiters <= 0.0
                error("The number of maxiters has to be a non-negative and non-zero number.")
            elseif !(isnothing(maxiters))
                maxiters = convert(Int, maxiters)
            end

            result = CMAEvolutionStrategy.minimize(_loss, prob.u0, 0.1; lower = prob.lb, upper = prob.ub, maxiter = maxiters, kwargs...)
            CMAEvolutionStrategy.print_header(result)
            CMAEvolutionStrategy.print_result(result)
            println("\n")
            criterion = true

            if (result.stop.reason === :maxtime) #this is an arbitrary choice of convergence (based on the stop.reason values)
                criterion = false
            end

            SciMLBase.build_solution(prob, opt, result.logger.xbest[end], result.logger.fbest[end]; original=result)
        end
    end
end
