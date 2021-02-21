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

function __solve(prob::OptimizationProblem, opt, data = DEFAULT_DATA;
                 maxiters::Number, cb = (args...) -> (false),
                 progress = false, save_best = true, kwargs...)

    if maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    else
        maxiters = convert(Int, maxiters)
    end

    # Flux is silly and doesn't have an abstract type on its optimizers, so assume
    # this is a Flux optimizer
    θ = copy(prob.u0)
    ps = Flux.params(θ)

    if data != DEFAULT_DATA
        maxiters = length(data)
    else
        data = take(data, maxiters)
    end

    t0 = time()

    local x, min_err, _loss
    min_err = typemax(eltype(prob.u0)) #dummy variables
    min_opt = 1

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    @withprogress progress name="Training" begin
      for (i,d) in enumerate(data)
        gs = Flux.Zygote.gradient(ps) do
            x = prob.f(θ,prob.p, d...)
            first(x)
          end
        x = f.f(θ, prob.p, d...)
        cb_call = cb(θ, x...)
        if !(typeof(cb_call) <: Bool)
          error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
          break
        end
        msg = @sprintf("loss: %.3g", x[1])
        progress && ProgressLogging.@logprogress msg i/maxiters
        Flux.update!(opt, ps, gs)

        if save_best
          if first(x) < first(min_err)  #found a better solution
            min_opt = opt
            min_err = x
          end
          if i == maxiters  #Last iteration, revert to best.
            opt = min_opt
            cb(θ,min_err...)
          end
        end
      end
    end

    _time = time()

    build_solution(prob, opt, θ, x[1])
    # here should be build_solution to create the output message
end


decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function __solve(prob::OptimizationProblem, opt::Optim.AbstractOptimizer,
                 data = DEFAULT_DATA;
                 maxiters = nothing,
                 cb = (args...) -> (false),
                 progress = false,
                 kwargs...)
    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = opt == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
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

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function(θ)
        x = f.f(θ, prob.p, cur...)
        return first(x)
    end

    fg! = function (G,θ)
        if G !== nothing
            f.grad(G, θ, cur...)
        end
        return _loss(θ)
    end

    if opt isa Optim.KrylovTrustRegion
        optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, (H,θ,v) -> f.hv(H,θ,v,cur...), prob.u0)
    else
        optim_f = TwiceDifferentiable(_loss, (G, θ) -> f.grad(G, θ, cur...), fg!, (H,θ) -> f.hess(H,θ,cur...), prob.u0)
    end

    original = Optim.optimize(optim_f, prob.u0, opt,
                              !(isnothing(maxiters)) ?
                                Optim.Options(;extended_trace = true,
                                               callback = _cb,
                                               iterations = maxiters,
                                               kwargs...) :
                                Optim.Options(;extended_trace = true,
                                               callback = _cb, kwargs...))
    build_solution(prob, opt, original.minimizer, original.minimum; original=original)
end

function __solve(prob::OptimizationProblem, opt::Union{Optim.Fminbox,Optim.SAMIN},
                 data = DEFAULT_DATA;
                 maxiters = nothing,
                 cb = (args...) -> (false),
                 progress = false,
                 kwargs...)

    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

      function _cb(trace)
          cb_call = !(opt isa Optim.SAMIN) && opt.method == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
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

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function(θ)
        x = f.f(θ, prob.p, cur...)
        return first(x)
    end
    fg! = function (G,θ)
        if G !== nothing
            f.grad(G, θ, cur...)
        end

        return _loss(θ)
    end
    optim_f = OnceDifferentiable(_loss, f.grad, fg!, prob.u0)

    Optim.optimize(optim_f, prob.lb, prob.ub, prob.u0, opt, !(isnothing(maxiters)) ? Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...)
                                                                                    : Optim.Options(;extended_trace = true, callback = _cb, kwargs...))
end


function __solve(prob::OptimizationProblem, opt::Optim.ConstrainedOptimizer,
                 data = DEFAULT_DATA;
                 maxiters = nothing,
                 cb = (args...) -> (false),
                 progress = false,
                 kwargs...)

    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

      function _cb(trace)
      cb_call = cb(decompose_trace(trace).metadata["x"],x...)
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

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p,prob.ucons === nothing ? 0 : length(prob.ucons))

    f.cons_j ===nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function(θ)
        x = f.f(θ, prob.p, cur...)
        return x[1]
    end
    fg! = function (G,θ)
        if G !== nothing
            f.grad(G, θ, cur...)
        end
        return _loss(θ)
    end
    optim_f = TwiceDifferentiable(_loss, (G, θ) -> f.grad(G, θ, cur...), fg!, (H,θ) -> f.hess(H, θ, cur...), prob.u0)

    cons! = (res, θ) -> res .= f.cons(θ);

    cons_j! = function(J, x)
        f.cons_j(J, x)
    end

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i]*res[i]
        end
    end

    lb = prob.lb === nothing ? [] : prob.lb
    ub = prob.ub === nothing ? [] : prob.ub
    optim_fc = TwiceDifferentiableConstraints(cons!, cons_j!, cons_hl!, lb, ub, prob.lcons, prob.ucons)

    Optim.optimize(optim_f, optim_fc, prob.u0, opt, !(isnothing(maxiters)) ? Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...)
                                                                            : Optim.Options(;extended_trace = true, callback = _cb, kwargs...))
end


function __init__()
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

            bboptre = !(isnothing(maxiters)) ? BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], MaxSteps = maxiters, CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...) : BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...)


            OptimizationSolution(opt.method,
                                [NaN],# initial_x,
                                BlackBoxOptim.best_candidate(bboptre), #pick_best_x(f_incr_pick, state),
                                BlackBoxOptim.best_fitness(bboptre), # pick_best_f(f_incr_pick, state, d),
                                bboptre.iterations, #iteration,
                                !(isnothing(maxiters)) ? bboptre.iterations >= maxiters : true, #iteration == options.iterations,
                                true,
                                bboptre.elapsed_time,
                                NamedTuple())

        end

        function __solve(prob::EnsembleOptimizationProblem, opt::BBO, data = DEFAULT_DATA;
                         cb = (args...) -> (false), maxiters = nothing,
                         progress = false, FitnessScheme=nothing, kwargs...)

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

            if !(opt.method == :borg_moea)
                error("Multi-Objective optimisation is only possible with BorgMOEA algorithm(:borg_moea).")
            end

            _loss = function(θ)
                x = ntuple(i->first(prob.prob[i].f(θ, prob.prob[i].p, cur...)),length(prob.prob))
                return x
            end

            if any([prob.prob[1].lb != i.lb || prob.prob[1].ub != i.ub for i in prob.prob[2:end]])
                error("Lower or upper bounds are not consistent between OptimizationProblem.")
            else
                multi_bounds = (lb=prob.prob[1].lb,ub=prob.prob[1].ub)
            end

            if isnothing(FitnessScheme)
                println("Warning: No FitnessScheme was defined, using default scheme: FitnessScheme=ParetoFitnessScheme{length(prob.prob)}(is_minimizing=true).")
                FitnessScheme=BlackBoxOptim.ParetoFitnessScheme{length(prob.prob)}(is_minimizing=true)
            end

            bboptre = !(isnothing(maxiters)) ? BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(multi_bounds.lb[i], multi_bounds.ub[i]) for i in 1:length(multi_bounds.lb)], MaxSteps = maxiters, CallbackFunction = _cb, CallbackInterval = 0.0, FitnessScheme=FitnessScheme, kwargs...) : BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(multi_bounds.lb[i], multi_bounds.ub[i]) for i in 1:length(multi_bounds.lb)], CallbackFunction = _cb, CallbackInterval = 0.0, FitnessScheme=FitnessScheme, kwargs...)


            OptimizationSolution(opt.method,
                            [NaN],# initial_x,
                            BlackBoxOptim.best_candidate(bboptre), #pick_best_x(f_incr_pick, state),
                            BlackBoxOptim.best_fitness(bboptre), # pick_best_f(f_incr_pick, state, d),
                            bboptre.iterations, #iteration,
                            !(isnothing(maxiters)) ? bboptre.iterations >= maxiters : true, #iteration == options.iterations,
                            true,
                            bboptre.elapsed_time,
                            NamedTuple())

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

            OptimizationSolution(opt.algorithm,
                                prob.u0,# initial_x,
                                minx, #pick_best_x(f_incr_pick, state),
                                minf, # pick_best_f(f_incr_pick, state, d),
                                Int(opt.numevals), #iteration,
                                !(isnothing(maxiters)) ? opt.numevals >= maxiters : true, #iteration == options.iterations,
                                ret,
                                _time-t0,
                                NamedTuple())
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

            OptimizationSolution(opt,
                                [NaN],# initial_x,
                                p.location, #pick_best_x(f_incr_pick, state),
                                p.value, # pick_best_f(f_incr_pick, state, d),
                                local_maxiters,
                                local_maxiters>=opt.maxeval, #not sure if that's correct
                                true,
                                t1 - t0,
                                NamedTuple())
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

            OptimizationSolution(opt,
                                [NaN],# initial_x,
                                QuadDIRECT.position(box, x0), #pick_best_x(f_incr_pick, state),
                                QuadDIRECT.value(box), # pick_best_f(f_incr_pick, state, d),
                                !(isnothing(maxiters)) ? maxiters : 0,
                                box.qnconverged, #not sure if that's correct
                                true,
                                t1 - t0,
                                NamedTuple())
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

            OptimizationSolution(summary(result),
                                 prob.u0, #initial_x
                                 Evolutionary.minimizer(result), #pick_best_x
                                 minimum(result), #pick_best_f
                                 Evolutionary.iterations(result), #iteration
                                 Evolutionary.converged(result), #convergence status
                                 true,
                                 t1 - t0,
                                 NamedTuple())
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

            OptimizationSolution(opt,
                                prob.u0,# initial_x,
                                result.logger.xbest[end], #pick_best_x(f_incr_pick, state),
                                result.logger.fbest[end], # pick_best_f(f_incr_pick, state, d),
                                length(result.logger.fbest),
                                criterion,
                                true,
                                result.logger.times[end] - result.logger.times[1],
                                NamedTuple())
        end
    end
end
