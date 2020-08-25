
function DiffEqBase.solve(prob::OptimizationProblem, opt, args...;kwargs...)
	__solve(prob, opt, args...; kwargs...)
end

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
  for x in xs
    gs[x] === nothing && continue
    update!(opt, x, gs[x])
  end
end

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

function __solve(prob::OptimizationProblem, opt;cb = (args...) -> (false), maxiters = 1000, progress = true, save_best = true, kwargs...)

	# Flux is silly and doesn't have an abstract type on its optimizers, so assume
	# this is a Flux optimizer
	θ = copy(prob.x)
	ps = Flux.params(θ)
  
	t0 = time()
  
	local x, min_err, _loss
	min_err = typemax(eltype(prob.x)) #dummy variables
	min_opt = 1
  
		  
	if prob.f isa OptimizationFunction 
		_loss = function(θ)
			x = prob.f.f(θ, prob.p)
		end
	else 
		_loss = function(θ)
			x = prob.f(θ, prob.p)
		end
	end

	@withprogress progress name="Training" begin
	  for i in 1:maxiters
		gs = Flux.Zygote.gradient(ps) do
		  x = _loss(θ)
		  first(x)
		end
		cb_call = cb(θ,x...)
		if !(typeof(cb_call) <: Bool)
		  error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
		elseif cb_call
		  break
		end
		msg = @sprintf("loss: %.3g", x[1])
		progress && ProgressLogging.@logprogress msg i/maxiters
		update!(opt, ps, gs)
  
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
  
	Optim.MultivariateOptimizationResults(opt,
										  prob.x,# initial_x,
										  θ, #pick_best_x(f_incr_pick, state),
										  first(x), # pick_best_f(f_incr_pick, state, d),
										  maxiters, #iteration,
										  maxiters >= maxiters, #iteration == options.iterations,
										  true, # x_converged,
										  0.0,#T(options.x_tol),
										  0.0,#T(options.x_tol),
										  NaN,# x_abschange(state),
										  NaN,# x_abschange(state),
										  true,# f_converged,
										  0.0,#T(options.f_tol),
										  0.0,#T(options.f_tol),
										  NaN,#f_abschange(d, state),
										  NaN,#f_abschange(d, state),
										  true,#g_converged,
										  0.0,#T(options.g_tol),
										  NaN,#g_residual(d),
										  false, #f_increased,
										  nothing,
										  maxiters,
										  maxiters,
										  0,
										  true,
										  NaN,
										  _time-t0)
end
  

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function __solve(prob::OptimizationProblem, opt::Optim.AbstractOptimizer;cb = (args...) -> (false), maxiters = 1000, kwargs...)
  	local x

	function _cb(trace)
		cb_call = opt == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
		if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process.")
		end
		cb_call
  	end
	
	if prob.f isa OptimizationFunction
		_loss = function(θ)
			x = prob.f.f(θ, prob.p)
			return x[1]
		end
		fg! = function (G,θ)
			if G !== nothing
				prob.f.grad(G, θ)
			end

			return _loss(θ)
		end
		if opt isa Optim.KrylovTrustRegion
			optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, prob.f.hv, prob.x)
		else
			optim_f = TwiceDifferentiable(_loss, prob.f.grad, fg!, prob.f.hess, prob.x)
		end
	else
		!(opt isa Optim.ZerothOrderOptimizer) && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")
		_loss = function(θ)
			x = prob.f(θ, prob.p)
			return x[1]
		end
		optim_f = _loss
	end

  	Optim.optimize(optim_f, prob.x, opt, Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...))
end

function __solve(prob::OptimizationProblem, opt::Union{Optim.Fminbox,Optim.SAMIN};cb = (args...) -> (false), maxiters = 1000, kwargs...)
	local x

  	function _cb(trace)
	  	cb_call = !(opt isa Optim.SAMIN) && opt.method == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
	  	if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process.")
	  	end
	  	cb_call
	end
  
  	if prob.f isa OptimizationFunction && !(opt isa Optim.SAMIN)
	  	_loss = function(θ)
			x = prob.f.f(θ, prob.p)
			return x[1]  
	  	end
	  	fg! = function (G,θ)
			if G !== nothing
			  	prob.f.grad(G, θ)
			end

			return _loss(θ)
		end
		optim_f = OnceDifferentiable(_loss, prob.f.grad, fg!, prob.x)
  	else
	  	!(opt isa Optim.ZerothOrderOptimizer) && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")
		_loss = function(θ)
			x = prob.f isa OptimizationFunction ? prob.f.f(θ, prob.p) : prob.f(θ, prob.p)
			return x[1]  
	  	end
	  	optim_f = _loss
  	end
  
	Optim.optimize(optim_f, prob.lb, prob.ub, prob.x, opt, Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...))
end

function __init__()
	@require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209" begin
		decompose_trace(opt::BlackBoxOptim.OptRunController) = BlackBoxOptim.best_candidate(opt)

		struct BBO
			method::Symbol         
		end

		BBO() = BBO(:adaptive_de_rand_1_bin)

		function __solve(prob::OptimizationProblem, opt::BBO; cb = (args...) -> (false), maxiters = 1000, kwargs...)
			local x, _loss
		  
			function _cb(trace)
			  cb_call = cb(decompose_trace(trace),x...)
			  if !(typeof(cb_call) <: Bool)
				error("The callback should return a boolean `halt` for whether to stop the optimization process.")
			  end
			  if cb_call == true
				BlackBoxOptim.shutdown_optimizer!(trace) #doesn't work
			  end
			  cb_call
			end

			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
					return x[1]
				end
			else 
				_loss = function(θ)
					x = prob.f(θ, prob.p)
					return x[1]
				end
			end

			bboptre = BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(prob.lb[i], prob.ub[i]) for i in 1:length(prob.lb)], MaxSteps = maxiters, CallbackFunction = _cb, CallbackInterval = 0.0, kwargs...)
		  
			Optim.MultivariateOptimizationResults(opt.method,
												  [NaN],# initial_x,
												  BlackBoxOptim.best_candidate(bboptre), #pick_best_x(f_incr_pick, state),
												  BlackBoxOptim.best_fitness(bboptre), # pick_best_f(f_incr_pick, state, d),
												  bboptre.iterations, #iteration,
												  bboptre.iterations >= maxiters, #iteration == options.iterations,
												  false, # x_converged,
												  0.0,#T(options.x_tol),
												  0.0,#T(options.x_tol),
												  NaN,# x_abschange(state),
												  NaN,# x_abschange(state),
												  false,# f_converged,
												  0.0,#T(options.f_tol),
												  0.0,#T(options.f_tol),
												  NaN,#f_abschange(d, state),
												  NaN,#f_abschange(d, state),
												  false,#g_converged,
												  0.0,#T(options.g_tol),
												  NaN,#g_residual(d),
												  false, #f_increased,
												  nothing,
												  maxiters,
												  maxiters,
												  0,
												  true,
												  NaN,
												  bboptre.elapsed_time)
		end
	end

	@require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" begin
		function __solve(prob::OptimizationProblem, opt::NLopt.Opt; maxiters = 1000, nstart = 1, local_method = nothing, kwargs...)	
			local x

			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
					return x[1]
				end
				fg! = function (θ,G)
					if length(G) > 0
						prob.f.grad(G, θ)
					end
					
					return _loss(θ)
				end
				NLopt.min_objective!(opt, fg!)
			else 
				_loss = function(θ,G)
					x = prob.f(θ, prob.p)
					return x[1]
				end
				NLopt.min_objective!(opt, _loss)
			end

			if prob.ub !== nothing
				NLopt.upper_bounds!(opt, prob.ub)				
			end
			if prob.lb !== nothing
				NLopt.lower_bounds!(opt, prob.lb)
			end

			if nstart > 1 && local_method !== nothing
				NLopt.local_optimizer!(opt, local_method)
				NLopt.maxeval!(opt, nstart * maxiters)
			end

            NLopt.maxeval!(opt, maxiters)

            t0= time()
            (minf,minx,ret) = NLopt.optimize(opt, prob.x)
            _time = time()

            Optim.MultivariateOptimizationResults(opt,
                                                    prob.x,# initial_x,
                                                    minx, #pick_best_x(f_incr_pick, state),
                                                    minf, # pick_best_f(f_incr_pick, state, d),
                                                    maxiters, #iteration,
													maxiters >= opt.numevals, #iteration == options.iterations,
                                                    false, # x_converged,
                                                    0.0,#T(options.x_tol),
                                                    0.0,#T(options.x_tol),
                                                    NaN,# x_abschange(state),
                                                    NaN,# x_abschange(state),
                                                    false,# f_converged,
                                                    0.0,#T(options.f_tol),
                                                    0.0,#T(options.f_tol),
                                                    NaN,#f_abschange(d, state),
                                                    NaN,#f_abschange(d, state),
                                                    false,#g_converged,
                                                    0.0,#T(options.g_tol),
                                                    NaN,#g_residual(d),
                                                    false, #f_increased,
                                                    nothing,
                                                    maxiters,
                                                    maxiters,
                                                    0,
                                                    ret,
                                                    NaN,
													_time-t0,)
		end	
	end

	@require MultistartOptimization = "3933049c-43be-478e-a8bb-6e0f7fd53575" begin
		function __solve(prob::OptimizationProblem, opt::MultistartOptimization.TikTak; local_method, local_maxiters = 1000, kwargs...)
			local x, _loss
		  
			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
					return x[1]
				end
			else 
				_loss = function(θ)
					x = prob.f(θ, prob.p)
					return x[1]
				end
			end

			t0 = time()

			P = MultistartOptimization.MinimizationProblem(_loss, prob.lb, prob.ub)
			multistart_method = opt
			local_method = MultistartOptimization.NLoptLocalMethod(local_method, maxeval = local_maxiters)
			p = MultistartOptimization.multistart_minimization(multistart_method, local_method, P)
  
			t1 = time()
		  
			Optim.MultivariateOptimizationResults(opt,
                                                [NaN],# initial_x,
                                                p.location, #pick_best_x(f_incr_pick, state),
                                                p.value, # pick_best_f(f_incr_pick, state, d),
                                                0, #iteration,
                                                false, #iteration == options.iterations,
                                                false, # x_converged,
                                                0.0,#T(options.x_tol),
                                                0.0,#T(options.x_tol),
                                                NaN,# x_abschange(state),
                                                NaN,# x_abschange(state),
                                                false,# f_converged,
                                                0.0,#T(options.f_tol),
                                                0.0,#T(options.f_tol),
                                                NaN,#f_abschange(d, state),
                                                NaN,#f_abschange(d, state),
                                                false,#g_converged,
                                                0.0,#T(options.g_tol),
                                                NaN,#g_residual(d),
                                                false, #f_increased,
                                                nothing,
                                                local_maxiters,
                                                local_maxiters,
                                                0,
                                                true,
                                                NaN,
                                                t1 - t0)
		end
	end

	@require QuadDIRECT = "dae52e8d-d666-5120-a592-9e15c33b8d7a" begin
		export QuadDirect
		
        struct QuadDirect
		end

		function __solve(prob::OptimizationProblem, opt::QuadDirect; splits, maxiters = 1000, kwargs...)
			local x, _loss
		  
			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
					return x[1]
				end
			else 
				_loss = function(θ)
					x = prob.f(θ, prob.p)
					return x[1]
				end
			end

			t0 = time()

			root, x0 = QuadDIRECT.analyze(_loss, splits, prob.lb, prob.ub; maxevals = maxiters, kwargs...)
			box = minimum(root)
           	t1 = time()

           	Optim.MultivariateOptimizationResults(opt,
                                                [NaN],# initial_x,
                                                QuadDIRECT.position(box, x0), #pick_best_x(f_incr_pick, state),
                                                QuadDIRECT.value(box), # pick_best_f(f_incr_pick, state, d),
                                                0, #iteration,
                                                false, #iteration == options.iterations,
                                                false, # x_converged,
                                                0.0,#T(options.x_tol),
                                                0.0,#T(options.x_tol),
                                                NaN,# x_abschange(state),
                                                NaN,# x_abschange(state),
                                                false,# f_converged,
                                                0.0,#T(options.f_tol),
                                                0.0,#T(options.f_tol),
                                                NaN,#f_abschange(d, state),
                                                NaN,#f_abschange(d, state),
                                                false,#g_converged,
                                                0.0,#T(options.g_tol),
                                                NaN,#g_residual(d),
                                                false, #f_increased,
                                                nothing,
                                                maxiters,
                                                maxiters,
                                                0,
                                                true,
                                                NaN,
                                                t1 - t0)
		end
	end

	@require Evolutionary="86b6b26d-c046-49b6-aa0b-5f0f74682bd6" begin
		decompose_trace(trace::Evolutionary.OptimizationTrace) = last(trace)

		function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::Evolutionary.AbstractOptimizer, options)
			record["x"] = population
		end

		function __solve(prob::OptimizationProblem, opt::Evolutionary.AbstractOptimizer; cb = (args...) -> (false), maxiters = 1000, kwargs...)
			local x, _loss
		  
			function _cb(trace)
				cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
				if !(typeof(cb_call) <: Bool)
					error("The callback should return a boolean `halt` for whether to stop the optimization process.")
				end
				cb_call
			end

			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
					return x[1]
				end
			else 
				_loss = function(θ)
					x = prob.f(θ, prob.p)
					return x[1]
				end
			end

			Evolutionary.optimize(_loss, prob.x, opt, Evolutionary.Options(;iterations = maxiters, callback = _cb, kwargs...))
		end
	end
	@require CMAEvolutionStrategy="8d3b24bd-414e-49e0-94fb-163cc3a3e411" begin

		struct CMAEvolutionStrategyOpt end

		function __solve(prob::OptimizationProblem, opt::CMAEvolutionStrategyOpt; cb = (args...) -> (false), maxiters = 1000, kwargs...)
			local x, _loss
		  
			function _cb(trace)
				cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
				if !(typeof(cb_call) <: Bool)
					error("The callback should return a boolean `halt` for whether to stop the optimization process.")
				end
				cb_call
			end

			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
					return x[1]
				end
			else 
				_loss = function(θ)
					x = prob.f(θ, prob.p)
					return x[1]
				end
			end

			result = CMAEvolutionStrategy.minimize(_loss, prob.x, 0.1; lower = prob.lb, upper = prob.ub, kwargs...)


           	Optim.MultivariateOptimizationResults(opt,
                                                prob.x,# initial_x,
												result.logger.xbest[end], #pick_best_x(f_incr_pick, state),
                                                result.logger.fbest[end], # pick_best_f(f_incr_pick, state, d),
                                                0, #iteration,
                                                false, #iteration == options.iterations,
                                                false, # x_converged,
                                                0.0,#T(options.x_tol),
                                                0.0,#T(options.x_tol),
                                                NaN,# x_abschange(state),
                                                NaN,# x_abschange(state),
                                                false,# f_converged,
                                                0.0,#T(options.f_tol),
                                                0.0,#T(options.f_tol),
                                                NaN,#f_abschange(d, state),
                                                NaN,#f_abschange(d, state),
                                                false,#g_converged,
                                                0.0,#T(options.g_tol),
                                                NaN,#g_residual(d),
                                                false, #f_increased,
                                                nothing,
                                                maxiters,
                                                maxiters,
                                                0,
                                                true,
                                                NaN,
                                                result.logger.times[end] - result.logger.times[1])
		end
	end
end
