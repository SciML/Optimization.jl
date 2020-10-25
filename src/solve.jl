
struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i=1) = nothing
Base.length(::NullData) = 0

get_maxiters(data) = Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
                     Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
                     typemax(Int) : length(data)

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
	update!(opt, xs[1], gs)
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

function __solve(prob::OptimizationProblem, opt, _data = DEFAULT_DATA;cb = (args...) -> (false), maxiters::Number = 1000, progress = true, save_best = true, kwargs...)
	if maxiters <= 0.0
		error("The number of maxiters has to be a non-negative and non-zero number.")
	else
		maxiters = convert(Int, maxiters)
	end

	# Flux is silly and doesn't have an abstract type on its optimizers, so assume
	# this is a Flux optimizer
	θ = copy(prob.u0)
	ps = Flux.params(θ)

	if _data == DEFAULT_DATA && maxiters == typemax(Int)
		error("For Flux optimizers, either a data iterator must be provided or the `maxiters` keyword argument must be set.")
	  elseif _data == DEFAULT_DATA && maxiters != typemax(Int)
		data = Iterators.repeated((), maxiters)
	  elseif maxiters != typemax(Int)
		data = take(_data, maxiters)
	  else
		data = _data
	end

	t0 = time()

	local x, min_err, _loss
	min_err = typemax(eltype(prob.u0)) #dummy variables
	min_opt = 1

	f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

	@withprogress progress name="Training" begin
	  for (i,d) in enumerate(data)
		gs = prob.f.adtype isa AutoFiniteDiff ? Array{Number}(undef,length(θ)) : DiffResults.GradientResult(θ)
		f.grad(gs, θ, d...) 
		x = f.f(θ, prob.p, d...)
		cb_call = cb(θ, x...)
		if !(typeof(cb_call) <: Bool)
		  error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
		elseif cb_call
		  break
		end
		msg = @sprintf("loss: %.3g", x[1])
		progress && ProgressLogging.@logprogress msg i/maxiters
		update!(opt, ps, prob.f.adtype isa AutoFiniteDiff ? gs : DiffResults.gradient(gs))

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
										  prob.u0,# initial_x,
										  θ, #pick_best_x(f_incr_pick, state),
										  save_best ? first(min_err) : first(x), # pick_best_f(f_incr_pick, state, d),
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
										  _time-t0,
										  NamedTuple())
end


decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function __solve(prob::OptimizationProblem, opt::Optim.AbstractOptimizer, data = DEFAULT_DATA;cb = (args...) -> (false), maxiters::Number = 1000, kwargs...)
  	local x, cur, state
	cur, state = iterate(data)

	function _cb(trace)
		cb_call = opt == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
		if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process.")
		end
		cur, state = iterate(data, state)
		cb_call
  	end

	if maxiters <= 0.0
		error("The number of maxiters has to be a non-negative and non-zero number.")
	else
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

  	Optim.optimize(optim_f, prob.u0, opt, Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...))
end

function __solve(prob::OptimizationProblem, opt::Union{Optim.Fminbox,Optim.SAMIN}, data = DEFAULT_DATA;cb = (args...) -> (false), maxiters::Number = 1000, kwargs...)
	local x, cur, state
	cur, state = iterate(data)

  	function _cb(trace)
	  	cb_call = !(opt isa Optim.SAMIN) && opt.method == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
	  	if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process.")
		end
		cur, state = iterate(data, state)  
	  	cb_call
	end

	if maxiters <= 0.0
		error("The number of maxiters has to be a non-negative and non-zero number.")
	else
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

	Optim.optimize(optim_f, prob.lb, prob.ub, prob.u0, opt, Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...))
end


function __solve(prob::OptimizationProblem, opt::Optim.ConstrainedOptimizer, data = DEFAULT_DATA;cb = (args...) -> (false), maxiters::Number = 1000, kwargs...)
	local x, cur, state
	cur, state = iterate(data)

  	function _cb(trace)
	  cb_call = cb(decompose_trace(trace).metadata["x"],x...)
	  if !(typeof(cb_call) <: Bool)
		  error("The callback should return a boolean `halt` for whether to stop the optimization process.")
	  end
	  cur, state = iterate(data, state)
	  cb_call
	end

	if maxiters <= 0.0
		error("The number of maxiters has to be a non-negative and non-zero number.")
	else
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

	Optim.optimize(optim_f, optim_fc, prob.u0, opt, Optim.Options(;extended_trace = true, callback = _cb, iterations = maxiters, kwargs...))
end


function __init__()
	@require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209" begin
		decompose_trace(opt::BlackBoxOptim.OptRunController) = BlackBoxOptim.best_candidate(opt)

		struct BBO
			method::Symbol
		end

		BBO() = BBO(:adaptive_de_rand_1_bin)

		function __solve(prob::OptimizationProblem, opt::BBO, data = DEFAULT_DATA; cb = (args...) -> (false), maxiters::Number = 1000, kwargs...)
			local x, cur, state
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

			if maxiters <= 0.0
				error("The number of maxiters has to be a non-negative and non-zero number.")
			else
				maxiters = convert(Int, maxiters)
			end

			_loss = function(θ)
				x = prob.f(θ, prob.p, cur...)
				return first(x)
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
												  bboptre.elapsed_time,
												  NamedTuple())
		end
	end

	@require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" begin
		function __solve(prob::OptimizationProblem, opt::NLopt.Opt; maxiters::Number = 1000, nstart = 1, local_method = nothing, kwargs...)
			local x

			if maxiters <= 0.0
				error("The number of maxiters has to be a non-negative and non-zero number.")
			else
				maxiters = convert(Int, maxiters)
			end

			f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

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

			if prob.ub !== nothing
				NLopt.upper_bounds!(opt, prob.ub)
			end
			if prob.lb !== nothing
				NLopt.lower_bounds!(opt, prob.lb)
			end

			NLopt.maxeval!(opt, maxiters)

			if nstart > 1 && local_method !== nothing
				NLopt.local_optimizer!(opt, local_method)
				NLopt.maxeval!(opt, nstart * maxiters)
			end

            t0= time()
            (minf,minx,ret) = NLopt.optimize(opt, prob.u0)
            _time = time()

            Optim.MultivariateOptimizationResults(opt,
                                                    prob.u0,# initial_x,
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
													_time-t0,
													NamedTuple())
		end
	end

	@require MultistartOptimization = "3933049c-43be-478e-a8bb-6e0f7fd53575" begin
		function __solve(prob::OptimizationProblem, opt::MultistartOptimization.TikTak; local_method, local_maxiters::Number = 1000, kwargs...)
			local x, _loss

			if local_maxiters <= 0.0
				error("The number of local_maxiters has to be a non-negative and non-zero number.")
			else
				local_maxiters = convert(Int, local_maxiters)
			end

			_loss = function(θ)
				x = prob.f(θ, prob.p)
				return first(x)
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
                                                t1 - t0,
												NamedTuple())
		end
	end

	@require QuadDIRECT = "dae52e8d-d666-5120-a592-9e15c33b8d7a" begin
		export QuadDirect

        struct QuadDirect
		end

		function __solve(prob::OptimizationProblem, opt::QuadDirect; splits, maxiters::Number = 1000, kwargs...)
			local x, _loss

			if maxiters <= 0.0
				error("The number of maxiters has to be a non-negative and non-zero number.")
			else
				maxiters = convert(Int, maxiters)
			end

			_loss = function(θ)
				x = prob.f(θ, prob.p)
				return first(x)
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
                                                t1 - t0,
												NamedTuple())
		end
	end

	@require Evolutionary="86b6b26d-c046-49b6-aa0b-5f0f74682bd6" begin
		decompose_trace(trace::Evolutionary.OptimizationTrace) = last(trace)

		function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::Evolutionary.AbstractOptimizer, options)
			record["x"] = population
		end

		function __solve(prob::OptimizationProblem, opt::Evolutionary.AbstractOptimizer, data = DEFAULT_DATA; cb = (args...) -> (false), maxiters::Number = 1000, kwargs...)
			local x, cur, state
			cur, state = iterate(data)

			function _cb(trace)
				cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
				if !(typeof(cb_call) <: Bool)
					error("The callback should return a boolean `halt` for whether to stop the optimization process.")
				end
				cur, state = iterate(data, state)
				cb_call
			end

			if maxiters <= 0.0
				error("The number of maxiters has to be a non-negative and non-zero number.")
			else
				maxiters = convert(Int, maxiters)
			end

			_loss = function(θ)
				x = prob.f(θ, prob.p, cur...)
				return first(x)
			end

			Evolutionary.optimize(_loss, prob.u0, opt, Evolutionary.Options(;iterations = maxiters, callback = _cb, kwargs...))
		end
	end
	@require CMAEvolutionStrategy="8d3b24bd-414e-49e0-94fb-163cc3a3e411" begin

		struct CMAEvolutionStrategyOpt end

		function __solve(prob::OptimizationProblem, opt::CMAEvolutionStrategyOpt, data = DEFAULT_DATA; cb = (args...) -> (false), maxiters::Number = 1000, kwargs...)
			local x, cur, state
			cur, state = iterate(data)

			function _cb(trace)
				cb_call = cb(decompose_trace(trace).metadata["x"],trace.value...)
				if !(typeof(cb_call) <: Bool)
					error("The callback should return a boolean `halt` for whether to stop the optimization process.")
				end
				cur, state = iterate(data, state)
				cb_call
			end

			if maxiters <= 0.0
				error("The number of maxiters has to be a non-negative and non-zero number.")
			else
				maxiters = convert(Int, maxiters)
			end

			_loss = function(θ)
				x = prob.f(θ, prob.p, cur...)
				return first(x)
			end

			result = CMAEvolutionStrategy.minimize(_loss, prob.u0, 0.1; lower = prob.lb, upper = prob.ub, kwargs...)

           	Optim.MultivariateOptimizationResults(opt,
                                                prob.u0,# initial_x,
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
                                                result.logger.times[end] - result.logger.times[1],
												NamedTuple())
		end
	end
end
