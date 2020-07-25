
function DiffEqBase.solve(prob::OptimizationProblem, opt, args...;kwargs...)
	__solve(prob, opt, args...; kwargs...)
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
		fg! = let res = DiffResults.GradientResult(prob.x)
			function (G,θ)
				if G !== nothing
					prob.f.grad(res, θ)
					G .= DiffResults.gradient(res)
				end
	
				return _loss(θ)
			end
		end
		if opt isa Optim.KrylovTrustRegion
			hv! = function (H,θ,v)
				res = Array{typeof(x[1])}(undef, length(θ), length(θ)) #DiffResults.HessianResult(θ)
				prob.f.hess(res, θ)
			  	H .= res*v
			end

			optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, hv!, prob.x)
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
	  	fg! = let res = DiffResults.GradientResult(prob.x)
		  	function (G,θ)
			  	if G !== nothing
				  	prob.f.grad(res, θ)
				  	G .= DiffResults.gradient(res)
			  	end
  
			  	return _loss(θ)
		  	end
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
				fg! = let res = DiffResults.GradientResult(prob.x)
					function (θ,G)
						if length(G) > 0
							prob.f.grad(res, θ)
							G .= DiffResults.gradient(res)
						end

						return _loss(θ)
					end
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
end
  