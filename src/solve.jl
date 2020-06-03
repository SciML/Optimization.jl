
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
			error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
		end
		cb_call
  	end
	
	if prob.f isa OptimizationFunction
		_loss = function(θ)
			x = prob.f.f(θ, prob.p)
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
			res = DiffResults.HessianResult(prob.x)

			hv! = function (H,θ,v)
			  prob.f.hess(res, θ)
			  H .= DiffResults.hessian(res)*v
			end

			optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, hv!, prob.x)
		else
			optim_f = TwiceDifferentiable(_loss, prob.f.grad, fg!, prob.f.hess, prob.x)
		end
	else
		!(opt isa Optim.ZerothOrderOptimizer) && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")
		_loss = function(θ)
			x = prob.f(θ, prob.p)
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
			error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
	  	end
	  	cb_call
	end
  
  	if prob.f isa OptimizationFunction && !(opt isa Optim.SAMIN)
	  	_loss = function(θ)
		  	x = prob.f.f(θ, prob.p)
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
				error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
			  end
			  if cb_call == true
				BlackBoxOptim.shutdown_optimizer!(trace) #doesn't work
			  end
			  cb_call
			end

			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
				end
			else 
				_loss = function(θ)
					x = prob.f(θ, prob.p)
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
		function __solve(prob::OptimizationProblem, opt::NLopt.Opt; cb = (args...) -> (false), maxiters = 1000, nstart = 1, localopt = nothing, kwargs...)	
			local x

			if prob.f isa OptimizationFunction 
				_loss = function(θ)
					x = prob.f.f(θ, prob.p)
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
				_loss = function(θ)
					x = prob.f(θ, prob.p)
				end
				NLopt.min_objective!(opt, _loss)
			end

			if prob.ub !== nothing
				NLopt.upper_bounds!(opt, prob.ub)				
			end
			if prob.lb !== nothing
				NLopt.lower_bounds!(opt, prob.lb)
			end

			if nstart > 1 && localopt !== nothing
				NLopt.local_optimizer!(opt, localopt)
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
end
  