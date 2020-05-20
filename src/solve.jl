
function DiffEqBase.solve(prob::OptimizationProblem, opt, args...;kwargs...)
	__solve(prob, opt, args...; kwargs...)
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function __solve(prob::OptimizationProblem, opt::Optim.AbstractOptimizer;cb = (args...) -> (false), kwargs...)
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
				if G != nothing
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
	
  	Optim.optimize(optim_f, prob.x, opt, Optim.Options(;extended_trace = true, callback = _cb, kwargs...))
end

function __solve(prob::OptimizationProblem, opt::Optim.AbstractConstrainedOptimizer;cb = (args...) -> (false), kwargs...)
	local x

  	function _cb(trace)
	  	cb_call = opt.method == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
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
			  	if G != nothing
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
		  	x = prob.f(θ, prob.p)
	  	end
	  	optim_f = _loss
  	end
  
	Optim.optimize(optim_f, prob.lb, prob.ub, prob.x, opt, Optim.Options(;extended_trace = true, callback = _cb, kwargs...))
end