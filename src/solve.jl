
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
		optim_f = TwiceDifferentiable(_loss, prob.f.grad, prob.f.hess, prob.x)
	else
		!(opt isa Optim.ZerothOrderOptimizer) && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")
		_loss = function(θ)
			x = prob.f(θ, prob.p)
		end
		optim_f = _loss
	end
	
  	Optim.optimize(optim_f, prob.x, opt, Optim.Options(;extended_trace = true, callback = _cb, kwargs...))
end
