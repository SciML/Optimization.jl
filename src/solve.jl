
function DiffEqBase.solve(prob::OptimizationProblem, opt, args...;kwargs...)
	__solve(prob, opt, args...; kwargs...)
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function __solve(prob::OptimizationProblem, opt::Optim.ZerothOrderOptimizer;cb = (args...) -> (false), kwargs...)
  	local x

	function _cb(trace)
		cb_call = opt == NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
		if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
		end
		cb_call
  	end

	function optim_f(θ)
		if prob.u0 !== nothing
			x = prob.f(prob.u0,θ)
		else
			x = prob.f(θ)
		end
		x
  	end

  	Optim.optimize(optim_f, prob.p, opt, Optim.Options(;extended_trace=true,callback = _cb, kwargs...))
end

function __solve(prob::OptimizationProblem, opt::Optim.FirstOrderOptimizer;cb = (args...) -> (false), kwargs...)
	local x

	if !(prob.f isa OptimizationFunction)
		error("Use OptimizationFunction to pass the gradient or automatically generate the gradient")
	end

  	function _cb(trace)
		cb_call = cb(decompose_trace(trace).metadata["x"],x...)
		if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
		end
		cb_call
  	end

	function optim_f(θ)
		if prob.u0 !== nothing
			x = prob.f.f(prob.u0,θ)
		else
			x = prob.f.f(θ)
		end
		x
  	end

  	Optim.optimize(optim_f, prob.f.grad, prob.p, opt, Optim.Options(;extended_trace=true,callback = _cb, kwargs...))	
end

function __solve(prob::OptimizationProblem, opt::Optim.SecondOrderOptimizer;cb = (args...) -> (false), kwargs...)
	local x

	if !(prob.f isa OptimizationFunction)
		error("Use OptimizationFunction to pass the gradient and hessian or automatically generate the gradient and hessian")
	end

  	function _cb(trace)
		cb_call = cb(decompose_trace(trace).metadata["x"],x...)
		if !(typeof(cb_call) <: Bool)
			error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
		end
		cb_call
  	end

	function optim_f(θ)
		if prob.u0 !== nothing
			x = prob.f.f(prob.u0,θ)
		else
			x = prob.f.f(θ)
		end
		x
  	end

  	Optim.optimize(optim_f, prob.f.grad, prob.f.hes, prob.p, opt, Optim.Options(;extended_trace=true,callback = _cb, kwargs...))	
end