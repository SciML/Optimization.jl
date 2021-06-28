const AbstractFluxOptimiser = Union{Flux.Momentum,
                                    Flux.Nesterov,
                                    Flux.RMSProp,
                                    Flux.ADAM,
                                    Flux.RADAM,
                                    Flux.AdaMax,
                                    Flux.OADAM,
                                    Flux.ADAGrad,
                                    Flux.ADADelta,
                                    Flux.AMSGrad,
                                    Flux.NADAM,
                                    Flux.AdaBelief,
                                    Flux.Optimiser}

function __solve(prob::OptimizationProblem, opt, data = DEFAULT_DATA;
                 maxiters::Number = 0, cb = (args...) -> (false),
                 progress = false, save_best = true, kwargs...)

    if data != DEFAULT_DATA
      maxiters = length(data)
    else
      if maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
      end
      data = take(data, maxiters)
    end

    # Flux is silly and doesn't have an abstract type on its optimizers, so assume
    # this is a Flux optimizer
    θ = copy(prob.u0)
    G = copy(θ)

    t0 = time()

    local x, min_err, _loss
    min_err = typemax(eltype(prob.u0)) #dummy variables
    min_opt = 1

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    @withprogress progress name="Training" begin
      for (i,d) in enumerate(data)
        f.grad(G, θ, d...)
        x = f.f(θ, prob.p, d...)
        cb_call = cb(θ, x...)
        if !(typeof(cb_call) <: Bool)
          error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
          break
        end
        msg = @sprintf("loss: %.3g", x[1])
        progress && ProgressLogging.@logprogress msg i/maxiters
        Flux.update!(opt, θ, G)

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

    SciMLBase.build_solution(prob, opt, θ, x[1])
    # here should be build_solution to create the output message
end

function Flux.update!(opt, xs::Flux.Zygote.Params, gs)
    Flux.update!(opt, xs[1], gs)
end

@require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
  function Flux.update!(x::AbstractArray, x̄::AbstractArray{<:ForwardDiff.Dual})
    x .-= x̄
  end

  function Flux.update!(x::AbstractArray, x̄)
    x .-= getindex.(ForwardDiff.partials.(x̄),1)
  end

  function Flux.update!(opt, x, x̄)
    x .-= Flux.Optimise.apply!(opt, x, x̄)
  end

  function Flux.update!(opt, x, x̄::AbstractArray{<:ForwardDiff.Dual})
    x .-= Flux.Optimise.apply!(opt, x, getindex.(ForwardDiff.partials.(x̄),1))
  end
end
