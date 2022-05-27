struct AutoReverseDiff <: AbstractADType end

function instantiate_function(f, x, adtype::AutoReverseDiff, p=SciMLBase.NullParameters(), num_cons = 0)
    num_cons != 0 && error("AutoReverseDiff does not currently support constraints")

    _f = (θ, args...) -> first(f.f(θ,p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ, ReverseDiff.GradientConfig(θ))
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                                ReverseDiff.gradient(x -> _f(x, args...), θ)[1]
                                            end)
            else
                res .=  ForwardDiff.jacobian(θ) do θ
                    ReverseDiff.gradient(x ->_f(x, args...), θ)
                  end
            end
        end
    else
        hess = f.hess
    end


    if f.hv === nothing
        hv = function (H,θ,v, args...)
            _θ = ForwardDiff.Dual.(θ,v)
            res = DiffResults.GradientResult(_θ)
            grad(res, _θ, args...)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)),1)
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{true}(f, adtype; grad=grad, hess=hess, hv=hv,
        cons=nothing, cons_j=nothing, cons_h=nothing,
        hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing)
end
