struct AutoFiniteDiff{T1,T2} <: AbstractADType
    fdtype::T1
    fdhtype::T2
end

AutoFiniteDiff(;fdtype = Val(:forward), fdhtype = Val(:hcentral)) =
                                                  AutoFiniteDiff(fdtype,fdhtype)

function instantiate_function(f, x, adtype::AutoFiniteDiff, p, num_cons = 0)
    num_cons != 0 && error("AutoFiniteDiff does not currently support constraints")
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        grad = (res, θ, args...) -> FiniteDiff.finite_difference_gradient!(res, x ->_f(x, args...), θ, FiniteDiff.GradientCache(res, x, adtype.fdtype))
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = (res, θ, args...) -> FiniteDiff.finite_difference_hessian!(res, x ->_f(x, args...), θ, FiniteDiff.HessianCache(x, adtype.fdhtype))
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ArrayInterfaceCore.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res*v
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{false,AutoFiniteDiff,typeof(f),typeof(grad),typeof(hess),typeof(hv),Nothing,Nothing,Nothing}(f,adtype,grad,hess,hv,nothing,nothing,nothing)
end
