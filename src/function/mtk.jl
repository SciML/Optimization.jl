struct AutoModelingToolkit <: AbstractADType
    obj_sparse::Bool
    cons_sparse::Bool
end

AutoModelingToolkit() = AutoModelingToolkit(false, false)

function instantiate_function(f, x, adtype::AutoModelingToolkit, p, num_cons=0)
    p = isnothing(p) ? SciMLBase.NullParameters() : p
    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p))

    hess_prototype, cons_jac_prototype, cons_hess_prototype = nothing, nothing, nothing

    if f.grad === nothing
        grad_oop, grad_iip = ModelingToolkit.generate_gradient(sys, expression=Val{false})
        grad(J, u) = (grad_iip(J, u, p); J)
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess_oop, hess_iip = ModelingToolkit.generate_hessian(sys, expression=Val{false}, sparse = adtype.obj_sparse)
        hess(H, u) = (hess_iip(H, u, p); H)
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = ad.obj_sparse ? hess_prototype : ArrayInterfaceCore.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (θ) -> f.cons(θ, p)
        cons_sys = ModelingToolkit.modelingtoolkitize(NonlinearProblem(f.cons, x, p))
    end

    if f.cons !== nothing && f.cons_j === nothing
        jac_oop, jac_iip = ModelingToolkit.generate_jacobian(cons_sys, expression=Val{false}, sparse=adtype.cons_sparse)
        cons_j = function (J, θ)
            jac_iip(J, θ, p)
        end
    else
        cons_j = f.cons_j
    end

    if f.cons !== nothing && f.cons_h === nothing
        cons_hess_oop, cons_hess_iip = ModelingToolkit.generate_hessian(cons_sys, expression=Val{false}, sparse=adtype.cons_sparse)
        cons_h = function (res, θ)
            cons_hess_iip(res, θ, p)
        end
    else
        cons_h = f.cons_h
    end

    if ad.obj_sparse
        _hess_prototype = ModelingToolkit.hessian_sparsity(sys)
        hess_prototype = sparse(findnz(_hess_prototype)[1], findnz(_hess_prototype)[2], zeros(nnz(_hess_prototype)))
    end

    if ad.cons_sparse
        _cons_jac_prototype = ModelingToolkit.jacobian_sparsity(cons_sys)
        cons_jac_prototype = sparse(findnz(_cons_jac_prototype)[1], findnz(_cons_jac_prototype)[2], zeros(nnz(_cons_jac_prototype)))
        _cons_hess_prototype = ModelingToolkit.hessian_sparsity(cons_sys)
        cons_hess_prototype = [sparse(findnz(_cons_hess_prototype[i])[1], findnz(_cons_hess_prototype[i])[2], zeros(nnz(_cons_hess_prototype[i]))) for i in eachindex(_cons_hess_prototype)]
    end

    return OptimizationFunction{true}(f.f, adtype; grad=grad, hess=hess, hv=hv, 
        cons=cons, cons_j=cons_j, cons_h=cons_h,
        hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing)
end
