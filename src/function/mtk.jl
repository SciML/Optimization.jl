"""
AutoModelingToolkit <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f, AutoModelingToolkit(); kwargs...)
```

This uses the [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
package's `modelingtookitize` functionality to generate the derivatives and other fields of an `OptimizationFunction`.
This backend creates the symbolic expressions for the objective and its derivatives as well as
the constraints and their derivatives. Through `structural_simplify`, it enforces symplifications
that can reduce the number of operations needed to compute the derivatives of the constraints. This automatically
generates the expression graphs that some solver interfaces through OptimizationMOI like
[AmplNLWriter.jl](https://github.com/jump-dev/AmplNLWriter.jl) require.

  - Compatible with GPUs
  - Compatible with Hessian-based optimization
  - Compatible with Hv-based optimization
  - Compatible with constraints

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not generated via ModelingToolkit.

## Constructor

```julia
AutoModelingToolkit(false, false)
```

  - `obj_sparse`: to indicate whether the objective hessian is sparse.
  - `cons_sparse`: to indicate whether the constraints' jacobian and hessian are sparse.

"""
struct AutoModelingToolkit <: AbstractADType
    obj_sparse::Bool
    cons_sparse::Bool
end

AutoModelingToolkit() = AutoModelingToolkit(false, false)

function instantiate_function(f, x, adtype::AutoModelingToolkit, p,
                              num_cons = 0)
    p = isnothing(p) ? SciMLBase.NullParameters() : p

    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p;
                                                                 lcons = fill(0.0,
                                                                              num_cons),
                                                                 ucons = fill(0.0,
                                                                              num_cons)))
    sys = ModelingToolkit.structural_simplify(sys)
    f = OptimizationProblem(sys, x, p, grad = true, hess = true,
                            sparse = adtype.obj_sparse, cons_j = true, cons_h = true,
                            cons_sparse = adtype.cons_sparse).f

    grad = (G, θ, args...) -> f.grad(G, θ, p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, p, args...)

    hv = function (H, θ, v, args...)
        res = adtype.obj_sparse ? (eltype(θ)).(f.hess_prototype) :
              ArrayInterface.zeromatrix(θ)
        hess(res, θ, args...)
        H .= res * v
    end

    cons = (res, θ) -> f.cons(res, θ, p)

    cons_j = (J, θ) -> f.cons_j(J, θ, p)

    cons_h = (res, θ) -> f.cons_h(res, θ, p)
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype,
                                      expr = symbolify(f.expr),
                                      cons_expr = symbolify.(f.cons_expr),
                                      observed = f.observed)
end

function instantiate_function(f, cache::ReInitCache,
                              adtype::AutoModelingToolkit, num_cons = 0)
    p = isnothing(cache.p) ? SciMLBase.NullParameters() : cache.p

    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, cache.u0, cache.p;
                                                                 lcons = fill(0.0,
                                                                              num_cons),
                                                                 ucons = fill(0.0,
                                                                              num_cons)))
    sys = ModelingToolkit.structural_simplify(sys)
    f = OptimizationProblem(sys, cache.u0, cache.p, grad = true, hess = true,
                            sparse = adtype.obj_sparse, cons_j = true, cons_h = true,
                            cons_sparse = adtype.cons_sparse).f

    grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)

    hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)

    hv = function (H, θ, v, args...)
        res = adtype.obj_sparse ? (eltype(θ)).(f.hess_prototype) :
              ArrayInterface.zeromatrix(θ)
        hess(res, θ, args...)
        H .= res * v
    end

    cons = (res, θ) -> f.cons(res, θ, cache.p)

    cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)

    cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = f.hess_prototype,
                                      cons_jac_prototype = f.cons_jac_prototype,
                                      cons_hess_prototype = f.cons_hess_prototype,
                                      expr = symbolify(f.expr),
                                      cons_expr = symbolify.(f.cons_expr),
                                      observed = f.observed)
end
