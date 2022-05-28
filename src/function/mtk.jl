"""
AutoModelingToolkit <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f,AutoModelingToolkit();kwargs...)
```

This uses the [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
symbolic system for automatically converting the `f` function into
a symbolic equation and uses symbolic differentiation in order to generate
a fast derivative code. Note that this will also compile a new version
of your `f` function that is automatically optimized. Because of the
required symbolic analysis, the state and parameters are required in
the function definition, i.e.:

Summary:

- Not compatible with GPUs
- Compatible with Hessian-based optimization
- Not compatible with Hv-based optimization
- Not compatible with constraint functions

## Constructor

```julia
OptimizationFunction(f,AutoModelingToolkit(),x0,p,
                     grad = false, hess = false, sparse = false,
                     checkbounds = false,
                     linenumbers = true,
                     parallel=SerialForm(),
                     kwargs...)
```

The special keyword arguments are as follows:

- `grad`: whether to symbolically generate the gradient function.
- `hess`: whether to symbolically generate the Hessian function.
- `sparse`: whether to use sparsity detection in the Hessian.
- `checkbounds`: whether to perform bounds checks in the generated code.
- `linenumbers`: whether to include line numbers in the generated code.
- `parallel`: whether to automatically parallelize the calculations.

For more information, see the [ModelingToolkit.jl `OptimizationSystem` documentation](https://mtk.sciml.ai/dev/systems/OptimizationSystem/)
"""
struct AutoModelingToolkit <: AbstractADType
    obj_sparse::Bool
    cons_sparse::Bool
end

AutoModelingToolkit() = AutoModelingToolkit(false, false)

function instantiate_function(f, x, adtype::AutoModelingToolkit, p, num_cons=0)
    p = isnothing(p) ? SciMLBase.NullParameters() : p
    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p))

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
            res = ArrayInterfaceCore.zeromatrix(θ)
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

    return OptimizationFunction{true}(f.f, adtype; grad=grad, hess=hess, hv=hv, 
        cons=cons, cons_j=cons_j, cons_h=cons_h,
        hess_prototype=nothing, cons_jac_prototype=nothing, cons_hess_prototype=nothing)
end
