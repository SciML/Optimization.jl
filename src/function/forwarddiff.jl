"""
AutoForwardDiff{chunksize} <: AbstractADType

An AbstractADType choice for use in OptimizationFunction for automatically
generating the unspecified derivative functions. Usage:

```julia
OptimizationFunction(f,AutoForwardDiff();kwargs...)
```

This uses the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
package. It is the fastest choice for small systems, especially with
heavy scalar interactions. It is easy to use and compatible with most
pure is Julia functions which have loose type restrictions. However,
because it's forward-mode, it scales poorly in comparison to other AD
choices. Hessian construction is suboptimal as it uses the forward-over-forward
approach.

- Compatible with GPUs
- Compatible with Hessian-based optimization
- Compatible with Hv-based optimization
- Compatible with constraints

Note that only the unspecified derivative functions are defined. For example,
if a `hess` function is supplied to the `OptimizationFunction`, then the
Hessian is not defined via ForwardDiff.
"""
struct AutoForwardDiff{chunksize} <: AbstractADType end

function AutoForwardDiff(chunksize = nothing)
    AutoForwardDiff{chunksize}()
end

function default_chunk_size(len)
    if len < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        len
    else
        ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    end
end

function instantiate_function(
    f::OptimizationFunction{true},
    x,
    adtype::AutoForwardDiff{_chunksize},
    p,
    num_cons = 0,
) where {_chunksize}

    chunksize = _chunksize === nothing ? default_chunk_size(length(x)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        gradcfg =
            (args...) -> ForwardDiff.GradientConfig(
                x -> _f(x, args...),
                x,
                ForwardDiff.Chunk{chunksize}(),
            )
        grad =
            (res, θ, args...) -> ForwardDiff.gradient!(
                res,
                x -> _f(x, args...),
                θ,
                gradcfg(args...),
                Val{false}(),
            )
    else
        grad = f.grad
    end

    if f.hess === nothing
        hesscfg =
            (args...) -> ForwardDiff.HessianConfig(
                x -> _f(x, args...),
                x,
                ForwardDiff.Chunk{chunksize}(),
            )
        hess =
            (res, θ, args...) -> ForwardDiff.hessian!(
                res,
                x -> _f(x, args...),
                θ,
                hesscfg(args...),
                Val{false}(),
            )
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
        cons = θ -> f.cons(θ, p)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            cjconfig = ForwardDiff.JacobianConfig(cons, θ, ForwardDiff.Chunk{chunksize}())
            ForwardDiff.jacobian!(J, cons, θ, cjconfig)
        end
    else
        cons_j = f.cons_j
    end

    if cons !== nothing && f.cons_h === nothing
        cons_h = function (res, θ)
            for i = 1:num_cons
                hess_config_cache = ForwardDiff.HessianConfig(
                    x -> cons(x)[i],
                    θ,
                    ForwardDiff.Chunk{chunksize}(),
                )
                ForwardDiff.hessian!(
                    res[i],
                    (x) -> cons(x)[i],
                    θ,
                    hess_config_cache,
                    Val{false}(),
                )
            end
        end
    else
        cons_h = f.cons_h
    end

    return OptimizationFunction{true}(
        f.f,
        adtype;
        grad = grad,
        hess = hess,
        hv = hv,
        cons = cons,
        cons_j = cons_j,
        cons_h = cons_h,
        hess_prototype = nothing,
        cons_jac_prototype = nothing,
        cons_hess_prototype = nothing,
    )
end
