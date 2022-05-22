struct AutoForwardDiff{chunksize} <: AbstractADType end
function AutoForwardDiff(chunksize=nothing)
    AutoForwardDiff{chunksize}()
end

function default_chunk_size(len)
    if len < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        len
    else
        ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    end
end

function instantiate_function(f::OptimizationFunction{true}, x, ::AutoForwardDiff{_chunksize}, p, num_cons = 0) where _chunksize

    chunksize = _chunksize === nothing ? default_chunk_size(length(x)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        gradcfg = (args...) -> ForwardDiff.GradientConfig(x -> _f(x, args...), x, ForwardDiff.Chunk{chunksize}())
        grad = (res, θ, args...) -> ForwardDiff.gradient!(res, x -> _f(x, args...), θ, gradcfg(args...), Val{false}())
    else
        grad = f.grad
    end

    if f.hess === nothing
        hesscfg = (args...) -> ForwardDiff.HessianConfig(x -> _f(x, args...), x, ForwardDiff.Chunk{chunksize}())
        hess = (res, θ, args...) -> ForwardDiff.hessian!(res, x -> _f(x, args...), θ, hesscfg(args...), Val{false}())
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H,θ,v, args...)
            res = ArrayInterfaceCore.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res*v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = θ -> f.cons(θ,p)
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
            for i in 1:num_cons
                hess_config_cache = ForwardDiff.HessianConfig(x -> cons(x)[i], θ,ForwardDiff.Chunk{chunksize}())
                ForwardDiff.hessian!(res[i], (x) -> cons(x)[i], θ, hess_config_cache,Val{false}())
            end
        end
    else
        cons_h = f.cons_h
    end

    return OptimizationFunction{true,AutoForwardDiff,typeof(f.f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f.f,AutoForwardDiff(),grad,hess,hv,cons,cons_j,cons_h)
end
