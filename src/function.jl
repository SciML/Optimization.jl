struct AutoForwardDiff{chunksize} <: AbstractADType end
function AutoForwardDiff(chunksize=nothing)
    AutoForwardDiff{chunksize}()
end

struct AutoReverseDiff <: AbstractADType end
struct AutoTracker <: AbstractADType end
struct AutoZygote <: AbstractADType end

struct AutoFiniteDiff{T1,T2} <: AbstractADType
    fdtype::T1
    fdhtype::T2
end
AutoFiniteDiff(;fdtype = Val(:forward), fdhtype = Val(:hcentral)) =
                                                  AutoFiniteDiff(fdtype,fdhtype)

function default_chunk_size(len)
    if len < DEFAULT_CHUNK_THRESHOLD
        len
    else
        DEFAULT_CHUNK_THRESHOLD
    end
end

function instantiate_function(f, x, ::AbstractADType, p)
    grad   = f.grad   === nothing ? nothing : (G,x)->f.grad(G,x,p)
    hess   = f.hess   === nothing ? nothing : (H,x)->f.hess(H,x,p)
    hv     = f.hv     === nothing ? nothing : (H,x,v)->f.hv(H,x,v,p)
    cons   = f.cons   === nothing ? nothing : (x)->f.cons(x,p)
    cons_j = f.cons_j === nothing ? nothing : (res,x)->f.cons_j(res,x,p)
    cons_h = f.cons_h === nothing ? nothing : (res,x)->f.cons_h(res,x,p)

    OptimizationFunction{true,DiffEqBase.NoAD,typeof(f.f),typeof(grad),
                         typeof(hess),typeof(hv),typeof(cons),
                         typeof(cons_j),typeof(cons_h)}(f.f,
                         DiffEqBase.NoAD(),grad,hess,hv,cons,
                         cons_j,cons_h,f.num_cons)
end

function instantiate_function(f, x, ::AutoForwardDiff{_chunksize}, p) where _chunksize

    chunksize = _chunksize === nothing ? default_chunk_size(length(x)) : _chunksize

    _f = θ -> first(f.f(θ,p))

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        grad = (res,θ) -> ForwardDiff.gradient!(res, _f, θ, gradcfg)
    else
        grad = f.grad
    end

    if f.hess === nothing
        hesscfg = ForwardDiff.HessianConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        hess = (res,θ) -> ForwardDiff.hessian!(res, _f, θ, hesscfg)
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H,θ,v)
            res = ArrayInterface.zeromatrix(θ)
            hess(res, θ)
            H .= res*v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
        cons! = nothing
    else
        cons = θ -> f.cons(θ,p)
        cons! = (res, θ) -> (res .= f.cons(θ,p); res)
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
            for i in 1:f.num_cons
                hess_config_cache = ForwardDiff.HessianConfig(x -> cons(x)[i], θ,ForwardDiff.Chunk{chunksize}())
                ForwardDiff.hessian!(res[i], (x) -> cons(x)[i], θ, hess_config_cache,Val{false}())
            end
        end
    else
        cons_h = f.cons_h
    end

    return OptimizationFunction{true,AutoForwardDiff,typeof(f.f),typeof(grad),typeof(hess),typeof(hv),typeof(cons),typeof(cons_j),typeof(cons_h)}(f.f,AutoForwardDiff(),grad,hess,hv,cons,cons_j,cons_h,f.num_cons)
end

function instantiate_function(f, x, ::AutoZygote, p)
    f.num_cons != 0 && error("AutoZygote does not currently support constraints")

    _f = θ -> f(θ,p)[1]
    if f.grad === nothing
        grad = (res,θ) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Zygote.gradient(_f, θ)[1]) : res .= Zygote.gradient(_f, θ)[1]
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = function (res,θ)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                                Zygote.gradient(_f,θ)[1]
                                            end)
            else
                res .=  ForwardDiff.jacobian(θ) do θ
                    Zygote.gradient(_f,θ)[1]
                  end
            end
        end
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H,θ,v)
            _θ = ForwardDiff.Dual.(θ,v)
            res = DiffResults.GradientResult(_θ)
            grad(res,_θ)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)),1)
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{false,AutoZygote,typeof(f),typeof(grad),typeof(hess),typeof(hv),Nothing,Nothing,Nothing}(f,AutoZygote(),grad,hess,hv,nothing,nothing,nothing,0)
end

function instantiate_function(f, x, ::AutoReverseDiff, p=DiffEqBase.NullParameters())
    f.num_cons != 0 && error("AutoReverseDiff does not currently support constraints")

    _f = θ -> f.f(θ,p)[1]

    if f.grad === nothing
        grad = (res,θ) -> ReverseDiff.gradient!(res, _f, θ, ReverseDiff.GradientConfig(θ))
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = function (res,θ)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                                ReverseDiff.gradient(_f,θ)[1]
                                            end)
            else
                res .=  ForwardDiff.jacobian(θ) do θ
                    ReverseDiff.gradient(_f,θ)
                  end
            end
        end
    else
        hess = f.hess
    end


    if f.hv === nothing
        hv = function (H,θ,v)
            _θ = ForwardDiff.Dual.(θ,v)
            res = DiffResults.GradientResult(_θ)
            grad(res,_θ)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)),1)
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{false,AutoReverseDiff,typeof(f),typeof(grad),typeof(hess),typeof(hv),Nothing,Nothing,Nothing}(f,AutoReverseDiff(),grad,hess,hv,nothing,nothing,nothing,0)
end


function instantiate_function(f, x, ::AutoTracker, p)
    f.num_cons != 0 && error("AutoTracker does not currently support constraints")
    _f = θ -> f.f(θ,p)[1]

    if f.grad === nothing
        grad = (res,θ) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Tracker.data(Tracker.gradient(_f, θ)[1])) : res .= Tracker.data(Tracker.gradient(_f, θ)[1])
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = (res, θ) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` kwarg")
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = (res, θ) -> error("Hessian based methods not supported with Tracker backend, pass in the `hess` and `hv` kwargs")
    else
        hv = f.hv
    end


    return OptimizationFunction{false,AutoTracker,typeof(f),typeof(grad),typeof(hess),typeof(hv),Nothing,Nothing,Nothing}(f,AutoTracker(),grad,hess,hv,nothing,nothing,nothing,0)
end

function instantiate_function(f, x, adtype::AutoFiniteDiff, p)

    f.num_cons != 0 && error("AutoFiniteDiff does not currently support constraints")
    _f = θ -> f.f(θ,p)[1]

    if f.grad === nothing
        grad = (res,θ) -> FiniteDiff.finite_difference_gradient!(res, _f, θ, FiniteDiff.GradientCache(res, x, adtype.fdtype))
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = (res,θ) -> FiniteDiff.finite_difference_hessian!(res, _f, θ, FiniteDiff.HessianCache(x, adtype.fdhtype))
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H,θ,v)
            res = ArrayInterface.zeromatrix(θ)
            hess(res, θ)
            H .= res*v
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{false,AutoFiniteDiff,typeof(f),typeof(grad),typeof(hess),typeof(hv),Nothing,Nothing,Nothing}(f,adtype,grad,hess,hv,nothing,nothing,nothing,0)
end
