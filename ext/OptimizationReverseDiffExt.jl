module OptimizationReverseDiffExt

import Optimization
import Optimization.SciMLBase: OptimizationFunction
import Optimization.ADTypes: AutoReverseDiff
# using SparseDiffTools, Symbolics
isdefined(Base, :get_extension) ? (using ReverseDiff, ReverseDiff.ForwardDiff) :
(using ..ReverseDiff, ..ReverseDiff.ForwardDiff)

struct OptimizationReverseDiffTag end

function default_chunk_size(len)
    if len < ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        len
    else
        ForwardDiff.DEFAULT_CHUNK_THRESHOLD
    end
end

function Optimization.instantiate_function(f, x, adtype::AutoReverseDiff,
    p = SciMLBase.NullParameters(),
    num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))
    
    chunksize = default_chunk_size(length(x))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, x)
            tape = ReverseDiff.compile(_tape)
            grad = function (res, θ, args...)
                ReverseDiff.gradient!(res, tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(x)
            grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ, cfg)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationReverseDiffTag(),eltype(x))
            xdual = ForwardDiff.Dual{typeof(T),eltype(x),chunksize}.(x, Ref(ForwardDiff.Partials((ones(eltype(x), chunksize)...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(θ)
                res1 = zeros(eltype(θ), length(θ))
                ReverseDiff.gradient!(res1, htape, θ)
            end
            jaccfg = ForwardDiff.JacobianConfig(g, x, ForwardDiff.Chunk{chunksize}(), T)
            hess = function (res, θ, args...)
                ForwardDiff.jacobian!(res, g, θ, jaccfg, Val{false}())
            end
        else
            hess = function (res, θ, args...)
                ReverseDiff.hessian!(res, x -> _f(x, args...), θ)
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        if adtype.compile
            _jac_tape = ReverseDiff.JacobianTape(cons_oop, x)
            jac_tape = ReverseDiff.compile(_jac_tape)
            cons_j = function (J, θ)
                ReverseDiff.jacobian!(J, jac_tape, θ)
            end
        else
            cjconfig = ReverseDiff.JacobianConfig(x)
            cons_j = function (J, θ)
                ReverseDiff.jacobian!(J, cons_oop, θ, cjconfig)
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        if adtype.compile
            consh_tapes = ReverseDiff.GradientTape.(fncs, Ref(xdual))
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(θ, htape)
                res1 = zeros(eltype(θ), length(θ))
                ReverseDiff.gradient!(res1, htape, θ)
            end
            gs = [x -> grad_cons(x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardDiff.JacobianConfig(gs[i], x, ForwardDiff.Chunk{chunksize}(), T) for i in 1:num_cons]
            cons_h = function (res, θ)
                for i in 1:num_cons
                    ForwardDiff.jacobian!(res[i], gs[i], θ, jaccfgs[i], Val{false}())
                end
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    ReverseDiff.hessian!(res[i], fncs[i], θ)
                end
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    end
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

function Optimization.instantiate_function(f, cache::Optimization.ReInitCache,
    adtype::AutoReverseDiff, num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    chunksize = default_chunk_size(length(cache.u0))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, cache.u0)
            tape = ReverseDiff.compile(_tape)
            grad = function (res, θ, args...)
                ReverseDiff.gradient!(res, tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(cache.u0)
            grad = (res, θ, args...) -> ReverseDiff.gradient!(res, x -> _f(x, args...), θ, cfg)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    if f.hess === nothing
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationReverseDiffTag(),eltype(cache.u0))
            xdual = ForwardDiff.Dual{typeof(T),eltype(cache.u0),chunksize}.(cache.u0, Ref(ForwardDiff.Partials((ones(eltype(cache.u0), chunksize)...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(θ)
                res1 = zeros(eltype(θ), length(θ))
                ReverseDiff.gradient!(res1, htape, θ)
            end
            jaccfg = ForwardDiff.JacobianConfig(g, cache.u0, ForwardDiff.Chunk{chunksize}(), T)
            hess = function (res, θ, args...)
                ForwardDiff.jacobian!(res, g, θ, jaccfg, Val{false}())
            end
        else
            hess = function (res, θ, args...)
                ReverseDiff.hessian!(res, x -> _f(x, args...), θ)
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        if adtype.compile
            _jac_tape = ReverseDiff.JacobianTape(cons_oop, cache.u0)
            jac_tape = ReverseDiff.compile(_jac_tape)
            cons_j = function (J, θ)
                ReverseDiff.jacobian!(J, jac_tape, θ)
            end
        else
            cjconfig = ReverseDiff.JacobianConfig(cache.u0)
            cons_j = function (J, θ)
                ReverseDiff.jacobian!(J, cons_oop, θ, cjconfig)
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        if adtype.compile
            consh_tapes = ReverseDiff.GradientTape.(fncs, Ref(xdual))
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(θ, htape)
                res1 = zeros(eltype(θ), length(θ))
                ReverseDiff.gradient!(res1, htape, θ)
            end
            gs = [x -> grad_cons(x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardDiff.JacobianConfig(gs[i], cache.u0, ForwardDiff.Chunk{chunksize}(), T) for i in 1:num_cons]
            cons_h = function (res, θ)
                for i in 1:num_cons
                    ForwardDiff.jacobian!(res[i], gs[i], θ, jaccfgs[i], Val{false}())
                end
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    ReverseDiff.hessian!(res[i], fncs[i], θ)
                end
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h, f.lag_hess_prototype)
end

end
