mutable struct IpoptCache{T, F <: OptimizationFunction, RC, LB, UB, I, S,
    JT <: DenseOrSparse{T}, HT <: DenseOrSparse{T},
    CHT <: DenseOrSparse{T}, CB, O} <: SciMLBase.AbstractOptimizationCache
    const f::F
    const n::Int
    const num_cons::Int
    const reinit_cache::RC
    const lb::LB
    const ub::UB
    const int::I
    const lcons::Vector{T}
    const ucons::Vector{T}
    const sense::S
    J::JT
    H::HT
    cons_H::Vector{CHT}
    const callback::CB
    const progress::Bool
    f_calls::Int
    f_grad_calls::Int
    iterations::Cint
    obj_expr::Union{Expr, Nothing}
    cons_expr::Union{Vector{Expr}, Nothing}
    const opt::O
    const solver_args::NamedTuple
end

function Base.getproperty(cache::IpoptCache, name::Symbol)
    if name in fieldnames(OptimizationBase.ReInitCache)
        return getfield(cache.reinit_cache, name)
    end
    return getfield(cache, name)
end
function Base.setproperty!(cache::IpoptCache, name::Symbol, x)
    if name in fieldnames(OptimizationBase.ReInitCache)
        return setfield!(cache.reinit_cache, name, x)
    end
    return setfield!(cache, name, x)
end

function SciMLBase.get_p(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {T, N, uType, C <: IpoptCache}
    sol.cache.p
end
function SciMLBase.get_observed(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {T, N, uType, C <: IpoptCache}
    sol.cache.f.observed
end
function SciMLBase.get_syms(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {T, N, uType, C <: IpoptCache}
    variable_symbols(sol.cache.f)
end
function SciMLBase.get_paramsyms(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {T, N, uType, C <: IpoptCache}
    parameter_symbols(sol.cache.f)
end

function IpoptCache(prob, opt;
        callback = nothing,
        progress = false,
        kwargs...)
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    if prob.f.adtype isa ADTypes.AutoSymbolics || (prob.f.adtype isa ADTypes.AutoSparse &&
        prob.f.adtype.dense_ad isa ADTypes.AutoSymbolics)
        f = Optimization.instantiate_function(
            prob.f, reinit_cache, prob.f.adtype, num_cons;
            g = true, h = true, cons_j = true, cons_h = true)
    else
        f = Optimization.instantiate_function(
            prob.f, reinit_cache, prob.f.adtype, num_cons;
            g = true, h = true, cons_j = true, cons_vjp = true, lag_h = true)
    end
    T = eltype(prob.u0)
    n = length(prob.u0)

    J = if isnothing(f.cons_jac_prototype)
        zeros(T, num_cons, n)
    else
        convert.(T, f.cons_jac_prototype)
    end
    lagh = !isnothing(f.lag_hess_prototype)
    H = if lagh # lag hessian takes precedence
        convert.(T, f.lag_hess_prototype)
    elseif !isnothing(f.hess_prototype)
        convert.(T, f.hess_prototype)
    else
        zeros(T, n, n)
    end
    cons_H = if lagh
        Matrix{T}[zeros(T, 0, 0) for i in 1:num_cons] # No need to allocate this up if using lag hessian
    elseif isnothing(f.cons_hess_prototype)
        Matrix{T}[zeros(T, n, n) for i in 1:num_cons]
    else
        [convert.(T, f.cons_hess_prototype[i]) for i in 1:num_cons]
    end
    lcons = prob.lcons === nothing ? fill(T(-Inf), num_cons) : prob.lcons
    ucons = prob.ucons === nothing ? fill(T(Inf), num_cons) : prob.ucons

    sys = f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing} ?
            nothing : f.sys
    obj_expr = f.expr
    cons_expr = f.cons_expr

    solver_args = NamedTuple(kwargs)

    return IpoptCache(
        f,
        n,
        num_cons,
        reinit_cache,
        prob.lb,
        prob.ub,
        prob.int,
        lcons,
        ucons,
        prob.sense,
        J,
        H,
        cons_H,
        callback,
        progress,
        0,
        0,
        Cint(0),
        obj_expr,
        cons_expr,
        opt,
        solver_args
    )
end

function eval_objective(cache::IpoptCache, x)
    l = cache.f(x, cache.p)
    cache.f_calls += 1
    return cache.sense === Optimization.MaxSense ? -l : l
end

function eval_constraint(cache::IpoptCache, g, x)
    cache.f.cons(g, x)
    return
end

function eval_objective_gradient(cache::IpoptCache, G, x)
    if cache.f.grad === nothing
        error("Use OptimizationFunction to pass the objective gradient or " *
              "automatically generate it with one of the autodiff backends." *
              "If you are using the ModelingToolkit symbolic interface, pass the `grad` kwarg set to `true` in `OptimizationProblem`.")
    end
    cache.f.grad(G, x)
    cache.f_grad_calls += 1

    if cache.sense === Optimization.MaxSense
        G .*= -one(eltype(G))
    end

    return
end

function jacobian_structure(cache::IpoptCache)
    if cache.J isa SparseMatrixCSC
        rows, cols, _ = findnz(cache.J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(cache.J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function eval_constraint_jacobian(cache::IpoptCache, j, x)
    if isempty(j)
        return
    elseif cache.f.cons_j === nothing
        error("Use OptimizationFunction to pass the constraints' jacobian or " *
              "automatically generate i with one of the autodiff backends." *
              "If you are using the ModelingToolkit symbolic interface, pass the `cons_j` kwarg set to `true` in `OptimizationProblem`.")
    end
    # Get and cache the Jacobian object here once. `evaluator.J` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    J = cache.J
    cache.f.cons_j(J, x)
    if J isa SparseMatrixCSC
        nnz = nonzeros(J)
        @assert length(j) == length(nnz)
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        j .= vec(J)
    end
    return
end

function hessian_lagrangian_structure(cache::IpoptCache)
    lagh = cache.f.lag_h !== nothing
    if cache.f.lag_hess_prototype isa SparseMatrixCSC
        rows, cols, _ = findnz(cache.f.lag_hess_prototype)
        return Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    end
    sparse_obj = cache.H isa SparseMatrixCSC
    sparse_constraints = all(H -> H isa SparseMatrixCSC, cache.cons_H)
    if !lagh && !sparse_constraints && any(H -> H isa SparseMatrixCSC, cache.cons_H)
        # Some constraint hessians are dense and some are sparse! :(
        error("Mix of sparse and dense constraint hessians are not supported")
    end
    N = length(cache.u0)
    inds = if sparse_obj
        rows, cols, _ = findnz(cache.H)
        Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    else
        Tuple{Int, Int}[(row, col) for col in 1:N for row in 1:col]
    end
    lagh && return inds
    if sparse_constraints
        for Hi in cache.cons_H
            r, c, _ = findnz(Hi)
            for (i, j) in zip(r, c)
                if i <= j
                    push!(inds, (i, j))
                end
            end
        end
    elseif !sparse_obj
        # Performance optimization. If both are dense, no need to repeat
    else
        for col in 1:N, row in 1:col
            push!(inds, (row, col))
        end
    end
    return inds
end

function eval_hessian_lagrangian(cache::IpoptCache{T},
        h,
        x,
        σ,
        μ) where {T}
    if cache.f.lag_h !== nothing
        cache.f.lag_h(h, x, σ, Vector(μ))

        if cache.sense === Optimization.MaxSense
            h .*= -one(eltype(h))
        end

        return
    end
    if cache.f.hess === nothing
        error("Use OptimizationFunction to pass the objective hessian or " *
              "automatically generate it with one of the autodiff backends." *
              "If you are using the ModelingToolkit symbolic interface, pass the `hess` kwarg set to `true` in `OptimizationProblem`.")
    end
    # Get and cache the Hessian object here once. `evaluator.H` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    H = cache.H
    fill!(h, zero(T))
    k = 0
    cache.f.hess(H, x)
    sparse_objective = H isa SparseMatrixCSC
    if sparse_objective
        rows, cols, _ = findnz(H)
        for (i, j) in zip(rows, cols)
            if i <= j
                k += 1
                h[k] = σ * H[i, j]
            end
        end
    else
        for i in 1:size(H, 1), j in 1:i
            k += 1
            h[k] = σ * H[i, j]
        end
    end
    # A count of the number of non-zeros in the objective Hessian is needed if
    # the constraints are dense.
    nnz_objective = k
    if !isempty(μ) && !all(iszero, μ)
        if cache.f.cons_h === nothing
            error("Use OptimizationFunction to pass the constraints' hessian or " *
                  "automatically generate it with one of the autodiff backends." *
                  "If you are using the ModelingToolkit symbolic interface, pass the `cons_h` kwarg set to `true` in `OptimizationProblem`.")
        end
        cache.f.cons_h(cache.cons_H, x)
        for (μi, Hi) in zip(μ, cache.cons_H)
            if Hi isa SparseMatrixCSC
                rows, cols, _ = findnz(Hi)
                for (i, j) in zip(rows, cols)
                    if i <= j
                        k += 1
                        h[k] += μi * Hi[i, j]
                    end
                end
            else
                # The constraints are dense. We only store one copy of the
                # Hessian, so reset `k` to where it starts. That will be
                # `nnz_objective` if the objective is sprase, and `0` otherwise.
                k = sparse_objective ? nnz_objective : 0
                for i in 1:size(Hi, 1), j in 1:i
                    k += 1
                    h[k] += μi * Hi[i, j]
                end
            end
        end
    end

    if cache.sense === Optimization.MaxSense
        h .*= -one(eltype(h))
    end

    return
end
