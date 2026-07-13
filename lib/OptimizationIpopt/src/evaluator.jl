"""
    IpoptEvaluator(cache::OptimizationCache)

Solver-specific state built inside `__solve`: preallocated derivative buffers
(constraint Jacobian `J`, Lagrangian/objective Hessian `H`, per-constraint
Hessians `cons_H`) plus evaluation counters. Ipopt's C callbacks close over
this object.
"""
mutable struct IpoptEvaluator{C <: OptimizationCache, JT, HT, CHT}
    const cache::C
    const J::JT
    const H::HT
    const cons_H::Vector{CHT}
    f_calls::Int
    f_grad_calls::Int
end

function IpoptEvaluator(cache::OptimizationCache)
    f = cache.f
    T = eltype(cache.u0)
    n = length(cache.u0)
    num_cons = cache.ucons === nothing ? 0 : length(cache.ucons)

    J = if isnothing(f.cons_jac_prototype)
        zeros(T, num_cons, n)
    else
        similar(f.cons_jac_prototype, T)
    end
    lagh = !isnothing(f.lag_h)
    H = if !isnothing(f.lag_hess_prototype) # lag hessian takes precedence
        similar(f.lag_hess_prototype, T)
    elseif !isnothing(f.hess_prototype)
        similar(f.hess_prototype, T)
    elseif isnothing(f.hess) && !lagh
        nothing # no second-order information (e.g. hessian_approximation = "limited-memory")
    else
        zeros(T, n, n)
    end
    cons_H = if lagh || isnothing(f.cons_h)
        Matrix{T}[] # not needed when using the Lagrangian hessian or no second order
    elseif isnothing(f.cons_hess_prototype)
        Matrix{T}[zeros(T, n, n) for _ in 1:num_cons]
    else
        [similar(f.cons_hess_prototype[i], T) for i in 1:num_cons]
    end

    return IpoptEvaluator(cache, J, H, cons_H, 0, 0)
end

# Forward the live `cache.p` (which `reinit!` may have replaced) to the
# instantiated derivative closures. The `AutoSymbolics` instantiation hardcodes
# the parameter object into its wrappers, the `NoAD` wrappers only accept a
# parameter argument when the problem has parameters, and `instantiate_function`
# normalizes `nothing` to `NullParameters()` (so passing the raw `nothing`
# through would invalidate the DifferentiationInterface preparation), so those
# paths have to be called without `p`.
function _accepts_live_p(evaluator::IpoptEvaluator)
    adtype = evaluator.cache.f.adtype
    return !(
        adtype isa ADTypes.AutoSymbolics ||
            (adtype isa ADTypes.AutoSparse && adtype.dense_ad isa ADTypes.AutoSymbolics) ||
            evaluator.cache.p === nothing ||
            evaluator.cache.p isa SciMLBase.NullParameters
    )
end

function eval_objective(evaluator::IpoptEvaluator, x)
    l = evaluator.cache.f(x, evaluator.cache.p)
    evaluator.f_calls += 1
    return l
end

function eval_constraint(evaluator::IpoptEvaluator, g, x)
    if _accepts_live_p(evaluator)
        evaluator.cache.f.cons(g, x, evaluator.cache.p)
    else
        evaluator.cache.f.cons(g, x)
    end
    return
end

function eval_objective_gradient(evaluator::IpoptEvaluator, G, x)
    if evaluator.cache.f.grad === nothing
        error(
            "Use OptimizationFunction to pass the objective gradient or " *
                "automatically generate it with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `grad` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    if _accepts_live_p(evaluator)
        evaluator.cache.f.grad(G, x, evaluator.cache.p)
    else
        evaluator.cache.f.grad(G, x)
    end
    evaluator.f_grad_calls += 1
    return
end

function jacobian_structure(evaluator::IpoptEvaluator)
    J = evaluator.J
    if J isa SparseMatrixCSC
        rows, cols, _ = findnz(J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function eval_constraint_jacobian(evaluator::IpoptEvaluator, j, x)
    if isempty(j)
        return
    elseif evaluator.cache.f.cons_j === nothing
        error(
            "Use OptimizationFunction to pass the constraints' jacobian or " *
                "automatically generate it with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `cons_j` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    J = evaluator.J
    if _accepts_live_p(evaluator)
        evaluator.cache.f.cons_j(J, x, evaluator.cache.p)
    else
        evaluator.cache.f.cons_j(J, x)
    end
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

function hessian_lagrangian_structure(evaluator::IpoptEvaluator)
    f = evaluator.cache.f
    lagh = f.lag_h !== nothing
    if f.lag_hess_prototype isa SparseMatrixCSC
        rows, cols, _ = findnz(f.lag_hess_prototype)
        return Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    end
    sparse_obj = evaluator.H isa SparseMatrixCSC
    sparse_constraints = all(H -> H isa SparseMatrixCSC, evaluator.cons_H)
    if !lagh && !sparse_constraints && any(H -> H isa SparseMatrixCSC, evaluator.cons_H)
        # Some constraint hessians are dense and some are sparse! :(
        error("Mix of sparse and dense constraint hessians are not supported")
    end
    N = length(evaluator.cache.u0)
    inds = if sparse_obj
        rows, cols, _ = findnz(evaluator.H)
        Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    else
        Tuple{Int, Int}[(row, col) for col in 1:N for row in 1:col]
    end
    lagh && return inds
    if sparse_constraints
        for Hi in evaluator.cons_H
            r, c, _ = findnz(Hi)
            for (i, j) in zip(r, c)
                if i <= j
                    push!(inds, (i, j))
                end
            end
        end
    elseif !sparse_obj
        # If both the objective and the constraint hessians are dense, the
        # dense upper triangle above already covers the union of the sparsity
        # patterns, so nothing needs to be appended.
    else
        for col in 1:N, row in 1:col
            push!(inds, (row, col))
        end
    end
    return inds
end

function eval_hessian_lagrangian(evaluator::IpoptEvaluator, h, x, σ, μ)
    f = evaluator.cache.f
    if f.lag_h !== nothing
        if _accepts_live_p(evaluator)
            f.lag_h(h, x, σ, Vector(μ), evaluator.cache.p)
        else
            f.lag_h(h, x, σ, Vector(μ))
        end
        return
    end
    if f.hess === nothing
        error(
            "Use OptimizationFunction to pass the objective hessian or " *
                "automatically generate it with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `hess` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    H = evaluator.H
    fill!(h, zero(eltype(h)))
    k = 0
    if _accepts_live_p(evaluator)
        f.hess(H, x, evaluator.cache.p)
    else
        f.hess(H, x)
    end
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
        if f.cons_h === nothing
            error(
                "Use OptimizationFunction to pass the constraints' hessian or " *
                    "automatically generate it with one of the autodiff backends." *
                    "If you are using the ModelingToolkit symbolic interface, pass the `cons_h` kwarg set to `true` in `OptimizationProblem`."
            )
        end
        f.cons_h(evaluator.cons_H, x)
        for (μi, Hi) in zip(μ, evaluator.cons_H)
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
                # `nnz_objective` if the objective is sparse, and `0` otherwise.
                k = sparse_objective ? nnz_objective : 0
                for i in 1:size(Hi, 1), j in 1:i
                    k += 1
                    h[k] += μi * Hi[i, j]
                end
            end
        end
    end
    return
end
