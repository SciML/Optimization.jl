module OptimizationNLPModels

using Reexport
@reexport using NLPModels, OptimizationBase, ADTypes
using SparseArrays

"""
    OptimizationFunction(nlpmodel::AbstractNLPModel, adtype::AbstractADType = NoAD())

Returns an `OptimizationFunction` from the `NLPModel` defined in `nlpmodel` where the
available derivatives are re-used from the model, and the rest are populated with the
Automatic Differentiation backend specified by `adtype`.
"""
function SciMLBase.OptimizationFunction(
        nlpmodel::AbstractNLPModel,
        adtype::ADTypes.AbstractADType = SciMLBase.NoAD(); kwargs...
    )
    f(x, p) = NLPModels.obj(nlpmodel, x)
    grad(G, u, p) = NLPModels.grad!(nlpmodel, u, G)
    hess(H, u, p) = (H .= NLPModels.hess(nlpmodel, u))
    hv(Hv, u, v, p) = NLPModels.hprod!(nlpmodel, u, v, Hv)

    if !unconstrained(nlpmodel) && !bound_constrained(nlpmodel)
        cons(res, x, p) = NLPModels.cons!(nlpmodel, x, res)
        cons_j(J, x, p) = (J .= NLPModels.jac(nlpmodel, x))
        cons_jvp(Jv, v, x, p) = NLPModels.jprod!(nlpmodel, x, v, Jv)

        return OptimizationFunction(
            f, adtype; grad, hess, hv, cons, cons_j, cons_jvp, kwargs...
        )
    end

    return OptimizationFunction(f, adtype; grad, hess, hv, kwargs...)
end

"""
    OptimizationProblem(nlpmodel::AbstractNLPModel, adtype::AbstractADType = NoAD())

Returns an `OptimizationProblem` with the bounds and constraints defined in `nlpmodel`.
The optimization function and its derivatives are re-used from `nlpmodel` when available
or populated wit the Automatic Differentiation backend specified by `adtype`.
"""
function SciMLBase.OptimizationProblem(
        nlpmodel::AbstractNLPModel,
        adtype::ADTypes.AbstractADType = SciMLBase.NoAD(); kwargs...
    )
    f = OptimizationFunction(nlpmodel, adtype; kwargs...)
    u0 = nlpmodel.meta.x0
    lb, ub = if has_bounds(nlpmodel)
        (nlpmodel.meta.lvar, nlpmodel.meta.uvar)
    else
        (nothing, nothing)
    end

    lcons, ucons = if has_inequalities(nlpmodel) || has_equalities(nlpmodel)
        (nlpmodel.meta.lcon, nlpmodel.meta.ucon)
    else
        (nothing, nothing)
    end
    sense = nlpmodel.meta.minimize ? OptimizationBase.MinSense : OptimizationBase.MaxSense

    # The number of variables, geometry of u0, etc.. are valid and were checked when the
    # nlpmodel was created.

    return OptimizationBase.OptimizationProblem(
        f, u0; lb = lb, ub = ub, lcons = lcons, ucons = ucons, sense = sense, kwargs...
    )
end

struct NLPModelsAdaptor{C, T, HB} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    cache::C
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    jac_buffer::AbstractMatrix{T}
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    hess_buffer::HB  # Can be Vector{T} or Matrix{T}
end

function _enumerate_dense_structure(ncon, nvar)
    nnz = ncon * nvar
    rows = Vector{Int}(undef, nnz)
    cols = Vector{Int}(undef, nnz)
    idx = 1
    for j in 1:nvar
        for i in 1:ncon
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function _enumerate_lower_triangle(n)
    nnz = div(n * (n + 1), 2)
    rows = Vector{Int}(undef, nnz)
    cols = Vector{Int}(undef, nnz)
    idx = 1
    for j in 1:n
        for i in j:n  # Only lower triangle
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function NLPModelsAdaptor(
        cache::C, meta::NLPModels.NLPModelMeta{T, Vector{T}}, counters
    ) where {C, T}
    # Extract Jacobian structure once
    jac_prototype = cache.f.cons_jac_prototype

    if jac_prototype isa SparseMatrixCSC
        jac_rows, jac_cols, _ = findnz(jac_prototype)
        jac_buffer = similar(jac_prototype)
    elseif jac_prototype isa AbstractMatrix
        ncon, nvar = size(jac_prototype)
        jac_rows, jac_cols = _enumerate_dense_structure(ncon, nvar)
        jac_buffer = similar(jac_prototype)
    else
        # Fallback: assume dense structure
        ncon, nvar = meta.ncon, meta.nvar
        jac_rows, jac_cols = _enumerate_dense_structure(ncon, nvar)
        jac_buffer = zeros(T, ncon, nvar)
    end

    ncon = !isnothing(cache.lcons) ? length(cache.lcons) : 0

    # Extract Hessian structure
    hess_proto = ncon > 0 ? cache.f.lag_hess_prototype : cache.f.hess_prototype

    if !isnothing(hess_proto) && hess_proto isa SparseMatrixCSC
        I, J, _ = findnz(hess_proto)
        # Keep only lower triangle
        lower_mask = I .>= J
        hess_rows = I[lower_mask]
        hess_cols = J[lower_mask]
        # Create a values buffer matching the number of lower triangle elements
        hess_buffer = zeros(T, sum(lower_mask))
    elseif !isnothing(hess_proto)
        # Dense Hessian
        n = size(hess_proto, 1)
        hess_rows, hess_cols = _enumerate_lower_triangle(n)
        # For dense, store the full matrix but we'll extract values later
        hess_buffer = similar(hess_proto, T)
    else
        # No prototype - create dense structure
        n = meta.nvar
        hess_rows, hess_cols = _enumerate_lower_triangle(n)
        hess_buffer = zeros(T, n, n)
    end

    return NLPModelsAdaptor{C, T, typeof(hess_buffer)}(
        cache, meta, counters,
        jac_rows, jac_cols, jac_buffer,
        hess_rows, hess_cols, hess_buffer
    )
end

function NLPModels.obj(nlp::NLPModelsAdaptor, x::AbstractVector)
    return nlp.cache.f(x, nlp.cache.p)
end

function NLPModels.grad!(nlp::NLPModelsAdaptor, x::AbstractVector, g::AbstractVector)
    nlp.cache.f.grad(g, x, nlp.cache.p)
    return g
end

function NLPModels.cons!(nlp::NLPModelsAdaptor, x::AbstractVector, c::AbstractVector)
    if !isempty(c)
        nlp.cache.f.cons(c, x)
    end
    return c
end

function NLPModels.jac_structure!(
        nlp::NLPModelsAdaptor, I::AbstractVector{T}, J::AbstractVector{T}
    ) where {T}
    copyto!(I, nlp.jac_rows)
    copyto!(J, nlp.jac_cols)
    return I, J
end

function NLPModels.jac_coord!(
        nlp::NLPModelsAdaptor, x::AbstractVector, vals::AbstractVector
    )
    if !isempty(vals)
        # Evaluate Jacobian into preallocated buffer
        nlp.cache.f.cons_j(nlp.jac_buffer, x)

        # Extract values in COO order
        if nlp.jac_buffer isa SparseMatrixCSC
            _, _, v = findnz(nlp.jac_buffer)
            copyto!(vals, v)
        else
            # Dense case: extract in column-major order matching structure
            for (idx, (i, j)) in enumerate(zip(nlp.jac_rows, nlp.jac_cols))
                vals[idx] = nlp.jac_buffer[i, j]
            end
        end
    end

    return vals
end

function NLPModels.hess_structure!(
        nlp::NLPModelsAdaptor, I::AbstractVector{T}, J::AbstractVector{T}
    ) where {T}
    copyto!(I, nlp.hess_rows)
    copyto!(J, nlp.hess_cols)
    return I, J
end

function NLPModels.hess_coord!(
        nlp::NLPModelsAdaptor, x, y, H::AbstractVector; obj_weight = 1.0
    )
    if !isnothing(nlp.cache.f.lag_h)
        # Use Lagrangian Hessian directly
        if nlp.hess_buffer isa AbstractVector
            # For sparse prototypes, hess_buffer is already a values vector
            nlp.cache.f.lag_h(nlp.hess_buffer, x, obj_weight, y)
        else
            # For dense matrices, we need to pass the full matrix and extract values
            nlp.cache.f.lag_h(nlp.hess_buffer, x, obj_weight, y)
        end
    else
        # Manual computation: objective + constraint Hessians
        nlp.cache.f.hess(nlp.hess_buffer, x)
        nlp.hess_buffer .*= obj_weight

        if !isnothing(nlp.cache.f.cons_h) && !isempty(y)
            # Add weighted constraint Hessians
            cons_hessians = [
                similar(nlp.hess_buffer, eltype(nlp.hess_buffer))
                    for _ in 1:length(y)
            ]
            nlp.cache.f.cons_h(cons_hessians, x)
            for (λ, H_cons) in zip(y, cons_hessians)
                nlp.hess_buffer .+= λ .* H_cons
            end
        end
    end

    if !isempty(H)
        # Extract values depending on buffer type
        if nlp.hess_buffer isa AbstractVector
            # For sparse, hess_buffer already contains just the values
            copyto!(H, nlp.hess_buffer)
        else
            # For dense matrices, extract lower triangle values
            for (idx, (i, j)) in enumerate(zip(nlp.hess_rows, nlp.hess_cols))
                H[idx] = nlp.hess_buffer[i, j]
            end
        end
    end

    return H
end

function NLPModels.jtprod!(
        nlp::NLPModelsAdaptor, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector
    )
    # Compute J^T * v using the AD-provided VJP (Vector-Jacobian Product)
    if !isnothing(nlp.cache.f.cons_vjp) && !isempty(Jtv)
        nlp.cache.f.cons_vjp(Jtv, x, v)
    end
    return Jtv
end

function NLPModels.jprod!(
        nlp::NLPModelsAdaptor, x::AbstractVector, v::AbstractVector, Jv::AbstractVector
    )
    # Compute J * v using the AD-provided JVP (Jacobian-Vector Product)
    if !isnothing(nlp.cache.f.cons_jvp) && !isempty(Jv)
        nlp.cache.f.cons_jvp(Jv, x, v)
    end
    return Jv
end

function _get_nnzj(f, ncon, nvar)
    jac_prototype = f.cons_jac_prototype

    if isnothing(jac_prototype)
        # No prototype - assume dense structure if there are constraints
        return ncon > 0 ? ncon * nvar : 0
    elseif jac_prototype isa SparseMatrixCSC
        nnz(jac_prototype)
    else
        prod(size(jac_prototype))
    end
end

function _get_nnzh(f, ncon, nvar)
    # For constrained problems, use Lagrangian Hessian; for unconstrained, use objective Hessian
    hess_proto = ncon > 0 ? f.lag_hess_prototype : f.hess_prototype

    if isnothing(hess_proto)
        # No prototype provided - assume dense Hessian
        return div(nvar * (nvar + 1), 2)
    elseif hess_proto isa SparseMatrixCSC
        # Only count lower triangle
        I, J, _ = findnz(hess_proto)
        return count(i >= j for (i, j) in zip(I, J))
    else
        # Dense: n(n+1)/2
        n = size(hess_proto, 1)
        return div(n * (n + 1), 2)
    end
end

function build_nlpmodel_meta(cache)
    T = eltype(cache.u0)

    nvar = length(cache.u0)
    ncon = !isnothing(cache.lcons) ? length(cache.lcons) : 0

    lvar = something(cache.lb, fill(-Inf, nvar))
    uvar = something(cache.ub, fill(Inf, nvar))
    lcon = something(cache.lcons, T[])
    ucon = something(cache.ucons, T[])

    return NLPModels.NLPModelMeta(
        nvar;
        ncon,
        nnzj = _get_nnzj(cache.f, ncon, nvar),
        nnzh = _get_nnzh(cache.f, ncon, nvar),
        x0 = cache.u0,
        y0 = zeros(eltype(cache.u0), ncon),
        lvar,
        uvar,
        lcon,
        ucon,
        minimize = cache.sense !== MaxSense  # Default to minimization when sense is nothing or MinSense
    )
end

end
