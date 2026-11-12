mutable struct MOIOptimizationNLPEvaluator{
        T, F <: OptimizationFunction, RC, LB, UB,
        I, JT <: AbstractMatrix{T}, HT <: AbstractMatrix{T}, CHT <: AbstractMatrix{T}, S, CB,
    } <:
    MOI.AbstractNLPEvaluator
    f::F
    const source_f::OptimizationFunction
    const source_cons_vjp::Bool
    reinit_cache::RC
    lb::LB
    ub::UB
    int::I
    lcons::Vector{T}
    ucons::Vector{T}
    sense::S
    J::JT
    H::HT
    cons_H::Vector{CHT}
    callback::CB
    iteration::Int
    obj_expr::Union{Expr, Nothing}
    cons_expr::Union{AbstractVector{Expr}, Nothing}
    # For a system-based problem `obj_expr`/`cons_expr` are only built by `MOI.initialize`
    # when the solver requests `:ExprGraph`; numeric-only solvers never pay for them.
    system_exprs_pending::Bool
    dual_start::Union{Nothing, NamedTuple}
end

function Base.getproperty(evaluator::MOIOptimizationNLPEvaluator, x::Symbol)
    if x in fieldnames(OptimizationBase.ReInitCache)
        return getfield(evaluator.reinit_cache, x)
    end
    return getfield(evaluator, x)
end

struct MOIOptimizationNLPCache{E <: MOIOptimizationNLPEvaluator, O, R} <:
    SciMLBase.AbstractOptimizationCache
    evaluator::E
    opt::O
    solver_args::NamedTuple
    dual_results::R
end

function Base.getproperty(cache::MOIOptimizationNLPCache{E}, name::Symbol) where {E}
    if name === :reinit_cache || name === :dual_start
        return getfield(getfield(cache, :evaluator), name)
    elseif name in fieldnames(E)
        return getfield(cache.evaluator, name)
    elseif name in fieldnames(OptimizationBase.ReInitCache)
        return getfield(cache.evaluator.reinit_cache, name)
    end
    return getfield(cache, name)
end
function Base.setproperty!(cache::MOIOptimizationNLPCache{E}, name::Symbol, x) where {E}
    if name in fieldnames(E)
        return setfield!(cache.evaluator, name, x)
    elseif name in fieldnames(OptimizationBase.ReInitCache)
        return setfield!(cache.evaluator.reinit_cache, name, x)
    end
    # The cache itself is immutable; its own fields delegate reads through
    # `getproperty` but are never reassigned, so there is no valid setter.
    throw(ArgumentError(lazy"field `$name` of `MOIOptimizationNLPCache` is immutable and cannot be set"))
end

function SciMLBase.get_p(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {
        T, N,
        uType,
        C <:
        MOIOptimizationNLPCache,
    }
    return sol.cache.evaluator.p
end
function SciMLBase.get_observed(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {
        T,
        N,
        uType,
        C <:
        MOIOptimizationNLPCache,
    }
    return sol.cache.evaluator.f.observed
end
function SciMLBase.get_syms(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {
        T,
        N,
        uType,
        C <:
        MOIOptimizationNLPCache,
    }
    return variable_symbols(sol.cache.evaluator.f)
end
function SciMLBase.get_paramsyms(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {
        T,
        N,
        uType,
        C <:
        MOIOptimizationNLPCache,
    }
    return parameter_symbols(sol.cache.evaluator.f)
end

function MOIOptimizationNLPCache(
        prob::OptimizationProblem,
        opt;
        mtkize = false,
        callback = nothing,
        dual_values = nothing,
        kwargs...
    )
    # `AutoSpecialize`/`AutoDespecialize`: hide the types of the functions (and of `p`)
    f_prob, p_prob = OptimizationBase.despecialize(prob.f, prob.p)
    callback = OptimizationBase.despecialize_callback(prob.f, callback)
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, p_prob) # everything that can be changed via `reinit`

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    adtype = prob.f.adtype
    symbolic = adtype isa ADTypes.AutoSymbolics ||
        (adtype isa ADTypes.AutoSparse && adtype.dense_ad isa ADTypes.AutoSymbolics)
    source_f = symbolic ? generate_exprs(prob) : f_prob
    source_cons_vjp = OptimizationBase._matrixfree_cons_vjp(prob.f, adtype)
    f = _instantiate_nlp_function(source_f, reinit_cache, num_cons, source_cons_vjp)
    T = eltype(prob.u0)
    J, H, cons_H = _nlp_buffers(f, T, length(prob.u0), num_cons)
    lcons = prob.lcons === nothing ? fill(T(-Inf), num_cons) : prob.lcons
    ucons = prob.ucons === nothing ? fill(T(Inf), num_cons) : prob.ucons

    if f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing} && mtkize
        try
            sys = MTK.modelingtoolkitize(prob)
        catch err
            throw(ArgumentError("Automatic symbolic expression generation with ModelingToolkit failed with error: $err.
            Try by setting `mtkize = false` instead if the solver doesn't require symbolic expressions."))
        end
        if !isnothing(prob.p) && !(prob.p isa SciMLBase.NullParameters)
            unames = variable_symbols(sys)
            pnames = parameter_symbols(sys)
            us = [unames[i] => prob.u0[i] for i in 1:length(prob.u0)]
            ps = [pnames[i] => prob.p[i] for i in 1:length(prob.p)]
            sysprob = OptimizationProblem(sys, us, ps)
        else
            unames = variable_symbols(sys)
            us = [unames[i] => prob.u0[i] for i in 1:length(prob.u0)]
            sysprob = OptimizationProblem(sys, us)
        end

        obj_expr = sysprob.f.expr
        cons_expr = sysprob.f.cons_expr
    else
        sys = f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing} ?
            nothing : f.sys
        obj_expr = f.expr
        cons_expr = f.cons_expr
    end

    if sys === nothing
        expr = obj_expr
        _cons_expr = cons_expr
    else
        expr = _cons_expr = nothing
    end

    evaluator = MOIOptimizationNLPEvaluator(
        f,
        source_f,
        source_cons_vjp,
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
        0,
        expr,
        _cons_expr,
        sys !== nothing,
        nothing
    )
    cache = MOIOptimizationNLPCache(evaluator, opt, NamedTuple(kwargs), nothing)
    _set_dual_start!(cache, dual_values)
    return cache
end

function _nlp_buffers(f, T, n, num_cons)
    J = isnothing(f.cons_jac_prototype) ? zeros(T, num_cons, n) : similar(f.cons_jac_prototype, T)
    lagh = !isnothing(f.lag_hess_prototype)
    H = if lagh
        similar(f.lag_hess_prototype, T)
    elseif !isnothing(f.hess_prototype)
        similar(f.hess_prototype, T)
    else
        zeros(T, n, n)
    end
    cons_H = if lagh
        Matrix{T}[zeros(T, 0, 0) for _ in 1:num_cons]
    elseif isnothing(f.cons_hess_prototype)
        Matrix{T}[zeros(T, n, n) for _ in 1:num_cons]
    else
        [similar(f.cons_hess_prototype[i], T) for i in 1:num_cons]
    end
    return J, H, cons_H
end

function _instantiate_nlp_function(f, reinit_cache, num_cons, cons_vjp)
    adtype = f.adtype
    if adtype isa ADTypes.AutoSymbolics ||
            (adtype isa ADTypes.AutoSparse && adtype.dense_ad isa ADTypes.AutoSymbolics)
        return OptimizationBase.instantiate_function(
            f, reinit_cache, adtype, num_cons; g = true, h = true, cons_j = true, cons_h = true)
    end
    return OptimizationBase.instantiate_function(
        f, reinit_cache, adtype, num_cons; g = true, h = true, cons_j = true,
        lag_h = true, cons_vjp)
end

function _set_dual_start!(cache::MOIOptimizationNLPCache, values)
    if values !== nothing
        values isa NamedTuple || throw(ArgumentError("dual_values must be a NamedTuple or nothing"))
        all(key -> key in (:constraints, :lower, :upper, :lower_primal, :upper_primal,
                :reduced_costs, :dual_objective, :constraint_primal), keys(values)) ||
            throw(ArgumentError("unrecognized dual_values field"))
        for (key, n) in ((:constraints, length(cache.lcons)), (:lower, length(cache.u0)),
                (:upper, length(cache.u0)), (:lower_primal, length(cache.u0)),
                (:upper_primal, length(cache.u0)))
            value = get(values, key, nothing)
            value === nothing && continue
            value isa AbstractVector && length(value) == n ||
                throw(ArgumentError("dual_values.$key must be a vector of length $n"))
        end
    end
    cache.dual_start = deepcopy(values)
    return cache
end

function SciMLBase.reinit!(cache::MOIOptimizationNLPCache; p = missing, u0 = missing,
        dual_values = missing, interpret_symbolicmap = true)
    evaluator = cache.evaluator
    old_p, old_u0, old_start = cache.p, cache.u0, cache.dual_start
    try
        if u0 !== missing && !(eltype(u0) <: Pair)
            length(u0) == length(old_u0) || throw(ArgumentError("u0 must keep the original number of variables"))
        end
        dual_values === missing || _set_dual_start!(cache, dual_values)
        invoke(SciMLBase.reinit!, Tuple{SciMLBase.AbstractOptimizationCache}, cache;
            p, u0, interpret_symbolicmap)
        if p !== missing
            f = _instantiate_nlp_function(evaluator.source_f, evaluator.reinit_cache,
                length(evaluator.lcons), evaluator.source_cons_vjp)
            J, H, cons_H = _nlp_buffers(f, eltype(evaluator.u0), length(evaluator.u0), length(evaluator.lcons))
            (f isa typeof(evaluator.f) && J isa typeof(evaluator.J) &&
                H isa typeof(evaluator.H) && cons_H isa typeof(evaluator.cons_H)) ||
                throw(ArgumentError("reinitializing changed the derivative types; create a new cache instead"))
            setfield!(evaluator, :f, f)
            setfield!(evaluator, :J, J)
            setfield!(evaluator, :H, H)
            setfield!(evaluator, :cons_H, cons_H)
        end
    catch
        evaluator.reinit_cache.p = old_p
        evaluator.reinit_cache.u0 = old_u0
        evaluator.dual_start = old_start
        rethrow()
    end
    return cache
end

"""
    get_dual(sol[, index])

Return a snapshot of MOI results as a named tuple. `constraints` contains problem
constraint multipliers in order; `lower` and `upper` contain bound
multipliers in variable order (zero for absent bounds), and `reduced_costs` is their
sum. `dual_objective` and `constraint_primal` are also included. Unavailable values
are `nothing`. `get_dual(sol, i::Integer)` selects a problem constraint,
`get_dual(sol, MOI.VariableIndex(i))` selects a reduced cost, and
`get_dual(sol, ci::MOI.ConstraintIndex)` selects a variable-bound or algebraic multiplier.
Multipliers follow MOI's sign convention. Results remain snapshots when a cache or
optimizer is reused; `sol.original` itself remains the live optimizer.

Pass this tuple as `dual_values` to `init`, `solve`, or `reinit!` to reuse available
dual starts. Alternatively provide a named tuple with `constraints`, `lower`, and/or
`upper` vectors. Optional `lower_primal` and `upper_primal` vectors set MOI bound
constraint primal starts. For affine/quadratic problems, `constraint_primal_start`
supplies primal starts in constraint order. Explicit starts require solver support;
NLPBlock has no constraint-primal-start attribute. Solver-specific options such as Ipopt's
`warm_start_init_point = "yes"` must be supplied separately.
"""
function get_dual end

function MOI.features_available(evaluator::MOIOptimizationNLPEvaluator)
    features = [:Grad, :Hess, :Jac, :JacVec]
    # Assume that if there are constraints and expr then cons_expr exists
    if evaluator.f.expr !== nothing
        push!(features, :ExprGraph)
    end
    return features
end

function MOI.initialize(
        evaluator::MOIOptimizationNLPEvaluator,
        requested_features::Vector{Symbol}
    )
    available_features = MOI.features_available(evaluator)
    for feat in requested_features
        if !(feat in available_features)
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
    if :ExprGraph in requested_features && evaluator.system_exprs_pending
        expr, cons_expr = process_system_exprs(evaluator.f, evaluator.lcons, evaluator.ucons)
        setfield!(evaluator, :obj_expr, expr)
        setfield!(evaluator, :cons_expr, cons_expr)
        setfield!(evaluator, :system_exprs_pending, false)
    end
    return
end

function MOI.eval_objective(evaluator::MOIOptimizationNLPEvaluator, x)
    if evaluator.callback === nothing
        return evaluator.f(x, evaluator.p)
    else
        l = evaluator.f(x, evaluator.p)
        evaluator.iteration += 1
        state = OptimizationBase.OptimizationState(
            iter = evaluator.iteration,
            u = x,
            p = evaluator.p,
            objective = l[1]
        )
        evaluator.callback(state, l)
        return l
    end
end

# Symbolic wrappers already bind their own parameter object, including variadic
# gradient wrappers for which `applicable` cannot distinguish an extra parameter.
function _eval_with_p(evaluator, f, args...)
    ad = evaluator.f.adtype
    symbolic = ad isa ADTypes.AutoSymbolics ||
        (ad isa ADTypes.AutoSparse && ad.dense_ad isa ADTypes.AutoSymbolics)
    return !symbolic && applicable(f, args..., evaluator.p) ?
        f(args..., evaluator.p) : f(args...)
end

function MOI.eval_constraint(evaluator::MOIOptimizationNLPEvaluator, g, x)
    _eval_with_p(evaluator, evaluator.f.cons, g, x)
    return
end

function MOI.eval_objective_gradient(evaluator::MOIOptimizationNLPEvaluator, G, x)
    if evaluator.f.grad === nothing
        error(
            "Use OptimizationFunction to pass the objective gradient or " *
                "automatically generate it with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `grad` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    _eval_with_p(evaluator, evaluator.f.grad, G, x)
    return
end

# This structure assumes the calculation of moiproblem.J is dense.
function MOI.jacobian_structure(evaluator::MOIOptimizationNLPEvaluator)
    if evaluator.J isa SparseMatrixCSC
        rows, cols, _ = findnz(evaluator.J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(evaluator.J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function MOI.eval_constraint_jacobian(evaluator::MOIOptimizationNLPEvaluator, j, x)
    if isempty(j)
        return
    elseif evaluator.f.cons_j === nothing
        error(
            "Use OptimizationFunction to pass the constraints' jacobian or " *
                "automatically generate i with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `cons_j` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    # Get and cache the Jacobian object here once. `evaluator.J` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    J = evaluator.J
    _eval_with_p(evaluator, evaluator.f.cons_j, J, x)
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

function MOI.eval_constraint_jacobian_product(
        evaluator::MOIOptimizationNLPEvaluator, y, x, w
    )
    if evaluator.f.cons_jvp !== nothing
        _eval_with_p(evaluator, evaluator.f.cons_jvp, y, x, w)
    elseif evaluator.f.cons_j !== nothing
        J = evaluator.J
        _eval_with_p(evaluator, evaluator.f.cons_j, J, x)
        mul!(y, J, w)
    else
        error("Thou shalt provide the v'J of the constraint jacobian, not doing so is associated with great misfortune and also no ice cream for you.")
    end
    return nothing
end

function MOI.eval_constraint_jacobian_transpose_product(
        evaluator::MOIOptimizationNLPEvaluator,
        y,
        x,
        w
    )
    if evaluator.f.cons_vjp !== nothing
        _eval_with_p(evaluator, evaluator.f.cons_vjp, y, x, w)
    elseif evaluator.f.cons_j !== nothing
        J = evaluator.J
        _eval_with_p(evaluator, evaluator.f.cons_j, J, x)
        mul!(y, J', w)
    else
        error("Thou shalt provide the v'J of the constraint jacobian, not doing so is associated with great misfortune and also no ice cream for you.")
    end
    return nothing
end

function MOI.hessian_lagrangian_structure(evaluator::MOIOptimizationNLPEvaluator)
    lagh = evaluator.f.lag_h !== nothing
    if evaluator.f.lag_hess_prototype isa SparseMatrixCSC
        rows, cols, _ = findnz(evaluator.f.lag_hess_prototype)
        return Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    end
    sparse_obj = evaluator.H isa SparseMatrixCSC
    sparse_constraints = all(H -> H isa SparseMatrixCSC, evaluator.cons_H)
    if !lagh && !sparse_constraints && any(H -> H isa SparseMatrixCSC, evaluator.cons_H)
        # Some constraint hessians are dense and some are sparse! :(
        error("Mix of sparse and dense constraint hessians are not supported")
    end
    N = length(evaluator.u0)
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
        # Performance OptimizationBase. If both are dense, no need to repeat
    else
        for col in 1:N, row in 1:col

            push!(inds, (row, col))
        end
    end
    return inds
end

function MOI.eval_hessian_lagrangian(
        evaluator::MOIOptimizationNLPEvaluator{T},
        h,
        x,
        σ,
        μ
    ) where {T}
    if evaluator.f.lag_h !== nothing
        _eval_with_p(evaluator, evaluator.f.lag_h, h, x, σ, Vector(μ))
        return
    end
    if evaluator.f.hess === nothing
        error(
            "Use OptimizationFunction to pass the objective hessian or " *
                "automatically generate it with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `hess` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    # Get and cache the Hessian object here once. `evaluator.H` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    H = evaluator.H
    fill!(h, zero(T))
    k = 0
    _eval_with_p(evaluator, evaluator.f.hess, H, x)
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
        if evaluator.f.cons_h === nothing
            error(
                "Use OptimizationFunction to pass the constraints' hessian or " *
                    "automatically generate it with one of the autodiff backends." *
                    "If you are using the ModelingToolkit symbolic interface, pass the `cons_h` kwarg set to `true` in `OptimizationProblem`."
            )
        end
        _eval_with_p(evaluator, evaluator.f.cons_h, evaluator.cons_H, x)
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
                # `nnz_objective` if the objective is sprase, and `0` otherwise.
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

# function MOI.eval_hessian_lagrangian_product(evaluator::MOIOptimizationNLPEvaluator, h, x, v, σ, μ)
#     if evaluator.f.lag_hvp !== nothing
#         evaluator.f.lag_hvp(h, x, v, σ, μ)
#     elseif evaluator.f.lag_h !== nothing
#         H = copy(h)
#         evaluator.f.lag_h(H, x, σ, μ)
#         mul!(h, H, v)
#     else
#         error("The hessian-lagrangian product ")
#     end
# end

function MOI.objective_expr(evaluator::MOIOptimizationNLPEvaluator)
    expr = deepcopy(evaluator.obj_expr)
    repl_getindex!(expr)
    _replace_parameter_indices!(expr, evaluator.p)
    expr = _normalize_integer_powers!(expr)
    _replace_variable_indices!(expr)
    return expr
end

function MOI.constraint_expr(evaluator::MOIOptimizationNLPEvaluator, i)
    cons_expr = evaluator.cons_expr[i]
    cons_expr = if Meta.isexpr(cons_expr, :comparison)
        deepcopy(cons_expr.args[3])
    else
        deepcopy(cons_expr.args[2])
    end
    repl_getindex!(cons_expr)
    _replace_parameter_indices!(cons_expr, evaluator.p)
    cons_expr = _normalize_integer_powers!(cons_expr)
    _replace_variable_indices!(cons_expr)
    lb, ub = Float64(evaluator.lcons[i]), Float64(evaluator.ucons[i])
    if lb == ub
        return Expr(:call, :(==), cons_expr, lb)
    else
        if lb == -Inf
            return Expr(:call, :(<=), cons_expr, ub)
        elseif ub == Inf
            return Expr(:call, :(>=), cons_expr, lb)
        else
            return Expr(:call, :between, cons_expr, lb, ub)
        end
    end
end

function _add_moi_variables!(opt_setup, evaluator::MOIOptimizationNLPEvaluator)
    num_variables = length(evaluator.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
    lower = fill!(Vector{Any}(undef, num_variables), nothing)
    upper = fill!(Vector{Any}(undef, num_variables), nothing)
    if evaluator.lb !== nothing
        eachindex(evaluator.lb) == Base.OneTo(num_variables) ||
            throw(ArgumentError("Expected `cache.lb` to be of the same length as the number of variables."))
    end
    if evaluator.ub !== nothing
        eachindex(evaluator.ub) == Base.OneTo(num_variables) ||
            throw(ArgumentError("Expected `cache.ub` to be of the same length as the number of variables."))
    end

    for i in 1:num_variables
        if evaluator.lb !== nothing && evaluator.lb[i] > -Inf
            lower[i] = MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(Float64(evaluator.lb[i])))
        end
        if evaluator.ub !== nothing && evaluator.ub[i] < Inf
            upper[i] = MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(Float64(evaluator.ub[i])))
        end
        if evaluator.int !== nothing && evaluator.int[i]
            if evaluator.lb !== nothing && evaluator.lb[i] == 0 &&
                    evaluator.ub !== nothing &&
                    evaluator.ub[i] == 1
                MOI.add_constraint(opt_setup, θ[i], MOI.ZeroOne())
            else
                MOI.add_constraint(opt_setup, θ[i], MOI.Integer())
            end
        end
    end

    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        eachindex(evaluator.u0) == Base.OneTo(num_variables) ||
            throw(ArgumentError("Expected `cache.u0` to be of the same length as the number of variables."))
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], evaluator.u0[i])
        end
    end
    if evaluator.dual_start !== nothing
        for (key, indices, attr) in ((:lower, lower, MOI.ConstraintDualStart()),
                (:upper, upper, MOI.ConstraintDualStart()),
                (:lower_primal, lower, MOI.ConstraintPrimalStart()),
                (:upper_primal, upper, MOI.ConstraintPrimalStart()))
            values = get(evaluator.dual_start, key, nothing)
            values === nothing && continue
            for (i, ci) in enumerate(indices)
                ci === nothing && continue
                values[i] === nothing && continue
                MOI.supports(opt_setup, attr, typeof(ci)) || throw(MOI.UnsupportedAttribute(attr))
                MOI.set(opt_setup, attr, ci, values[i])
            end
        end
    end
    return θ, lower, upper
end

# Result attributes do not have a `supports` query; unavailable results must not
# turn an otherwise successful solve into an error.
function _moi_result(opt, attr, indices...)
    try
        return MOI.get(opt, attr, indices...)
    catch e
        (e isa MOI.UnsupportedAttribute || e isa MOI.GetAttributeNotAllowed) || rethrow()
        return nothing
    end
end

function _nlp_results(opt, evaluator, minimizer, θ, lower, upper, has_primal, has_dual)
    constraints = has_dual ? _moi_result(opt, MOI.NLPBlockDual()) : nothing
    constraints === nothing || (constraints = copy(constraints))
    dual_objective = has_dual ? _moi_result(opt, MOI.DualObjectiveValue()) : nothing
    bounds = Dict{Any, Any}()
    function bound_duals(indices)
        values = zeros(Float64, length(indices))
        available = has_dual
        for (i, ci) in enumerate(indices)
            ci === nothing && continue
            value = has_dual ? _moi_result(opt, MOI.ConstraintDual(), ci) : nothing
            bounds[ci] = value
            if value === nothing
                available = false
            else
                values[i] = value
            end
        end
        return available ? values : nothing
    end
    lower_values, upper_values = bound_duals(lower), bound_duals(upper)
    reduced_costs = lower_values === nothing || upper_values === nothing ? nothing : lower_values + upper_values
    constraint_primal = if has_primal
        values = zeros(Float64, length(evaluator.lcons))
        try
            isempty(values) || MOI.eval_constraint(evaluator, values, minimizer)
            values
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
    else
        nothing
    end
    values = (; constraints, lower = lower_values, upper = upper_values,
        reduced_costs, dual_objective, constraint_primal)
    variables = Dict(index => i for (i, index) in enumerate(θ))
    return (; values, bounds, variables)
end

_is_point(status) = status ∉ (MOI.NO_SOLUTION, MOI.INFEASIBILITY_CERTIFICATE,
    MOI.NEARLY_INFEASIBILITY_CERTIFICATE, MOI.REDUCTION_CERTIFICATE,
    MOI.NEARLY_REDUCTION_CERTIFICATE)

function SciMLBase.__solve(cache::MOIOptimizationNLPCache)
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_setup = __map_optimizer_args(
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...
    )

    θ, lower, upper = _add_moi_variables!(opt_setup, cache.evaluator)
    MOI.set(
        opt_setup,
        MOI.ObjectiveSense(),
        cache.evaluator.sense === OptimizationBase.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE
    )

    xor(isnothing(cache.evaluator.lcons), isnothing(cache.evaluator.ucons)) &&
        throw(ArgumentError("Expected `cache.evaluator.lcons` and `cache.evaluator.lcons` to be supplied both or none."))
    if isnothing(cache.evaluator.lcons) && isnothing(cache.evaluator.ucons)
        con_bounds = MOI.NLPBoundsPair[]
    else
        con_bounds = MOI.NLPBoundsPair.(
            Float64.(cache.evaluator.lcons),
            Float64.(cache.evaluator.ucons)
        )
    end
    MOI.set(
        opt_setup,
        MOI.NLPBlock(),
        MOI.NLPBlockData(con_bounds, cache.evaluator, true)
    )
    if cache.dual_start !== nothing
        values = get(cache.dual_start, :constraints, nothing)
        if values !== nothing
            MOI.supports(opt_setup, MOI.NLPBlockDualStart()) ||
                throw(MOI.UnsupportedAttribute(MOI.NLPBlockDualStart()))
            MOI.set(opt_setup, MOI.NLPBlockDualStart(), Float64.(values))
        end
    end

    if cache.evaluator.callback !== nothing
        MOI.set(opt_setup, MOI.Silent(), true)
    end

    MOI.optimize!(opt_setup)
    has_result = MOI.get(opt_setup, MOI.ResultCount()) >= 1
    has_primal = has_result && _is_point(MOI.get(opt_setup, MOI.PrimalStatus()))
    has_dual = has_result && _is_point(MOI.get(opt_setup, MOI.DualStatus()))
    if has_primal
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
    else
        minimizer = fill(NaN, length(θ))
        minimum = NaN
    end
    opt_ret = __moi_status_to_ReturnCode(MOI.get(opt_setup, MOI.TerminationStatus()))

    results = _nlp_results(opt_setup, cache.evaluator, minimizer, θ, lower, upper, has_primal, has_dual)

    # check if the solver supports BarrierIterations
    iterations = try
        MOI.get(opt_setup, MOI.BarrierIterations())
    catch e
        if !(e isa MOI.GetAttributeNotAllowed)
            rethrow(e)
        end
        0
    end

    stats = OptimizationBase.OptimizationStats(;
        time = MOI.get(opt_setup, MOI.SolveTimeSec()),
        iterations
    )
    return SciMLBase.build_solution(
        MOIOptimizationNLPCache(cache.evaluator, cache.opt, cache.solver_args, results),
        cache.opt,
        minimizer,
        minimum;
        original = opt_setup,
        retcode = opt_ret,
        stats = stats
    )
end
