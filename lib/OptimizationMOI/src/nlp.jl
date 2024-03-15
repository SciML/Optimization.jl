mutable struct MOIOptimizationNLPEvaluator{T, F <: OptimizationFunction, RC, LB, UB,
    I,
    JT <: DenseOrSparse{T}, HT <: DenseOrSparse{T},
    CHT <: DenseOrSparse{T}, S, CB} <:
               MOI.AbstractNLPEvaluator
    f::F
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
    cons_expr::Union{Vector{Expr}, Nothing}
end

function Base.getproperty(evaluator::MOIOptimizationNLPEvaluator, x::Symbol)
    if x in fieldnames(OptimizationBase.ReInitCache)
        return getfield(evaluator.reinit_cache, x)
    end
    return getfield(evaluator, x)
end

struct MOIOptimizationNLPCache{E <: MOIOptimizationNLPEvaluator, O} <:
       SciMLBase.AbstractOptimizationCache
    evaluator::E
    opt::O
    solver_args::NamedTuple
end

function Base.getproperty(cache::MOIOptimizationNLPCache{E}, name::Symbol) where {E}
    if name in fieldnames(E)
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
    return setfield!(cache, name, x)
end

function SciMLBase.get_p(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {T, N,
        uType,
        C <:
        MOIOptimizationNLPCache
}
    sol.cache.evaluator.p
end
function SciMLBase.get_observed(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {
        T,
        N,
        uType,
        C <:
        MOIOptimizationNLPCache
}
    sol.cache.evaluator.f.observed
end
function SciMLBase.get_syms(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {T,
        N,
        uType,
        C <:
        MOIOptimizationNLPCache
}
    variable_symbols(sol.cache.evaluator.f)
end
function SciMLBase.get_paramsyms(sol::SciMLBase.OptimizationSolution{
        T,
        N,
        uType,
        C
}) where {
        T,
        N,
        uType,
        C <:
        MOIOptimizationNLPCache
}
    parameter_symbols(sol.cache.evaluator.f)
end

function MOIOptimizationNLPCache(prob::OptimizationProblem,
        opt;
        mtkize = false,
        callback = nothing,
        kwargs...)
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype, num_cons)
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
        expr_map = get_expr_map(sys)
        expr = convert_to_expr(obj_expr, expr_map; expand_expr = false)
        expr = repl_getindex!(expr)
        cons = MTK.constraints(sys)
        _cons_expr = Vector{Expr}(undef, length(cons))
        for i in eachindex(cons)
            _cons_expr[i] = repl_getindex!(convert_to_expr(cons_expr[i],
                expr_map;
                expand_expr = false))
        end
    end

    evaluator = MOIOptimizationNLPEvaluator(f,
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
        _cons_expr)
    return MOIOptimizationNLPCache(evaluator, opt, NamedTuple(kwargs))
end

function MOI.features_available(evaluator::MOIOptimizationNLPEvaluator)
    features = [:Grad, :Hess, :Jac]
    # Assume that if there are constraints and expr then cons_expr exists
    if evaluator.f.expr !== nothing
        push!(features, :ExprGraph)
    end
    return features
end

function MOI.initialize(evaluator::MOIOptimizationNLPEvaluator,
        requested_features::Vector{Symbol})
    available_features = MOI.features_available(evaluator)
    for feat in requested_features
        if !(feat in available_features)
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
    return
end

function MOI.eval_objective(evaluator::MOIOptimizationNLPEvaluator, x)
    if evaluator.callback === nothing
        return evaluator.f(x, evaluator.p)
    else
        l = evaluator.f(x, evaluator.p)
        evaluator.iteration += 1
        state = Optimization.OptimizationState(iter = evaluator.iteration,
            u = x,
            objective = l[1])
        evaluator.callback(state, l)
        return l
    end
end

function MOI.eval_constraint(evaluator::MOIOptimizationNLPEvaluator, g, x)
    evaluator.f.cons(g, x)
    return
end

function MOI.eval_objective_gradient(evaluator::MOIOptimizationNLPEvaluator, G, x)
    if evaluator.f.grad === nothing
        error("Use OptimizationFunction to pass the objective gradient or " *
              "automatically generate it with one of the autodiff backends." *
              "If you are using the ModelingToolkit symbolic interface, pass the `grad` kwarg set to `true` in `OptimizationProblem`.")
    end
    evaluator.f.grad(G, x)
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
        error("Use OptimizationFunction to pass the constraints' jacobian or " *
              "automatically generate i with one of the autodiff backends." *
              "If you are using the ModelingToolkit symbolic interface, pass the `cons_j` kwarg set to `true` in `OptimizationProblem`.")
    end
    # Get and cache the Jacobian object here once. `evaluator.J` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    J = evaluator.J
    evaluator.f.cons_j(J, x)
    if J isa SparseMatrixCSC
        nnz = nonzeros(J)
        @assert length(j) == length(nnz)
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        for i in eachindex(j)
            j[i] = J[i]
        end
    end
    return
end

function MOI.hessian_lagrangian_structure(evaluator::MOIOptimizationNLPEvaluator)
    lagh = evaluator.f.lag_h !== nothing
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
        # Performance optimization. If both are dense, no need to repeat
    else
        for col in 1:N, row in 1:col
            push!(inds, (row, col))
        end
    end
    return inds
end

function MOI.eval_hessian_lagrangian(evaluator::MOIOptimizationNLPEvaluator{T},
        h,
        x,
        σ,
        μ) where {T}
    if evaluator.f.lag_h !== nothing
        return evaluator.f.lag_h(h, x, σ, μ)
    end
    if evaluator.f.hess === nothing
        error("Use OptimizationFunction to pass the objective hessian or " *
              "automatically generate it with one of the autodiff backends." *
              "If you are using the ModelingToolkit symbolic interface, pass the `hess` kwarg set to `true` in `OptimizationProblem`.")
    end
    # Get and cache the Hessian object here once. `evaluator.H` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    H = evaluator.H
    fill!(h, zero(T))
    k = 0
    evaluator.f.hess(H, x)
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
            error("Use OptimizationFunction to pass the constraints' hessian or " *
                  "automatically generate it with one of the autodiff backends." *
                  "If you are using the ModelingToolkit symbolic interface, pass the `cons_h` kwarg set to `true` in `OptimizationProblem`.")
        end
        evaluator.f.cons_h(evaluator.cons_H, x)
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

function MOI.objective_expr(evaluator::MOIOptimizationNLPEvaluator)
    expr = deepcopy(evaluator.obj_expr)
    repl_getindex!(expr)
    _replace_parameter_indices!(expr, evaluator.p)
    _replace_variable_indices!(expr)
    return expr
end

function MOI.constraint_expr(evaluator::MOIOptimizationNLPEvaluator, i)
    # expr has the form f(x,p) == 0 or f(x,p) <= 0
    cons_expr = deepcopy(evaluator.cons_expr[i].args[2])
    repl_getindex!(cons_expr)
    _replace_parameter_indices!(cons_expr, evaluator.p)
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
            MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(evaluator.lb[i]))
        end
        if evaluator.ub !== nothing && evaluator.ub[i] < Inf
            MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(evaluator.ub[i]))
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
    return θ
end

function SciMLBase.__solve(cache::MOIOptimizationNLPCache)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_setup = __map_optimizer_args(cache,
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    θ = _add_moi_variables!(opt_setup, cache.evaluator)
    MOI.set(opt_setup,
        MOI.ObjectiveSense(),
        cache.evaluator.sense === Optimization.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE)
    xor(isnothing(cache.evaluator.lcons), isnothing(cache.evaluator.ucons)) &&
        throw(ArgumentError("Expected `cache.evaluator.lcons` and `cache.evaluator.lcons` to be supplied both or none."))
    if isnothing(cache.evaluator.lcons) && isnothing(cache.evaluator.ucons)
        con_bounds = MOI.NLPBoundsPair[]
    else
        con_bounds = MOI.NLPBoundsPair.(Float64.(cache.evaluator.lcons),
            Float64.(cache.evaluator.ucons))
    end
    MOI.set(opt_setup,
        MOI.NLPBlock(),
        MOI.NLPBlockData(con_bounds, cache.evaluator, true))

    if cache.evaluator.callback !== nothing
        MOI.set(opt_setup, MOI.Silent(), true)
    end

    MOI.optimize!(opt_setup)
    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = __moi_status_to_ReturnCode(MOI.get(opt_setup, MOI.TerminationStatus()))
    else
        minimizer = fill(NaN, length(θ))
        minimum = NaN
        opt_ret = SciMLBase.ReturnCode.Default
    end
    stats = Optimization.OptimizationStats()
    return SciMLBase.build_solution(cache,
        cache.opt,
        minimizer,
        minimum;
        original = opt_setup,
        retcode = opt_ret,
        stats = stats)
end
