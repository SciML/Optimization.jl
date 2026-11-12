struct MOIOptimizationCache{
        F <: OptimizationFunction, RC, LB, UB, I, S, EX,
        CEX, O, R,
    } <: SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    int::I
    sense::S
    expr::EX
    cons_expr::CEX
    opt::O
    solver_args::NamedTuple
    dual_start::Ref{Union{Nothing, NamedTuple}}
    dual_results::R
end

function MOIOptimizationCache(prob::OptimizationProblem, opt; dual_values = nothing, kwargs...)
    f = prob.f
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p)
    if isnothing(f.sys)
        if f.adtype isa OptimizationBase.AutoSymbolics
            num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
            f = generate_exprs(prob)
            f = OptimizationBase.instantiate_function(
                f,
                reinit_cache,
                prob.f.adtype,
                num_cons
            )
        else
            throw(ArgumentError("Expected an `OptimizationProblem` that was setup via an `OptimizationSystem`, or AutoSymbolics ad choice"))
        end
    end

    # TODO: check if the problem is at most bilinear, i.e. affine and or quadratic terms in two variables
    if f.sys !== nothing
        expr, cons_expr = process_system_exprs(prob, f)
        f = remake(f; expr, cons_expr)
    end
    cache = MOIOptimizationCache(
        f,
        reinit_cache,
        prob.lb,
        prob.ub,
        prob.int,
        prob.sense,
        f.expr,
        f.cons_expr,
        opt,
        NamedTuple(kwargs),
        Ref{Union{Nothing, NamedTuple}}(nothing),
        nothing
    )
    _set_dual_start!(cache, dual_values)
    return cache
end

const MOIResultCache = Union{MOIOptimizationNLPCache, MOIOptimizationCache}

function get_dual(sol::SciMLBase.OptimizationSolution{T, N, U, C}) where {T, N, U, C <: MOIResultCache}
    return sol.cache.dual_results.values
end
function get_dual(sol::SciMLBase.OptimizationSolution{T, N, U, C}, i::Integer) where {T, N, U, C <: MOIResultCache}
    values = get_dual(sol).constraints
    return values === nothing ? nothing : values[i]
end
function get_dual(sol::SciMLBase.OptimizationSolution{T, N, U, C}, i::MOI.VariableIndex) where {T, N, U, C <: MOIResultCache}
    values = get_dual(sol).reduced_costs
    index = sol.cache.dual_results.variables[i]
    return values === nothing ? nothing : values[index]
end
function get_dual(sol::SciMLBase.OptimizationSolution{T, N, U, C}, i::MOI.ConstraintIndex) where {T, N, U, C <: MOIResultCache}
    return sol.cache.dual_results.bounds[i]
end

function _set_dual_start!(cache::MOIOptimizationCache, values)
    if values !== nothing
        values isa NamedTuple || throw(ArgumentError("dual_values must be a NamedTuple or nothing"))
        all(
            key -> key in (
                :constraints, :lower, :upper, :lower_primal, :upper_primal,
                :constraint_primal_start, :reduced_costs, :dual_objective, :constraint_primal,
            ), keys(values)
        ) ||
            throw(ArgumentError("unrecognized dual_values field"))
        ncons = cache.cons_expr === nothing ? 0 : length(cache.cons_expr)
        for (key, n) in (
                (:constraints, ncons), (:constraint_primal_start, ncons),
                (:lower, length(cache.u0)), (:upper, length(cache.u0)),
                (:lower_primal, length(cache.u0)), (:upper_primal, length(cache.u0)),
            )
            value = get(values, key, nothing)
            value === nothing && continue
            value isa AbstractVector && length(value) == n ||
                throw(ArgumentError("dual_values.$key must be a vector of length $n"))
        end
    end
    cache.dual_start[] = deepcopy(values)
    return cache
end

function SciMLBase.reinit!(
        cache::MOIOptimizationCache; p = missing, u0 = missing,
        dual_values = missing, interpret_symbolicmap = true
    )
    old_p, old_u0, old_start = cache.p, cache.u0, cache.dual_start[]
    try
        if u0 !== missing && !(eltype(u0) <: Pair)
            length(u0) == length(old_u0) || throw(ArgumentError("u0 must keep the original number of variables"))
        end
        dual_values === missing || _set_dual_start!(cache, dual_values)
        invoke(
            SciMLBase.reinit!, Tuple{SciMLBase.AbstractOptimizationCache}, cache;
            p, u0, interpret_symbolicmap
        )
    catch
        cache.reinit_cache.p = old_p
        cache.reinit_cache.u0 = old_u0
        cache.dual_start[] = old_start
        rethrow()
    end
    return cache
end

struct MalformedExprException <: Exception
    msg::String
end
function Base.showerror(io::IO, e::MalformedExprException)
    return print(io, "MalformedExprException: ", e.msg)
end

function _add_moi_variables!(opt_setup, cache::MOIOptimizationCache)
    num_variables = length(cache.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
    lower = fill!(Vector{Any}(undef, num_variables), nothing)
    upper = fill!(Vector{Any}(undef, num_variables), nothing)
    if cache.lb !== nothing
        eachindex(cache.lb) == Base.OneTo(num_variables) ||
            throw(ArgumentError("Expected `cache.lb` to be of the same length as the number of variables."))
    end
    if cache.ub !== nothing
        eachindex(cache.ub) == Base.OneTo(num_variables) ||
            throw(ArgumentError("Expected `cache.ub` to be of the same length as the number of variables."))
    end

    for i in 1:num_variables
        if cache.lb !== nothing && cache.lb[i] > -Inf
            lower[i] = MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(Float64(cache.lb[i])))
        end
        if cache.ub !== nothing && cache.ub[i] < Inf
            upper[i] = MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(Float64(cache.ub[i])))
        end
        if cache.int !== nothing && cache.int[i]
            if cache.lb !== nothing && cache.lb[i] == 0 && cache.ub !== nothing &&
                    cache.ub[i] == 1
                MOI.add_constraint(opt_setup, θ[i], MOI.ZeroOne())
            else
                MOI.add_constraint(opt_setup, θ[i], MOI.Integer())
            end
        end
    end

    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        eachindex(cache.u0) == Base.OneTo(num_variables) ||
            throw(ArgumentError("Expected `cache.u0` to be of the same length as the number of variables."))
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], Float64(cache.u0[i]))
        end
    end
    starts = cache.dual_start[]
    if starts !== nothing
        for (key, indices, attr) in (
                (:lower, lower, MOI.ConstraintDualStart()),
                (:upper, upper, MOI.ConstraintDualStart()),
                (:lower_primal, lower, MOI.ConstraintPrimalStart()),
                (:upper_primal, upper, MOI.ConstraintPrimalStart()),
            )
            _set_constraint_starts!(opt_setup, indices, get(starts, key, nothing), attr)
        end
    end
    return θ, lower, upper
end

function _set_constraint_starts!(opt, indices, values, attr, offsets = nothing)
    values === nothing && return
    for (i, ci) in enumerate(indices)
        (ci === nothing || values[i] === nothing) && continue
        MOI.supports(opt, attr, typeof(ci)) || throw(MOI.UnsupportedAttribute(attr))
        value = offsets === nothing ? values[i] : values[i] - offsets[i]
        MOI.set(opt, attr, ci, value)
    end
    return
end

function _constraint_parts(expr::Expr, p)
    if expr.head === :comparison && length(expr.args) == 5 && expr.args[2] === :(<=) && expr.args[4] === :(<=)
        lower, lhs, upper = expr.args[1], expr.args[3], expr.args[5]
    elseif expr.head === :call && length(expr.args) == 3
        op, lhs, rhs = expr.args
        if op in (:(==), :(=))
            lower = upper = rhs
        elseif op === :(<=)
            lower, upper = -Inf, rhs
        elseif op === :(>=)
            lower, upper = rhs, Inf
        else
            throw(MalformedExprException("$expr"))
        end
    else
        throw(MalformedExprException("$expr"))
    end
    function bound(x)
        value = fixpoint_simplify_and_expand!(_replace_parameter_indices!(repl_getindex!(deepcopy(x)), p))
        # Symbolification of a comparison can turn its leading numeric bound into a Symbol.
        value isa Symbol && (value = tryparse(Float64, String(value)))
        value isa Real || throw(ArgumentError("constraint bound must evaluate to a real number"))
        return Float64(value)
    end
    lhs = _replace_parameter_indices!(repl_getindex!(deepcopy(lhs)), p)
    return fixpoint_simplify_and_expand!(lhs), bound(lower), bound(upper)
end

function _affine_results(opt, θ, lower, upper, rows, offsets, has_primal, has_dual)
    bounds = Dict{Any, Any}()
    function duals(indices)
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
    constraints, lower_values, upper_values = duals(rows), duals(lower), duals(upper)
    reduced_costs = lower_values === nothing || upper_values === nothing ? nothing : lower_values + upper_values
    constraint_primal = if has_primal
        values = [_moi_result(opt, MOI.ConstraintPrimal(), ci) for ci in rows]
        any(isnothing, values) ? nothing : values .+ offsets
    else
        nothing
    end
    dual_objective = has_dual ? _moi_result(opt, MOI.DualObjectiveValue()) : nothing
    values = (;
        constraints, lower = lower_values, upper = upper_values,
        reduced_costs, dual_objective, constraint_primal,
    )
    variables = Dict(index => i for (i, index) in enumerate(θ))
    return (; values, bounds, variables)
end

function SciMLBase.__solve(cache::MOIOptimizationCache)
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

    Theta, lower, upper = _add_moi_variables!(opt_setup, cache)
    MOI.set(
        opt_setup,
        MOI.ObjectiveSense(),
        cache.sense === OptimizationBase.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE
    )

    rows = Any[]
    offsets = Float64[]
    if !isnothing(cache.cons_expr)
        for cons_expr in cache.cons_expr
            expr, lb, ub = _constraint_parts(cons_expr, cache.p)
            func, c = try
                get_moi_function(expr)
            catch e
                if e isa MalformedExprException
                    rethrow(e)
                    #rethrow(MalformedExprException("$expr"))
                else
                    rethrow(e)
                end
            end
            func = MOI.Utilities.map_indices(i -> Theta[i.value], func)
            set = if lb == ub
                MOI.EqualTo(lb - c)
            elseif lb == -Inf
                MOI.LessThan(ub - c)
            elseif ub == Inf
                MOI.GreaterThan(lb - c)
            else
                MOI.Interval(lb - c, ub - c)
            end
            push!(rows, MOI.add_constraint(opt_setup, func, set))
            push!(offsets, c)
        end
    end

    starts = cache.dual_start[]
    if starts !== nothing
        _set_constraint_starts!(opt_setup, rows, get(starts, :constraints, nothing), MOI.ConstraintDualStart())
        _set_constraint_starts!(
            opt_setup, rows, get(starts, :constraint_primal_start, nothing),
            MOI.ConstraintPrimalStart(), offsets
        )
    end

    # objective
    expr = _replace_parameter_indices!(repl_getindex!(deepcopy(cache.expr)), cache.p)
    expr = fixpoint_simplify_and_expand!(expr)
    func, c = try
        get_moi_function(expr)
    catch e
        if e isa MalformedExprException
            rethrow(MalformedExprException("$expr"))
        else
            rethrow(e)
        end
    end
    func = MOI.Utilities.map_indices(i -> Theta[i.value], func)
    func = MOI.Utilities.operate(+, Float64, func, c)
    MOI.set(opt_setup, MOI.ObjectiveFunction{typeof(func)}(), func)

    MOI.optimize!(opt_setup)
    has_result = MOI.get(opt_setup, MOI.ResultCount()) >= 1
    has_primal = has_result && _is_point(MOI.get(opt_setup, MOI.PrimalStatus()))
    has_dual = has_result && _is_point(MOI.get(opt_setup, MOI.DualStatus()))
    if has_primal
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), Theta)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
    else
        minimizer = fill(NaN, length(Theta))
        minimum = NaN
    end
    opt_ret = __moi_status_to_ReturnCode(MOI.get(opt_setup, MOI.TerminationStatus()))
    results = _affine_results(opt_setup, Theta, lower, upper, rows, offsets, has_primal, has_dual)
    stats = OptimizationBase.OptimizationStats()
    return SciMLBase.build_solution(
        MOIOptimizationCache(
            cache.f, cache.reinit_cache, cache.lb, cache.ub, cache.int,
            cache.sense, cache.expr, cache.cons_expr, cache.opt, cache.solver_args,
            cache.dual_start, results
        ),
        cache.opt,
        minimizer,
        minimum;
        original = opt_setup,
        retcode = opt_ret,
        stats = stats
    )
end

function get_moi_function(expr)
    affine_terms = MOI.ScalarAffineTerm{Float64}[]
    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    constant = Ref(0.0)
    collect_moi_terms!(
        expr,
        affine_terms,
        quadratic_terms,
        constant
    )
    func = if isempty(quadratic_terms)
        MOI.ScalarAffineFunction(affine_terms, 0.0)
    else
        MOI.ScalarQuadraticFunction(quadratic_terms, affine_terms, 0.0)
    end
    return func, constant[]
end

simplify_and_expand!(expr::T) where {T} = expr
simplify_and_expand!(expr::Rational) = Float64(expr)

"""
Simplify and expands the given expression. All computations on numbers are evaluated and simplified.
After successive application the resulting expression should only contain terms of the form `:(a * x[i])` or `:(a * x[i] * x[j])`.
Also mutates the given expression in-place, however incorrectly!
"""
function simplify_and_expand!(expr::Expr) # looks awful but this is actually much faster than `Metatheory.jl`
    if expr.head == :call && length(expr.args) == 3
        if expr.args[1] == :(*) && expr.args[2] isa Number && expr.args[3] isa Number # a::Number * b::Number => a * b
            return expr.args[2] * expr.args[3]
        elseif expr.args[1] == :(+) && expr.args[2] isa Number && expr.args[3] isa Number # a::Number + b::Number => a + b
            return expr.args[2] + expr.args[3]
        elseif expr.args[1] == :(^) && expr.args[2] isa Number && expr.args[3] isa Number  # a::Number^b::Number => a^b
            return expr.args[2]^expr.args[3]
        elseif expr.args[1] == :(/) && expr.args[2] isa Number && expr.args[3] isa Number  # a::Number/b::Number => a/b
            return expr.args[2] / expr.args[3]
        elseif expr.args[1] == :(//) && expr.args[2] isa Number && expr.args[3] isa Number  # a::Number//b::Number => a/b
            return expr.args[2] / expr.args[3]
        elseif expr.args[1] == :(*) && isa(expr.args[2], Real) && isone(expr.args[2]) # 1 * x => x
            return expr.args[3]
        elseif expr.args[1] == :(*) && isa(expr.args[3], Real) && isone(expr.args[3]) # x * 1 => x
            return expr.args[2]
        elseif expr.args[1] == :(*) && isa(expr.args[2], Real) && iszero(expr.args[2]) # 0 * x => 0
            return 0
        elseif expr.args[1] == :(*) && isa(expr.args[3], Real) && iszero(expr.args[3]) # x * 0 => x
            return 0
        elseif expr.args[1] == :(+) && isa(expr.args[2], Real) && iszero(expr.args[2]) # 0 + x => x
            return expr.args[3]
        elseif expr.args[1] == :(+) && isa(expr.args[3], Real) && iszero(expr.args[3]) # x + 0 => x
            return expr.args[2]
        elseif expr.args[1] == :(/) && isa(expr.args[3], Real) && isone(expr.args[3]) # x / 1 => x
            return expr.args[2]
        elseif expr.args[1] == :// && isa(expr.args[3], Real) && isone(expr.args[3]) # x // 1 => x
            return expr.args[2]
        elseif expr.args[1] == :(^) && isa(expr.args[3], Int) && expr.args[3] == 2 # x^2 => x * x
            if isa(expr.args[2], Expr) && expr.args[2].head == :call &&
                    expr.args[2].args[1] == :+ # (x + y)^2 => (x^2 + ((2 * (x * y)) + y^2))
                return Expr(
                    :call, :+,
                    Expr(:call, :^, expr.args[2].args[2], 2),
                    Expr(
                        :call, :+,
                        Expr(
                            :call, :*, 2,
                            Expr(
                                :call, :*, expr.args[2].args[2],
                                expr.args[2].args[3]
                            )
                        ),
                        Expr(:call, :^, expr.args[2].args[3], 2)
                    )
                )
            else
                return Expr(:call, :*, expr.args[2], expr.args[2]) # x^2 => x * x
            end
        elseif expr.args[1] == :(^) && isa(expr.args[3], Int) && expr.args[3] > 2 # x^n => x * x^(n-1)
            return Expr(
                :call, :*, Expr(:call, :^, expr.args[2], expr.args[3] - 1),
                expr.args[2]
            )
        elseif expr.args[1] == :(*) && isa(expr.args[3], Number) # x * a::Number => a * x
            return Expr(:call, :*, expr.args[3], expr.args[2])
        elseif expr.args[1] == :(+) && isa(expr.args[3], Number) # x + a::Number => a + x
            return Expr(:call, :+, expr.args[3], expr.args[2])
        elseif expr.args[1] == :(*) && isa(expr.args[3], Expr) &&
                expr.args[3].head == :call && expr.args[3].args[1] == :(+) # (x * (y + z)) => ((x * y) + (x * z))
            return Expr(
                :call, :+,
                Expr(:call, :*, expr.args[2], expr.args[3].args[2]),
                Expr(:call, :*, expr.args[2], expr.args[3].args[3])
            )
        elseif expr.args[1] == :(*) && isa(expr.args[2], Expr) &&
                expr.args[2].head == :call && expr.args[2].args[1] == :(+) # ((y + z) * x) => ((x * y) + (x * z))
            return Expr(
                :call, :+,
                Expr(:call, :*, expr.args[3], expr.args[2].args[2]),
                Expr(:call, :*, expr.args[3], expr.args[2].args[3])
            )
        elseif expr.args[1] == :(*) && expr.args[2] isa Number && isa(expr.args[3], Expr) &&
                expr.args[3].head == :call && expr.args[3].args[1] == :(*) &&
                expr.args[3].args[2] isa Number # a::Number * (b::Number * c) => (a * b) * c
            return Expr(
                :call, :*, expr.args[2] * expr.args[3].args[2],
                expr.args[3].args[3]
            )
        elseif expr.args[1] == :(+) && isa(expr.args[3], Expr) &&
                isa(expr.args[2], Number) &&
                expr.args[3].head == :call && expr.args[3].args[1] == :(+) &&
                isa(expr.args[3].args[2], Number) # a::Number + (b::Number + x)  => (a+b) + x
            return Expr(
                :call, :+, expr.args[2] + expr.args[3].args[2],
                expr.args[3].args[3]
            )
        elseif expr.args[1] == :(*) && isa(expr.args[3], Expr) &&
                expr.args[3].head == :call && expr.args[3].args[1] == :(*) &&
                isa(expr.args[3].args[2], Number) # x * (a::Number * y) => a * (x * y)
            return Expr(
                :call, :*, expr.args[3].args[2],
                Expr(:call, :*, expr.args[2], expr.args[3].args[3])
            )
        elseif expr.args[1] == :(*) && isa(expr.args[2], Expr) &&
                expr.args[2].head == :call && expr.args[2].args[1] == :(*) &&
                isa(expr.args[2].args[2], Number) # (a::Number * x) * y => a * (x * y)
            return Expr(
                :call, :*, expr.args[2].args[2],
                Expr(:call, :*, expr.args[2].args[3], expr.args[3])
            )
        end
    elseif expr.head == :call && all(isa.(expr.args[2:end], Number)) # func(a::Number...)
        return eval(expr)
    end
    for i in 1:length(expr.args)
        expr.args[i] = simplify_and_expand!(expr.args[i])
    end
    return expr
end

"""
Simplifies the given expression until a fixed-point is reached and the expression no longer changes.
Will not terminate if a cycle occurs!
"""
function fixpoint_simplify_and_expand!(expr; iter_max = typemax(Int) - 1)
    expr = _normalize_integer_powers!(expr)
    i = 0
    iter_max >= 0 || throw(ArgumentError("Expected `iter_max` to be positive."))
    while i <= iter_max
        expr_old = deepcopy(expr)
        expr = simplify_and_expand!(expr)
        expr_old == expr && break # might not return if a cycle is reached
        i += 1
    end
    return expr
end

function collect_moi_terms!(expr::Real, affine_terms, quadratic_terms, constant)
    (isnan(expr) || isinf(expr)) && throw(MalformedExprException("$expr"))
    return constant[] += expr
end

function collect_moi_terms!(expr::Expr, affine_terms, quadratic_terms, constant)
    if expr.head == :call
        length(expr.args) == 3 || throw(MalformedExprException("$expr"))
        if expr.args[1] == :(+)
            for i in 2:length(expr.args)
                collect_moi_terms!(expr.args[i], affine_terms, quadratic_terms, constant)
            end
        elseif expr.args[1] == :(*)
            if isa(expr.args[2], Number) && isa(expr.args[3], Expr)
                if expr.args[3].head == :call && expr.args[3].args[1] == :(*) # a::Number * (x[i] * x[j])
                    x1 = _get_variable_index_from_expr(expr.args[3].args[2])
                    x2 = _get_variable_index_from_expr(expr.args[3].args[3])
                    factor = x1 == x2 ? 2.0 : 1.0
                    c = factor * Float64(expr.args[2])
                    (isnan(c) || isinf(c)) && throw(MalformedExprException("$expr"))
                    push!(quadratic_terms, MOI.ScalarQuadraticTerm(c, x1, x2))
                elseif expr.args[3].head == :ref # a::Number * x[i]
                    x = _get_variable_index_from_expr(expr.args[3])
                    c = Float64(expr.args[2])
                    (isnan(c) || isinf(c)) && throw(MalformedExprException("$expr"))
                    push!(affine_terms, MOI.ScalarAffineTerm(c, x))
                else
                    throw(MalformedExprException("$expr"))
                end
            elseif isa(expr.args[2], Number) && isa(expr.args[3], Number) # a::Number * b::Number
                c = expr.args[2] * expr.args[3]
                (isnan(c) || isinf(c)) && throw(MalformedExprException("$expr"))
                constant[] += c
            elseif isa(expr.args[2], Expr) && isa(expr.args[3], Expr)
                if expr.args[2].head == :call && expr.args[2].args[1] == :(*) &&
                        isa(expr.args[2].args[2], Number) # (a::Number * x[i]) * x[j]
                    x1 = _get_variable_index_from_expr(expr.args[2].args[3])
                    x2 = _get_variable_index_from_expr(expr.args[3])
                    factor = x1 == x2 ? 2.0 : 1.0
                    c = factor * Float64(expr.args[2].args[2])
                    (isnan(c) || isinf(c)) && throw(MalformedExprException("$expr"))
                    push!(quadratic_terms, MOI.ScalarQuadraticTerm(c, x1, x2))
                else # x[i] * x[j]
                    x1 = _get_variable_index_from_expr(expr.args[2])
                    x2 = _get_variable_index_from_expr(expr.args[3])
                    factor = x1 == x2 ? 2.0 : 1.0
                    push!(
                        quadratic_terms,
                        MOI.ScalarQuadraticTerm(
                            factor,
                            x1, x2
                        )
                    )
                end
            else
                throw(MalformedExprException("$expr"))
            end
        elseif expr.args[1] == :^
            throw(MalformedExprException("$expr"))
        end
    elseif expr.head == :ref # x[i]
        expr.args[1] == :x || throw(MalformedExprException("$expr"))
        push!(affine_terms, MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(expr.args[2])))
    else
        throw(MalformedExprException("$expr"))
    end

    return
end
