struct MOIOptimizationCache{F <: OptimizationFunction, RC, LB, UB, I, S, EX,
    CEX, O} <: SciMLBase.AbstractOptimizationCache
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
end

function MOIOptimizationCache(prob::OptimizationProblem, opt; kwargs...)
    f = prob.f
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p)
    if isnothing(f.sys)
        if f.adtype isa OptimizationBase.AutoSymbolics
            @info "AutoSymbolics"
            num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
            @info "generate_exprs"
            f = generate_exprs(prob)
            @info "instantiate_fn"
            f = OptimizationBase.instantiate_function(f,
                reinit_cache,
                prob.f.adtype,
                num_cons)
        else
            throw(ArgumentError("Expected an `OptimizationProblem` that was setup via an `OptimizationSystem`, or AutoSymbolics ad choice"))
        end
    end

    # TODO: check if the problem is at most bilinear, i.e. affine and or quadratic terms in two variables
    if f.sys !== nothing
        @info "process_system_exprs"
        expr, cons_expr = process_system_exprs(prob, f)
        f = remake(f; expr, cons_expr)
    end
    return MOIOptimizationCache(f,
        reinit_cache,
        prob.lb,
        prob.ub,
        prob.int,
        prob.sense,
        f.expr,
        f.cons_expr,
        opt,
        NamedTuple(kwargs))
end

struct MalformedExprException <: Exception
    msg::String
end
function Base.showerror(io::IO, e::MalformedExprException)
    print(io, "MalformedExprException: ", e.msg)
end

function _add_moi_variables!(opt_setup, cache::MOIOptimizationCache)
    num_variables = length(cache.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
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
            MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(Float64(cache.lb[i])))
        end
        if cache.ub !== nothing && cache.ub[i] < Inf
            MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(Float64(cache.ub[i])))
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
    return θ
end

function SciMLBase.__solve(cache::MOIOptimizationCache)
    maxiters = OptimizationBase._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_setup = __map_optimizer_args(cache,
        cache.opt;
        abstol = cache.solver_args.abstol,
        reltol = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime = maxtime,
        cache.solver_args...)

    Theta = _add_moi_variables!(opt_setup, cache)
    MOI.set(opt_setup,
        MOI.ObjectiveSense(),
        cache.sense === OptimizationBase.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE)

    if !isnothing(cache.cons_expr)
        for cons_expr in cache.cons_expr
            expr = _replace_parameter_indices!(deepcopy(cons_expr.args[2]), # f(x) == 0 or f(x) <= 0
                cache.p)
            expr = fixpoint_simplify_and_expand!(expr)
            func, c = try
                get_moi_function(expr) # find: f(x) + c == 0 or f(x) + c <= 0
            catch e
                if e isa MalformedExprException
                    rethrow(e)
                    #rethrow(MalformedExprException("$expr"))
                else
                    rethrow(e)
                end
            end
            if is_eq(cons_expr)
                MOI.add_constraint(opt_setup, func, MOI.EqualTo(Float64(-c)))
            elseif is_leq(cons_expr)
                MOI.add_constraint(opt_setup, func, MOI.LessThan(Float64(-c)))
            else
                throw(MalformedExprException("$expr"))
            end
        end
    end

    # objective
    expr = _replace_parameter_indices!(deepcopy(cache.expr), cache.p)
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
    MOI.set(opt_setup, MOI.ObjectiveFunction{typeof(func)}(), func)

    MOI.optimize!(opt_setup)
    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), Theta)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = __moi_status_to_ReturnCode(MOI.get(opt_setup, MOI.TerminationStatus()))
    else
        minimizer = fill(NaN, length(Theta))
        minimum = NaN
        opt_ret = SciMLBase.ReturnCode.Default
    end
    stats = OptimizationBase.OptimizationStats()
    return SciMLBase.build_solution(cache,
        cache.opt,
        minimizer,
        minimum;
        original = opt_setup,
        retcode = opt_ret,
        stats = stats)
end

function get_moi_function(expr)
    affine_terms = MOI.ScalarAffineTerm{Float64}[]
    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    constant = Ref(0.0)
    collect_moi_terms!(expr,
        affine_terms,
        quadratic_terms,
        constant)
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
                return Expr(:call, :+,
                    Expr(:call, :^, expr.args[2].args[2], 2),
                    Expr(:call, :+,
                        Expr(:call, :*, 2,
                            Expr(:call, :*, expr.args[2].args[2],
                                expr.args[2].args[3])),
                        Expr(:call, :^, expr.args[2].args[3], 2)))
            else
                return Expr(:call, :*, expr.args[2], expr.args[2]) # x^2 => x * x
            end
        elseif expr.args[1] == :(^) && isa(expr.args[3], Int) && expr.args[3] > 2 # x^n => x * x^(n-1)
            return Expr(:call, :*, Expr(:call, :^, expr.args[2], expr.args[3] - 1),
                expr.args[2])
        elseif expr.args[1] == :(*) && isa(expr.args[3], Number) # x * a::Number => a * x
            return Expr(:call, :*, expr.args[3], expr.args[2])
        elseif expr.args[1] == :(+) && isa(expr.args[3], Number) # x + a::Number => a + x
            return Expr(:call, :+, expr.args[3], expr.args[2])
        elseif expr.args[1] == :(*) && isa(expr.args[3], Expr) &&
               expr.args[3].head == :call && expr.args[3].args[1] == :(+) # (x * (y + z)) => ((x * y) + (x * z))
            return Expr(:call, :+,
                Expr(:call, :*, expr.args[2], expr.args[3].args[2]),
                Expr(:call, :*, expr.args[2], expr.args[3].args[3]))
        elseif expr.args[1] == :(*) && isa(expr.args[2], Expr) &&
               expr.args[2].head == :call && expr.args[2].args[1] == :(+) # ((y + z) * x) => ((x * y) + (x * z))
            return Expr(:call, :+,
                Expr(:call, :*, expr.args[3], expr.args[2].args[2]),
                Expr(:call, :*, expr.args[3], expr.args[2].args[3]))
        elseif expr.args[1] == :(*) && expr.args[2] isa Number && isa(expr.args[3], Expr) &&
               expr.args[3].head == :call && expr.args[3].args[1] == :(*) &&
               expr.args[3].args[2] isa Number # a::Number * (b::Number * c) => (a * b) * c
            return Expr(:call, :*, expr.args[2] * expr.args[3].args[2],
                expr.args[3].args[3])
        elseif expr.args[1] == :(+) && isa(expr.args[3], Expr) &&
               isa(expr.args[2], Number) &&
               expr.args[3].head == :call && expr.args[3].args[1] == :(+) &&
               isa(expr.args[3].args[2], Number) # a::Number + (b::Number + x)  => (a+b) + x
            return Expr(:call, :+, expr.args[2] + expr.args[3].args[2],
                expr.args[3].args[3])
        elseif expr.args[1] == :(*) && isa(expr.args[3], Expr) &&
               expr.args[3].head == :call && expr.args[3].args[1] == :(*) &&
               isa(expr.args[3].args[2], Number) # x * (a::Number * y) => a * (x * y)
            return Expr(:call, :*, expr.args[3].args[2],
                Expr(:call, :*, expr.args[2], expr.args[3].args[3]))
        elseif expr.args[1] == :(*) && isa(expr.args[2], Expr) &&
               expr.args[2].head == :call && expr.args[2].args[1] == :(*) &&
               isa(expr.args[2].args[2], Number) # (a::Number * x) * y => a * (x * y)
            return Expr(:call, :*, expr.args[2].args[2],
                Expr(:call, :*, expr.args[2].args[3], expr.args[3]))
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
    constant[] += expr
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
                    push!(quadratic_terms,
                        MOI.ScalarQuadraticTerm(factor,
                            x1, x2))
                end
            else
                throw(MalformedExprException("$expr"))
            end
        end
    elseif expr.head == :ref # x[i]
        expr.args[1] == :x || throw(MalformedExprException("$expr"))
        push!(affine_terms, MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(expr.args[2])))
    else
        throw(MalformedExprException("$expr"))
    end

    return
end
