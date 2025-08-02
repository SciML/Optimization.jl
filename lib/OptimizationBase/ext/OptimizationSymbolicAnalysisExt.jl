module OptimizationSymbolicAnalysisExt

using OptimizationBase, SciMLBase, SymbolicAnalysis, SymbolicAnalysis.Symbolics,
      OptimizationBase.ArrayInterface
using SymbolicAnalysis: AnalysisResult
import SymbolicAnalysis.Symbolics: variable, Equation, Inequality, unwrap, @variables

function OptimizationBase.symify_cache(
        f::OptimizationFunction{iip, AD, F, G, FG, H, FGH, HV, C, CJ, CJV, CVJ, CH, HP,
            CJP, CHP, O, EX, CEX, SYS, LH, LHP, HCV, CJCV, CHCV, LHCV},
        prob, num_cons,
        manifold) where {
        iip, AD, F, G, FG, H, FGH, HV, C, CJ, CJV, CVJ, CH, HP, CJP, CHP, O,
        EX <: Nothing, CEX <: Nothing, SYS, LH, LHP, HCV, CJCV, CHCV, LHCV}
    obj_expr = f.expr
    cons_expr = f.cons_expr === nothing ? nothing : getfield.(f.cons_expr, Ref(:lhs))

    if obj_expr === nothing || cons_expr === nothing
        try
            vars = if prob.u0 isa Matrix
                @variables X[1:size(prob.u0, 1), 1:size(prob.u0, 2)]
            else
                ArrayInterface.restructure(
                    prob.u0, [variable(:x, i) for i in eachindex(prob.u0)])
            end
            params = if prob.p isa SciMLBase.NullParameters
                []
            elseif prob.p isa MTK.MTKParameters
                [variable(:α, i) for i in eachindex(vcat(p...))]
            else
                ArrayInterface.restructure(p, [variable(:α, i) for i in eachindex(p)])
            end

            if prob.u0 isa Matrix
                vars = vars[1]
            end

            if obj_expr === nothing
                obj_expr = f.f(vars, params)
            end

            if cons_expr === nothing && SciMLBase.isinplace(prob) && !isnothing(prob.f.cons)
                lhs = Array{Symbolics.Num}(undef, num_cons)
                f.cons(lhs, vars)
                cons = Union{Equation, Inequality}[]

                if !isnothing(prob.lcons)
                    for i in 1:num_cons
                        if !isinf(prob.lcons[i])
                            if prob.lcons[i] != prob.ucons[i]
                                push!(cons, prob.lcons[i] ≲ lhs[i])
                            else
                                push!(cons, lhs[i] ~ prob.ucons[i])
                            end
                        end
                    end
                end

                if !isnothing(prob.ucons)
                    for i in 1:num_cons
                        if !isinf(prob.ucons[i]) && prob.lcons[i] != prob.ucons[i]
                            push!(cons, lhs[i] ≲ prob.ucons[i])
                        end
                    end
                end
                if (isnothing(prob.lcons) || all(isinf, prob.lcons)) &&
                   (isnothing(prob.ucons) || all(isinf, prob.ucons))
                    throw(ArgumentError("Constraints passed have no proper bounds defined.
                    Ensure you pass equal bounds (the scalar that the constraint should evaluate to) for equality constraints
                    or pass the lower and upper bounds for inequality constraints."))
                end
                cons_expr = lhs
            elseif cons_expr === nothing && !isnothing(prob.f.cons)
                cons_expr = f.cons(vars, params)
            end
        catch err
            throw(ArgumentError("Automatic symbolic expression generation with failed with error: $err.
            Try by setting `structural_analysis = false` instead if the solver doesn't require symbolic expressions."))
        end
    end

    if obj_expr !== nothing
        obj_expr = obj_expr |> Symbolics.unwrap
        if manifold === nothing
            obj_res = analyze(obj_expr)
        else
            obj_res = analyze(obj_expr, manifold)
        end
        @info "Objective Euclidean curvature: $(obj_res.curvature)"
        if obj_res.gcurvature !== nothing
            @info "Objective Geodesic curvature: $(obj_res.gcurvature)"
        end
    else
        obj_res = nothing
    end

    if cons_expr !== nothing
        cons_expr = cons_expr .|> Symbolics.unwrap
        if manifold === nothing
            cons_res = analyze.(cons_expr)
        else
            cons_res = analyze.(cons_expr, Ref(manifold))
        end
        for i in 1:num_cons
            @info "Constraints Euclidean curvature: $(cons_res[i].curvature)"

            if cons_res[i].gcurvature !== nothing
                @info "Constraints Geodesic curvature: $(cons_res[i].gcurvature)"
            end
        end
    else
        cons_res = nothing
    end

    return obj_res, cons_res
end

end
