import LinearAlgebra: opnorm, norm, tr, det, rank
using Symbolics: @register_symbolic

@register_symbolic opnorm(a)
@register_symbolic norm(a)
@register_symbolic tr(a)
@register_symbolic det(a)
@register_symbolic rank(a)

@testset "runtime symbolic coefficients" begin
    @variables x
    @parameters a
    objective_cases = ((3.0, 0.0, 0.0), (0.0, 1.0, -2.0), (-3.0, 0.0, 0.0), (0.0, 1.0, -2.0))
    constraint_cases = ((2.0, 0.5, -0.5), (4.0, 0.25, -0.25), (-2.0, 0.5, -0.5), (4.0, 0.25, -0.25))
    scalar_cases = ((3.0, 0.0, 0.0), (0.0, 1.0, -2.0))
    rank_cases = ((3.0, 0.0, 0.0), (0.0, 1.0, -0.5))
    @testset "$kind" for (kind, objective, constraints, cases) in (
            (:objective, (opnorm(a) - 2) * x, [], objective_cases),
            (:constraint, -x, [opnorm(a) * x ≲ 1.0], constraint_cases),
            (:norm, (norm(a) - 2) * x, [], scalar_cases),
            (:tr, (tr(a) - 2) * x, [], scalar_cases),
            (:det, (det(a) - 2) * x, [], scalar_cases),
            (:rank, (rank(a) - 0.5) * x, [], rank_cases),
        )
        sys = complete(OptimizationSystem(objective, [x], [a]; constraints, name = kind))
        optimizer = HiGHS.Optimizer()
        @testset "a = $parameter" for (parameter, expected_x, expected_objective) in cases
            prob = OptimizationProblem(
                sys, [x => 0.5, a => parameter]; lb = [0.0], ub = [1.0],
                grad = false, hess = false
            )
            cache = OptimizationMOI.init(prob, optimizer)
            expr = deepcopy(cache.expr)
            cons_expr = deepcopy(cache.cons_expr)
            parameters = copy(cache.p)
            for _ in 1:2
                sol = OptimizationMOI.solve!(cache)
                @test sol.retcode == ReturnCode.Success
                @test sol.u ≈ [expected_x] atol = 1.0e-8
                @test sol.objective ≈ expected_objective atol = 1.0e-8
                @test cache.expr == expr
                @test cache.cons_expr == cons_expr
                @test cache.p == parameters
            end
        end
    end
end
