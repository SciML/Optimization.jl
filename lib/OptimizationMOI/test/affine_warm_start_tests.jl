# Included by core_tests.jl, which supplies OptimizationMOI, OptimizationBase,
# HiGHS, MathOptInterface, ModelingToolkitBase, and Test.

# Isolate this specialization from the other MockOptimizer tests: this model
# stores all start attributes, but intentionally does not advertise NLPBlock.
const AffineStartMock = MathOptInterface.Utilities.MockOptimizer{
    MathOptInterface.Utilities.UniversalFallback{MathOptInterface.Utilities.Model{Float64}},
    Float32,
}
MathOptInterface.supports(::AffineStartMock, ::MathOptInterface.NLPBlock) = false

@testset "Affine MOI duals and warm starts" begin
    MOI = MathOptInterface
    getdual = OptimizationMOI.get_dual
    AF = MOI.ScalarAffineFunction{Float64}
    GT = MOI.GreaterThan{Float64}
    LT = MOI.LessThan{Float64}
    indices(model, F, S) = MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
    function new_mock()
        mock = MOI.Utilities.MockOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), Float32
        )
        # No solution is needed to inspect the model and its start attributes.
        MOI.Utilities.set_mock_optimize!(mock, m -> MOI.Utilities.mock_optimize!(m, MOI.INFEASIBLE))
        return mock
    end

    # min x1 + 2x2 - 3x3 + 7; x1+x2+3 >= 5, x1+5 <= 8.
    # The optimum (2,0,1) has multipliers +1, 0, lower(x2)=+1,
    # upper(x3)=-3. Every bound is finite to exercise all start entries.
    f = OptimizationFunction(
        (x, p) -> x[1] + 2x[2] - 3x[3] + 7, AutoSymbolics();
        cons = (out, x, p) -> (out .= [x[1] + x[2] + 3, x[1] + 5])
    )
    prob = OptimizationProblem(
        f, [1.0, 1.0, 0.0];
        lb = [0.0, 0.0, -5.0], ub = [5.0, 5.0, 1.0],
        lcons = [5.0, -Inf], ucons = [Inf, 8.0]
    )

    @testset "LP solution, indices and snapshots" begin
        optimizer = HiGHS.Optimizer()
        cache = init(prob, optimizer)
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [2.0, 0.0, 1.0] atol = 1.0e-7
        @test sol.objective ≈ 6.0 atol = 1.0e-7
        d = getdual(sol)
        @test keys(d) == (:constraints, :lower, :upper, :reduced_costs, :dual_objective, :constraint_primal)
        @test d.constraints ≈ [1.0, 0.0] atol = 1.0e-7
        @test d.lower ≈ [0.0, 1.0, 0.0] atol = 1.0e-7
        @test d.upper ≈ [0.0, 0.0, -3.0] atol = 1.0e-7
        @test d.reduced_costs ≈ d.lower + d.upper atol = 1.0e-7
        @test d.dual_objective ≈ 6.0 atol = 1.0e-7
        # Stationarity with MOI's minimization sign convention.
        @test [1.0, 2.0, -3.0] ≈
            [d.constraints[1] + d.constraints[2], d.constraints[1], 0.0] + d.reduced_costs atol = 1.0e-7
        model = sol.original
        vars = MOI.get(model, MOI.ListOfVariableIndices())
        lo = indices(model, MOI.VariableIndex, GT)
        hi = indices(model, MOI.VariableIndex, LT)
        # Use the actual indices returned by the model, not presumed row numbers.
        c1 = only(indices(model, AF, GT))
        c2 = only(indices(model, AF, LT))
        @test getdual(sol, 1) ≈ d.constraints[1]
        @test getdual(sol, 2) ≈ d.constraints[2]
        @test getdual(sol, c1) ≈ d.constraints[1]
        @test getdual(sol, c2) ≈ d.constraints[2]
        for i in eachindex(vars)
            @test getdual(sol, vars[i]) ≈ d.reduced_costs[i]
            @test getdual(sol, lo[i]) ≈ d.lower[i]
            @test getdual(sol, hi[i]) ≈ d.upper[i]
        end
        # The public values retain the original row constants even if MOI
        # moved them from the stored function to its set.
        @test d.constraint_primal ≈ [5.0, 7.0] atol = 1.0e-7
        @test MOI.get(model, MOI.ConstraintPrimal(), c1) - MOI.get(model, MOI.ConstraintSet(), c1).lower ≈ 0.0 atol = 1.0e-7
        @test MOI.get(model, MOI.ConstraintPrimal(), c2) - MOI.get(model, MOI.ConstraintSet(), c2).upper ≈ -1.0 atol = 1.0e-7

        saved = deepcopy(d)
        OptimizationMOI.reinit!(cache; u0 = [1.5, 0.5, 1.0])
        sol2 = solve!(cache)
        @test sol2.u ≈ sol.u atol = 1.0e-7
        @test getdual(sol) == saved
        @test getdual(sol, c1) ≈ saved.constraints[1]
        # A new solve with the same live optimizer must not mutate sol's snapshot.
        other = OptimizationProblem(
            OptimizationFunction((x, p) -> 4x[1], AutoSymbolics()), [0.5];
            lb = [1.0], ub = [2.0]
        )
        sol3 = solve(other, optimizer)
        @test sol3.u ≈ [1.0] atol = 1.0e-7
        @test getdual(sol3).lower ≈ [4.0] atol = 1.0e-7
        @test getdual(sol) == saved
        @test getdual(sol, vars[2]) ≈ saved.reduced_costs[2]
        @test getdual(sol, lo[2]) ≈ saved.lower[2]
    end

    @testset "Maximization and equality signs" begin
        mf = OptimizationFunction(
            (x, p) -> 2x[1] + x[2] + 3, AutoSymbolics();
            cons = (out, x, p) -> (out[1] = x[1] + x[2])
        )
        mp = OptimizationProblem(
            mf, [1.0, 2.0]; sense = OptimizationBase.MaxSense,
            lb = [0.0, 0.0], ub = [2.0, Inf], lcons = [3.0], ucons = [3.0]
        )
        sol = solve(mp, HiGHS.Optimizer())
        @test sol.u ≈ [2.0, 1.0] atol = 1.0e-7
        @test sol.objective ≈ 8.0 atol = 1.0e-7
        d = getdual(sol)
        # MOI's dual cone convention is independent of objective sense.
        @test d.constraints ≈ [-1.0] atol = 1.0e-7
        @test d.lower ≈ [0.0, 0.0] atol = 1.0e-7
        @test d.upper ≈ [-1.0, 0.0] atol = 1.0e-7
        @test d.reduced_costs ≈ [-1.0, 0.0] atol = 1.0e-7
        @test d.dual_objective ≈ 8.0 atol = 1.0e-7
        @test d.constraint_primal ≈ [3.0] atol = 1.0e-7
        ci = only(indices(sol.original, AF, MOI.EqualTo{Float64}))
        @test getdual(sol, ci) ≈ -1.0 atol = 1.0e-7
    end

    @testset "Quadratic and ranged AutoSymbolics constraints" begin
        qf = OptimizationFunction(
            (x, p) -> (x[1] - 3)^2 + 4, AutoSymbolics();
            cons = (out, x, p) -> (out[1] = x[1] + 2)
        )
        qp = OptimizationProblem(
            qf, [0.5]; lb = [-2.0], ub = [4.0],
            lcons = [-Inf], ucons = [3.0]
        )
        sol = solve(qp, HiGHS.Optimizer())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0] atol = 1.0e-5
        @test sol.objective ≈ 8.0 atol = 1.0e-5
        d = getdual(sol)
        @test d.constraints ≈ [-4.0] atol = 1.0e-5
        @test d.reduced_costs ≈ [0.0] atol = 1.0e-5
        @test 2 * (sol.u[1] - 3) ≈ d.constraints[1] + d.reduced_costs[1] atol = 1.0e-5
        if d.dual_objective !== nothing
            # HiGHS' MOI wrapper computes this attribute using row and bound
            # multipliers and may not include the quadratic contribution.
            @test d.dual_objective ≈ MOI.get(sol.original, MOI.DualObjectiveValue())
        end
        # A finite lower and upper bound generates a comparison expression.
        ranged = OptimizationProblem(
            qf, [0.5]; lb = [-2.0], ub = [4.0],
            lcons = [2.0], ucons = [3.0]
        )
        ranged_sol = solve(ranged, HiGHS.Optimizer())
        @test ranged_sol.u ≈ [1.0] atol = 1.0e-5
        @test ranged_sol.objective ≈ 8.0 atol = 1.0e-5
        @test getdual(ranged_sol).constraints ≈ [-4.0] atol = 1.0e-5
        @test getdual(ranged_sol).constraint_primal ≈ [3.0] atol = 1.0e-5
    end

    @testset "MTK system and parameter update" begin
        @variables z
        @parameters a
        sys = complete(
            OptimizationSystem(
                a * z + 5, [z], [a];
                constraints = [z ≳ 2.0], name = :affine_dual_system
            )
        )
        p = OptimizationProblem(
            sys, [z => 0.0, a => 2.0];
            lb = [-1.0], ub = [4.0], grad = false, hess = false
        )
        sol = solve(p, HiGHS.Optimizer())
        @test sol.u ≈ [2.0] atol = 1.0e-7
        @test sol.objective ≈ 9.0 atol = 1.0e-7
        @test abs(only(getdual(sol).constraints)) ≈ 2.0 atol = 1.0e-7

        # A vector parameter avoids MTKParameters-specific reinitialization.
        pf = OptimizationFunction(
            (x, p) -> p[1] * x[1] + 5, AutoSymbolics();
            cons = (out, x, p) -> (out[1] = x[1] + p[2])
        )
        pp = OptimizationProblem(
            pf, [1.0], [2.0, 1.0];
            lb = [0.0], ub = [5.0], lcons = [3.0], ucons = [Inf]
        )
        cache = init(pp, HiGHS.Optimizer())
        before = solve!(cache)
        saved = deepcopy(getdual(before))
        OptimizationMOI.reinit!(cache; p = [3.0, 0.0], u0 = [2.0])
        after = solve!(cache)
        @test before.u ≈ [2.0] atol = 1.0e-7
        @test after.u ≈ [3.0] atol = 1.0e-7
        @test after.objective ≈ 14.0 atol = 1.0e-7
        @test getdual(after).constraints ≈ [3.0] atol = 1.0e-7
        @test getdual(before) == saved
    end

    @testset "Unavailable duals" begin
        mip = OptimizationProblem(
            OptimizationFunction((x, p) -> x[1], AutoSymbolics()), [0.0];
            lb = [0.0], ub = [1.0], int = [true]
        )
        sol = solve(mip, HiGHS.Optimizer())
        @test sol.retcode == ReturnCode.Success
        @test getdual(sol).constraints === nothing
        @test getdual(sol).lower === nothing
        @test getdual(sol).upper === nothing
        @test getdual(sol).reduced_costs === nothing
        @test getdual(sol).dual_objective === nothing
        @test getdual(sol, 1) === nothing
        var = only(MOI.get(sol.original, MOI.ListOfVariableIndices()))
        lower = only(indices(sol.original, MOI.VariableIndex, GT))
        @test getdual(sol, var) === nothing
        @test getdual(sol, lower) === nothing

        infeasible = OptimizationProblem(
            OptimizationFunction(
                (x, p) -> x[1], AutoSymbolics();
                cons = (out, x, p) -> (out[1] = x[1])
            ), [0.0];
            lb = [0.0], ub = [1.0], lcons = [2.0], ucons = [Inf]
        )
        bad = solve(infeasible, HiGHS.Optimizer())
        @test MOI.get(bad.original, MOI.TerminationStatus()) == MOI.INFEASIBLE
        # A dual infeasibility certificate is not an optimal dual solution.
        @test getdual(bad).constraints === nothing
        @test getdual(bad).reduced_costs === nothing
        # HiGHS may still return an infeasible primal point for diagnostics.
    end

    @testset "All starts propagate through init, solve and reinit" begin
        starts = (
            constraints = [1.25, -0.5], lower = [0.1, 0.2, 0.3],
            upper = [-0.1, -0.2, -0.3], lower_primal = [0.4, 0.5, 0.6],
            upper_primal = [0.7, 0.8, 0.9],
        )
        row_starts = [2.5, 1.5]
        function check_starts(model, x0, d, rows)
            vars = MOI.get(model, MOI.ListOfVariableIndices())
            lo = indices(model, MOI.VariableIndex, GT)
            hi = indices(model, MOI.VariableIndex, LT)
            c1 = only(indices(model, AF, GT))
            c2 = only(indices(model, AF, LT))
            @test MOI.get(model, MOI.VariablePrimalStart(), vars) ≈ x0
            @test MOI.get(model, MOI.ConstraintDualStart(), lo) ≈ d.lower
            @test MOI.get(model, MOI.ConstraintDualStart(), hi) ≈ d.upper
            @test MOI.get(model, MOI.ConstraintPrimalStart(), lo) ≈ d.lower_primal
            @test MOI.get(model, MOI.ConstraintPrimalStart(), hi) ≈ d.upper_primal
            @test MOI.get(model, MOI.ConstraintDualStart(), [c1, c2]) ≈ d.constraints
            # These MOI functions have had +3 and +5 moved into their sets.
            @test MOI.get(model, MOI.ConstraintPrimalStart(), [c1, c2]) ≈ rows - [3.0, 5.0]
        end
        cache = init(prob, new_mock(); dual_values = (; starts..., constraint_primal_start = row_starts))
        sol = solve!(cache)
        check_starts(sol.original, prob.u0, starts, row_starts)
        newstarts = (
            constraints = [2.0, -1.0], lower = [0.4, 0.5, 0.6],
            upper = [-0.4, -0.5, -0.6], lower_primal = [1.0, 1.1, 1.2],
            upper_primal = [1.3, 1.4, 1.5],
        )
        x0 = [2.0, 0.5, 0.8]
        newrows = [1.0, 2.0]
        OptimizationMOI.reinit!(
            cache; u0 = x0,
            dual_values = (; newstarts..., constraint_primal_start = newrows)
        )
        sol2 = solve!(cache)
        check_starts(sol2.original, x0, newstarts, newrows)
        OptimizationMOI.reinit!(cache; u0 = prob.u0)
        kept = solve!(cache)
        check_starts(kept.original, prob.u0, newstarts, newrows)
        OptimizationMOI.reinit!(cache; dual_values = nothing)
        cleared = solve!(cache).original
        @test MOI.get(cleared, MOI.ConstraintDualStart(), only(indices(cleared, AF, GT))) === nothing
        @test MOI.get(cleared, MOI.ConstraintPrimalStart(), only(indices(cleared, AF, GT))) === nothing
        sol3 = solve(prob, new_mock(); dual_values = (; starts..., constraint_primal_start = row_starts))
        check_starts(sol3.original, prob.u0, starts, row_starts)

        # A complete result tuple can be reused without forwarding its
        # result-only fields as primal starts.
        result = getdual(solve(prob, HiGHS.Optimizer()))
        reused = solve(prob, new_mock(); dual_values = result).original
        lo = indices(reused, MOI.VariableIndex, GT)
        hi = indices(reused, MOI.VariableIndex, LT)
        c1 = only(indices(reused, AF, GT))
        c2 = only(indices(reused, AF, LT))
        @test MOI.get(reused, MOI.ConstraintDualStart(), lo) ≈ result.lower
        @test MOI.get(reused, MOI.ConstraintDualStart(), hi) ≈ result.upper
        @test MOI.get(reused, MOI.ConstraintDualStart(), [c1, c2]) ≈ result.constraints
        @test MOI.get(reused, MOI.ConstraintPrimalStart(), c1) === nothing
    end

    @testset "Start validation and unsupported attributes" begin
        for field in (:constraints, :constraint_primal_start, :lower, :upper, :lower_primal, :upper_primal)
            expected = field in (:constraints, :constraint_primal_start) ? 2 : 3
            for n in (expected - 1, expected + 1)
                bad = NamedTuple{(field,)}((zeros(n),))
                @test_throws ArgumentError begin
                    cache = init(prob, new_mock(); dual_values = bad)
                    solve!(cache)
                end
                @test_throws ArgumentError begin
                    cache = init(prob, new_mock())
                    OptimizationMOI.reinit!(cache; dual_values = bad)
                    solve!(cache)
                end
            end
        end
        cache = init(prob, new_mock())
        @test_throws ArgumentError OptimizationMOI.reinit!(cache; u0 = [1.0])
        @test length(cache.u0) == 3
        # Check only attributes that this HiGHS version explicitly rejects.
        h = HiGHS.Optimizer()
        for (attr, CI, kwargs) in (
                (MOI.ConstraintDualStart(), MOI.ConstraintIndex{AF, GT}, (dual_values = (constraints = [1.0, 0.0],),)),
                (MOI.ConstraintDualStart(), MOI.ConstraintIndex{MOI.VariableIndex, GT}, (dual_values = (lower = zeros(3),),)),
                (MOI.ConstraintDualStart(), MOI.ConstraintIndex{MOI.VariableIndex, LT}, (dual_values = (upper = zeros(3),),)),
                (MOI.ConstraintPrimalStart(), MOI.ConstraintIndex{MOI.VariableIndex, GT}, (dual_values = (lower_primal = zeros(3),),)),
                (MOI.ConstraintPrimalStart(), MOI.ConstraintIndex{MOI.VariableIndex, LT}, (dual_values = (upper_primal = zeros(3),),)),
                (MOI.ConstraintPrimalStart(), MOI.ConstraintIndex{AF, GT}, (dual_values = (constraint_primal_start = zeros(2),),)),
            )
            if !MOI.supports(h, attr, CI)
                @test_throws MOI.UnsupportedAttribute begin
                    cache = init(prob, HiGHS.Optimizer(); kwargs...)
                    solve!(cache)
                end
            end
        end
    end
end
