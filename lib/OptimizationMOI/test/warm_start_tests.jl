# This file is included by core_tests.jl, which supplies the test and solver imports.
@testset "MOI dual solutions and warm starts" begin
    MOI = MathOptInterface
    getdual = OptimizationMOI.get_dual

    # The first constraint is active, the second inactive. At the optimum the
    # third variable is at its lower bound and the fourth at its upper bound.
    objective(x, p) = sum((x[i] - p[i])^2 for i in eachindex(x))
    constraints!(res, x, p) = (res .= [x[1] + x[2], x[1]^2 + x[2]^2])
    f = OptimizationFunction(objective, AutoForwardDiff(); cons = constraints!)
    function make_problem(p)
        return OptimizationProblem(
            f, [0.5, 1.5, 0.5, 0.5], p;
            lb = [-5.0, -5.0, 0.0, -5.0], ub = [5.0, 5.0, 5.0, 1.0],
            lcons = [2.0, -Inf], ucons = [2.0, 10.0]
        )
    end
    prob = make_problem([2.0, 2.0, -1.0, 3.0])
    lower(i) = MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}(i)
    upper(i) = MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}(i)

    @testset "Ipopt results and indexed access" begin
        sol = solve(prob, Ipopt.Optimizer(); print_level = 0)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0, 1.0, 0.0, 1.0] atol = 1.0e-5
        dual = getdual(sol)
        @test dual isa NamedTuple
        @test all(
            k -> hasproperty(dual, k), (
                :constraints, :lower, :upper, :reduced_costs,
                :dual_objective, :constraint_primal,
            )
        )
        # MOI uses a positive sign for a lower bound and negative for an upper.
        @test dual.constraints ≈ [-2.0, 0.0] atol = 1.0e-5
        @test dual.lower ≈ [0.0, 0.0, 2.0, 0.0] atol = 1.0e-5
        @test dual.upper ≈ [0.0, 0.0, 0.0, -4.0] atol = 1.0e-5
        @test dual.reduced_costs ≈ dual.lower + dual.upper atol = 1.0e-7
        for i in 1:2
            @test getdual(sol, i) ≈ dual.constraints[i] atol = 1.0e-8
        end
        for i in 1:4
            @test getdual(sol, MOI.VariableIndex(i)) ≈ dual.reduced_costs[i] atol = 1.0e-8
            @test getdual(sol, lower(i)) ≈ dual.lower[i] atol = 1.0e-8
            @test getdual(sol, upper(i)) ≈ dual.upper[i] atol = 1.0e-8
        end
        @test dual.constraint_primal ≈ [2.0, 2.0] atol = 1.0e-5
        @test dual.dual_objective === nothing # Ipopt does not expose this attribute
    end

    @testset "start propagation, kwargs and reinit" begin
        starts = (
            constraints = [-1.5, -0.25],
            lower = [0.1, 0.2, 1.5, 0.4],
            upper = [-0.1, -0.2, -0.3, -3.5],
        )
        cache = init(
            prob, Ipopt.Optimizer(); dual_values = starts,
            print_level = 0, warm_start_init_point = "yes"
        )
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0, 1.0, 0.0, 1.0] atol = 1.0e-5
        # Convergence alone would not establish that the starts reached MOI.
        @test MOI.get(sol.original, MOI.RawOptimizerAttribute("warm_start_init_point")) == "yes"
        @test MOI.get(sol.original, MOI.NLPBlockDualStart()) == starts.constraints
        for i in 1:4
            @test MOI.get(sol.original, MOI.ConstraintDualStart(), lower(i)) == starts.lower[i]
            @test MOI.get(sol.original, MOI.ConstraintDualStart(), upper(i)) == starts.upper[i]
        end

        dual = getdual(sol)
        saved = deepcopy(dual)
        # The whole output tuple is reusable: result-only fields are ignored.
        OptimizationMOI.reinit!(cache; u0 = copy(sol.u), dual_values = dual)
        sol2 = solve!(cache)
        @test sol2.retcode == ReturnCode.Success
        @test sol2.u ≈ sol.u atol = 1.0e-5
        @test MOI.get(sol2.original, MOI.NLPBlockDualStart()) ≈ saved.constraints atol = 1.0e-8
        @test getdual(sol) == saved

        # A partial tuple must not require any of the other start fields.
        OptimizationMOI.reinit!(cache; u0 = [0.8, 1.2, 0.1, 0.8], dual_values = (constraints = [-2.0, 0.0],))
        sol3 = solve!(cache)
        @test sol3.retcode == ReturnCode.Success
        @test MOI.get(sol3.original, MOI.NLPBlockDualStart()) == [-2.0, 0.0]
        @test getdual(sol) == saved

        OptimizationMOI.reinit!(
            cache; p = [1.0, 1.0, -2.0, 4.0], u0 = sol3.u,
            dual_values = getdual(sol3)
        )
        sol4 = solve!(cache)
        @test sol4.u ≈ [1.0, 1.0, 0.0, 1.0] atol = 1.0e-5
        @test getdual(sol4).lower[3] ≈ 4.0 atol = 1.0e-5
        @test getdual(sol) == saved

        sol5 = solve(
            prob, Ipopt.Optimizer(); dual_values = (constraints = [-2.0, 0.0],),
            print_level = 0, warm_start_init_point = "yes"
        )
        @test MOI.get(sol5.original, MOI.NLPBlockDualStart()) == [-2.0, 0.0]
    end

    @testset "bound constraint primal starts" begin
        starts = (
            lower_primal = [0.6, 1.4, 0.1, 0.8],
            upper_primal = [0.7, 1.3, 0.2, 0.9],
        )
        cache = init(prob, Ipopt.Optimizer(); dual_values = starts, print_level = 0)
        model = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        _, lo, hi = OptimizationMOI._add_moi_variables!(model, cache.evaluator)
        for i in 1:4
            @test MOI.get(model, MOI.ConstraintPrimalStart(), lo[i]) == starts.lower_primal[i]
            @test MOI.get(model, MOI.ConstraintPrimalStart(), hi[i]) == starts.upper_primal[i]
        end
        @test_throws MOI.UnsupportedAttribute solve!(cache)
    end

    @testset "dual objective snapshot" begin
        mock = MOI.Utilities.MockOptimizer(
            MOI.Utilities.Model{Float64}();
            eval_dual_objective_value = false
        )
        MOI.set(mock, MOI.ResultCount(), 1)
        MOI.set(mock, MOI.NLPBlockDual(), [1.0, 2.0])
        MOI.set(mock, MOI.DualObjectiveValue(), 3.0)
        cache = init(prob, Ipopt.Optimizer(); print_level = 0)
        results = OptimizationMOI._nlp_results(
            mock, cache.evaluator, prob.u0,
            MOI.VariableIndex.([3, 7, 11, 15]), fill(nothing, 4), fill(nothing, 4), false, true
        )
        @test results.values.constraints == [1.0, 2.0]
        @test results.values.dual_objective == 3.0
        @test results.variables[MOI.VariableIndex(11)] == 3
        MOI.set(mock, MOI.NLPBlockDual(), [4.0, 5.0])
        @test results.values.constraints == [1.0, 2.0]

        badf = OptimizationFunction(
            (x, p) -> sum(abs2, x), NoAD();
            cons = (out, x, p) -> error("constraint unavailable")
        )
        badprob = OptimizationProblem(badf, [1.0]; lcons = [0.0], ucons = [1.0])
        badcache = init(badprob, Ipopt.Optimizer(); print_level = 0)
        results = OptimizationMOI._nlp_results(
            mock, badcache.evaluator, [1.0],
            [MOI.VariableIndex(7)], [nothing], [nothing], true, true
        )
        @test results.values.constraint_primal === nothing
        @test OptimizationMOI._is_point(MOI.FEASIBLE_POINT)
        @test !OptimizationMOI._is_point(MOI.INFEASIBILITY_CERTIFICATE)
        @test !OptimizationMOI._is_point(MOI.NO_SOLUTION)
    end

    @testset "solution snapshots survive optimizer reuse" begin
        optimizer = Ipopt.Optimizer()
        sol1 = solve(prob, optimizer; print_level = 0)
        dual1 = deepcopy(getdual(sol1))
        # Reuse empties the same optimizer but yields different multipliers.
        sol2 = solve(make_problem([1.0, 1.0, -2.0, 4.0]), optimizer; print_level = 0)
        @test sol2.retcode == ReturnCode.Success
        @test getdual(sol2).lower[3] ≈ 4.0 atol = 1.0e-5
        @test getdual(sol2).upper[4] ≈ -6.0 atol = 1.0e-5
        @test getdual(sol1) == dual1
        @test getdual(sol1, 1) ≈ dual1.constraints[1] atol = 1.0e-8
        @test getdual(sol1, MOI.VariableIndex(3)) ≈ dual1.reduced_costs[3] atol = 1.0e-8
        @test getdual(sol1, lower(3)) ≈ dual1.lower[3] atol = 1.0e-8
    end

    @testset "parameter-dependent constraints after reinit" begin
        for ad in (AutoForwardDiff(), AutoZygote())
            param_f = OptimizationFunction(
                (x, p) -> (x[1] - 2)^2, ad;
                cons = (res, x, p) -> (res[1] = p[1] * x[1]^2)
            )
            param_prob = OptimizationProblem(param_f, [1.0], [1.0]; lcons = [-Inf], ucons = [4.0])
            param_cache = init(param_prob, Ipopt.Optimizer(); print_level = 0)
            new_p = [4.0]
            OptimizationMOI.reinit!(param_cache; p = new_p)
            @test param_cache.p === new_p
            @test param_prob.p == [1.0]
            evaluator = param_cache.evaluator
            cons = zeros(1)
            jac = zeros(1)
            vjp = zeros(1)
            hess = zeros(1)
            MOI.eval_constraint(evaluator, cons, [1.0])
            MOI.eval_constraint_jacobian(evaluator, jac, [1.0])
            MOI.eval_constraint_jacobian_transpose_product(evaluator, vjp, [1.0], [1.0])
            MOI.eval_hessian_lagrangian(evaluator, hess, [1.0], 0.0, [1.0])
            @test cons ≈ [4.0]
            @test jac ≈ [8.0]
            @test vjp ≈ [8.0]
            @test hess ≈ [8.0]
            @test solve!(param_cache).u ≈ [1.0] atol = 1.0e-5
        end
    end

    @testset "invalid start lengths" begin
        # Validate every input field, including otherwise optional fields; the
        # error may be raised by init or by solve! when the MOI model is built.
        for field in (:constraints, :lower, :upper, :lower_primal, :upper_primal)
            expected = field === :constraints ? 2 : 4
            for n in (expected - 1, expected + 1)
                bad = NamedTuple{(field,)}((zeros(n),))
                @test_throws ArgumentError begin
                    cache = init(prob, Ipopt.Optimizer(); dual_values = bad, print_level = 0)
                    solve!(cache)
                end
                cache = init(prob, Ipopt.Optimizer(); print_level = 0)
                @test_throws ArgumentError begin
                    OptimizationMOI.reinit!(cache; u0 = copy(prob.u0), dual_values = bad)
                    solve!(cache)
                end
            end
        end
        cache = init(prob, Ipopt.Optimizer(); print_level = 0)
        @test_throws ArgumentError OptimizationMOI.reinit!(cache; u0 = [1.0])
        @test length(cache.u0) == 4
    end

    @testset "NLopt without dual support" begin
        opt = MOI.OptimizerWithAttributes(
            NLopt.Optimizer, "algorithm" => :LD_SLSQP, "xtol_rel" => 1.0e-8
        )
        sol = solve(prob, opt)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0, 1.0, 0.0, 1.0] atol = 1.0e-4
        # NLopt supplies a primal solution but no MOI dual solution. Asking for
        # the aggregate must not accidentally query unavailable attributes.
        dual = getdual(sol)
        @test dual.constraints === nothing
        @test dual.lower === nothing
        @test dual.upper === nothing
        @test dual.reduced_costs === nothing
        @test dual.dual_objective === nothing
        @test getdual(sol, 1) === nothing
        @test getdual(sol, MOI.VariableIndex(1)) === nothing
        @test getdual(sol, lower(1)) === nothing

        # Explicit unsupported starts must not be silently ignored.
        cache = init(
            prob, opt; dual_values = (
                constraints = [-2.0, 0.0], lower = [0.0, 0.0, 2.0, 0.0],
                upper = [0.0, 0.0, 0.0, -4.0],
            )
        )
        @test_throws MOI.UnsupportedAttribute solve!(cache)
    end
end
