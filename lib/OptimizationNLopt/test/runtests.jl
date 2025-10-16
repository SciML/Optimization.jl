using OptimizationNLopt, OptimizationBase, Zygote, ReverseDiff
using Test, Random

@testset "OptimizationNLopt.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), OptimizationBase.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = OptimizationBase.MaxSense)
    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p)

    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])

    sol = solve(prob, NLopt.Opt(:LD_LBFGS, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.Opt(:G_MLSL_LDS, 2), local_method = NLopt.Opt(:LD_LBFGS, 2),
        maxiters = 10000)
    @test sol.retcode == ReturnCode.MaxIters
    @test 10 * sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, NLopt.LN_BOBYQA())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.LD_LBFGS())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    # XTOL_REACHED
    sol = solve(prob, NLopt.LD_LBFGS(), xtol_abs = 1e10)
    @test sol.retcode == ReturnCode.Success

    # STOPVAL_REACHED
    sol = solve(prob, NLopt.LD_LBFGS(), stopval = 1e10)
    @test sol.retcode == ReturnCode.Success

    prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
    sol = solve(prob, NLopt.LD_LBFGS())
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(),
        local_maxiters = 10000, maxiters = 10000, population = 10)
    @test sol.retcode == ReturnCode.MaxIters
    @test 10 * sol.objective < l1

    @testset "cache" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]

        optf = OptimizationFunction(objective, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optf, x0, p)
        cache = OptimizationBase.init(prob, NLopt.Opt(:LD_LBFGS, 1))
        sol = OptimizationBase.solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u≈[1.0] atol=1e-3

        cache = OptimizationBase.reinit!(cache; p = [2.0])
        sol = OptimizationBase.solve!(cache)
        # @test sol.retcode == ReturnCode.Success
        @test sol.u≈[2.0] atol=1e-3
    end

    @testset "callback" begin
        cbstopping = function (state, loss)
            println(state.iter, " ", state.u, " ", state.objective)
            return state.objective < 0.7
        end

        sol = solve(prob, NLopt.LD_LBFGS())
        #nlopt gives the last best not the one where callback stops
        @test sol.objective < 0.8
    end

    @testset "MAXTIME_REACHED" begin
        # without maxtime=... this will take time
        n = 2000
        A, b = rand(n, n), rand(n)
        system(x, p) = sum((A * x - b) .^ 2)
        x0 = zeros(n)
        __p = Float64[]
        optprob = OptimizationFunction((x, p) -> -system(x, p), OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optprob, x0, __p; sense = OptimizationBase.MaxSense)
        sol = solve(prob, NLopt.Opt(:LD_LBFGS, n), maxtime = 1e-6)
        @test sol.retcode == ReturnCode.MaxTime
    end

    @testset "dual_ftol_rel parameter" begin
        # Test that dual_ftol_rel parameter can be passed to NLopt without errors
        # This parameter is specific to MMA/CCSA algorithms for dual optimization tolerance
        x0_test = zeros(2)
        optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optprob, x0_test, _p)

        # Test with NLopt.Opt interface
        opt = NLopt.Opt(:LD_MMA, 2)
        # This should not throw an error - the PR fixed the UndefVarError
        sol = solve(prob, opt, dual_ftol_rel = 1e-6, maxiters = 100)
        @test sol.retcode ∈ [ReturnCode.Success, ReturnCode.MaxIters]

        # Test with direct algorithm interface
        sol = solve(prob, NLopt.LD_MMA(), dual_ftol_rel = 1e-5, maxiters = 100)
        @test sol.retcode ∈ [ReturnCode.Success, ReturnCode.MaxIters]

        # Verify it works with other solver options
        sol = solve(prob, NLopt.LD_MMA(), dual_ftol_rel = 1e-4, ftol_rel = 1e-6,
            xtol_rel = 1e-6, maxiters = 100)
        @test sol.retcode ∈ [ReturnCode.Success, ReturnCode.MaxIters]
    end

    @testset "constrained" begin
        Random.seed!(1)
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        x0 = zeros(2)
        optprob = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote();
            cons = cons)
        prob = OptimizationProblem(optprob, x0, _p, lcons = [0.0], ucons = [0.0])
        sol = solve(prob, NLopt.LN_COBYLA())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        Random.seed!(1)
        prob = OptimizationProblem(optprob, rand(2), _p,
            lcons = [0.0], ucons = [0.0])

        sol = solve(prob, NLopt.LD_SLSQP())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        Random.seed!(1)
        prob = OptimizationProblem(optprob, rand(2), _p,
            lcons = [0.0], ucons = [0.0])
        sol = solve(prob, NLopt.AUGLAG(), local_method = NLopt.LD_LBFGS())
        # @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        # Test that AUGLAG without local_method throws an error
        @test_throws ErrorException solve(prob, NLopt.LN_AUGLAG())
        @test_throws ErrorException solve(prob, NLopt.LD_AUGLAG())

        function con2_c(res, x, p)
            res .= [x[1]^2 + x[2]^2 - 1.0, x[2] * sin(x[1]) - x[1] - 2.0]
        end

        # FTOL_REACHED
        optprob = OptimizationFunction(
            rosenbrock, OptimizationBase.AutoForwardDiff(); cons = con2_c)
        Random.seed!(1)
        prob = OptimizationProblem(
            optprob, rand(2), _p, lcons = [0.0, -Inf], ucons = [0.0, 0.0])
        sol = solve(prob, NLopt.LD_AUGLAG(), local_method = NLopt.LD_LBFGS())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1

        Random.seed!(1)
        prob = OptimizationProblem(optprob, [0.5, 0.5], _p, lcons = [-Inf, -Inf],
            ucons = [0.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])
        sol = solve(prob, NLopt.GN_ISRES(), maxiters = 1000)
        @test sol.retcode == ReturnCode.MaxIters
        @test sol.objective < l1
    end

    @testset "gradient-based algorithm without AD backend" begin
        # Test that gradient-based algorithms throw a helpful error when no AD backend is specified
        # This reproduces the issue from https://discourse.julialang.org/t/error-when-using-multistart-optimization/133174
        rosenbrock_test(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
        x0_test = zeros(2)
        p_test = [1.0, 100.0]

        # Create OptimizationFunction WITHOUT specifying an AD backend
        f_no_ad = OptimizationFunction(rosenbrock_test)
        prob_no_ad = OptimizationProblem(
            f_no_ad, x0_test, p_test, lb = [-1.0, -1.0], ub = [1.5, 1.5])

        # Test with LD_LBFGS (gradient-based algorithm) - should throw IncompatibleOptimizerError
        @test_throws OptimizationBase.IncompatibleOptimizerError solve(prob_no_ad, NLopt.LD_LBFGS())

        # Test with NLopt.Opt interface - should also throw IncompatibleOptimizerError
        @test_throws OptimizationBase.IncompatibleOptimizerError solve(prob_no_ad, NLopt.Opt(:LD_LBFGS, 2))

        # Test that gradient-free algorithms still work without AD backend
        sol = solve(prob_no_ad, NLopt.LN_NELDERMEAD())
        @test sol.retcode == ReturnCode.Success

        # Test that with AD backend, gradient-based algorithms work correctly
        f_with_ad = OptimizationFunction(rosenbrock_test, OptimizationBase.AutoZygote())
        prob_with_ad = OptimizationProblem(
            f_with_ad, x0_test, p_test, lb = [-1.0, -1.0], ub = [1.5, 1.5])
        sol = solve(prob_with_ad, NLopt.LD_LBFGS())
        @test sol.retcode == ReturnCode.Success
        @test sol.objective < 1.0
    end
end
