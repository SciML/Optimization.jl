using OptimizationNLopt, Optimization, Zygote, ReverseDiff
using Test, Random

@testset "OptimizationNLopt.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)
    sol = solve(prob, NLopt.Opt(:LN_BOBYQA, 2))
    @test sol.retcode == ReturnCode.Success
    @test 10 * sol.objective < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
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

        optf = OptimizationFunction(objective, Optimization.AutoZygote())
        prob = OptimizationProblem(optf, x0, p)
        cache = Optimization.init(prob, NLopt.Opt(:LD_LBFGS, 1))
        sol = Optimization.solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u≈[1.0] atol=1e-3

        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
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
        optprob = OptimizationFunction((x, p) -> -system(x, p), Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, __p; sense = Optimization.MaxSense)
        sol = solve(prob, NLopt.Opt(:LD_LBFGS, n), maxtime = 1e-6)
        @test sol.retcode == ReturnCode.MaxTime
    end

    @testset "constrained" begin
        Random.seed!(1)
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        x0 = zeros(2)
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote();
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

        function con2_c(res, x, p)
            res .= [x[1]^2 + x[2]^2 - 1.0, x[2] * sin(x[1]) - x[1] - 2.0]
        end

        # FTOL_REACHED
        optprob = OptimizationFunction(
            rosenbrock, Optimization.AutoForwardDiff(); cons = con2_c)
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
        @test 10 * sol.objective < l1
    end
end
