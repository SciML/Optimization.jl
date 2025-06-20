using OptimizationSciPy, Optimization, Zygote, ReverseDiff, ForwardDiff
using Test, Random
using Optimization.SciMLBase: ReturnCode, NonlinearLeastSquaresProblem
using PythonCall

function rosenbrock(x, p)
    (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
end

function rosenbrock_hess(H, x, p)
    H[1,1] = 2 - 400*p[2]*x[2] + 1200*p[2]*x[1]^2
    H[1,2] = -400*p[2]*x[1]
    H[2,1] = -400*p[2]*x[1]
    H[2,2] = 200*p[2]
    return nothing
end

@testset "OptimizationSciPy.jl" begin
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    @testset "MaxSense" begin
        optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)
        sol = solve(prob, ScipyNelderMead())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "unconstrained with gradient" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyBFGS())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyLBFGSB())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "bounded" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
        sol = solve(prob, ScipyLBFGSB())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "global optimization" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
        sol = solve(prob, ScipyDifferentialEvolution(), maxiters = 100)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyBasinhopping(), maxiters = 50)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyDualAnnealing(), maxiters = 100)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyShgo())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyDirect(), maxiters = 1000)
        @test sol.retcode in (ReturnCode.Success, ReturnCode.Failure)
        if sol.retcode == ReturnCode.Success
            @test 10 * sol.objective < l1
        end
        sol = solve(prob, ScipyBrute(), Ns = 10)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "various methods" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyNelderMead())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyPowell())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyCG())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyTNC())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "with Hessian" begin
        optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); hess = rosenbrock_hess)
        prob = OptimizationProblem(optf, x0, _p)
        sol = solve(prob, ScipyNewtonCG(), maxiters = 200)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "bounded optimization" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
        sol = solve(prob, ScipyLBFGSB())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        sol = solve(prob, ScipyTNC())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
    end

    @testset "trust region with Hessian" begin
        optf_hess = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); hess = rosenbrock_hess)
        x0_trust = [0.5, 0.5]
        prob = OptimizationProblem(optf_hess, x0_trust, _p)
        for method in [ScipyDogleg(), ScipyTrustNCG(), ScipyTrustKrylov(), ScipyTrustExact()]
            sol = solve(prob, method, maxiters = 2000)
            @test sol.retcode in (ReturnCode.Success, ReturnCode.MaxIters, ReturnCode.Unstable, ReturnCode.Infeasible)
            if sol.retcode == ReturnCode.Success
                @test 10 * sol.objective < sol.original.fun
            end
        end
    end

    @testset "COBYQA method" begin
        optf_no_grad = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optf_no_grad, x0, _p)
        sol = solve(prob, ScipyCOBYQA(), maxiters = 10000)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        prob_bounded = OptimizationProblem(optf_no_grad, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
        sol = solve(prob_bounded, ScipyCOBYQA())
        @test sol.retcode == ReturnCode.Success
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        optf_cons = OptimizationFunction(rosenbrock; cons = cons)
        prob_cons = OptimizationProblem(optf_cons, [0.5, 0.5], _p, lcons = [-0.01], ucons = [0.01])
        sol = solve(prob_cons, ScipyCOBYQA())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "ScipyMinimizeScalar" begin
        f_scalar(x, p) = (x[1] - p[1])^2 + sin(x[1])
        x0_scalar = [0.0]
        p_scalar = [2.0]
        optf = OptimizationFunction(f_scalar)
        prob = OptimizationProblem(optf, x0_scalar, p_scalar)
        sol = solve(prob, ScipyBrent())
        @test sol.retcode == ReturnCode.Success
        @test length(sol.u) == 1
        @test abs(2*(sol.u[1] - p_scalar[1]) + cos(sol.u[1])) < 1e-6
        sol = solve(prob, ScipyGolden())
        @test sol.retcode == ReturnCode.Success
        @test abs(2*(sol.u[1] - p_scalar[1]) + cos(sol.u[1])) < 1e-6
        prob_bounded = OptimizationProblem(optf, x0_scalar, p_scalar, lb = [0.0], ub = [3.0])
        sol = solve(prob_bounded, ScipyBounded())
        @test sol.retcode == ReturnCode.Success
        @test 0.0 <= sol.u[1] <= 3.0
        prob_multidim = OptimizationProblem(rosenbrock, x0, _p)
        @test_throws ArgumentError solve(prob_multidim, ScipyMinimizeScalar("brent"))
        @test_throws ArgumentError solve(prob, ScipyBounded())
        optf_grad = OptimizationFunction(f_scalar, Optimization.AutoZygote())
        prob_grad = OptimizationProblem(optf_grad, x0_scalar, p_scalar)
        sol = solve(prob_grad, ScipyBrent())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "ScipyRootScalar" begin
        f_root(x, p) = x[1]^3 - 2*x[1] - 5
        x0_root = [2.0]
        optf = OptimizationFunction(f_root)
        prob_bracket = OptimizationProblem(optf, x0_root, nothing, lb = [2.0], ub = [3.0])
        sol = solve(prob_bracket, ScipyRootScalar("brentq"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        sol = solve(prob_bracket, ScipyRootScalar("brenth"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        sol = solve(prob_bracket, ScipyRootScalar("bisect"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        sol = solve(prob_bracket, ScipyRootScalar("ridder"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        prob_no_bracket = OptimizationProblem(optf, x0_root)
        sol = solve(prob_no_bracket, ScipyRootScalar("secant"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        f_root_grad(g, x, p) = g[1] = 3*x[1]^2 - 2
        optf_grad = OptimizationFunction(f_root; grad = f_root_grad)
        prob_newton = OptimizationProblem(optf_grad, x0_root)
        sol = solve(prob_newton, ScipyRootScalar("newton"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        f_root_hess(H, x, p) = H[1,1] = 6*x[1]
        optf_halley = OptimizationFunction(f_root; grad = f_root_grad, hess = f_root_hess)
        prob_halley = OptimizationProblem(optf_halley, x0_root)
        sol = solve(prob_halley, ScipyRootScalar("halley"))
        @test sol.retcode == ReturnCode.Success
        @test abs(f_root(sol.u, nothing)) < 1e-10
        prob_multidim = OptimizationProblem(rosenbrock, x0, _p)
        @test_throws ArgumentError solve(prob_multidim, ScipyRootScalar("brentq"))
        @test_throws ArgumentError solve(prob_no_bracket, ScipyRootScalar("brentq"))
    end

    @testset "ScipyRoot" begin
        function system(x, p)
            return [x[1]^2 + x[2]^2 - 1.0, x[2] - x[1]^2]
        end
        x0_system = [0.5, 0.5]
        optf = OptimizationFunction(system)
        prob = OptimizationProblem(optf, x0_system)
        sol = solve(prob, ScipyRoot("hybr"))
        @test sol.retcode == ReturnCode.Success
        res = system(sol.u, nothing)
        @test all(abs.(res) .< 1e-10)
        sol = solve(prob, ScipyRoot("lm"))
        @test sol.retcode == ReturnCode.Success
        res = system(sol.u, nothing)
        @test all(abs.(res) .< 1e-10)
        for method in ["broyden1", "broyden2", "anderson", "linearmixing", 
                      "diagbroyden", "excitingmixing", "krylov", "df-sane"]
            sol = solve(prob, ScipyRoot(method))
            @test sol.retcode in (ReturnCode.Success, ReturnCode.Failure)
            if sol.retcode == ReturnCode.Success
                res = system(sol.u, nothing)
                @test all(abs.(res) .< 1e-4)
            end
        end
    end

    @testset "ScipyLinprog" begin
        function linear_obj(x, p)
            c = [-1.0, -2.0]
            return c
        end
        x0_lp = [0.0, 0.0]
        optf = OptimizationFunction(linear_obj)
        prob = OptimizationProblem(optf, x0_lp, nothing, 
                                 lb = [0.0, 0.0], ub = [4.0, 2.0])
        for method in ["highs", "highs-ds", "highs-ipm"]
            sol = solve(prob, ScipyLinprog(method))
            @test sol.retcode in (ReturnCode.Success, ReturnCode.Failure)
            if sol.retcode == ReturnCode.Success
                @test sol.u[1] >= 0.0
                @test sol.u[2] >= 0.0
                @test sol.u[1] <= 4.0
                @test sol.u[2] <= 2.0
            end
        end
    end

    @testset "ScipyMilp" begin
        function milp_obj(x, p)
            c = [-1.0, -2.0]
            return c
        end
        x0_milp = [0.0, 0.0]
        optf = OptimizationFunction(milp_obj)
        prob = OptimizationProblem(optf, x0_milp, nothing,
                                 lb = [0.0, 0.0], ub = [4.0, 2.0])
        sol = solve(prob, ScipyMilp())
        @test sol.retcode in (ReturnCode.Success, ReturnCode.Failure, ReturnCode.Infeasible)
        if sol.retcode == ReturnCode.Success
            @test sol.u[1] >= 0.0
            @test sol.u[2] >= 0.0
            @test sol.u[1] <= 4.0
            @test sol.u[2] <= 2.0
        end
    end

    @testset "cache interface" begin
        objective(x, p) = (p[1] - x[1])^2
        x0 = zeros(1)
        p = [1.0]
        optf = OptimizationFunction(objective, Optimization.AutoZygote())
        prob = OptimizationProblem(optf, x0, p)
        cache = Optimization.init(prob, ScipyBFGS())
        sol = Optimization.solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0] atol=1e-3
        cache = Optimization.reinit!(cache; p = [2.0])
        sol = Optimization.solve!(cache)
        @test sol.u ≈ [2.0] atol=1e-3
    end

    @testset "callback" begin
        cbstopping = function (state, loss)
            return state.objective < 0.7
        end
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        @test_throws ErrorException solve(prob, ScipyBFGS(), callback = cbstopping)
    end

    @testset "constrained optimization" begin
        Random.seed!(1)
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        cons_j = (res, x, p) -> begin
            res[1,1] = 2*x[1]
            res[1,2] = 2*x[2]
        end
        x0 = zeros(2)
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = cons, cons_j = cons_j)
        prob_cobyla = OptimizationProblem(optprob, x0, _p, lcons = [-1e-6], ucons = [1e-6])
        sol = solve(prob_cobyla, ScipyCOBYLA(), maxiters = 10000)
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        Random.seed!(42)
        prob = OptimizationProblem(optprob, rand(2), _p, lcons = [0.0], ucons = [0.0])
        sol = solve(prob, ScipySLSQP())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        Random.seed!(123)
        prob = OptimizationProblem(optprob, rand(2), _p, lcons = [0.0], ucons = [0.0])
        sol = solve(prob, ScipyTrustConstr())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        function con2_c(res, x, p)
            res .= [x[1]^2 + x[2]^2 - 1.0, x[2] * sin(x[1]) - x[1] - 2.0]
        end
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = con2_c)
        Random.seed!(456)
        prob = OptimizationProblem(optprob, rand(2), _p, lcons = [0.0, -Inf], ucons = [0.0, 0.0])
        sol = solve(prob, ScipySLSQP())
        @test sol.retcode == ReturnCode.Success
        @test 10 * sol.objective < l1
        Random.seed!(789)
        prob = OptimizationProblem(optprob, [0.5, 0.5], _p, lcons = [-Inf, -Inf],
            ucons = [0.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])
        sol = solve(prob, ScipyShgo(), n = 50, iters = 1)
        @test sol.retcode == ReturnCode.Success
        @test sol.objective < l1
    end

    @testset "method-specific options" begin
        simple_optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        unconstrained_prob = OptimizationProblem(simple_optprob, x0, _p, lb = [-1.0, -1.0], ub = [1.0, 1.0])
        sol = solve(unconstrained_prob, ScipyDifferentialEvolution(), 
                   popsize = 10, mutation = (0.5, 1.0), recombination = 0.7)
        @test sol.retcode == ReturnCode.Success
        sol = solve(unconstrained_prob, ScipyBasinhopping(), T = 1.0, stepsize = 0.5, niter = 10)
        @test sol.retcode == ReturnCode.Success
        sol = solve(unconstrained_prob, ScipyDualAnnealing(), 
                   initial_temp = 5000.0, restart_temp_ratio = 2e-5)
        @test sol.retcode == ReturnCode.Success
        sol = solve(unconstrained_prob, ScipyShgo(), n = 50, sampling_method = "simplicial")
        @test sol.retcode == ReturnCode.Success
        sol = solve(unconstrained_prob, ScipyDirect(), eps = 0.001, locally_biased = true)
        @test sol.retcode == ReturnCode.Success
        sol = solve(unconstrained_prob, ScipyBrute(), Ns = 5, workers = 1)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "gradient-free methods" begin
        optf_no_grad = OptimizationFunction(rosenbrock)
        prob = OptimizationProblem(optf_no_grad, x0, _p)
        sol = solve(prob, ScipyCOBYLA(), maxiters = 10000)
        @test sol.retcode == ReturnCode.Success
        sol = solve(prob, ScipyNelderMead())
        @test sol.retcode == ReturnCode.Success
        sol = solve(prob, ScipyPowell())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "AutoDiff backends" begin
        for adtype in [Optimization.AutoZygote(), 
                      Optimization.AutoReverseDiff(), 
                      Optimization.AutoForwardDiff()]
            optf = OptimizationFunction(rosenbrock, adtype)
            prob = OptimizationProblem(optf, x0, _p)
            sol = solve(prob, ScipyBFGS())
            @test sol.retcode == ReturnCode.Success
            @test 10 * sol.objective < l1
        end
    end

    @testset "optimization stats" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyBFGS())
        @test sol.stats.time > 0
    end

    @testset "original result access" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyBFGS())
        @test !isnothing(sol.original)
        @test pyhasattr(sol.original, "success")
        @test pyhasattr(sol.original, "message")
    end

    @testset "tolerance settings" begin
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        sol = solve(prob, ScipyNelderMead(), abstol = 1e-8)
        @test sol.objective < 1e-7
        sol = solve(prob, ScipyBFGS(), reltol = 1e-8)
        @test sol.objective < 1e-7
    end

    @testset "constraint satisfaction" begin
        cons = (res, x, p) -> res .= [x[1]^2 + x[2]^2 - 1.0]
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = cons)
        prob = OptimizationProblem(optprob, [0.5, 0.5], _p, lcons = [-0.01], ucons = [0.01])
        sol = solve(prob, ScipySLSQP())
        @test sol.retcode == ReturnCode.Success
        cons_val = [0.0]
        cons(cons_val, sol.u, _p)
        @test abs(cons_val[1]) < 0.011
    end

    @testset "invalid method" begin
        @test_throws ArgumentError ScipyMinimize("InvalidMethodName")
        @test_throws ArgumentError ScipyMinimizeScalar("InvalidMethodName")
        @test_throws ArgumentError ScipyLeastSquares(method="InvalidMethodName")
        @test_throws ArgumentError ScipyLeastSquares(loss="InvalidLossName")
        @test_throws ArgumentError ScipyRootScalar("InvalidMethodName")
        @test_throws ArgumentError ScipyRoot("InvalidMethodName")
        @test_throws ArgumentError ScipyLinprog("InvalidMethodName")
    end

    @testset "Edge cases" begin
        f_simple(x, p) = (x[1] - p[1])^2
        prob = OptimizationProblem(f_simple, [0.0], [3.0])
        sol = solve(prob, ScipyBFGS())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [3.0] atol=1e-6
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0, _p)
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyDifferentialEvolution())
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyDirect())
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyDualAnnealing())
        @test_throws SciMLBase.IncompatibleOptimizerError solve(prob, ScipyBrute())
        @test_throws ArgumentError solve(prob, ScipyBrent())
        @test_throws ArgumentError solve(prob, ScipyRootScalar("brentq"))
    end

    @testset "Type stability" begin
        x0_f32 = Float32[0.0, 0.0]
        p_f32 = Float32[1.0, 100.0]
        optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
        prob = OptimizationProblem(optprob, x0_f32, p_f32)
        sol = solve(prob, ScipyBFGS())
        @test sol.retcode == ReturnCode.Success
        @test eltype(sol.u) == Float32
    end

    @testset "ScipyLinprog matrix constraints" begin
        # Minimize c^T x subject to A_ub * x <= b_ub and simple bounds
        c_vec(x, p) = [1.0, 1.0]  # constant cost vector
        x0_lp = [0.0, 0.0]
        optf_lp = OptimizationFunction(c_vec)
        prob_lp = OptimizationProblem(optf_lp, x0_lp)

        A_ub = [1.0 1.0]               # x1 + x2 <= 5
        b_ub = [5.0]
        sol = solve(prob_lp, ScipyLinprog("highs"),
                     A_ub = A_ub, b_ub = b_ub,
                     lb = [0.0, 0.0], ub = [10.0, 10.0])
        @test sol.retcode == ReturnCode.Success
        @test sol.u[1] + sol.u[2] ≤ 5.0 + 1e-8
    end

    @testset "ScipyMilp matrix constraints" begin
        # Mixed-integer LP: first variable binary, second continuous
        c_vec_milp(x, p) = [-1.0, -2.0]  # maximize -> minimize negative
        x0_milp = [0.0, 0.0]
        optf_milp = OptimizationFunction(c_vec_milp)
        prob_milp = OptimizationProblem(optf_milp, x0_milp)

        A = [1.0 1.0]                   # x1 + x2 >= 1  -> lb = 1, ub = Inf
        lb_con = [1.0]
        ub_con = [Inf]
        integrality = [1, 0]            # binary, continuous

        sol = solve(prob_milp, ScipyMilp();
                     A = A, lb_con = lb_con, ub_con = ub_con,
                     integrality = integrality,
                     lb = [0.0, 0.0], ub = [1.0, 10.0])
        @test sol.retcode in (ReturnCode.Success, ReturnCode.Failure)
        if sol.retcode == ReturnCode.Success
            @test sol.u[1] in (0.0, 1.0)
            @test isapprox(sol.u[1] + sol.u[2], 1.0; atol = 1e-6) || sol.u[1] + sol.u[2] > 1.0
        end
    end
end