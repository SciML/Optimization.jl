using Optimization, OptimizationOptimJL, OptimizationMOI, OptimizationLBFGSB
using Ipopt, Test
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker, Mooncake
using Enzyme, Random

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

function g!(G, x, p = nothing)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    return G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x, p = nothing)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    return H[2, 2] = 200.0
end

@testset "No AD" begin
    optf = OptimizationFunction(rosenbrock; grad = g!, hess = h!)

    prob = OptimizationProblem(optf, x0)
    sol = solve(prob, OptimizationLBFGSB.LBFGSB())

    @test 10 * sol.objective < l1
    @test sol.retcode == ReturnCode.Success

    sol = solve(prob, Optim.Newton())
    @test 10 * sol.objective < l1
    @test sol.retcode == ReturnCode.Success
end

@testset "No constraint" begin
    @testset "$adtype" for adtype in [
            AutoEnzyme(), AutoForwardDiff(), AutoZygote(), AutoReverseDiff(),
            AutoFiniteDiff(), AutoSymbolics(), AutoSparse(AutoForwardDiff()),
            AutoSparse(AutoReverseDiff()), AutoSparse(AutoZygote()), AutoSparse(AutoSymbolics()), AutoMooncake(),
        ]
        optf = OptimizationFunction(rosenbrock, adtype)

        prob = OptimizationProblem(optf, x0)

        sol = solve(prob, Optim.BFGS())
        @test 10 * sol.objective < l1
        if adtype != AutoFiniteDiff()
            @test sol.retcode == ReturnCode.Success
        end

        # `Newton` requires Hession, which Mooncake doesn't support at the moment.
        if adtype != AutoMooncake()
            sol = solve(prob, Optim.Newton())
            @test 10 * sol.objective < l1
            if adtype != AutoFiniteDiff()
                @test sol.retcode == ReturnCode.Success
            end
        end

        # Requires Hession, which Mooncake doesn't support at the moment.
        # Enzyme Hessian-Free seems to have an issue that is hard to track down.
        # https://github.com/SciML/Optimization.jl/issues/1030
        if adtype != AutoMooncake() && adtype != AutoEnzyme()
            sol = solve(prob, Optim.KrylovTrustRegion())
            @test 10 * sol.objective < l1
            if adtype != AutoFiniteDiff()
                @test sol.retcode == ReturnCode.Success
            end
        end

        sol = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)
        @test 10 * sol.objective < l1
        @test sol.retcode == ReturnCode.Success
    end
end

@testset "One constraint" begin
    @testset "$adtype" for adtype in [
            AutoEnzyme(), AutoForwardDiff(), AutoZygote(), AutoReverseDiff(),
            AutoFiniteDiff(), AutoSymbolics(), AutoSparse(AutoForwardDiff()),
            AutoSparse(AutoReverseDiff()), AutoSparse(AutoZygote()), AutoSparse(AutoSymbolics()), AutoMooncake(),
        ]
        cons = (res, x, p) -> (res[1] = x[1]^2 + x[2]^2 - 1.0; return nothing)
        optf = OptimizationFunction(rosenbrock, adtype, cons = cons)

        prob = OptimizationProblem(
            optf, x0, lb = [-1.0, -1.0], ub = [1.0, 1.0], lcons = [0.0], ucons = [0.0]
        )

        sol = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)
        @test 10 * sol.objective < l1

        # Requires Hession, which Mooncake doesn't support at the moment.
        if adtype != AutoMooncake()
            sol = solve(prob, Ipopt.Optimizer(), max_iter = 1000; print_level = 0)
            @test 10 * sol.objective < l1
        end
    end
end

@testset "Two constraints" begin
    @testset "$adtype" for adtype in [
            AutoForwardDiff(), AutoZygote(), AutoReverseDiff(),
            AutoFiniteDiff(), AutoSymbolics(), AutoSparse(AutoForwardDiff()),
            AutoSparse(AutoReverseDiff()), AutoSparse(AutoZygote()), AutoSparse(AutoSymbolics()), AutoMooncake(),
        ]
        function con2_c(res, x, p)
            res[1] = x[1]^2 + x[2]^2
            res[2] = x[2] * sin(x[1]) - x[1]
            return nothing
        end
        optf = OptimizationFunction(rosenbrock, adtype, cons = con2_c)

        prob = OptimizationProblem(
            optf, x0, lb = [-1.0, -1.0], ub = [1.0, 1.0],
            lcons = [1.0, -2.0], ucons = [1.0, 2.0]
        )

        sol = solve(prob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)
        @test 10 * sol.objective < l1

        # Requires Hession, which Mooncake doesn't support at the moment.
        if adtype != AutoMooncake()
            sol = solve(prob, Ipopt.Optimizer(), max_iter = 1000; print_level = 0)
            @test 10 * sol.objective < l1
        end
    end
end
