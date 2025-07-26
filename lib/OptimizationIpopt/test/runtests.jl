using Optimization, OptimizationIpopt
using Zygote
using Symbolics
using Test
using SparseArrays
using ModelingToolkit
using ReverseDiff

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]
l1 = rosenbrock(x0, _p)

optfunc = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
prob = OptimizationProblem(optfunc, x0, _p; sense = Optimization.MaxSense)

callback = function (state, l)
    display(l)
    return false
end

sol = solve(prob, IpoptOptimizer(); callback, hessian_approximation = "exact")
@test SciMLBase.successful_retcode(sol)
@test sol ≈ [1, 1]

sol = solve(prob, IpoptOptimizer(); callback, hessian_approximation = "limited-memory")
@test SciMLBase.successful_retcode(sol)
@test sol ≈ [1, 1]

function _test_sparse_derivatives_hs071(backend, optimizer)
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
        ]
    end
    prob = OptimizationProblem(
        OptimizationFunction(objective, backend; cons = constraints),
        [1.0, 5.0, 5.0, 1.0];
        sense = Optimization.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0])
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; atol = 1e-6)
    x = [1.0, 4.7429996418092970, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.u, x; atol = 1e-6)
    @test prod(sol.u) >= 25.0 - 1e-6
    @test isapprox(sum(sol.u .^ 2), 40.0; atol = 1e-6)
    return
end

@testset "backends" begin
    backends = (
        AutoForwardDiff(),
        AutoReverseDiff(),
        AutoSparse(AutoForwardDiff())
    )
    for backend in backends
        @testset "$backend" begin
            _test_sparse_derivatives_hs071(backend, IpoptOptimizer())
        end
    end
end

@testset "MTK cache" begin
    @variables x
    @parameters a = 1.0
    @named sys = OptimizationSystem((x - a)^2, [x], [a];)
    sys = complete(sys)
    prob = OptimizationProblem(sys, [x => 0.0]; grad = true, hess = true)
    cache = init(prob, IpoptOptimizer(); verbose = false)
    @test cache isa OptimizationIpopt.IpoptCache
    sol = solve!(cache)
    @test sol.u ≈ [1.0] # ≈ [1]

    @test_broken begin # needs reinit/remake fixes
        cache = Optimization.reinit!(cache; p = [2.0])
        sol = solve!(cache)
        @test sol.u ≈ [2.0]  # ≈ [2]
    end
end

@testset "tutorial" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 1.0]

    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    function lagh(res, x, sigma, mu, p)
        lH = sigma * [2 + 8(x[1]^2) * p[2]-4(x[2] - (x[1]^2)) * p[2] -4p[2]*x[1]
              -4p[2]*x[1] 2p[2]] .+ [2mu[1] mu[2]
              mu[2] 2mu[1]]
        res .= lH[[1, 3, 4]]
    end
    lag_hess_prototype = sparse([1 1; 0 1])

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();
        cons = cons, lag_h = lagh, lag_hess_prototype)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])
    sol = solve(prob, IpoptOptimizer())

    @test SciMLBase.successful_retcode(sol)
end

# Include additional tests based on Ipopt examples
include("additional_tests.jl")
include("advanced_features.jl")
include("problem_types.jl")
