using OptimizationEvolutionary, Optimization, Random
using Test

Random.seed!(1234)
@testset "OptimizationEvolutionary.jl" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)
    optprob = OptimizationFunction(rosenbrock)
    prob = Optimization.OptimizationProblem(optprob, x0, _p)
    sol = solve(prob, CMAES(μ = 40, λ = 100), abstol = 1e-15)
    @test 10 * sol.objective < l1

    x0 = [-0.7, 0.3]
    prob = Optimization.OptimizationProblem(optprob, x0, _p, lb = [0.0, 0.0],
        ub = [0.5, 0.5])
    sol = solve(prob, CMAES(μ = 50, λ = 60))
    @test sol.u == zeros(2)

    x0 = zeros(2)
    cons_circ = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock; cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [0.25^2])
    sol = solve(prob, CMAES(μ = 40, λ = 100))
    res = zeros(1)
    cons_circ(res, sol.u, nothing)
    @test res[1]≈0.0625 atol=1e-5
    @test sol.objective < l1

    prob = OptimizationProblem(optprob, x0, _p, lcons = [-Inf], ucons = [5.0],
        lb = [0.0, 1.0], ub = [Inf, Inf])
    sol = solve(prob, CMAES(μ = 40, λ = 100))
    res = zeros(1)
    cons_circ(res, sol.u, nothing)
    @test sol.objective < l1

    function cb(state, args...)
        if state.iter % 10 == 0
            println(state.u)
        end
        return false
    end
    sol = solve(prob, CMAES(μ = 40, λ = 100), callback = cb, maxiters = 100)
    
    #test that `store_trace=true` works now. Threw ""type Array has no field value" before.
    solve(prob, CMAES(μ = 40, λ = 100), store_trace = true)
end
