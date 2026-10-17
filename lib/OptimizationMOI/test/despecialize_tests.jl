using OptimizationMOI, OptimizationBase, Ipopt, ForwardDiff, SciMLBase, Test
using SciMLBase: AutoDespecialize, DespecializedParameters, FullSpecialize

struct DespecializationTestParams
    a::Float64
    b::Float64
end

@testset "despecialization" begin
    rosenbrock(x, p) = (p.a - x[1])^2 + p.b * (x[2] - x[1]^2)^2
    circle!(res, x, p) = (res[1] = x[1]^2 + x[2]^2; nothing)
    p = DespecializationTestParams(1.0, 100.0)
    solve_with(spec; kwargs...) = solve(
        OptimizationProblem(
            OptimizationFunction{true, spec}(rosenbrock, AutoForwardDiff(); cons = circle!),
            zeros(2), p; lcons = [-Inf], ucons = [0.8]
        ),
        Ipopt.Optimizer(); print_level = 0, kwargs...
    )
    reference = solve_with(FullSpecialize)
    seen = []
    callback = (state, loss) -> (push!(seen, state.p); false)
    sol = solve_with(AutoDespecialize; callback)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ reference.u rtol = 1.0e-8
    evaluator = sol.cache.evaluator
    @test evaluator.f.f isa OptimizationBase.DespecializedFunction
    @test evaluator.p isa DespecializedParameters
    @test !isempty(seen)
    @test all(q -> q === p, seen)
end
