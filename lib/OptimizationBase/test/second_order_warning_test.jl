using DifferentiationInterface: SecondOrder
using ForwardDiff

struct ExactHessianOptimizer end

SciMLBase.requireshessian(::ExactHessianOptimizer) = true

@testset "explicit sparse second-order AD" begin
    objective(x, p) = sum(abs2, x)
    sparse_second_order = AutoSparse(
        SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    )
    explicit_f = OptimizationFunction(objective, sparse_second_order)
    explicit_prob = OptimizationProblem(explicit_f, [1.0, 1.0])

    @test_nowarn OptimizationCache(explicit_prob, ExactHessianOptimizer())

    implicit_f = OptimizationFunction(objective, AutoSparse(AutoForwardDiff()))
    implicit_prob = OptimizationProblem(implicit_f, [1.0, 1.0])
    @test_logs (:warn, r"missing_second_order_ad") match_mode = :any OptimizationCache(
        implicit_prob, ExactHessianOptimizer()
    )
end
