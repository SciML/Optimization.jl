using OptimizationBase
using SciMLBase: NoAD, requireshessian
using Test

struct ExactHessianOptimizer end
requireshessian(::ExactHessianOptimizer) = true

@testset "missing_second_order_ad with analytical Hessian" begin
    objective(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    function objective_grad!(G, x, p)
        G[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
        G[2] = 200 * (x[2] - x[1]^2)
        return nothing
    end
    function objective_hess!(H, x, p)
        H[1, 1] = 2 - 400 * (x[2] - 3 * x[1]^2)
        H[1, 2] = -400 * x[1]
        H[2, 1] = -400 * x[1]
        H[2, 2] = 200
        return nothing
    end

    f_analytical = OptimizationFunction(
        objective, NoAD();
        grad = objective_grad!, hess = objective_hess!
    )
    prob_analytical = OptimizationProblem(f_analytical, zeros(2))
    @test_nowarn OptimizationCache(prob_analytical, ExactHessianOptimizer())

    f_missing = OptimizationFunction(objective, NoAD())
    prob_missing = OptimizationProblem(f_missing, zeros(2))
    @test_logs (:warn, r"missing_second_order_ad") match_mode = :any OptimizationCache(
        prob_missing, ExactHessianOptimizer()
    )
end
