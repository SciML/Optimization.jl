using DifferentiationInterface: SecondOrder
using ForwardDiff
using SciMLBase
using LinearAlgebra
using Zygote

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

struct ExactConstraintHessianOptimizer end
struct ExactLagrangianHessianOptimizer end
struct ExactObjectiveAndConstraintHessianOptimizer end

SciMLBase.requiresconshess(::ExactConstraintHessianOptimizer) = true
SciMLBase.requireslagh(::ExactLagrangianHessianOptimizer) = true
SciMLBase.requireshessian(::ExactObjectiveAndConstraintHessianOptimizer) = true
SciMLBase.requiresconshess(::ExactObjectiveAndConstraintHessianOptimizer) = true

@testset "analytical second-order derivatives" begin
    objective(x, p) = sum(abs2, x)
    hess!(H, x, p) = (H .= 2 .* Matrix{eltype(x)}(I, length(x), length(x)))
    cons!(c, x, p) = (c[1] = sum(abs2, x))
    cons_h!(H, x, p) = hess!(H[1], x, p)
    function lag_h!(H, x, sigma, mu, p)
        hess!(H, x, p)
        return H .*= sigma + mu[1]
    end

    for (opt, derivatives) in (
            (ExactHessianOptimizer(), (; hess = hess!)),
            (ExactConstraintHessianOptimizer(), (; cons = cons!, cons_h = cons_h!)),
            (ExactLagrangianHessianOptimizer(), (; cons = cons!, lag_h = lag_h!)),
            (
                ExactObjectiveAndConstraintHessianOptimizer(),
                (; hess = hess!, cons = cons!, cons_h = cons_h!),
            ),
        )
        f = OptimizationFunction(objective; derivatives...)
        prob = OptimizationProblem(f, [1.0, 1.0])
        @test_logs OptimizationCache(prob, opt)
    end

    for adtype in (AutoForwardDiff(), AutoZygote())
        f = OptimizationFunction(objective, adtype; hess = hess!)
        prob = OptimizationProblem(f, [1.0, 1.0])
        @test_logs OptimizationCache(prob, ExactHessianOptimizer())
    end

    for opt in (
            ExactHessianOptimizer(), ExactConstraintHessianOptimizer(),
            ExactLagrangianHessianOptimizer(),
        )
        prob = OptimizationProblem(OptimizationFunction(objective), [1.0, 1.0])
        @test_logs (:warn, r"missing_second_order_ad") OptimizationCache(prob, opt)
    end

    f = OptimizationFunction(objective; hess = hess!, cons = cons!)
    prob = OptimizationProblem(f, [1.0, 1.0])
    @test_logs (:warn, r"missing_second_order_ad") OptimizationCache(
        prob, ExactObjectiveAndConstraintHessianOptimizer()
    )
end
