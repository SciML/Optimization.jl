using OptimizationBase, LinearAlgebra, ForwardDiff, Zygote, FiniteDiff,
      DifferentiationInterface, SparseConnectivityTracer
using Test, ReverseDiff

@testset "Matrix Valued" begin
    for adtype in [AutoForwardDiff(), SecondOrder(AutoForwardDiff(), AutoZygote()),
        SecondOrder(AutoForwardDiff(), AutoFiniteDiff()),
        AutoSparse(AutoForwardDiff(), sparsity_detector = TracerLocalSparsityDetector()),
        AutoSparse(SecondOrder(AutoForwardDiff(), AutoZygote()),
            sparsity_detector = TracerLocalSparsityDetector()),
        AutoSparse(SecondOrder(AutoForwardDiff(), AutoFiniteDiff()),
            sparsity_detector = TracerLocalSparsityDetector())]
        # 1. Matrix Factorization
        @show adtype
        function matrix_factorization_objective(X, A)
            U, V = @view(X[1:size(A, 1), 1:Int(size(A, 2) / 2)]),
            @view(X[1:size(A, 1), (Int(size(A, 2) / 2) + 1):size(A, 2)])
            return norm(A - U * V')
        end
        function non_negative_constraint(X, A)
            U, V = X
            return [all(U .>= 0) && all(V .>= 0)]
        end
        A_mf = rand(4, 4)  # Original matrix
        U_mf = rand(4, 2)  # Factor matrix U
        V_mf = rand(4, 2)  # Factor matrix V

        optf = OptimizationFunction{false}(
            matrix_factorization_objective, adtype, cons = non_negative_constraint)
        optf = OptimizationBase.instantiate_function(
            optf, hcat(U_mf, V_mf), adtype, A_mf, g = true, h = true,
            cons_j = true, cons_h = true)
        optf.grad(hcat(U_mf, V_mf))
        optf.hess(hcat(U_mf, V_mf))
        if !(adtype isa ADTypes.AutoSparse)
            optf.cons_j(hcat(U_mf, V_mf))
            optf.cons_h(hcat(U_mf, V_mf))
        end

        # 2. Principal Component Analysis (PCA)
        function pca_objective(X, A)
            return -tr(X' * A * X)  # Minimize the negative of the trace for maximization
        end
        function orthogonality_constraint(X, A)
            return [norm(X' * X - I) < 1e-6]
        end
        A_pca = rand(4, 4)  # Covariance matrix (can be symmetric positive definite)
        X_pca = rand(4, 2)  # Matrix to hold principal components

        optf = OptimizationFunction{false}(
            pca_objective, adtype, cons = orthogonality_constraint)
        optf = OptimizationBase.instantiate_function(
            optf, X_pca, adtype, A_pca, g = true, h = true,
            cons_j = true, cons_h = true)
        optf.grad(X_pca)
        optf.hess(X_pca)
        if !(adtype isa ADTypes.AutoSparse)
            optf.cons_j(X_pca)
            optf.cons_h(X_pca)
        end

        # 3. Matrix Completion
        function matrix_completion_objective(X, P)
            A, Omega = P
            return norm(Omega .* (A - X))
        end
        # r = 2  # Rank of the matrix to be completed
        # function rank_constraint(X, P)
        #     return [rank(X) <= r]
        # end
        A_mc = rand(4, 4)  # Original matrix with missing entries
        Omega_mc = rand(4, 4) .> 0.5  # Mask for observed entries (boolean matrix)
        X_mc = rand(4, 4)  # Matrix to be completed
        optf = OptimizationFunction{false}(
            matrix_completion_objective, adtype)
        optf = OptimizationBase.instantiate_function(
            optf, X_mc, adtype, (A_mc, Omega_mc), g = true, h = true)
        optf.grad(X_mc)
        optf.hess(X_mc)
    end
end
