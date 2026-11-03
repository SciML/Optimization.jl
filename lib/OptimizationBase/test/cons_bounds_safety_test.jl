using OptimizationBase, Test, ADTypes, ForwardDiff, ReverseDiff
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm

@testset "zero-initialized OOP constraint buffer when cons! undersizes writes" begin
    # num_cons = 3 while cons! writes only 2 entries: the out-of-place AD wrapper
    # must not leave uninitialized memory in Jacobian row 3.
    rosen(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    cons!(res, x, p) = (res[1] = sum(abs2, x); res[2] = x[1] * x[2]; nothing)
    x = [0.5, 0.5]
    p = [1.0, 100.0]
    @testset "$(nameof(typeof(adtype)))" for adtype in [
            AutoForwardDiff(),
            AutoReverseDiff(),
            AutoSparse(
                AutoForwardDiff();
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = GreedyColoringAlgorithm()
            ),
        ]
        optf = OptimizationFunction(rosen, adtype; cons = cons!)
        fi = OptimizationBase.instantiate_function(
            optf, OptimizationBase.ReInitCache(x, p), adtype, 3;
            g = true, cons_j = true
        )
        J = fill(NaN, 3, 2)
        fi.cons_j(J, x)
        @test J[1:2, :] ≈ [1.0 1.0; 0.5 0.5]
        @test J[3, :] == [0.0, 0.0]
        @test all(isfinite, J)
    end
end
