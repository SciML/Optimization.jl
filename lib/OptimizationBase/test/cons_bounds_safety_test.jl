using OptimizationBase, Test, ADTypes, ForwardDiff

@testset "zero-initialized OOP constraint buffer when cons! undersizes writes" begin
    # num_cons comes from length(ucons). If cons! writes fewer entries, the AD
    # out-of-place wrapper must not leave uninitialized memory in the Jacobian.
    rosen(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    cons!(res, x, p) = (res[1] = sum(abs2, x); res[2] = x[1] * x[2]; nothing)
    optf = OptimizationFunction(rosen, AutoForwardDiff(); cons = cons!)
    x = [0.5, 0.5]
    p = [1.0, 100.0]
    fi = OptimizationBase.instantiate_function(
        optf, OptimizationBase.ReInitCache(x, p), AutoForwardDiff(), 3;
        g = true, cons_j = true
    )
    J = fill(NaN, 3, 2)
    fi.cons_j(J, x)
    @test J[1:2, :] ≈ [1.0 1.0; 0.5 0.5]
    @test J[3, :] == [0.0, 0.0]
    @test all(isfinite, J)
end
