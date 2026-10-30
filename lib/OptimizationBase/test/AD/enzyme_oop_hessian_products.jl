using OptimizationBase, Enzyme, ChainRulesCore, Test

@testset "Enzyme out-of-place Hessian and Hessian-vector products" begin
    rosen(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    ad = AutoEnzyme()
    optf = OptimizationBase.instantiate_function(
        OptimizationFunction{false}(rosen, ad), zeros(2), ad, nothing;
        g = true, fg = true, h = true, hv = true, fgh = true
    )

    for x in ([0.5, 0.7], [1.0, 1.0]), v in ([1.0, 1.0], [2.0, -1.0])
        expected_hessian = [
            2 - 400 * x[2] + 1200 * x[1]^2 -400 * x[1]
            -400 * x[1] 200
        ]
        expected_product = expected_hessian * v
        @test optf.hv(x, v) ≈ expected_product
        @test optf.hess(x) ≈ expected_hessian
        @test optf.hv(x, v) ≈ expected_product
        _, hessian = optf.fgh(x)
        @test hessian ≈ expected_hessian
        @test optf.hv(x, v) ≈ expected_product
    end
end
