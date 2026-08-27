using ChainRulesCore, Enzyme, OptimizationBase, Test

function check_lagrangian_hessian(N)
    h = 1 / N
    alpha = 350
    x_offset = N + 1
    u_offset = 2(N + 1)
    function objective(x, p)
        return sum(
            0.5 * h * (x[u_offset + i + 1]^2 + x[u_offset + i]^2) +
                0.5 * alpha * h * (cos(x[i + 1]) + cos(x[i])) for i in 1:N
        ) + x[1] * x[2]
    end
    function constraint!(res, x, p)
        for i in 1:N
            res[i] = x[x_offset + i + 1] - x[x_offset + i] -
                0.5 * h * (sin(x[i + 1]) + sin(x[i]))
            res[N + i] = x[i + 1] - x[i] -
                0.5 * h * (x[u_offset + i + 1] + x[u_offset + i])
        end
        return nothing
    end

    x = zeros(3(N + 1))
    f = OptimizationFunction(objective, AutoEnzyme(); cons = constraint!)
    instantiated = OptimizationBase.instantiate_function(
        f, x, AutoEnzyme(), nothing, 2N; lag_h = true
    )
    multipliers = ones(2N)
    expected = zeros(length(x), length(x))
    for i in 1:(N + 1)
        expected[i, i] = (i == 1 || i == N + 1) ? -0.5alpha * h : -alpha * h
        expected[u_offset + i, u_offset + i] =
            (i == 1 || i == N + 1) ? h : 2h
    end
    expected[1, 2] = expected[2, 1] = 1

    packed = zeros(length(x) * (length(x) + 1) ÷ 2)
    instantiated.lag_h(packed, x, 1.0, multipliers)
    @test packed ≈ [expected[i, j] for i in axes(expected, 1) for j in 1:i]

    dense = zeros(length(x), length(x))
    instantiated.lag_h(dense, x, 1.0, multipliers)
    @test dense ≈ expected
    return nothing
end

@testset "Enzyme Lagrangian Hessian" begin
    @testset "N = $N" for N in (1:10..., 20, 40, 60)
        check_lagrangian_hessian(N)
    end
end
