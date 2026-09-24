using OptimizationNLPModels, OptimizationBase, NLPModelsTest, Ipopt, OptimizationMOI, Zygote,
    ReverseDiff, OptimizationLBFGSB, OptimizationOptimJL
using Test

@testset "NLPModels" begin
    # First problem: Problem 5 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS5
    # Problem with box bounds
    hs5f(u, p) = sin(u[1] + u[2]) + (u[1] - u[2])^2 - (3 / 2) * u[1] + (5 / 2)u[2] + 1
    f = OptimizationBase.OptimizationFunction(hs5f, OptimizationBase.AutoZygote())
    lb = [-1.5; -3]
    ub = [4.0; 3.0]
    u0 = [0.0; 0.0]
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lb = lb, ub = ub, sense = OptimizationBase.MinSense
    )

    nlpmo = NLPModelsTest.HS5()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, OptimizationBase.AutoZygote())

    sol_native = solve(oprob, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)
    sol_converted = solve(converted, OptimizationLBFGSB.LBFGSB(), maxiters = 1000)

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Second problem: Brown and Dennis function
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.BROWNDEN
    # Problem without bounds
    function brown_dennis(u, p)
        return sum([((u[1] + (i / 5) * u[2] - exp(i / 5))^2 + (u[3] + sin(i / 5) * u[4] - cos(i / 5))^2)^2 for i in 1:20])
    end
    f = OptimizationBase.OptimizationFunction(brown_dennis, OptimizationBase.AutoZygote())
    u0 = [25.0; 5.0; -5.0; -1.0]
    oprob = OptimizationBase.OptimizationProblem(f, u0, sense = OptimizationBase.MinSense)

    nlpmo = NLPModelsTest.BROWNDEN()
    converted = OptimizationNLPModels.OptimizationProblem(nlpmo, OptimizationBase.AutoZygote())

    sol_native = solve(oprob, OptimizationOptimJL.Optim.BFGS())
    sol_converted = solve(converted, OptimizationOptimJL.Optim.BFGS())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Third problem: Problem 10 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS10
    # Problem with inequality bounds
    hs10(u, p) = u[1] - u[2]
    hs10_cons(res, u, p) = (res .= -3.0 * u[1]^2 + 2.0 * u[1] * u[2] - u[2]^2 + 1.0)
    lcons = [0.0]
    ucons = [Inf]
    u0 = [-10.0; 10.0]
    f = OptimizationBase.OptimizationFunction(
        hs10, OptimizationBase.AutoForwardDiff(); cons = hs10_cons
    )
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lcons = lcons, ucons = ucons, sense = OptimizationBase.MinSense
    )

    nlpmo = NLPModelsTest.HS10()
    converted = OptimizationNLPModels.OptimizationProblem(
        nlpmo, OptimizationBase.AutoForwardDiff()
    )

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Fourth problem: Problem 13 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS13
    # Problem with box & inequality bounds
    hs13(u, p) = (u[1] - 2.0)^2 + u[2]^2
    hs13_cons(res, u, p) = (res .= (1.0 - u[1])^3 - u[2])
    lcons = [0.0]
    ucons = [Inf]
    lb = [0.0; 0.0]
    ub = [Inf; Inf]
    u0 = [-2.0; -2.0]
    f = OptimizationBase.OptimizationFunction(
        hs13, OptimizationBase.AutoForwardDiff(); cons = hs13_cons
    )
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lb = lb, ub = ub, lcons = lcons,
        ucons = ucons, sense = OptimizationBase.MinSense
    )

    nlpmo = NLPModelsTest.HS13()
    converted = OptimizationNLPModels.OptimizationProblem(
        nlpmo, OptimizationBase.AutoForwardDiff()
    )

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u

    # Fifth problem: Problem 14 in the Hock-Schittkowski suite
    # https://jso.dev/NLPModelsTest.jl/dev/reference/#NLPModelsTest.HS14
    # Problem with mixed equality & inequality constraints
    hs14(u, p) = (u[1] - 2.0)^2 + (u[2] - 1.0)^2
    hs14_cons(res, u, p) = (
        res .= [
            u[1] - 2.0 * u[2];
            -0.25 * u[1]^2 - u[2]^2 + 1.0
        ]
    )
    lcons = [-1.0; 0.0]
    ucons = [-1.0; Inf]
    u0 = [2.0; 2.0]
    f = OptimizationBase.OptimizationFunction(
        hs14, OptimizationBase.AutoForwardDiff(); cons = hs14_cons
    )
    oprob = OptimizationBase.OptimizationProblem(
        f, u0, lcons = lcons, ucons = ucons, sense = OptimizationBase.MinSense
    )

    nlpmo = NLPModelsTest.HS14()
    converted = OptimizationNLPModels.OptimizationProblem(
        nlpmo, OptimizationBase.AutoForwardDiff()
    )

    sol_native = solve(oprob, Ipopt.Optimizer())
    sol_converted = solve(converted, Ipopt.Optimizer())

    @test sol_converted.retcode == sol_native.retcode
    @test sol_converted.u ≈ sol_native.u
end

@testset "sparse lag_h structure/value ordering (canonical contract)" begin
    # Regression test for the COO ordering bug: the adaptor declared the
    # lower triangle in CSC order while OptimizationBase's vector lag_h
    # writes the upper triangle in CSC order — a different enumeration for
    # any pattern with >= 3 coupled columns, so every off-diagonal value
    # landed on a wrong index. A 2x2 pattern CANNOT catch this (the two
    # enumerations coincide by symmetry); this problem has three coupled
    # variables on purpose.
    using SparseArrays
    import ADTypes
    import ForwardDiff
    import DifferentiationInterface

    obj(x, p) = (1 - x[1])^2 + 10.0 * (x[2] - x[1]^2)^2 + (x[3] - x[2])^2
    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[3]])
    x0 = [0.4, -0.7, 1.3]
    σ = 0.7
    λ = [1.3, -2.1]

    sparse_ad = ADTypes.AutoSparse(
        DifferentiationInterface.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff())
    )
    dense_ad = DifferentiationInterface.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff())

    f = OptimizationBase.OptimizationFunction(obj, sparse_ad; cons = cons)
    inst = OptimizationBase.instantiate_function(
        f, x0, sparse_ad, nothing, 2; lag_h = true, cons_j = true
    )

    fd = OptimizationBase.OptimizationFunction(obj, dense_ad; cons = cons)
    inst_dense = OptimizationBase.instantiate_function(
        fd, x0, dense_ad, nothing, 2; lag_h = true
    )

    # Independent dense reference Lagrangian Hessian
    Href = zeros(3, 3)
    inst_dense.lag_h(Href, x0, σ, λ)

    proto = inst.lag_hess_prototype
    @test proto isa SparseMatrixCSC

    # 1. The canonical helper matches the order the vector lag_h writes.
    rows, cols = OptimizationBase.lag_hess_structure(proto)
    h = zeros(length(rows))
    inst.lag_h(h, x0, σ, λ)
    for k in eachindex(h)
        @test h[k] ≈ Href[rows[k], cols[k]] atol = 1.0e-10
    end

    # 2. Guard the test's power: on this pattern the canonical (mirrored)
    # enumeration must differ from the naive lower-triangle CSC enumeration
    # that caused the bug — otherwise this test could not detect it.
    I_, J_, _ = findnz(proto)
    lower_mask = I_ .>= J_
    @test !(cols == I_[lower_mask] && rows == J_[lower_mask])

    # 3. Adaptor end-to-end: declared structure + coord values agree with the
    # dense reference entry-wise.
    meta = NLPModels.NLPModelMeta(
        3; ncon = 2, x0 = x0, y0 = zeros(2),
        lcon = [1.0, 0.5], ucon = [1.0, Inf],
        nnzj = nnz(inst.cons_jac_prototype), nnzh = length(rows), minimize = true
    )
    cache_like = (f = inst, p = nothing, lcons = [1.0, 0.5], ucons = [1.0, Inf])
    nlp = OptimizationNLPModels.NLPModelsAdaptor(cache_like, meta, NLPModels.Counters())

    srows = zeros(Int, length(rows))
    scols = zeros(Int, length(rows))
    NLPModels.hess_structure!(nlp, srows, scols)
    # NLPModels convention: lower-triangle coordinates (mirrored canonical)
    @test all(srows .>= scols)
    @test srows == cols && scols == rows

    vals = zeros(length(rows))
    NLPModels.hess_coord!(nlp, x0, λ, vals; obj_weight = σ)
    for k in eachindex(vals)
        @test vals[k] ≈ Href[srows[k], scols[k]] atol = 1.0e-10
    end
end
