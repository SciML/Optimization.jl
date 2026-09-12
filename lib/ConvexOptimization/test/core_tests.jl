using ConvexOptimization
using SciMLBase
using SciMLBase: ConvexOptimizationProblem, OptimizationSolution
import MathOptInterface as MOI
import Clarabel
using LinearAlgebra
using Test

# minimize  x1 + 2 x2   s.t.  x1 + x2 == 1,  x >= 0
# analytic optimum: x* = (1, 0), obj = 1
# LP duals: equality multiplier y = 1; nonneg-cone dual s = c - Aᵀy = (0, 1) >= 0
c = [1.0, 2.0]

@testset "LP solve: primal, objective, retcode" begin
    optf = OptimizationFunction((u, p) -> c[1] * u[1] + c[2] * u[2])
    cons = [
        ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1)),        # sum(x) == 1
        ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2)),        # x >= 0
    ]
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5]; constraints = cons)

    sol = solve(prob, ConvexMOI(Clarabel.Optimizer))

    @test sol isa OptimizationSolution
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [1.0, 0.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, 1.0; atol = 1.0e-6)

    @testset "dual: one entry per constraint, in user variables" begin
        @test sol.dual !== nothing
        @test length(sol.dual) == 2
        @test isapprox(only(sol.dual[1]), 1.0; atol = 1.0e-6)      # equality multiplier
        @test isapprox(sol.dual[2], [0.0, 1.0]; atol = 1.0e-6)     # nonneg-cone dual
    end

    @testset "cache carries the SciMLBase glue fields" begin
        @test sol.cache isa SciMLBase.AbstractOptimizationCache
        @test sol.cache.p isa SciMLBase.NullParameters
        @test sol.cache.u0 == [0.5, 0.5]
    end
end

@testset "non-convex objective is rejected, not solved incorrectly" begin
    # x1*x2 is neither convex nor concave -> certification must error.
    optf = OptimizationFunction((u, p) -> u[1] * u[2])
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5])
    @test_throws Exception solve(prob, ConvexMOI(Clarabel.Optimizer))
end

@testset "SOC constraint lowers and solves" begin
    # minimize t  s.t.  || (x1, x2) ||_2 <= t,  x == (3, 4)  ->  t* = 5
    optf = OptimizationFunction((u, p) -> u[3])   # u = (x1, x2, t)
    cons = [
        ConeConstraint((u, p) -> [u[1] - 3.0, u[2] - 4.0], MOI.Zeros(2)),   # x fixed
        ConeConstraint((u, p) -> [u[3], u[1], u[2]], MOI.SecondOrderCone(3)), # (t, x) in SOC
    ]
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0, 0.0]; constraints = cons)
    sol = solve(prob, ConvexMOI(Clarabel.Optimizer))
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.objective, 5.0; atol = 1.0e-5)   # ||(3,4)|| = 5
    @test isapprox(sol.u[3], 5.0; atol = 1.0e-5)
end

# Cross-check the LP primal AND dual against Convex.jl (same solver). The exit
# gate for the vertical slice: matching Convex.jl on both, to solver tolerance.
@testset "matches Convex.jl (primal + dual)" begin
    convex_available = try
        @eval import Convex
        true
    catch
        @info "Convex.jl not available in this environment; skipping cross-check " *
            "(the analytic assertions above already pin the same values)."
        false
    end
    if convex_available
        xc = Convex.Variable(2)
        pc = Convex.minimize(c[1] * xc[1] + c[2] * xc[2], [sum(xc) == 1, xc >= 0])
        Convex.solve!(pc, Clarabel.Optimizer; silent = true)
        @test isapprox(Convex.evaluate(xc), [1.0, 0.0]; atol = 1.0e-6)
        @test isapprox(pc.optval, 1.0; atol = 1.0e-6)
    end
end

# minimize ||A*u - b||_2 with A = [1;1], b = [0;2]
# analytic optimum: u* = 1, residual (1,-1), objective sqrt(2).
# The atom is lowered through its epigraph, so the user sees no epigraph variable:
# sol.u has length 1 and sol.dual stays 1:1 with the (here empty) user constraints.
@testset "norm objective is lowered through its epigraph" begin
    A = reshape([1.0, 1.0], 2, 1)
    b = [0.0, 2.0]
    optf = OptimizationFunction((u, p) -> norm(A * u - b, 2))
    prob = ConvexOptimizationProblem(optf, [0.0])

    sol = solve(prob, ConvexMOI(Clarabel.Optimizer))

    @test SciMLBase.successful_retcode(sol.retcode)
    @test length(sol.u) == 1                       # epigraph variable is internal
    @test isapprox(sol.u[1], 1.0; atol = 1.0e-6)
    @test isapprox(sol.objective, sqrt(2); atol = 1.0e-6)
end

@testset "norm objective with constraints keeps duals 1:1 with user constraints" begin
    # minimize ||u - [1,1]||_2  s.t.  sum(u) == 1, u >= 0
    optf = OptimizationFunction((u, p) -> norm(u .- 1.0, 2))
    cons = [
        ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1)),
        ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2)),
    ]
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5]; constraints = cons)

    sol = solve(prob, ConvexMOI(Clarabel.Optimizer))

    @test SciMLBase.successful_retcode(sol.retcode)
    @test length(sol.u) == 2
    # projection of [1,1] onto the simplex sum(u)==1, u>=0 is (1/2, 1/2)
    @test isapprox(sol.u, [0.5, 0.5]; atol = 1.0e-6)
    @test isapprox(sol.objective, norm([0.5, 0.5] .- 1.0, 2); atol = 1.0e-6)
    # one dual entry per user constraint, in order — the epigraph cone is not among them
    @test sol.dual !== nothing
    @test length(sol.dual) == 2
    @test length(sol.dual[1]) == 1
    @test length(sol.dual[2]) == 2
end

@testset "invalid epigraph lowering is rejected, not silently solved" begin
    A = reshape([1.0, 1.0], 2, 1)
    b = [0.0, 2.0]

    # maximizing a norm is not concave: the epigraph bounds the atom from above only.
    optf = OptimizationFunction((u, p) -> norm(A * u - b, 2))
    prob = ConvexOptimizationProblem(optf, [0.0]; sense = SciMLBase.MaxSense)
    @test_throws Exception solve(prob, ConvexMOI(Clarabel.Optimizer))

    # a norm entering with a negative coefficient is concave, not convex.
    optf2 = OptimizationFunction((u, p) -> -norm(A * u - b, 2))
    prob2 = ConvexOptimizationProblem(optf2, [0.0])
    @test_throws Exception solve(prob2, ConvexMOI(Clarabel.Optimizer))

    # p = 3 has no corresponding MOI cone (1, 2 and Inf do).
    optf3 = OptimizationFunction((u, p) -> norm(A * u - b, 3))
    prob3 = ConvexOptimizationProblem(optf3, [0.0])
    @test_throws Exception solve(prob3, ConvexMOI(Clarabel.Optimizer))

    # a non-affine norm argument cannot be lowered.
    optf4 = OptimizationFunction((u, p) -> norm(u .^ 2 .- 1.0, 2))
    prob4 = ConvexOptimizationProblem(optf4, [0.5, 0.5])
    @test_throws Exception solve(prob4, ConvexMOI(Clarabel.Optimizer))
end

# l1 and linf norms lower to NormOneCone / NormInfinityCone, which share the
# (tau, w...) row layout of the second-order cone.
# Both minimize ||u - 1||_p subject to sum(u) == 1. Substituting u = (t, 1-t):
#   p = 1:   |t-1| + |t|          -> 1   for any t in [0, 1]
#   p = Inf: max(|t-1|, |t|)      -> 1/2 at t = 1/2
@testset "l1 and linf norm objectives lower to their cones" begin
    cons = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))]

    optf1 = OptimizationFunction((u, p) -> norm(u .- 1.0, 1))
    prob1 = ConvexOptimizationProblem(optf1, [0.5, 0.5]; constraints = cons)
    sol1 = solve(prob1, ConvexMOI(Clarabel.Optimizer))
    @test SciMLBase.successful_retcode(sol1.retcode)
    @test isapprox(sol1.objective, 1.0; atol = 1.0e-6)
    @test isapprox(sum(sol1.u), 1.0; atol = 1.0e-6)
    @test length(sol1.dual) == 1        # epigraph cone stays out of the user duals

    optfi = OptimizationFunction((u, p) -> norm(u .- 1.0, Inf))
    probi = ConvexOptimizationProblem(optfi, [0.5, 0.5]; constraints = cons)
    soli = solve(probi, ConvexMOI(Clarabel.Optimizer))
    @test SciMLBase.successful_retcode(soli.retcode)
    @test isapprox(soli.objective, 0.5; atol = 1.0e-6)
    @test isapprox(soli.u, [0.5, 0.5]; atol = 1.0e-6)
end

# exp and log lower to the exponential cone. exp is convex so it is bounded
# above (epigraph); log is concave so it is bounded below (hypograph) and must
# enter the objective negatively.
@testset "exp and log objectives lower to the exponential cone" begin
    # minimize exp(u1) + u2  s.t.  u1 == 1, u2 == 0  ->  e
    optfe = OptimizationFunction((u, p) -> exp(u[1]) + u[2])
    conse = [ConeConstraint((u, p) -> [u[1] - 1.0, u[2]], MOI.Zeros(2))]
    sole = solve(
        ConvexOptimizationProblem(optfe, [0.0, 0.0]; constraints = conse),
        ConvexMOI(Clarabel.Optimizer)
    )
    @test SciMLBase.successful_retcode(sole.retcode)
    @test isapprox(sole.objective, exp(1); atol = 1.0e-6)

    # log barrier: minimize -log(u1) s.t. u1 == 2  ->  -log(2)
    optfl = OptimizationFunction((u, p) -> -log(u[1]))
    consl = [ConeConstraint((u, p) -> [u[1] - 2.0], MOI.Zeros(1))]
    soll = solve(
        ConvexOptimizationProblem(optfl, [1.0]; constraints = consl),
        ConvexMOI(Clarabel.Optimizer)
    )
    @test SciMLBase.successful_retcode(soll.retcode)
    @test isapprox(soll.objective, -log(2); atol = 1.0e-6)
    @test length(soll.dual) == 1        # epigraph cone stays out of the user duals

    # a bare `log` objective is concave: minimizing it is not a convex program.
    optfb = OptimizationFunction((u, p) -> log(u[1]))
    @test_throws Exception solve(
        ConvexOptimizationProblem(optfb, [1.0]; constraints = consl),
        ConvexMOI(Clarabel.Optimizer)
    )
end
using ConvexOptimization
using SciMLBase
using SciMLBase: ConvexOptimizationProblem, OptimizationSolution
import MathOptInterface as MOI
import Clarabel
using LinearAlgebra
using Test

const ALG = ConvexMOI(Clarabel.Optimizer)

# minimize p1*u1 + u2  s.t.  u1 + u2 == 1 + p2,  u >= 0.
# With s = 1 + p2 >= 0 the optimum is a vertex of the scaled simplex:
#   p1 < 1: u* = (s, 0), obj = p1*s;   p1 > 1: u* = (0, s), obj = s.
# LP duals: equality multiplier y = min(p1, 1); nonneg-cone dual = (p1 - y, 1 - y).
function lp_analytic(θ)
    s = 1 + θ[2]
    return θ[1] < 1 ? ([s, 0.0], θ[1] * s, [θ[1]], [0.0, 1 - θ[1]]) :
        ([0.0, s], s, [1.0], [θ[1] - 1, 0.0])
end
lp_cons() = [
    ConeConstraint((u, p) -> [u[1] + u[2] - 1.0 - p[2]], MOI.Zeros(1)),
    ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2)),
]
const LP_SWEEP = ([0.5, 0.0], [2.0, 1.0], [-1.0, 0.5], [0.0, 3.0], [0.5, 0.0])

@testset "parametric LP: reinit! matches the analytic optimum and a cold solve" begin
    optf = OptimizationFunction((u, p) -> p[1] * u[1] + u[2])
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5], [0.5, 0.0]; constraints = lp_cons())

    cache = init(prob, ALG)
    for θ in LP_SWEEP
        cache = reinit!(cache; p = θ)
        sol = solve!(cache)
        u★, obj★, d1★, d2★ = lp_analytic(θ)

        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, u★; atol = 1.0e-6)
        @test isapprox(sol.objective, obj★; atol = 1.0e-6)
        @test isapprox(sol.dual[1], d1★; atol = 1.0e-6)
        @test isapprox(sol.dual[2], d2★; atol = 1.0e-6)
        @test SciMLBase.parameter_values(cache) == θ

        # the fast path must agree with re-canonicalizing from scratch
        cold = solve(SciMLBase.remake(prob; p = θ), ALG)
        @test isapprox(sol.u, cold.u; atol = 1.0e-8)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
        @test all(isapprox.(sol.dual, cold.dual; atol = 1.0e-8))

        # the 1:1 dual contract survives every re-solve
        @test length(sol.dual) == length(prob.constraints)
        @test length.(sol.dual) == [MOI.dimension(c.set) for c in prob.constraints]
        @test sol.dual isa Vector{Vector{Float64}}
    end
end

@testset "re-solving at a new p does no symbolic work" begin
    # `prob.f.f` and each `con.g` are evaluated exactly once per `_trace_problem`,
    # and every symbolic stage (epigraph lowering, certification, the
    # parameter-affine extraction) is downstream of that one trace. So counting
    # traces counts symbolic work.
    ntrace = Ref(0)
    optf = OptimizationFunction((u, p) -> (ntrace[] += 1; p[1] * u[1] + u[2]))
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5], [0.5, 0.0]; constraints = lp_cons())

    cache = init(prob, ALG)
    @test ntrace[] == 1
    dpp = cache.dpp

    for θ in LP_SWEEP
        cache = reinit!(cache; p = θ)
        solve!(cache)
    end
    @test ntrace[] == 1                 # no re-trace
    @test cache.dpp === dpp             # no re-extraction: the same tensors throughout
    @test cache.analysis === nothing    # parametric path is certified structurally
end

@testset "reinit! is specialized; the generic SciMLBase method must not run" begin
    # SciMLBase's generic `reinit!(::AbstractOptimizationCache)` only writes
    # `cache.reinit_cache.p`; it would leave the MOI model stale and return the
    # previous solution labelled with the new p. Two independent guards: the cache
    # deliberately has no `reinit_cache` field, and dispatch resolves here.
    optf = OptimizationFunction((u, p) -> p[1] * u[1] + u[2])
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5], [0.5, 0.0]; constraints = lp_cons())
    cache = init(prob, ALG)
    @test !SciMLBase.has_reinit(cache)
    @test !hasfield(typeof(cache), :reinit_cache)
    @test which(SciMLBase.reinit!, Tuple{typeof(cache)}).module === ConvexOptimization
end

@testset "an atom scaled by a parameter is admitted, and rejected only where invalid" begin
    # min ||A u - b||_2 + p1 * ||u||_1 — the lasso path, the canonical DPP problem.
    # The epigraph coefficient of the second atom IS p1, so the monotonicity guard's
    # verdict is p-dependent: it must be re-checked at every p, not once at init.
    A = [1.0 0.5; 0.2 1.0; 0.7 0.7]
    b = [1.0, 2.0, 0.5]
    optf = OptimizationFunction((u, p) -> norm(A * u - b, 2) + p[1] * norm(u, 1))
    prob = ConvexOptimizationProblem(
        optf, [0.0, 0.0], [0.0]; lb = [-10.0, -10.0], ub = [10.0, 10.0]
    )

    cache = init(prob, ALG)
    for λ in (0.0, 0.25, 0.5, 2.0, 10.0)
        cache = reinit!(cache; p = [λ])
        sol = solve!(cache)
        cold = solve(SciMLBase.remake(prob; p = [λ]), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, cold.u; atol = 1.0e-7)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-7)
    end
    # large λ shrinks u to 0; the objective is then ||b||
    cache = reinit!(cache; p = [10.0])
    @test isapprox(solve!(cache).u, [0.0, 0.0]; atol = 1.0e-6)

    # λ < 0 makes -|u| enter a minimization: the epigraph lowering is invalid there.
    @test_throws "Lowering an atom through its epigraph" reinit!(cache; p = [-1.0])
    # a rejected p must leave the cache untouched and usable
    @test cache.p == [10.0]
    @test isapprox(solve!(cache).u, [0.0, 0.0]; atol = 1.0e-6)
end

@testset "a parameter inside an atom argument is a θ-affine cone constant" begin
    # min ||u - p||_2 s.t. sum(u) == 1  ->  u* = projection of p onto the hyperplane
    optf = OptimizationFunction((u, p) -> norm(u .- p, 2))
    cons = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))]
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0], [1.0, 1.0]; constraints = cons)

    cache = init(prob, ALG)
    for θ in ([1.0, 1.0], [3.0, -1.0], [0.0, 0.0])
        cache = reinit!(cache; p = θ)
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, θ .+ (1 - sum(θ)) / 2; atol = 1.0e-6)
        @test length(sol.dual) == 1        # the atom's SOC stays out of the user duals
    end
end

@testset "an objective coefficient may cross zero between re-solves" begin
    # Built at p1 = 0, where u1 carries no objective term at all. The emitted term
    # list is recomputed from c(p) on every assembly, so it must reappear at p1 != 0.
    optf = OptimizationFunction((u, p) -> p[1] * u[1] + u[2])
    cons = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))]
    prob = ConvexOptimizationProblem(
        optf, [0.5, 0.5], [0.0];
        constraints = cons, lb = [-10.0, -10.0], ub = [10.0, 10.0]
    )
    # u2 = 1 - u1 with u1 in [-9, 10]: obj = (p1 - 1) u1 + 1
    truth(p1) = p1 > 1 ? -9 * p1 + 10 : 10 * p1 - 9
    cache = init(prob, ALG)
    for p1 in (0.0, 3.0, -5.0, 0.0)
        cache = reinit!(cache; p = [p1])
        @test isapprox(solve!(cache).objective, truth(p1); atol = 1.0e-6)
    end
end

@testset "MaxSense parametric re-solve" begin
    optf = OptimizationFunction((u, p) -> p[1] * u[1] + u[2] - norm(u .- 1.0, 2))
    prob = ConvexOptimizationProblem(
        optf, [0.0, 0.0], [1.0];
        sense = SciMLBase.MaxSense, lb = [-5.0, -5.0], ub = [5.0, 5.0]
    )
    cache = init(prob, ALG)
    for θ in ([1.0], [-2.0], [0.0])
        cache = reinit!(cache; p = θ)
        sol = solve!(cache)
        cold = solve(SciMLBase.remake(prob; p = θ), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-7)
        @test isapprox(sol.u, cold.u; atol = 1.0e-7)
    end
end

@testset "the MOI index layout is identical at every p" begin
    # `conrefs` is read back after each rebuild, so the constraint numbering must be
    # reproduced exactly; a p-dependent emission of any variable or cone would
    # re-point them at another cone's dual without any error.
    optf = OptimizationFunction((u, p) -> norm(u .- p[1:2], 2) + p[3] * u[1])
    cons = [
        ConeConstraint((u, p) -> [u[1] + u[2] - p[3]], MOI.Zeros(1)),
        ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2)),
    ]
    prob = ConvexOptimizationProblem(
        optf, [0.0, 0.0], [1.0, 1.0, 0.0];
        constraints = cons, lb = [-10.0, -10.0], ub = [10.0, 10.0]
    )
    cache = init(prob, ALG)
    sig(c) = (
        getfield.(c.xvars, :value),
        [(typeof(r), r.value) for r in c.conrefs],
        [(typeof(r), r.value) for r in c.atomrefs],
    )
    ref = sig(cache)
    for θ in ([1.0, 1.0, 0.0], [3.0, -1.0, 2.0], [0.0, 0.0, -4.0])
        cache = reinit!(cache; p = θ)
        solve!(cache)
        @test sig(cache) == ref
    end
end

@testset "a failed solve yields NaN duals of the right shape, not a false certificate" begin
    # min p1*u1 + u2 s.t. u >= 0: unbounded below whenever p1 < 0. Clarabel then
    # reports DualStatus INFEASIBLE_POINT — a vector of the right length that is
    # not a dual solution — so it must not be handed back as `sol.dual`.
    optf = OptimizationFunction((u, p) -> p[1] * u[1] + u[2])
    cons = [ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2))]
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0], [1.0]; constraints = cons)

    cache = init(prob, ALG)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.dual[1], [1.0, 1.0]; atol = 1.0e-6)

    cache = reinit!(cache; p = [-1.0])
    bad = solve!(cache)
    @test !SciMLBase.successful_retcode(bad.retcode)
    @test bad.dual isa Vector{Vector{Float64}}
    @test length(bad.dual) == 1
    @test length(bad.dual[1]) == 2
    @test all(isnan, bad.dual[1])

    # and the cache recovers at the next good p
    cache = reinit!(cache; p = [2.0])
    good = solve!(cache)
    @test SciMLBase.successful_retcode(good.retcode)
    @test isapprox(good.dual[1], [2.0, 1.0]; atol = 1.0e-6)
end

@testset "problem data that is not disciplined-parametrized is rejected" begin
    Z1 = MOI.Zeros(1)
    box = (lb = [-10.0, -10.0], ub = [10.0, 10.0])
    pprob(f, p; kw...) = ConvexOptimizationProblem(
        OptimizationFunction(f), [0.5, 0.5], p; box..., kw...
    )
    lin = (u, p) -> u[1] + u[2]

    # not affine in the optimization variables at all
    @test_throws "affine in the optimization variables" solve(
        pprob((u, p) -> p[1] * u[1]^2, [1.0]), ALG
    )
    @test_throws "not affine in the variables" solve(
        pprob(lin, [1.0]; constraints = [ConeConstraint((u, p) -> [u[1]^2 + p[1] * u[2] - 1.0], Z1)]), ALG
    )
    @test_throws "must be affine in the optimization variables" solve(
        pprob((u, p) -> norm(u .^ 2 .- p[1], 2), [1.0]), ALG
    )

    # a product of optimization variables: `linear_expansion` reports islin = true
    # with a symbolic coefficient, so only the coefficient check catches it. This is
    # what replaces `analyze` as the convexity gate on the parametric path.
    @test_throws "still contains the optimization variable" solve(
        pprob((u, p) -> u[1] * u[2] + p[1] * u[1], [1.0]), ALG
    )
    @test_throws "still contains the optimization variable" solve(
        pprob(lin, [1.0]; constraints = [ConeConstraint((u, p) -> [u[1] * u[2] + p[1] - 1.0], Z1)]), ALG
    )

    # sound at every p, but the cone matrix moves with p, so it is not cacheable
    @test_throws "parameter-dependent coefficient" solve(
        pprob(lin, [1.0]; constraints = [ConeConstraint((u, p) -> [p[1] * u[1] + u[2] - 1.0], Z1)]), ALG
    )
    @test_throws "parameter-dependent coefficient" solve(
        pprob((u, p) -> norm(p[1] .* u .- 1.0, 2), [1.0]), ALG
    )

    # problem data not affine in p
    @test_throws "not affine in the parameters" solve(
        pprob((u, p) -> p[1] * p[2] * u[1] + u[2], [1.0, 2.0]), ALG
    )
    @test_throws "not affine in the parameters" solve(
        pprob((u, p) -> u[1] + p[1]^2, [2.0]), ALG
    )
    @test_throws "not affine in the parameters" solve(
        pprob(lin, [2.0]; constraints = [ConeConstraint((u, p) -> [u[1] + u[2] - 1 / p[1]], Z1)]), ALG
    )
    @test_throws "scales a lowered atom" solve(
        pprob((u, p) -> exp(p[1]) * u[1] + u[2], [1.0]), ALG
    )

    # the epigraph-sign guard, already at the initial p
    @test_throws "Lowering an atom through its epigraph" solve(
        pprob((u, p) -> u[1] + p[1] * norm(u .- 1.0, 2), [-2.0]), ALG
    )
    # …and the same objective solves at a p where the lowering is valid
    @test SciMLBase.successful_retcode(
        solve(pprob((u, p) -> u[1] + p[1] * norm(u .- 1.0, 2), [2.0]), ALG).retcode
    )

    # p must be a vector: anything else is silently linearized by `eachindex`
    @test_throws "must be an `AbstractVector`" solve(pprob(lin, 3.0), ALG)
    @test_throws "must be an `AbstractVector`" solve(pprob(lin, [1.0 2.0; 3.0 4.0]), ALG)
end

@testset "reinit! validates p without doing symbolic work" begin
    optf = OptimizationFunction((u, p) -> p[1] * u[1] + u[2])
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5], [0.5, 0.0]; constraints = lp_cons())
    cache = init(prob, ALG)

    @test_throws "symbolic maps" reinit!(cache; p = [:a => 1.0])
    @test_throws "canonicalized for 2 parameters" reinit!(cache; p = [1.0, 2.0, 3.0])
    @test_throws "non-finite" reinit!(cache; p = [NaN, 0.0])
    @test_throws "non-finite" reinit!(cache; p = [Inf, 0.0])
    @test_throws "AbstractVector" reinit!(cache; p = 3.0)
    @test_throws "canonicalized for 2 optimization variables" reinit!(cache; u0 = [1.0, 2.0, 3.0])

    # every rejection leaves the cache exactly as it was
    @test cache.p == [0.5, 0.0]
    @test isapprox(solve!(cache).objective, 0.5; atol = 1.0e-6)

    # u0 is stored but does not move a global conic solve
    cache = reinit!(cache; u0 = [0.1, 0.9])
    @test cache.u0 == [0.1, 0.9]
    @test isapprox(solve!(cache).objective, 0.5; atol = 1.0e-6)

    # a NullParameters problem has nothing to update
    nprob = ConvexOptimizationProblem(
        OptimizationFunction((u, p) -> u[1] + 2u[2]), [0.5, 0.5];
        constraints = [
            ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1)),
            ConeConstraint((u, p) -> [u[1], u[2]], MOI.Nonnegatives(2)),
        ]
    )
    ncache = init(nprob, ALG)
    @test ncache.p isa SciMLBase.NullParameters
    @test ncache.dpp.m == 0
    @test ncache.analysis isa NamedTuple        # non-parametric path still runs `analyze`
    @test_throws "NullParameters" reinit!(ncache; p = [1.0])
    @test reinit!(ncache) === ncache
    @test isapprox(solve!(ncache).objective, 1.0; atol = 1.0e-6)
end

@testset "a captured parameter vector is warned about, not silently swept" begin
    p0 = [2.0]
    optf = OptimizationFunction((u, p) -> u[1] + p0[1] * u[2])   # note: p0, not p
    prob = ConvexOptimizationProblem(
        optf, [0.5, 0.5], p0;
        constraints = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))],
        lb = [-10.0, -10.0], ub = [10.0, 10.0]
    )
    @test_logs (:warn, r"do not depend on it") init(prob, ALG)
end
