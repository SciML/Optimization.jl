using ConvexOptimization
using SciMLBase
using SciMLBase: ConvexOptimizationProblem, OptimizationSolution
import MathOptInterface as MOI
import Clarabel
using LinearAlgebra
using StaticArrays: SVector
using Test

# A successful solve must be self-consistent: the reported objective equals
# the user's f evaluated at the returned u.
function solve_checked(prob, alg; atol = 1.0e-5)
    sol = solve(prob, alg)
    SciMLBase.successful_retcode(sol.retcode) &&
        @test isapprox(prob.f.f(sol.u, prob.p), sol.objective; atol = atol)
    return sol
end

function solve!_checked(cache; atol = 1.0e-5)
    sol = solve!(cache)
    SciMLBase.successful_retcode(sol.retcode) &&
        @test isapprox(cache.f.f(sol.u, cache.p), sol.objective; atol = atol)
    return sol
end

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

    sol = solve_checked(prob, ConvexMOI(Clarabel.Optimizer))

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
    @test_throws Exception solve_checked(prob, ConvexMOI(Clarabel.Optimizer))
end

@testset "SOC constraint lowers and solves" begin
    # minimize t  s.t.  || (x1, x2) ||_2 <= t,  x == (3, 4)  ->  t* = 5
    optf = OptimizationFunction((u, p) -> u[3])   # u = (x1, x2, t)
    cons = [
        ConeConstraint((u, p) -> [u[1] - 3.0, u[2] - 4.0], MOI.Zeros(2)),   # x fixed
        ConeConstraint((u, p) -> [u[3], u[1], u[2]], MOI.SecondOrderCone(3)), # (t, x) in SOC
    ]
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0, 0.0]; constraints = cons)
    sol = solve_checked(prob, ConvexMOI(Clarabel.Optimizer))
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

    sol = solve_checked(prob, ConvexMOI(Clarabel.Optimizer))

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

    sol = solve_checked(prob, ConvexMOI(Clarabel.Optimizer))

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
    @test_throws Exception solve_checked(prob, ConvexMOI(Clarabel.Optimizer))

    # a norm entering with a negative coefficient is concave, not convex.
    optf2 = OptimizationFunction((u, p) -> -norm(A * u - b, 2))
    prob2 = ConvexOptimizationProblem(optf2, [0.0])
    @test_throws Exception solve_checked(prob2, ConvexMOI(Clarabel.Optimizer))

    # p = 3 has no corresponding MOI cone (1, 2 and Inf do).
    optf3 = OptimizationFunction((u, p) -> norm(A * u - b, 3))
    prob3 = ConvexOptimizationProblem(optf3, [0.0])
    @test_throws Exception solve_checked(prob3, ConvexMOI(Clarabel.Optimizer))

    # a non-affine norm argument cannot be lowered.
    optf4 = OptimizationFunction((u, p) -> norm(u .^ 2 .- 1.0, 2))
    prob4 = ConvexOptimizationProblem(optf4, [0.5, 0.5])
    @test_throws Exception solve_checked(prob4, ConvexMOI(Clarabel.Optimizer))
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
    sol1 = solve_checked(prob1, ConvexMOI(Clarabel.Optimizer))
    @test SciMLBase.successful_retcode(sol1.retcode)
    @test isapprox(sol1.objective, 1.0; atol = 1.0e-6)
    @test isapprox(sum(sol1.u), 1.0; atol = 1.0e-6)
    @test length(sol1.dual) == 1        # epigraph cone stays out of the user duals

    optfi = OptimizationFunction((u, p) -> norm(u .- 1.0, Inf))
    probi = ConvexOptimizationProblem(optfi, [0.5, 0.5]; constraints = cons)
    soli = solve_checked(probi, ConvexMOI(Clarabel.Optimizer))
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
    sole = solve_checked(
        ConvexOptimizationProblem(optfe, [0.0, 0.0]; constraints = conse),
        ConvexMOI(Clarabel.Optimizer)
    )
    @test SciMLBase.successful_retcode(sole.retcode)
    @test isapprox(sole.objective, exp(1); atol = 1.0e-6)

    # log barrier: minimize -log(u1) s.t. u1 == 2  ->  -log(2)
    optfl = OptimizationFunction((u, p) -> -log(u[1]))
    consl = [ConeConstraint((u, p) -> [u[1] - 2.0], MOI.Zeros(1))]
    soll = solve_checked(
        ConvexOptimizationProblem(optfl, [1.0]; constraints = consl),
        ConvexMOI(Clarabel.Optimizer)
    )
    @test SciMLBase.successful_retcode(soll.retcode)
    @test isapprox(soll.objective, -log(2); atol = 1.0e-6)
    @test length(soll.dual) == 1        # epigraph cone stays out of the user duals

    # a bare `log` objective is concave: minimizing it is not a convex program.
    optfb = OptimizationFunction((u, p) -> log(u[1]))
    @test_throws Exception solve_checked(
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
        sol = solve!_checked(cache)
        u★, obj★, d1★, d2★ = lp_analytic(θ)

        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, u★; atol = 1.0e-6)
        @test isapprox(sol.objective, obj★; atol = 1.0e-6)
        @test isapprox(sol.dual[1], d1★; atol = 1.0e-6)
        @test isapprox(sol.dual[2], d2★; atol = 1.0e-6)
        @test SciMLBase.parameter_values(cache) == θ

        # the fast path must agree with re-canonicalizing from scratch
        cold = solve_checked(SciMLBase.remake(prob; p = θ), ALG)
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
        solve!(cache)                   # raw solve!: the consistency check would call f
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
        sol = solve!_checked(cache)
        cold = solve_checked(SciMLBase.remake(prob; p = [λ]), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, cold.u; atol = 1.0e-7)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-7)
    end
    # large λ shrinks u to 0; the objective is then ||b||
    cache = reinit!(cache; p = [10.0])
    @test isapprox(solve!_checked(cache).u, [0.0, 0.0]; atol = 1.0e-6)

    # λ < 0 makes -|u| enter a minimization: the epigraph lowering is invalid there.
    @test_throws "Lowering an atom through its epigraph" reinit!(cache; p = [-1.0])
    # a rejected p must leave the cache untouched and usable
    @test cache.p == [10.0]
    @test isapprox(solve!_checked(cache).u, [0.0, 0.0]; atol = 1.0e-6)
end

@testset "a parameter inside an atom argument is a θ-affine cone constant" begin
    # min ||u - p||_2 s.t. sum(u) == 1  ->  u* = projection of p onto the hyperplane
    optf = OptimizationFunction((u, p) -> norm(u .- p, 2))
    cons = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))]
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0], [1.0, 1.0]; constraints = cons)

    cache = init(prob, ALG)
    for θ in ([1.0, 1.0], [3.0, -1.0], [0.0, 0.0])
        cache = reinit!(cache; p = θ)
        sol = solve!_checked(cache)
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
        @test isapprox(solve!_checked(cache).objective, truth(p1); atol = 1.0e-6)
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
        sol = solve!_checked(cache)
        cold = solve_checked(SciMLBase.remake(prob; p = θ), ALG)
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
        solve!_checked(cache)
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
    sol = solve!_checked(cache)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.dual[1], [1.0, 1.0]; atol = 1.0e-6)

    cache = reinit!(cache; p = [-1.0])
    bad = solve!_checked(cache)
    @test !SciMLBase.successful_retcode(bad.retcode)
    @test bad.dual isa Vector{Vector{Float64}}
    @test length(bad.dual) == 1
    @test length(bad.dual[1]) == 2
    @test all(isnan, bad.dual[1])

    # and the cache recovers at the next good p
    cache = reinit!(cache; p = [2.0])
    good = solve!_checked(cache)
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

    # not affine in the optimization variables at all (`u[1]^2` would now be a
    # lowerable quadratic atom; `u[1]^3` still is not)
    @test_throws "affine in the optimization variables" solve_checked(
        pprob((u, p) -> p[1] * u[1]^3, [1.0]), ALG
    )
    @test_throws "not affine in the variables" solve_checked(
        pprob(lin, [1.0]; constraints = [ConeConstraint((u, p) -> [u[1]^2 + p[1] * u[2] - 1.0], Z1)]), ALG
    )
    # u[i]^2 inside the norm is a nested atom; the element can change sign, so
    # certification rejects it.
    @test_throws "not certified convex" solve_checked(
        pprob((u, p) -> norm(u .^ 2 .- p[1], 2), [1.0]), ALG
    )

    # a product of optimization variables: `linear_expansion` reports islin = true
    # with a symbolic coefficient, so only the coefficient check catches it. This is
    # what replaces `analyze` as the convexity gate on the parametric path.
    @test_throws "still contains the optimization variable" solve_checked(
        pprob((u, p) -> u[1] * u[2] + p[1] * u[1], [1.0]), ALG
    )
    @test_throws "still contains the optimization variable" solve_checked(
        pprob(lin, [1.0]; constraints = [ConeConstraint((u, p) -> [u[1] * u[2] + p[1] - 1.0], Z1)]), ALG
    )

    # sound at every p, but the cone matrix moves with p, so it is not cacheable
    @test_throws "parameter-dependent coefficient" solve_checked(
        pprob(lin, [1.0]; constraints = [ConeConstraint((u, p) -> [p[1] * u[1] + u[2] - 1.0], Z1)]), ALG
    )
    @test_throws "parameter-dependent coefficient" solve_checked(
        pprob((u, p) -> norm(p[1] .* u .- 1.0, 2), [1.0]), ALG
    )

    # problem data not affine in p
    @test_throws "not affine in the parameters" solve_checked(
        pprob((u, p) -> p[1] * p[2] * u[1] + u[2], [1.0, 2.0]), ALG
    )
    @test_throws "not affine in the parameters" solve_checked(
        pprob((u, p) -> u[1] + p[1]^3, [2.0]), ALG
    )
    @test_throws "not affine in the parameters" solve_checked(
        pprob(lin, [2.0]; constraints = [ConeConstraint((u, p) -> [u[1] + u[2] - 1 / p[1]], Z1)]), ALG
    )
    prob = pprob((u, p) -> exp(p[1]) * u[1] + u[2], [1.0])
    cache = init(prob, ALG)
    for p in ([1.0], [-1.0])
        cache = reinit!(cache; p)
        sol = solve!_checked(cache)
        @test isapprox(sol.u, [-10.0, -10.0]; atol = 1.0e-5)
        @test isapprox(sol.objective, -10 * (exp(p[1]) + 1); atol = 1.0e-5)
    end

    # the epigraph-sign guard, already at the initial p
    @test_throws "Lowering an atom through its epigraph" solve_checked(
        pprob((u, p) -> u[1] + p[1] * norm(u .- 1.0, 2), [-2.0]), ALG
    )
    # …and the same objective solves at a p where the lowering is valid
    @test SciMLBase.successful_retcode(
        solve_checked(pprob((u, p) -> u[1] + p[1] * norm(u .- 1.0, 2), [2.0]), ALG).retcode
    )

    # p must be a vector: anything else is silently linearized by `eachindex`
    @test_throws "must be an `AbstractVector`" solve_checked(pprob(lin, 3.0), ALG)
    @test_throws "must be an `AbstractVector`" solve_checked(pprob(lin, [1.0 2.0; 3.0 4.0]), ALG)
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
    @test isapprox(solve!_checked(cache).objective, 0.5; atol = 1.0e-6)

    # u0 is stored but does not move a global conic solve
    cache = reinit!(cache; u0 = [0.1, 0.9])
    @test cache.u0 == [0.1, 0.9]
    @test isapprox(solve!_checked(cache).objective, 0.5; atol = 1.0e-6)

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
    @test isapprox(solve!_checked(ncache).objective, 1.0; atol = 1.0e-6)
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

import SymbolicAnalysis

# Tighter solver tolerances for the quadratic tests: the objective is flat at
# the optimum, so the primal error scales like sqrt(objective error); default
# Clarabel settings leave ~1e-4 in u.
const ALG_TIGHT = ConvexMOI(
    MOI.OptimizerWithAttributes(
        Clarabel.Optimizer, "tol_gap_abs" => 1.0e-10, "tol_gap_rel" => 1.0e-10,
        "tol_feas" => 1.0e-10, "tol_ktratio" => 1.0e-10
    )
)
const ALG_TIGHTEST = ConvexMOI(
    MOI.OptimizerWithAttributes(
        Clarabel.Optimizer, "tol_gap_abs" => 1.0e-12, "tol_gap_rel" => 1.0e-12,
        "tol_feas" => 1.0e-12, "tol_ktratio" => 1.0e-12, "max_iter" => 200
    )
)

# min ||A u - b||^2 with a 4x3 A lowers to one rotated second-order cone. The
# oracle is the direct solve `A \ b`, not this backend's own output.
const QLS_A = Float64[1 0 2; 0 1 1; 1 1 0; 2 0 1]
const QLS_B = Float64[1, 2, 3, 4]

@testset "sum of squares lowers least squares through the rotated SOC" begin
    u★ = QLS_A \ QLS_B
    obj★ = sum(abs2.(QLS_A * u★ .- QLS_B))
    for spelling in (
            (u, p) -> sum(abs2.(QLS_A * u .- QLS_B)),
            (u, p) -> sum((QLS_A * u .- QLS_B) .^ 2),
            (u, p) -> sum(abs2, QLS_A * u .- QLS_B),
        )
        prob = ConvexOptimizationProblem(OptimizationFunction(spelling), zeros(3))
        sol = solve_checked(prob, ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test length(sol.u) == 3                     # epigraph variable is internal
        @test isapprox(sol.u, u★; atol = 1.0e-6)
        @test isapprox(sol.objective, obj★; atol = 1.0e-6)
        @test isempty(sol.dual)                      # the RSOC is not a user dual
    end

    # scalar spellings: abs2(w) and w^2 for affine scalar w
    optf = OptimizationFunction((u, p) -> abs2(u[1] - 1.0) + (u[2] + 2.0)^2)
    sol = solve_checked(ConvexOptimizationProblem(optf, [0.0, 0.0]), ALG)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [1.0, -2.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, 0.0; atol = 1.0e-6)
end

@testset "least squares with an equality constraint matches the KKT solution" begin
    # min ||A u - b||^2  s.t.  C u = d.  KKT:  [AᵀA Cᵀ; C 0] [u; λ] = [Aᵀb; d]
    C = Float64[1 -1 0; 0 0 1]
    d = [0.25, 1.0]
    u★ = ([QLS_A' * QLS_A C'; C zeros(2, 2)] \ [QLS_A' * QLS_B; d])[1:3]
    obj★ = sum(abs2.(QLS_A * u★ .- QLS_B))

    cons = [ConeConstraint((u, p) -> C * u .- d, MOI.Zeros(2))]
    optf = OptimizationFunction((u, p) -> sum(abs2.(QLS_A * u .- QLS_B)))
    prob = ConvexOptimizationProblem(optf, zeros(3); constraints = cons)
    sol = solve_checked(prob, ALG_TIGHT)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, u★; atol = 1.0e-6)
    @test isapprox(sol.objective, obj★; atol = 1.0e-6)
    @test length(sol.dual) == 1                      # only the user constraint
end

# min u'P u + c'u, P symmetric PSD: u'Pu sees only sym(P) = LᵀL and lowers to
# ||L u||² <= τ in the same rotated cone.
const QP_P = Float64[2 0 0; 0 3 1; 0 1 1]
const QP_C = Float64[1, -2, 0.5]

@testset "quadratic forms (u'Pu, quad_form) lower to the rotated SOC" begin
    # unconstrained: ∇ = 2 P u + c = 0  ->  u★ = -P \ c / 2
    u★ = -(QP_P \ QP_C) / 2
    obj★ = u★' * QP_P * u★ + QP_C' * u★
    optf = OptimizationFunction((u, p) -> u' * QP_P * u + QP_C' * u)
    sol = solve_checked(ConvexOptimizationProblem(optf, zeros(3)), ALG_TIGHT)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, u★; atol = 1.0e-6)
    @test isapprox(sol.objective, obj★; atol = 1.0e-6)
    @test isempty(sol.dual)

    # equality-constrained QP through the SymbolicAnalysis atom, against the
    # KKT oracle  [2P Cᵀ; C 0] [u; λ] = [-c; d]
    C2 = Float64[1 1 1]
    d2 = [1.0]
    u★2 = ([2QP_P C2'; C2 0] \ [-QP_C; d2])[1:3]
    obj★2 = u★2' * QP_P * u★2 + QP_C' * u★2
    cons = [ConeConstraint((u, p) -> C2 * u .- d2, MOI.Zeros(1))]
    optf2 = OptimizationFunction(
        (u, p) -> SymbolicAnalysis.quad_form(u, QP_P) + QP_C' * u
    )
    sol2 = solve_checked(ConvexOptimizationProblem(optf2, zeros(3); constraints = cons), ALG_TIGHT)
    @test SciMLBase.successful_retcode(sol2.retcode)
    @test isapprox(sol2.u, u★2; atol = 1.0e-6)
    @test isapprox(sol2.objective, obj★2; atol = 1.0e-6)
    @test length(sol2.dual) == 1

    # a scalar multiple of the form composes through the objective coefficients
    optf3 = OptimizationFunction((u, p) -> 2 * (u' * QP_P * u) + QP_C' * u)
    sol3 = solve_checked(ConvexOptimizationProblem(optf3, zeros(3)), ALG_TIGHT)
    u★3 = -(QP_P \ QP_C) / 4
    @test isapprox(sol3.u, u★3; atol = 1.0e-6)
end

@testset "invalid quadratic atoms are rejected, not silently solved" begin
    Pbad = Float64[1 0; 0 -1]
    cases = (
        # -abs2 under MinSense is concave: the epigraph bound goes the wrong
        # way; a convex atom under MaxSense cannot be bounded below either.
        (
            "Lowering an atom through its epigraph",
            (u, p) -> -abs2(u[1]), [0.5], nothing, (;),
        ),
        (
            "Lowering an atom through its epigraph",
            (u, p) -> abs2(u[1]), [0.5], nothing, (sense = SciMLBase.MaxSense,),
        ),
        # atom arguments must be affine in u
        (
            "not affine in the optimization variables",
            (u, p) -> abs2(u[1] * u[2]), [0.5, 0.5], nothing, (;),
        ),

        # indefinite P makes the quadratic form non-convex
        (
            "positive semidefinite",
            (u, p) -> u' * Pbad * u, [0.5, 0.5], nothing, (;),
        ),
        # a P built from p moves the cone matrix with θ
        (
            "constant numeric matrix",
            (u, p) -> SymbolicAnalysis.quad_form(u, p[1] * QP_P), zeros(3), [1.0], (;),
        ),
        # w^3 is not a square and stays for the affinity check to reject
        (
            "requires an objective that is affine",
            (u, p) -> u[1]^3 + u[2], [0.5, 0.5], nothing, (;),
        ),
    )
    for (msg, f, u0, p, kw) in cases
        prob = p === nothing ?
            ConvexOptimizationProblem(OptimizationFunction(f), u0; kw...) :
            ConvexOptimizationProblem(OptimizationFunction(f), u0, p; kw...)
        @test_throws msg solve_checked(prob, ALG)
    end
end

@testset "sum-of-squares atom: reinit! with a parameter in the argument" begin
    # min ||A u - p||^2: b = p enters the RSOC constant, affine in θ.
    optf = OptimizationFunction((u, p) -> sum(abs2.(QLS_A * u .- p)))
    prob = ConvexOptimizationProblem(optf, zeros(3), [1.0, 2.0, 3.0, 4.0])
    cache = init(prob, ALG)
    for θ in ([1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0], [4.0, 3.0, 2.0, 1.0])
        cache = reinit!(cache; p = θ)
        sol = solve!_checked(cache)
        cold = solve_checked(SciMLBase.remake(prob; p = θ), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, QLS_A \ θ; atol = 1.0e-6)     # analytic oracle
        @test isapprox(sol.u, cold.u; atol = 1.0e-8)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
        @test isempty(sol.dual)
    end
end

# Scalar `*` products over array arguments (`c'*u`, `(M*u)'*c`) are evaluated
# to affine expressions, so pure LPs solve. The box makes them bounded.
@testset "scalar products over arrays lower pure LPs" begin
    c = [1.0, -2.0]
    M = [1.0 2.0; 3.0 4.0]
    box = (lb = [-1.0, -1.0], ub = [1.0, 1.0])
    for (f, obj★) in (
            ((u, p) -> c' * u, -3.0),
            ((u, p) -> LinearAlgebra.dot(c, u), -3.0),
            ((u, p) -> c' * M * u, -11.0),
            ((u, p) -> (M * u)' * c, -11.0),
            ((u, p) -> sum(c .* u), -3.0),
            ((u, p) -> c' * (u .- 1.0), -2.0),
        )
        prob = ConvexOptimizationProblem(OptimizationFunction(f), [0.0, 0.0]; box...)
        sol = solve_checked(prob, ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.objective, obj★; atol = 1.0e-6)
    end
end

@testset "self-products v' * v lower as sums of squares" begin
    sol = solve_checked(ConvexOptimizationProblem(OptimizationFunction((u, p) -> u' * u), [0.5, 0.5]), ALG)
    @test isapprox(sol.objective, 0.0; atol = 1.0e-6)
    # (A*u)'*(A*u) flattens to adjoint(A*u)*A*u: still ||A*u||²
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> (QLS_A * u)' * (QLS_A * u)), zeros(3)
        ), ALG
    )
    @test isapprox(sol.u, zeros(3); atol = 1.0e-6)
    # (A*u - b)'*(A*u - b) is the least-squares problem in self-product form
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> (QLS_A * u .- QLS_B)' * (QLS_A * u .- QLS_B)),
            zeros(3)
        ), ALG_TIGHT
    )
    @test isapprox(sol.u, QLS_A \ QLS_B; atol = 1.0e-6)
    # v' * P * v with a precomputed P = A'A; the RSOC primal error scales as
    # sqrt(dobj/λmin), so u gets that bound and the objective is pinned.
    M2 = QLS_A' * QLS_A
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction(
                (u, p) -> u' * M2 * u + QP_C' * u
            ), zeros(3)
        ), ALG_TIGHTEST
    )
    us = -(M2 \ QP_C) / 2
    @test isapprox(sol.objective, us' * M2 * us + QP_C' * us; atol = 1.0e-9)
    λmin = minimum(eigvals(Symmetric(M2)))
    @test norm(sol.u - us, Inf) <= 2 * sqrt(1.0e-9 / λmin)
end

@testset "squares of parameters are p-dependent constants, not atoms" begin
    # min u1 + p1² over [-10, 10] at p = 2 is -10 + 4
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u[1] + p[1]^2), [0.0], [2.0];
            lb = [-10.0], ub = [10.0]
        ), ALG
    )
    @test isapprox(sol.u, [-10.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, -6.0; atol = 1.0e-6)
    # -p1² is the same constant with the opposite sign: -4, not an epigraph
    # sign error.
    optf = OptimizationFunction((u, p) -> u[1]^2 + u[2]^2 - p[1]^2)
    prob = ConvexOptimizationProblem(optf, [0.5, 0.5], [2.0])
    sol = solve_checked(prob, ALG)
    @test isapprox(sol.u, [0.0, 0.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, -4.0; atol = 1.0e-6)
    cache = init(prob, ALG)
    cache = reinit!(cache; p = [3.0])
    sol3 = solve!_checked(cache)
    cold = solve_checked(SciMLBase.remake(prob; p = [3.0]), ALG)
    @test isapprox(sol3.objective, -9.0; atol = 1.0e-6)
    @test isapprox(sol3.objective, cold.objective; atol = 1.0e-8)
    # inside an atom argument the lifted square stays θ-affine
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> abs2(u[1] - p[1]^2)), [0.0], [2.0]
        ), ALG
    )
    @test isapprox(sol.u, [4.0]; atol = 1.0e-6)
    # …and inside a cone constraint (u1 >= p1²)
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u[1]), [0.0], [2.0];
            constraints = [ConeConstraint((u, p) -> [p[1]^2 - u[1]], MOI.Nonpositives(1))]
        ), ALG
    )
    @test isapprox(sol.u, [4.0]; atol = 1.0e-6)
    # lifted squares evaluate through a compiled function, so non-polynomial
    # p-only arguments work: min e^{2p1}·u1 + u1² + u2², u★ = -e^{2p1}/2,
    # u within the sqrt(dobj/λmin) bound (λmin = 2).
    prob = ConvexOptimizationProblem(
        OptimizationFunction(
            (u, p) -> exp(p[1])^2 * u[1] + u[1]^2 + u[2]^2
        ), [0.5, 0.5], [1.0, 2.0]
    )
    sol = solve_checked(prob, ALG_TIGHTEST)
    us = [-exp(2.0) / 2, 0.0]
    @test isapprox(sol.objective, -exp(4.0) / 4; atol = 1.0e-9)
    @test norm(sol.u - us, Inf) <= 2 * sqrt(1.0e-9 / 2)
    cache = reinit!(init(prob, ALG_TIGHTEST); p = [2.0, 3.0])
    sol2 = solve!_checked(cache)
    cold2 = solve_checked(SciMLBase.remake(prob; p = [2.0, 3.0]), ALG_TIGHTEST)
    @test isapprox(sol2.objective, -exp(8.0) / 4; rtol = 1.0e-8)
    @test isapprox(sol2.u, cold2.u; atol = 1.0e-8)
    @test isapprox(sol2.objective, cold2.objective; rtol = 1.0e-10)
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> abs2(sin(p[1])) + sum(abs2, u)),
            [0.5, 0.5], [1.0, 2.0]
        ), ALG
    )
    @test isapprox(sol.objective, sin(1.0)^2; atol = 1.0e-6)
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> (p[1] / p[2])^2 + sum(abs2, u)),
            [0.5, 0.5], [1.0, 2.0]
        ), ALG
    )
    @test isapprox(sol.objective, 0.25; atol = 1.0e-6)
end

@testset "quadratic-form edge cases: asymmetric, singular, near-PSD P" begin
    # asymmetric P contributes only sym(P) = 2I: min 2‖u‖² + c'u, u★ = -c/4
    Pa = Float64[2 1; -1 2]
    c = [1.0, -2.0]
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * Pa * u + c' * u), [0.5, 0.5]
        ), ALG_TIGHTEST
    )
    @test isapprox(sol.u, -c / 4; atol = 1.0e-6)
    @test isapprox(sol.objective, -5 / 8; atol = 1.0e-6)
    # singular PSD P = ones(2,2): min (u1+u2)² + u1+u2 -> s = -1/2
    sol = @test_logs match_mode = :any solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * ones(2, 2) * u + ones(2)' * u),
            [0.5, 0.5]
        ), ALG_TIGHT
    )
    @test isapprox(sum(sol.u), -0.5; atol = 1.0e-6)
    @test isapprox(sol.objective, -0.25; atol = 1.0e-6)
    # λmin = -1e-16 is inside the n·eps·λmax tolerance: clamped with a debug
    # log (routine on rank-deficient BᵀB, so no warn). Diagonal P makes the
    # eigenvalue deterministic; u2 is free, so only the objective is asserted.
    Ptol = diagm([1.0, -1.0e-16])
    sol = @test_logs (:debug, r"clamp") min_level = Base.CoreLogging.Debug solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * Ptol * u + [1.0, 0.0]' * u), [0.5, 0.5]
        ), ALG
    )
    @test isapprox(sol.objective, -0.25; atol = 1.0e-6)
    # a materially negative eigenvalue is rejected even at large scale
    @test_throws "positive semidefinite" solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * [1.0e6 0.0; 0.0 -1.0e-6] * u - u[2]),
            [0.5, 0.5]; lb = [-10.0, -10.0], ub = [10.0, 10.0]
        ), ALG
    )
end

@testset "reinit! rechecks the epigraph sign of a θ-scaled atom" begin
    # min p1·u1² + u2² + u2: valid for p1 > 0, unbounded (rejected) at p1 = -1
    optf = OptimizationFunction((u, p) -> abs2(u[1]) * p[1] + u[2]^2 + u[2])
    prob = ConvexOptimizationProblem(
        optf, [0.5, 0.5], [1.0]; lb = [-10.0, -10.0], ub = [10.0, 10.0]
    )
    cache = init(prob, ALG)
    sol = solve!_checked(cache)
    @test isapprox(sol.u, [0.0, -0.5]; atol = 1.0e-6)
    @test isapprox(sol.objective, -0.25; atol = 1.0e-6)
    @test_throws "epigraph" reinit!(cache; p = [-1.0])
    @test cache.p == [1.0]                     # rejected p leaves the cache intact
    cache = reinit!(cache; p = [3.0])
    sol3 = solve!_checked(cache)
    @test isapprox(sol3.u, [0.0, -0.5]; atol = 1.0e-6)
    @test isapprox(sol3.objective, -0.25; atol = 1.0e-6)
end

@testset "scalar-scaled and multi-matrix quadratic products" begin
    # all flatten to *(2, adjoint(u), u) = 2‖u‖²: u★ = (1/4, 0), obj = -1/8
    for f in (
            (u, p) -> u' * (2 * u) - u[1],
            (u, p) -> (2 * u)' * u - u[1],
            (u, p) -> u' * u * 2 - u[1],
        )
        sol = solve_checked(
            ConvexOptimizationProblem(OptimizationFunction(f), [0.5, 0.5]), ALG_TIGHT
        )
        @test isapprox(sol.u, [0.25, 0.0]; atol = 1.0e-6)
        @test isapprox(sol.objective, -0.125; atol = 1.0e-6)
    end
    # the scale folds into P: min 2u'Pu - u1 -> 4Pu★ = e1, obj = -u★1/2
    P2 = Float64[2 1; 1 2]
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> (2 * u)' * P2 * u - u[1]), [0.5, 0.5]
        ), ALG_TIGHT
    )
    us = (P2 \ [1.0, 0.0]) / 4
    @test isapprox(sol.u, us; atol = 1.0e-6)
    @test isapprox(sol.objective, -us[1] / 2; atol = 1.0e-6)
    # a negative scale is concave, not a quadratic atom to lower
    @test_throws "positive semidefinite" solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * (-2 * u) - u[1]), [0.5, 0.5];
            lb = [-10.0, -10.0], ub = [10.0, 10.0]
        ), ALG
    )
    # un-parenthesized u'*A'*A*u is a (1,)-shaped term that still lowers
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * QLS_A' * QLS_A * u), zeros(3)
        ), ALG
    )
    @test isapprox(sol.u, zeros(3); atol = 1.0e-6)
    # u'*A*B*u with indefinite sym(A*B) reaches the PSD check, not a shape error
    Bi = Float64[1 0; 0 -1]
    @test_throws "positive semidefinite" solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction(
                (u, p) -> u' * Matrix{Float64}(I, 2, 2) * Bi * u
            ), [0.5, 0.5]; lb = [-10.0, -10.0], ub = [10.0, 10.0]
        ), ALG
    )
end

@testset "scalar factors on self-products keep the right objective" begin
    # k·u'Mu - c'u has obj★ = -c'(M\c)/(4k); the scale may sit on either
    # factor or outside the product.
    Asp = Float64[1 2; 0 1; 1 1]
    csp = [1.0, -2.0]
    Msp = Asp' * Asp
    for k in (0.5, 2.0)
        obj = -csp' * (Msp \ csp) / (4k)
        for f in (
                (u, p) -> (Asp * u)' * (k * (Asp * u)) - csp' * u,
                (u, p) -> (k * (Asp * u))' * (Asp * u) - csp' * u,
                (u, p) -> k * ((Asp * u)' * (Asp * u)) - csp' * u,
                (u, p) -> (Asp * u)' * (k * Asp * u) - csp' * u,
                (u, p) -> (k * Asp * u)' * (Asp * u) - csp' * u,
                (u, p) -> u' * (k * Msp) * u - csp' * u,
            )
            sol = solve_checked(
                ConvexOptimizationProblem(OptimizationFunction(f), [0.0, 0.0]),
                ALG_TIGHT
            )
            @test isapprox(sol.objective, obj; atol = 1.0e-6)
        end
    end
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * (3 * u) - csp' * u), [0.0, 0.0]
        ), ALG_TIGHT
    )
    @test isapprox(sol.objective, -csp' * csp / 12; atol = 1.0e-6)
    # (kA)u fused into a numeric matrix is a v'*M*w cross term: refused, not
    # lowered with a wrong scale
    for f in (
            (u, p) -> (Asp * u)' * ((2Asp) * u),
            (u, p) -> ((2Asp) * u)' * (Asp * u),
        )
        @test_throws "certified" solve_checked(
            ConvexOptimizationProblem(OptimizationFunction(f), [0.5, 0.5]), ALG
        )
    end
    # negative scales are concave: refused with a clear message either way
    box = (lb = [-10.0, -10.0], ub = [10.0, 10.0])
    for (m, f) in (
            ("positive semidefinite", (u, p) -> (Asp * u)' * (-(Asp * u))),
            ("positive semidefinite", (u, p) -> (-(Asp * u))' * (Asp * u)),
            ("Concave", (u, p) -> u' * (-u)),
            ("epigraph", (u, p) -> -(u' * u)),
        )
        @test_throws m solve_checked(
            ConvexOptimizationProblem(
                OptimizationFunction(f), [0.5, 0.5]; box...
            ), ALG
        )
    end
end

@testset "uncanonicalizable products get clear errors, not internal ones" begin
    A2 = Float64[1 0; 0 1]
    B2 = Float64[2 0; 1 1]
    @test_throws "could not be traced" solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * A2 * B2 * u - u[1]), [0.5, 0.5]
        ), ALG
    )
    P2 = Float64[2 1; 1 2]
    @test_throws "constant numeric matrix" solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> u' * (p[1] * P2) * u), [0.5, 0.5], [1.0]
        ), ALG
    )
end

# Shared 5x2 regression data. Reference values are Convex.jl with Clarabel on
# the same problems, not this backend's output; the LAD optimum is also exact
# by hand (it interpolates rows 1 and 5, and |r| sums to 1.495):
#   min maximum(abs.(PWL_A*u - PWL_b)) -> u* = (1.0270270232, -0.0926640934), obj = 0.5806949812
#   min sum(abs.(PWL_A*u - PWL_b))     -> u* = (0.55, -0.3), obj = 1.495
const PWL_A = [1.0 0.5; -0.3 1.2; 0.7 -0.4; 0.1 0.9; -0.8 0.2]
const PWL_b = [0.4, -1.0, 0.3, 0.6, -0.5]

@testset "scalar abs atoms lower through the 1-norm cone" begin
    # min |u1 - 2| + |u2 + 1|  s.t.  u1 + u2 == 1  ->  u* = (2, -1), obj = 0
    optf = OptimizationFunction((u, p) -> abs(u[1] - 2.0) + abs(u[2] + 1.0))
    cons = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))]
    sol = solve_checked(ConvexOptimizationProblem(optf, [0.0, 0.0]; constraints = cons), ALG)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [2.0, -1.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, 0.0; atol = 1.0e-6)
    @test length(sol.dual) == 1        # epigraph cones stay out of the user duals
end

@testset "Chebyshev fit: min maximum(abs.(A*u - b)) lowers to NormInfinityCone" begin
    optf = OptimizationFunction((u, p) -> maximum(abs.(PWL_A * u - PWL_b)))
    cons = [ConeConstraint((u, p) -> [u[1] + u[2] - 3.0], MOI.Nonpositives(1))]
    sol = solve_checked(ConvexOptimizationProblem(optf, [0.0, 0.0]; constraints = cons), ALG)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [1.0270270232, -0.0926640934]; atol = 1.0e-6)
    @test isapprox(sol.objective, 0.5806949812; atol = 1.0e-6)
    @test length(sol.u) == 2           # epigraph variable is internal
    @test length(sol.dual) == length(cons)
end

@testset "LAD fit: min sum(abs.(A*u - b)) lowers to NormOneCone" begin
    optf = OptimizationFunction((u, p) -> sum(abs.(PWL_A * u - PWL_b)))
    sol = solve_checked(ConvexOptimizationProblem(optf, [0.0, 0.0]), ALG)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [0.55, -0.3]; atol = 1.0e-6)
    @test isapprox(sol.objective, 1.495; atol = 1.0e-6)
    @test length(sol.dual) == 0        # no user constraints, no duals
end

@testset "max/min of three affine scalars" begin
    # min_u max(u1, u2, 1 - u1 - u2): balanced at u* = (1/3, 1/3), obj = 1/3.
    # The variadic spelling traces to nested binary `max` calls.
    optf = OptimizationFunction((u, p) -> max(u[1], u[2], 1.0 - u[1] - u[2]))
    sol = solve_checked(ConvexOptimizationProblem(optf, [0.0, 0.0]), ALG)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, [1 / 3, 1 / 3]; atol = 1.0e-6)
    @test isapprox(sol.objective, 1 / 3; atol = 1.0e-6)

    # `min` is concave: max_u min(u1, u2, 2 - u1 - u2) -> u* = (2/3, 2/3), obj = 2/3.
    optfm = OptimizationFunction((u, p) -> min(u[1], u[2], 2.0 - u[1] - u[2]))
    solm = solve_checked(
        ConvexOptimizationProblem(optfm, [0.0, 0.0]; sense = SciMLBase.MaxSense), ALG
    )
    @test SciMLBase.successful_retcode(solm.retcode)
    @test isapprox(solm.u, [2 / 3, 2 / 3]; atol = 1.0e-6)
    @test isapprox(solm.objective, 2 / 3; atol = 1.0e-6)
end

@testset "minimum solves under MaxSense and is rejected under MinSense" begin
    # max min(u)  s.t.  sum(u) == 1, u >= 0: u* = (1/3, 1/3, 1/3), obj = 1/3.
    cons = [
        ConeConstraint((u, p) -> [u[1] + u[2] + u[3] - 1.0], MOI.Zeros(1)),
        ConeConstraint((u, p) -> [u[1], u[2], u[3]], MOI.Nonnegatives(3)),
    ]
    optf = OptimizationFunction((u, p) -> minimum(u))
    sol = solve_checked(
        ConvexOptimizationProblem(
            optf, [0.3, 0.3, 0.3]; sense = SciMLBase.MaxSense, constraints = cons
        ), ALG
    )
    @test SciMLBase.successful_retcode(sol.retcode)
    @test isapprox(sol.u, fill(1 / 3, 3); atol = 1.0e-6)
    @test isapprox(sol.objective, 1 / 3; atol = 1.0e-6)
    @test length(sol.dual) == 2

    @test_throws "Lowering an atom through its hypograph" solve(
        ConvexOptimizationProblem(optf, [0.3, 0.3, 0.3]; constraints = cons), ALG
    )
end

@testset "piecewise-linear atoms reject what cannot be lowered soundly" begin
    cases = [
        ("nondecreasing in it for MinSense", (u, p) -> -abs2(u[1]), [0.5], SciMLBase.MinSense),
        ("Lowering an atom through its epigraph", (u, p) -> -abs(u[1]), [0.5], SciMLBase.MinSense),
        ("Lowering an atom through its epigraph", (u, p) -> max(u[1], u[2]), [0.0, 0.0], SciMLBase.MaxSense),
        ("Lowering an atom through its hypograph", (u, p) -> min(u[1], u[2]), [0.0, 0.0], SciMLBase.MinSense),
        ("Lowering an atom through its hypograph", (u, p) -> minimum(u .- 1.0), [0.0, 0.0], SciMLBase.MinSense),
        ("neither convex nor concave", (u, p) -> minimum(abs.(PWL_A * u - PWL_b)), [0.0, 0.0], SciMLBase.MinSense),
        # `sum(abs, w)` is the mapreduce spelling, out of scope.
        ("broadcast form", (u, p) -> sum(abs, u .- 1.0), [0.0, 0.0], SciMLBase.MinSense),
        ("only over all elements", (u, p) -> maximum(u; init = 0.0), [0.0, 0.0], SciMLBase.MinSense),
        ("only over all elements", (u, p) -> maximum(u; dims = 1), [0.0, 0.0], SciMLBase.MinSense),
        ("only over all elements", (u, p) -> sum(abs.(u); dims = 1), [0.0, 0.0], SciMLBase.MinSense),
        ("only over all elements", (u, p) -> sum(abs.(u); init = 1.0), [0.0, 0.0], SciMLBase.MinSense),
    ]
    for (msg, f, u0, sense) in cases
        prob = ConvexOptimizationProblem(OptimizationFunction(f), u0; sense)
        @test_throws msg solve(prob, ALG)
    end

    # the parametric path has no `analyze` gate; this must be refused deliberately.
    pprob = ConvexOptimizationProblem(
        OptimizationFunction((u, p) -> minimum(abs.(u .- p))), [0.0, 0.0], [1.0, 2.0]
    )
    @test_throws "neither convex nor concave" solve(pprob, ALG)
end

@testset "atoms with parameters inside the argument re-solve through reinit!" begin
    # a parameter inside the abs broadcast is a theta-affine cone constant.
    optf = OptimizationFunction((u, p) -> sum(abs.(u .- p)))
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0], [1.0, 1.0])
    cache = init(prob, ALG)
    for θ in ([1.0, 1.0], [3.0, -1.0], [-2.0, 0.5], [0.0, 0.0])
        cache = reinit!(cache; p = θ)
        sol = solve!_checked(cache)
        cold = solve_checked(SciMLBase.remake(prob; p = θ), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, cold.u; atol = 1.0e-8)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
        @test isapprox(sol.u, θ; atol = 1.0e-6)   # min sum|u - θ| is attained at u = θ
    end

    # the scalar `max` optimum is not unique at p = [1, -1] (any u1 <= -3 with
    # u2 = -5 attains obj = -4), so only the objective value is compared.
    optf2 = OptimizationFunction((u, p) -> max(u[1] - p[1], u[2] - p[2]))
    prob2 = ConvexOptimizationProblem(
        optf2, [0.0, 0.0], [0.0, 0.0];
        lb = [-5.0, -5.0], ub = [5.0, 5.0]
    )
    cache2 = init(prob2, ALG)
    for θ in ([0.0, 0.0], [1.0, -1.0], [-2.0, 3.0])
        cache2 = reinit!(cache2; p = θ)
        sol = solve!_checked(cache2)
        cold = solve_checked(SciMLBase.remake(prob2; p = θ), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
    end
end

@testset "quadratic and piecewise-linear atoms mix in one objective" begin
    # (u1-2)^2 + u2^2 + |u1| + |u2| is separable: u* = (1.5, 0), obj = 1.75.
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction(
                (u, p) -> sum(abs2.(u .- [2.0, 0.0])) + sum(abs.(u))
            ), [0.0, 0.0]; lb = [-5.0, -5.0], ub = [5.0, 5.0]
        ), ALG_TIGHT
    )
    @test isapprox(sol.u, [1.5, 0.0]; atol = 1.0e-5)
    @test isapprox(sol.objective, 1.75; atol = 1.0e-6)

    # max(u) + (u1-1)^2 on [-1,1]^2: the objective depends on u2 only through
    # max(u), so any u2 <= u1 is optimal; u1 + (u1-1)^2 is minimized at
    # u1 = 1/2 -> obj = 0.75.
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction(
                (u, p) -> maximum(u) + abs2(u[1] - 1.0)
            ), [0.0, 0.0]; lb = [-1.0, -1.0], ub = [1.0, 1.0]
        ), ALG_TIGHT
    )
    @test isapprox(sol.u[1], 0.5; atol = 1.0e-6)
    @test isapprox(sol.objective, 0.75; atol = 1.0e-6)

    # u'P2u + |u1 - 0.5| - u1: optimum sits at the |u1 - 0.5| kink, u* = (0.5, -0.25),
    # obj = -0.125 (0.5 offset direction balances the kink's subgradient).
    P2m = Float64[2 1; 1 2]
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction(
                (u, p) -> u' * P2m * u + abs(u[1] - 0.5) - u[1]
            ), [0.0, 0.0]; lb = [-5.0, -5.0], ub = [5.0, 5.0]
        ), ALG_TIGHT
    )
    @test isapprox(sol.u, [0.5, -0.25]; atol = 1.0e-6)
    @test isapprox(sol.objective, -0.125; atol = 1.0e-6)
end

# Atoms inside other atoms lower innermost-first; certification runs `analyze`
# on the original expression, so a lowered-but-nonconvex graph still errors.
@testset "nested atoms lower innermost-first and certify the original expression" begin
    # Same least squares as the sum-of-squares spelling above, via a norm atom.
    u★ = QLS_A \ QLS_B
    obj★ = norm(QLS_A * u★ - QLS_B)^2
    for spelling in (
            (u, p) -> norm(QLS_A * u - QLS_B)^2,
            (u, p) -> abs2(norm(QLS_A * u - QLS_B)),
        )
        sol = solve_checked(
            ConvexOptimizationProblem(OptimizationFunction(spelling), zeros(3)), ALG
        )
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, u★; atol = 1.0e-6)
        @test isapprox(sol.objective, obj★; atol = 1.0e-6)
    end

    # (objective, u0, kwargs, expected objective, expected u or nothing)
    cases = [
        # min exp(||u - c||): exp is increasing, so u* = c, obj = e^0 = 1.
        ((u, p) -> exp(norm(u .- [1.0, 2.0])), [0.0, 0.0], (;), 1.0, [1.0, 2.0]),
        # max(||u||, 1) on sum(u) = 1: unconstrained norm min is 0.707 < 1; u* not unique.
        (
            (u, p) -> max(norm(u), 1.0), [0.4, 0.6],
            (constraints = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))],),
            1.0, nothing,
        ),
        # (|u1| + |u2|)^2 on sum(u) = 1: |u1| + |u2| >= |u1 + u2| = 1; u* not unique.
        (
            (u, p) -> sum(abs.(u))^2, [0.4, 0.6],
            (constraints = [ConeConstraint((u, p) -> [u[1] + u[2] - 1.0], MOI.Zeros(1))],),
            1.0, nothing,
        ),
        # |norm(u - 1)| = norm(u - 1): u* = 1, obj = 0.
        ((u, p) -> abs(norm(u .- 1.0)), [0.0, 0.0], (;), 0.0, [1.0, 1.0]),
        # max(u1, |u2|) on the box: u* is not unique; only the objective is pinned.
        (
            (u, p) -> max(u[1], abs(u[2])), [0.3, -0.4],
            (lb = [-1.0, -1.0], ub = [1.0, 1.0]), 0.0, nothing,
        ),
        # ||u|| + exp(||u||) on [1, 2]^2 is increasing in u: u* = (1, 1).
        (
            (u, p) -> norm(u) + exp(norm(u)), [1.5, 1.5],
            (lb = [1.0, 1.0], ub = [2.0, 2.0]), sqrt(2.0) + exp(sqrt(2.0)), [1.0, 1.0],
        ),
        # -log(u1) + ||u[2:3]|| with u1 <= 1: -log pushes u1 up, u* = (1, 0, 0).
        (
            (u, p) -> -log(u[1]) + norm(u[2:3]), [0.5, 0.5, 0.5],
            (constraints = [ConeConstraint((u, p) -> [u[1] - 1.0], MOI.Nonpositives(1))],),
            0.0, [1.0, 0.0, 0.0],
        ),
        # max(u1, -log(u2)) with u1 pinned: the -log branch dominates, u2* = 2.
        (
            (u, p) -> max(u[1], -log(u[2])), [-5.0, 1.0],
            (lb = [-5.0, 0.0], ub = [-5.0, 2.0]), -log(2.0), [-5.0, 2.0],
        ),
        # pinned compositions: the objective is f evaluated at the pin.
        ((u, p) -> abs(u[1]^2), [1.0], (lb = [2.0], ub = [2.0]), 4.0, [2.0]),
        ((u, p) -> max(u[1], u[1]^2), [1.0], (lb = [2.0], ub = [2.0]), 4.0, [2.0]),
        ((u, p) -> maximum(u .^ 2), [1.0, 0.5], (lb = [2.0, 1.0], ub = [2.0, 1.0]), 4.0, [2.0, 1.0]),
        ((u, p) -> sum(abs.(u .^ 2)), [1.0, 0.5], (lb = [2.0, 1.0], ub = [2.0, 1.0]), 5.0, [2.0, 1.0]),
        ((u, p) -> norm(u .^ 2), [1.0, 0.5], (lb = [2.0, 1.0], ub = [2.0, 1.0]), sqrt(17.0), [2.0, 1.0]),
        (
            (u, p) -> norm([exp(u[1]), u[2]]), [0.4, 0.9],
            (lb = [0.5, 1.0], ub = [0.5, 1.0]), sqrt(exp(1.0) + 1.0), [0.5, 1.0],
        ),
        ((u, p) -> exp(exp(u[1])), [0.4], (lb = [0.5], ub = [0.5]), exp(exp(0.5)), [0.5]),
    ]
    for (f, u0, kw, obj★, u★) in cases
        sol = solve_checked(
            ConvexOptimizationProblem(OptimizationFunction(f), u0; kw...), ALG
        )
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.objective, obj★; atol = 1.0e-6)
        u★ === nothing || @test isapprox(sol.u, u★; atol = 1.0e-6)
    end

    # Nested concave under MaxSense: max_u min(u1, log(u2)), pinned finite.
    solm = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> min(u[1], log(u[2]))), [1.0, 2.0];
            sense = SciMLBase.MaxSense, lb = [log(4.0), 4.0], ub = [log(4.0), 4.0]
        ), ALG
    )
    @test SciMLBase.successful_retcode(solm.retcode)
    @test isapprox(solm.objective, log(4.0); atol = 1.0e-6)

    # These lower fine but the original composition is not DCP for the sense.
    bad = [
        ((u, p) -> abs2(max(u[1], u[2])), [0.5, 0.5], (;)),
        ((u, p) -> abs(max(u[1], u[2])), [0.5, 0.5], (;)),
        ((u, p) -> max(u[1], min(u[2], 0.0)), [0.5, 0.5], (;)),
        ((u, p) -> max(u[1], log(u[2])), [0.5, 0.5], (;)),
        ((u, p) -> exp(-norm(u)), [0.5, 0.5], (;)),
        ((u, p) -> -log(norm(u) + 1.0), [0.5, 0.5], (;)),
        ((u, p) -> norm(u .^ 2 .- 1.0), [0.5, 0.5], (;)),
        ((u, p) -> norm([max(u[1], u[2]), u[3]]), [0.5, 0.5, 0.5], (;)),
        # nested concave under MinSense / nested convex under MaxSense
        ((u, p) -> min(u[1], log(u[2])), [0.5, 0.5], (;)),
        ((u, p) -> norm(u)^2, [0.5, 0.5], (sense = SciMLBase.MaxSense,)),
    ]
    for (f, u0, kw) in bad
        prob = ConvexOptimizationProblem(OptimizationFunction(f), u0; kw...)
        @test_throws "not certified convex" solve(prob, ALG)
    end
end

@testset "nested atoms with parameters re-solve through reinit!" begin
    # min ||A u - p||^2: the parameter stays an affine constant in the cone.
    optf = OptimizationFunction((u, p) -> norm(QLS_A * u .- p)^2)
    prob = ConvexOptimizationProblem(optf, zeros(3), QLS_B)
    cache = init(prob, ALG)
    for θ in (QLS_B, [2.0, 1.0, 0.0, -1.0], zeros(4))
        cache = reinit!(cache; p = θ)
        sol = solve!_checked(cache)
        cold = solve_checked(SciMLBase.remake(prob; p = θ), ALG)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, cold.u; atol = 1.0e-8)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
    end

    # exp(||u - p||): u* = θ, obj = 1 at every θ.
    optf2 = OptimizationFunction((u, p) -> exp(norm(u .- p)))
    prob2 = ConvexOptimizationProblem(optf2, [0.0, 0.0], [1.0, 2.0])
    cache2 = init(prob2, ALG)
    for θ in ([1.0, 2.0], [-1.0, 0.5], [0.0, 0.0])
        cache2 = reinit!(cache2; p = θ)
        sol = solve!_checked(cache2)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, θ; atol = 1.0e-6)
        @test isapprox(sol.objective, 1.0; atol = 1.0e-6)
    end

    # (exp(u1) + p)^2 is DCP at p = 0 and nonconvex at p = -2; with parameters
    # symbolic in the certificate it refuses at init for every p.
    pprob = (f, p; kw...) -> ConvexOptimizationProblem(
        OptimizationFunction(f), [0.0, 0.0], p;
        constraints = [ConeConstraint((u, p) -> [u[1], u[2]], MOI.Zeros(2))],
        kw...
    )
    @test_throws "not certified convex" solve_checked(
        pprob((u, p) -> (exp(u[1]) + p[1])^2, [0.0]), ALG
    )
    # (exp(u1) + p^2)^2 is convex for every θ — the certificate must see p^2 ≥ 0.
    sol = solve_checked(pprob((u, p) -> (exp(u[1]) + p[1]^2)^2, [0.0]), ALG)
    @test isapprox(sol.objective, 1.0; atol = 1.0e-5)
end

@testset "nested certification holds for every p" begin
    # The inner argument changes sign with p: convex at some θ only → refused.
    pprob1 = (f, p; kw...) -> ConvexOptimizationProblem(
        OptimizationFunction(f), [0.5], p; kw...
    )
    pprob2 = (f, p; kw...) -> ConvexOptimizationProblem(
        OptimizationFunction(f), [0.5, 0.5], p; kw...
    )
    @test_throws "not certified convex" solve_checked(
        pprob1((u, p) -> abs2(abs(u[1]) + p[1]), [0.0]), ALG
    )
    @test_throws "not certified convex" solve_checked(
        pprob2((u, p) -> abs2(norm(u) + p[1]), [0.0]), ALG
    )
    @test_throws "not certified convex" solve_checked(
        pprob2((u, p) -> abs2(p[1] * norm(u)), [2.0]), ALG
    )
    @test_throws "not certified convex" solve_checked(
        pprob2((u, p) -> p[1] * norm(u .- 1.0)^2, [2.0]), ALG
    )
    # max(·, 0) is not provably nonnegative to SymbolicAnalysis: conservative refusal.
    @test_throws "not certified convex" solve_checked(
        pprob2((u, p) -> max(norm(u) + p[1], 0.0)^2, [2.0]), ALG
    )

    optf = OptimizationFunction((u, p) -> abs(u[1] - p[1])^2 + abs(u[2] - p[2])^2)
    prob = ConvexOptimizationProblem(optf, [0.0, 0.0], [1.0, -1.0])
    cache = init(prob, ALG)
    for θ in ([1.0, -1.0], [-2.0, 0.5], [0.0, 0.0])
        cache = reinit!(cache; p = θ)
        sol = solve!_checked(cache)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test isapprox(sol.u, θ; atol = 1.0e-5)
        @test isapprox(sol.objective, 0.0; atol = 1.0e-5)
    end
end

@testset "θ-only subterms certify as constants" begin
    # A p-only subterm certifies as an unknown-sign constant; p[i]^2 keeps its sign.
    pprob = (f, p; kw...) -> ConvexOptimizationProblem(
        OptimizationFunction(f), [0.0, 0.0], p; kw...
    )
    cases = (
        ((u, p) -> norm(u)^2 - p[1]^2, [1.0], θ -> -θ[1]^2, θ -> [0.0, 0.0]),
        (
            (u, p) -> norm(u .- p)^2 - norm(p)^2, [1.0, -1.0],
            θ -> -sum(abs2, θ), θ -> θ,
        ),
        ((u, p) -> p[1]^2 * norm(u .- 1.0)^2, [2.0], θ -> 0.0, θ -> [1.0, 1.0]),
    )
    thetas = ([2.0, -1.0], [-1.5, 0.5], [0.0, 0.0])
    for (f, p0, expected_obj, expected_u) in cases
        prob = pprob(f, p0)
        cache = init(prob, ALG)
        for θ in thetas
            θp = θ[1:length(p0)]
            cache = reinit!(cache; p = θp)
            sol = solve!_checked(cache)
            cold = solve_checked(SciMLBase.remake(prob; p = θp), ALG)
            @test isapprox(sol.objective, expected_obj(θp); atol = 1.0e-6)
            @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
            all(==(0.0), θp) && continue
            @test isapprox(sol.u, expected_u(θp); atol = 1.0e-6)
        end
    end

    prob = pprob((u, p) -> abs(p[1]) * norm(u)^2, [1.0])
    cache = init(prob, ALG)
    for p in ([1.0], [-2.0], [0.0])
        cache = reinit!(cache; p)
        sol = solve!_checked(cache)
        @test isapprox(sol.objective, 0.0; atol = 1.0e-6)
    end
    # A bare parameter of unknown sign still decides curvature: refused.
    @test_throws "not certified convex" solve_checked(
        pprob((u, p) -> p[1] * norm(u)^2, [1.0]), ALG
    )
end

@testset "parameter-only atoms use exact data" begin
    cases = (
        (
            (u, p) -> abs(u[1] + abs(p[1]) - 1) + u[1]^2,
            [2.0], ([2.0], [-0.25], [0.0], [-1.0]),
            p -> [clamp(1 - abs(p[1]), -0.5, 0.5)],
        ),
        (
            (u, p) -> abs(u[1] + exp(p[1]) - 2) + u[1]^2,
            [0.0], ([0.0], [log(2.25)], [log(1.5)]),
            p -> [clamp(2 - exp(p[1]), -0.5, 0.5)],
        ),
        (
            (u, p) -> u[1]^2 + norm(p),
            [1.0], ([1.0], [-2.0], [0.0]),
            p -> [0.0],
        ),
        (
            (u, p) -> max(p[1], p[2]) * 1 + norm(u .- p),
            [1.0, -2.0], ([1.0, -2.0], [-2.0, 3.0], [0.0, 0.0]),
            p -> p,
        ),
    )
    for (f, p0, thetas, expected_u) in cases
        prob = ConvexOptimizationProblem(OptimizationFunction(f), zeros(length(expected_u(p0))), p0)
        cache = init(prob, ALG_TIGHT)
        for p in thetas
            cache = reinit!(cache; p)
            sol = solve!_checked(cache)
            @test SciMLBase.successful_retcode(sol.retcode)
            @test isapprox(sol.u, expected_u(p); atol = 1.0e-5)
            @test isapprox(sol.objective, f(expected_u(p), p); atol = 1.0e-5)
        end
    end
end

@testset "parameter vectors are converted to Float64 vectors" begin
    for p in (SVector{0, Float64}(), Any[])
        prob = ConvexOptimizationProblem(OptimizationFunction((u, p) -> abs(u[1] - 1)), [0.0], p)
        sol = solve_checked(prob, ALG)
        @test isapprox(sol.u, [1.0]; atol = 1.0e-5)
    end
    prob = ConvexOptimizationProblem(
        OptimizationFunction((u, p) -> abs(u[1] - p[1])), [0.0], SVector(2.0)
    )
    cache = init(prob, ALG)
    for p in (SVector(2.0), SVector(-1.0))
        cache = reinit!(cache; p)
        sol = solve!_checked(cache)
        @test isapprox(sol.u, collect(p); atol = 1.0e-5)
    end
end

@testset "nested atoms in open boxes hit the analytic minimizer" begin
    # min ‖abs.(u)‖ on u ∈ [-3,-2]^2: |u| ∈ [2,3]^2, minimized at u = (-2,-2).
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> norm(abs.(u))), [-2.5, -2.5];
            lb = [-3.0, -3.0], ub = [-2.0, -2.0]
        ), ALG
    )
    @test isapprox(sol.u, [-2.0, -2.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, 2 * sqrt(2.0); atol = 1.0e-6)

    # exp(‖u‖ - 3) is increasing in ‖u‖: same minimizer.
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> exp(norm(u) - 3.0)), [0.0, 0.0];
            lb = [-3.0, -3.0], ub = [-2.0, -2.0]
        ), ALG
    )
    @test isapprox(sol.u, [-2.0, -2.0]; atol = 1.0e-6)
    @test isapprox(sol.objective, exp(2 * sqrt(2.0) - 3.0); atol = 1.0e-6)

    # (‖u‖ + p^2)^2: the argument is nonnegative at every θ.
    optf = OptimizationFunction((u, p) -> (norm(u) + p[1]^2)^2)
    prob = ConvexOptimizationProblem(
        optf, [0.0, 0.0], [0.0]; lb = [-3.0, -3.0], ub = [-2.0, -2.0]
    )
    cache = init(prob, ALG)
    for θ in ([0.0], [1.0], [-2.0], [0.5])
        cache = reinit!(cache; p = θ)
        sol = solve!_checked(cache)
        cold = solve_checked(SciMLBase.remake(prob; p = θ), ALG)
        @test isapprox(sol.u, [-2.0, -2.0]; atol = 1.0e-6)
        @test isapprox(sol.objective, (2 * sqrt(2.0) + θ[1]^2)^2; atol = 1.0e-5)
        @test isapprox(sol.objective, cold.objective; atol = 1.0e-8)
    end
end

@testset "an empty AbstractVector p behaves like NullParameters" begin
    sol = solve_checked(
        ConvexOptimizationProblem(
            OptimizationFunction((u, p) -> norm(u .- 1.0)^2 + norm(u)^2),
            [1.0, 2.0], Float64[]
        ), ALG
    )
    @test isapprox(sol.objective, 1.0; atol = 1.0e-5)
end
