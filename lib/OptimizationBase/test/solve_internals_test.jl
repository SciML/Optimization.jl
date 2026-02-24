using OptimizationBase, Test, SciMLBase

import OptimizationBase: solve_up, solve_call, get_concrete_problem, extract_opt_alg
using SciMLBase: ReturnCode

# ============================================================
# Mock solver infrastructure
# ============================================================

struct MockSolver end

# Track what the mock solver was called with
const _mock_ncalls = Ref(0)
const _mock_kwargs = Ref{NamedTuple}((;))
const _mock_u0 = Ref{Vector{Float64}}(Float64[])

function OptimizationBase.__solve(
        prob::SciMLBase.OptimizationProblem, ::MockSolver, args...; kwargs...
    )
    _mock_ncalls[] += 1
    _mock_kwargs[] = NamedTuple(kwargs)
    _mock_u0[] = copy(prob.u0)
    cache = SciMLBase.DefaultOptimizationCache(prob.f, prob.p)
    stats = SciMLBase.OptimizationStats(; iterations = 1, time = 0.0, fevals = 1)
    return SciMLBase.build_solution(
        cache, MockSolver(), prob.u0, prob.f.f(prob.u0, prob.p);
        retcode = ReturnCode.Success,
        stats = stats
    )
end

function reset_mock!()
    _mock_ncalls[] = 0
    _mock_kwargs[] = (;)
    return _mock_u0[] = Float64[]
end

# Helper: build a simple unconstrained problem
function simple_prob(u0 = [1.0, 2.0]; kwargs...)
    f = OptimizationFunction((x, p) -> sum(x .^ 2), SciMLBase.NoAD())
    return OptimizationProblem(f, u0; kwargs...)
end

# ============================================================
# extract_opt_alg
# ============================================================

@testset "extract_opt_alg" begin
    alg = MockSolver()

    # alg provided as first positional argument
    @test extract_opt_alg((alg,), (;), (;)) === alg

    # nothing as first positional arg falls through to solve kwargs
    @test extract_opt_alg((nothing,), (; alg), (;)) === alg

    # empty positional args → use solve kwargs
    @test extract_opt_alg((), (; alg), (;)) === alg

    # not in solve kwargs → fall back to prob kwargs
    @test extract_opt_alg((), (;), (; alg)) === alg

    # no alg anywhere → returns nothing
    @test isnothing(extract_opt_alg((), (;), (;)))

    # solve kwargs take priority over prob kwargs
    alg2 = MockSolver()
    @test extract_opt_alg((), (; alg), (; alg = alg2)) === alg
end

# ============================================================
# get_concrete_problem
# ============================================================

@testset "get_concrete_problem" begin
    prob = simple_prob([1.0, 2.0])

    # returns a problem with the same u0 by default
    new_prob = get_concrete_problem(prob)
    @test new_prob.u0 ≈ [1.0, 2.0]

    # u0 can be overridden via kwargs
    new_prob2 = get_concrete_problem(prob; u0 = [3.0, 4.0])
    @test new_prob2.u0 ≈ [3.0, 4.0]

    # p can be overridden via kwargs
    f = OptimizationFunction((x, p) -> sum(x) + sum(p), SciMLBase.NoAD())
    prob_p = OptimizationProblem(f, [1.0, 2.0], [10.0])
    new_prob3 = get_concrete_problem(prob_p; p = [20.0])
    @test new_prob3.p == [20.0]

    # overriding u0 does not modify the original problem
    @test prob.u0 ≈ [1.0, 2.0]
end

# ============================================================
# solve_call
# ============================================================

@testset "solve_call" begin
    prob = simple_prob()
    alg = MockSolver()

    # basic: calls __solve and returns a successful solution
    reset_mock!()
    sol = solve_call(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test _mock_ncalls[] == 1

    # solution u is the prob's u0 (mock solver does not change it)
    @test sol.u ≈ [1.0, 2.0]

    # kwargs stored in prob are forwarded to __solve
    reset_mock!()
    prob_kw = simple_prob(; maxiters = 100)
    solve_call(prob_kw, alg)
    @test _mock_kwargs[][:maxiters] == 100

    # kwargs passed to solve_call override those in the problem
    reset_mock!()
    solve_call(prob_kw, alg; maxiters = 200)
    @test _mock_kwargs[][:maxiters] == 200

    # non-concrete eltype in u0 throws NonConcreteEltypeError
    prob_bad = OptimizationProblem(
        OptimizationFunction((x, p) -> sum(x), SciMLBase.NoAD()),
        Number[1, 2]
    )
    @test_throws SciMLBase.NonConcreteEltypeError solve_call(prob_bad, alg)
end

# ============================================================
# solve_up
# ============================================================

@testset "solve_up" begin
    prob = simple_prob()
    alg = MockSolver()

    # basic: alg passed as first positional argument
    reset_mock!()
    sol = solve_up(prob, nothing, prob.u0, prob.p, alg)
    @test SciMLBase.successful_retcode(sol)
    @test _mock_ncalls[] == 1

    # u0 override is propagated to the concrete problem passed to __solve
    reset_mock!()
    solve_up(prob, nothing, [5.0, 6.0], prob.p, alg)
    @test _mock_u0[] ≈ [5.0, 6.0]

    # p override is used when building the concrete problem
    f = OptimizationFunction((x, p) -> sum(x) + sum(p), SciMLBase.NoAD())
    prob_p = OptimizationProblem(f, [1.0, 2.0], [10.0])
    reset_mock!()
    sol_p = solve_up(prob_p, nothing, prob_p.u0, [20.0], alg)
    @test SciMLBase.successful_retcode(sol_p)
    @test sol_p.objective ≈ sum([1.0, 2.0]) + sum([20.0])  # 1+2+20 = 23

    # alg can be supplied via prob kwargs instead of positional args
    prob_alg = OptimizationProblem(
        OptimizationFunction((x, p) -> sum(x .^ 2), SciMLBase.NoAD()),
        [1.0, 2.0]; alg = MockSolver()
    )
    reset_mock!()
    sol2 = solve_up(prob_alg, nothing, prob_alg.u0, prob_alg.p)
    @test SciMLBase.successful_retcode(sol2)
    @test _mock_ncalls[] == 1
end

# ============================================================
# Mock solver with has_init = true (init/solve! routing path)
# ============================================================

struct MockSolverWithInit end
SciMLBase.has_init(::MockSolverWithInit) = true

# Use the default SciMLBase.__init to build an OptimizationCache; only __solve is needed.
function OptimizationBase.__solve(
        cache::OptimizationBase.OptimizationCache{MockSolverWithInit}
    )
    stats = SciMLBase.OptimizationStats(; iterations = 1, time = 0.0, fevals = 1)
    u = cache.reinit_cache.u0
    return SciMLBase.build_solution(
        cache, cache.opt, u, cache.f.f(u, cache.reinit_cache.p);
        retcode = ReturnCode.Success,
        stats = stats
    )
end

# Mock solver that allows constraints (for _check_opt_alg tests)
struct MockAlgWithCons end
OptimizationBase.allowsconstraints(::MockAlgWithCons) = true

function OptimizationBase.__solve(
        prob::SciMLBase.OptimizationProblem, ::MockAlgWithCons, args...; kwargs...
    )
    cache = SciMLBase.DefaultOptimizationCache(prob.f, prob.p)
    stats = SciMLBase.OptimizationStats(; iterations = 1, time = 0.0, fevals = 1)
    return SciMLBase.build_solution(
        cache, MockAlgWithCons(), prob.u0, prob.f.f(prob.u0, prob.p);
        retcode = ReturnCode.Success,
        stats = stats
    )
end

# ============================================================
# init (public) — validation before reaching __init dispatch
# ============================================================

@testset "init validation" begin
    # non-concrete eltype in u0 throws before reaching __init
    prob_bad = OptimizationProblem(
        OptimizationFunction((x, p) -> sum(x), SciMLBase.NoAD()),
        Number[1, 2]
    )
    @test_throws SciMLBase.NonConcreteEltypeError init(prob_bad, MockSolver())

    # algorithm without bounds support rejects a bounded problem
    struct MockAlgNoBounds end
    # allowsbounds defaults to false, so a problem with lb/ub should throw
    prob_bounds = OptimizationProblem(
        OptimizationFunction((x, p) -> sum(x), SciMLBase.NoAD()),
        [1.0, 2.0]; lb = [0.0, 0.0], ub = [2.0, 2.0]
    )
    @test_throws OptimizationBase.IncompatibleOptimizerError init(
        prob_bounds, MockAlgNoBounds()
    )

    # algorithm that requires bounds rejects an unbounded problem
    struct MockAlgReqBounds end
    OptimizationBase.requiresbounds(::MockAlgReqBounds) = true
    @test_throws OptimizationBase.IncompatibleOptimizerError init(
        simple_prob(), MockAlgReqBounds()
    )

    # algorithm without constraint support rejects a constrained problem
    struct MockAlgNoCons end
    cons_f = OptimizationFunction(
        (x, p) -> sum(x), SciMLBase.NoAD();
        cons = (res, x, p) -> (res .= x[1]^2 + x[2]^2)
    )
    prob_cons = OptimizationProblem(cons_f, [1.0, 2.0])
    @test_throws OptimizationBase.IncompatibleOptimizerError init(
        prob_cons, MockAlgNoCons()
    )
end

# ============================================================
# solve_call — has_init routing (init/solve! path)
# ============================================================

@testset "solve_call has_init routing" begin
    prob = simple_prob()

    # when has_init returns true, solve_call routes through init then solve!
    sol = solve_call(prob, MockSolverWithInit())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 2.0]
end

# ============================================================
# _check_opt_alg — missing lcons/ucons with constrained algorithm
# ============================================================

@testset "_check_opt_alg missing lcons/ucons" begin
    f_cons = OptimizationFunction(
        (x, p) -> sum(x), SciMLBase.NoAD();
        cons = (res, x, p) -> (res .= x[1]^2 + x[2]^2)
    )

    # constraints present but lcons missing → ArgumentError
    prob_no_lcons = OptimizationProblem(f_cons, [1.0, 2.0]; ucons = [1.0])
    @test_throws ArgumentError solve_call(prob_no_lcons, MockAlgWithCons())

    # constraints present but ucons missing → ArgumentError
    prob_no_ucons = OptimizationProblem(f_cons, [1.0, 2.0]; lcons = [-Inf])
    @test_throws ArgumentError solve_call(prob_no_ucons, MockAlgWithCons())

    # both lcons and ucons provided → succeeds
    prob_ok = OptimizationProblem(f_cons, [1.0, 2.0]; lcons = [-Inf], ucons = [1.0])
    sol = solve_call(prob_ok, MockAlgWithCons())
    @test SciMLBase.successful_retcode(sol)
end

# ============================================================
# solve (public API)
# ============================================================

@testset "solve public API" begin
    prob = simple_prob()
    alg = MockSolver()

    # basic end-to-end solve
    reset_mock!()
    sol = solve(prob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test _mock_ncalls[] == 1

    # wrap=Val(false) skips wrap_sol but still returns a valid solution
    reset_mock!()
    sol_unwrapped = solve(prob, alg; wrap = Val(false))
    @test SciMLBase.successful_retcode(sol_unwrapped)

    # u0 kwarg overrides the problem's initial point
    reset_mock!()
    solve(prob, alg; u0 = [5.0, 6.0])
    @test _mock_u0[] ≈ [5.0, 6.0]

    # sensealg stored in prob.kwargs is extracted without error
    prob_sa = OptimizationProblem(
        OptimizationFunction((x, p) -> sum(x .^ 2), SciMLBase.NoAD()),
        [1.0, 2.0]; sensealg = nothing
    )
    reset_mock!()
    @test SciMLBase.successful_retcode(solve(prob_sa, alg))
end

# ============================================================
# solve_up — extra positional args (length(args) > 1 branch)
# ============================================================

@testset "solve_up extra args" begin
    prob = simple_prob()
    alg = MockSolver()

    # passing an extra positional argument after alg takes the Base.tail branch
    reset_mock!()
    sol = solve_up(prob, nothing, prob.u0, prob.p, alg, :extra_arg)
    @test SciMLBase.successful_retcode(sol)
    @test _mock_ncalls[] == 1
end
