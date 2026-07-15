using OptimizationBase, OptimizationIpopt
using Zygote
using Symbolics
using Test
using SparseArrays
using ModelingToolkit
using ReverseDiff

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]
l1 = rosenbrock(x0, _p)

optfunc = OptimizationFunction((x, p) -> -rosenbrock(x, p), OptimizationBase.AutoZygote())
prob = OptimizationProblem(optfunc, x0, _p; sense = OptimizationBase.MaxSense)

callback = function (state, l)
    display(l)
    return false
end

sol = solve(prob, IpoptOptimizer(hessian_approximation = "exact"); callback)
@test SciMLBase.successful_retcode(sol)
@test sol ≈ [1, 1]

sol = solve(prob, IpoptOptimizer(hessian_approximation = "limited-memory"); callback)
@test SciMLBase.successful_retcode(sol)
@test sol ≈ [1, 1]

# HS071 translated from the Ipopt examples (examples/hs071_cpp),
# https://github.com/coin-or/Ipopt, licensed under Eclipse Public License - v 2.0
# https://github.com/coin-or/Ipopt/blob/stable/3.14/LICENSE
function _test_sparse_derivatives_hs071(backend, optimizer)
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        return res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2,
        ]
    end
    prob = OptimizationProblem(
        OptimizationFunction(objective, backend; cons = constraints),
        [1.0, 5.0, 5.0, 1.0];
        sense = OptimizationBase.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0]
    )
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; atol = 1.0e-6)
    x = [1.0, 4.742999641809297, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.u, x; atol = 1.0e-6)
    @test prod(sol.u) >= 25.0 - 1.0e-6
    @test isapprox(sum(sol.u .^ 2), 40.0; atol = 1.0e-6)
    return
end

@testset "backends" begin
    backends = (
        AutoForwardDiff(),
        AutoReverseDiff(),
        AutoSparse(AutoForwardDiff()),
    )
    for backend in backends
        @testset "$backend" begin
            _test_sparse_derivatives_hs071(backend, IpoptOptimizer())
        end
    end
end

@testset "sparse constraint Jacobian with parameters" begin
    objective(x, p) = sum(abs2, x)
    function constraints!(res, x, p)
        res[1] = p[1] * x[1]
        return nothing
    end
    function constraint_jacobian!(J, x, p)
        J[1, 1] = p[1]
        return nothing
    end

    @testset "analytic = $analytic" for analytic in (false, true)
        jacobian_kwargs = analytic ?
            (;
                cons_j = constraint_jacobian!,
                cons_jac_prototype = sparse([1], [1], [1.0], 1, 2),
            ) : (;)
        f = OptimizationFunction{true}(
            objective, AutoSparse(AutoForwardDiff());
            cons = constraints!, jacobian_kwargs...
        )
        prob = OptimizationProblem(
            f, [1.0, 1.0], [2.0]; lcons = [0.0], ucons = [0.0]
        )
        sol = solve(
            prob,
            IpoptOptimizer(;
                hessian_approximation = "limited-memory",
                additional_options = Dict{String, Any}("print_level" => 0)
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test isapprox(sol.u[1], 0.0; atol = 1.0e-8)
    end
end

include("additional_tests.jl")
include("advanced_features.jl")
include("problem_types.jl")


@testset "tutorial" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 1.0]

    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    function lagh(res, x, sigma, mu, p)
        lH = sigma * [
            2 + 8(x[1]^2) * p[2] - 4(x[2] - (x[1]^2)) * p[2] -4p[2] * x[1]
            -4p[2] * x[1] 2p[2]
        ] .+ [
            2mu[1] mu[2]
            mu[2] 2mu[1]
        ]
        res .= lH[[1, 3, 4]]
    end
    lag_hess_prototype = sparse([1 1; 0 1])

    optprob = OptimizationFunction(
        rosenbrock, OptimizationBase.AutoForwardDiff();
        cons = cons, lag_h = lagh, lag_hess_prototype
    )
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])
    sol = solve(prob, IpoptOptimizer())

    @test SciMLBase.successful_retcode(sol)
end

@testset "MTK cache" begin
    @variables x
    @parameters a = 1.0
    @named sys = OptimizationSystem((x - a)^2, [x], [a])
    sys = complete(sys)
    prob = OptimizationProblem(sys, [x => 0.0]; grad = true, hess = true)
    cache = init(prob, IpoptOptimizer(); verbose = false)
    @test cache isa OptimizationBase.OptimizationCache
    sol = solve!(cache)
    @test sol.u ≈ [1.0] # ≈ [1]

    # reinit! with new parameters is still blocked upstream for MTK problems:
    # a plain vector cannot be `convert`ed into the concretely typed
    # `ReInitCache{_, MTKParameters}` field, and a symbolic map hits the
    # generic `process_p_u0_symbolic` fallback in SciMLBase ("Symbolic remake
    # for OptimizationCache ... is currently not supported").
    @test_broken begin
        cache = OptimizationBase.reinit!(cache; p = [a => 2.0])
        sol = solve!(cache)
        sol.u ≈ [2.0]
    end
end

@testset "reinit!" begin
    optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
    prob = OptimizationProblem(optfunc, zeros(2), [1.0, 100.0])
    cache = init(prob, IpoptOptimizer())
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-6

    # The new parameters must be picked up by the objective and all the
    # derivative callbacks
    cache = OptimizationBase.reinit!(cache; p = [2.0, 100.0])
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [2.0, 4.0] atol = 1.0e-5
end

@testset "MaxSense objective sign" begin
    # Maximization is handled through Ipopt's obj_scaling_factor, so the
    # reported objective (and the value passed to callbacks) must keep the
    # user's sign convention. Optimum: 5 at (1, 0).
    maxf(x, p) = -(x[1] - 1.0)^2 - x[2]^2 + 5.0
    callback_objs = Float64[]
    cb = (state, l) -> (push!(callback_objs, l); false)
    prob = OptimizationProblem(
        OptimizationFunction(maxf, OptimizationBase.AutoForwardDiff()), [3.0, 2.0];
        sense = OptimizationBase.MaxSense
    )
    sol = solve(prob, IpoptOptimizer(); callback = cb)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 0.0] atol = 1.0e-6
    @test sol.objective ≈ 5.0 atol = 1.0e-6
    @test last(callback_objs) ≈ 5.0 atol = 1.0e-4
end

@testset "limited-memory skips Hessian generation" begin
    optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
    prob = OptimizationProblem(optfunc, zeros(2), [1.0, 100.0])

    opt = IpoptOptimizer(hessian_approximation = "limited-memory")
    @test !SciMLBase.requireshessian(opt)
    @test !SciMLBase.requireslagh(opt)
    @test !SciMLBase.requiresconshess(opt)
    # also when requested through additional_options
    opt2 = IpoptOptimizer(
        additional_options = Dict("hessian_approximation" => "limited-memory")
    )
    @test !SciMLBase.requireshessian(opt2)

    cache = init(prob, opt)
    @test cache.f.hess === nothing
    @test cache.f.lag_h === nothing
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-5
end

@testset "Additional Options and Common Interface" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    p = [1.0, 100.0]

    @testset "additional_options dictionary" begin
        optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, x0, p)

        # Test with various option types
        opt = IpoptOptimizer(
            additional_options = Dict(
                "derivative_test" => "first-order",  # String
                "derivative_test_tol" => 1.0e-4,       # Float64
                "derivative_test_print_all" => "yes" # String
            )
        )
        sol = solve(prob, opt)
        @test SciMLBase.successful_retcode(sol)

        # Test options not in struct fields
        opt2 = IpoptOptimizer(
            additional_options = Dict(
                "fixed_variable_treatment" => "make_parameter",
                "required_infeasibility_reduction" => 0.9,
                "alpha_for_y" => "primal"
            )
        )
        sol2 = solve(prob, opt2)
        @test SciMLBase.successful_retcode(sol2)
    end

    @testset "Common interface arguments override" begin
        optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, x0, p)

        # Test that reltol overrides default tolerance
        sol1 = solve(prob, IpoptOptimizer(); reltol = 1.0e-12)
        @test SciMLBase.successful_retcode(sol1)
        @test sol1.u ≈ [1.0, 1.0] atol = 1.0e-10

        # Test that maxiters limits iterations
        sol2 = solve(prob, IpoptOptimizer(); maxiters = 5)
        # May not converge with only 5 iterations
        @test sol2.stats.iterations <= 5

        # Test verbose levels
        for verbose in [false, true, 0, 3, 5]
            sol = solve(prob, IpoptOptimizer(); verbose = verbose, maxiters = 10)
            @test sol isa SciMLBase.OptimizationSolution
        end

        # Test maxtime
        sol3 = solve(prob, IpoptOptimizer(); maxtime = 10.0)
        @test SciMLBase.successful_retcode(sol3)
    end

    @testset "Priority: additional_options > solve args" begin
        optfunc = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
        prob = OptimizationProblem(optfunc, x0, p)

        # An explicit Ipopt option in additional_options wins over the common
        # solve argument: rosenbrock from zeros needs far more than 5
        # iterations, so hitting max_iter = 5 proves which setting was applied
        opt = IpoptOptimizer(
            additional_options = Dict("max_iter" => 5)
        )
        sol = solve(prob, opt; maxiters = 100)
        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
        @test sol.stats.iterations <= 5

        # Without the explicit option, the solve argument reaches Ipopt
        sol2 = solve(prob, IpoptOptimizer(); maxiters = 5)
        @test sol2.retcode == SciMLBase.ReturnCode.MaxIters
        @test sol2.stats.iterations <= 5
    end
end
