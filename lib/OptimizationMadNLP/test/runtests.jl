using OptimizationMadNLP
using OptimizationBase
using MadNLP
using Test
using ADTypes
import Zygote, ForwardDiff, ReverseDiff
using SparseArrays
using DifferentiationInterface: SecondOrder
using Random

@testset "rosenbrock" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    # MadNLP requires second-order derivatives
    ad = SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())
    optfunc = OptimizationFunction(
        (x, p) -> -rosenbrock(x, p), ad
    )
    prob = OptimizationProblem(optfunc, x0, _p; sense = OptimizationBase.MaxSense)

    sol = solve(prob, MadNLPOptimizer(), verbose = true)

    @test sol ≈ [1, 1]
end

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
        # MadNLP uses lower triangle. For symmetric sparse([1 1; 1 1]), lower triangle has [1,1], [2,1], and [2,2]
        res[1] = lH[1, 1]  # Position [1,1]
        res[2] = lH[2, 1]  # Position [2,1] (off-diagonal)
        res[3] = lH[2, 2]  # Position [2,2]
    end
    lag_hess_prototype = sparse([1 1; 1 1])  # Symmetric sparse pattern for Hessian

    # Use SecondOrder AD for MadNLP
    ad = SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())
    optprob = OptimizationFunction(
        rosenbrock, ad;
        cons = cons, lag_h = lagh, lag_hess_prototype
    )
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])

    opts = [
        MadNLPOptimizer(),
        MadNLPOptimizer(linear_solver = LapackCPUSolver),
    ]

    for opt in opts
        sol = solve(prob, opt)
        @test SciMLBase.successful_retcode(sol)

        # compare against Ipopt results
        @test sol ≈ [0.7071678163428006, 0.7070457460302945] rtol = 1.0e-4
    end
end

@testset "cache" begin
    objective(x, p) = (p[1] - x[1])^2
    x0 = zeros(1)
    p = [1.0]

    # Use SecondOrder AD for MadNLP
    @testset "$ad" for ad in [
            SecondOrder(AutoZygote(), AutoZygote()),
            SecondOrder(AutoForwardDiff(), AutoZygote()),
            SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
        ]
        optf = OptimizationFunction(objective, ad)
        prob = OptimizationProblem(optf, x0, p)
        cache = OptimizationBase.init(prob, MadNLPOptimizer())
        sol = OptimizationBase.solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0] atol = 1.0e-3

        cache = OptimizationBase.reinit!(cache; p = [2.0])
        sol = OptimizationBase.solve!(cache)
        # @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [2.0] atol = 1.0e-3
    end
end

@testset "constraints & AD" begin
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2,
        ]
    end

    x0 = [1.0, 5.0, 5.0, 1.0]

    @testset "$ad" for ad in [
            AutoSparse(SecondOrder(AutoForwardDiff(), AutoZygote())),
            AutoSparse(SecondOrder(AutoForwardDiff(), AutoReverseDiff())),
        ]
        optfunc = OptimizationFunction(objective, ad, cons = constraints)
        prob = OptimizationProblem(
            optfunc, x0; sense = OptimizationBase.MinSense,
            lb = [1.0, 1.0, 1.0, 1.0],
            ub = [5.0, 5.0, 5.0, 5.0],
            lcons = [25.0, 40.0],
            ucons = [Inf, 40.0]
        )

        cache = init(prob, MadNLPOptimizer())

        sol = OptimizationBase.solve!(cache)

        @test SciMLBase.successful_retcode(sol)

        @test isapprox(sol.objective, 17.014017145179164; atol = 1.0e-6)
        x = [1.0, 4.742999641809297, 3.8211499817883077, 1.3794082897556983]
        @test isapprox(sol.u, x; atol = 1.0e-6)
        @test prod(sol.u) >= 25.0 - 1.0e-6
        @test isapprox(sum(sol.u .^ 2), 40.0; atol = 1.0e-6)
    end

    # AutoSparse(SecondOrder(AutoForwardDiff(), AutoForwardDiff())) fails due to
    # gradient dispatch MethodError. See GitHub issues #1137 and #1140 for tracking.
    @testset "AutoSparse(SecondOrder(AutoForwardDiff(), AutoForwardDiff())) - broken" begin
        ad = AutoSparse(SecondOrder(AutoForwardDiff(), AutoForwardDiff()))
        optfunc = OptimizationFunction(objective, ad, cons = constraints)
        prob = OptimizationProblem(
            optfunc, x0; sense = OptimizationBase.MinSense,
            lb = [1.0, 1.0, 1.0, 1.0],
            ub = [5.0, 5.0, 5.0, 5.0],
            lcons = [25.0, 40.0],
            ucons = [Inf, 40.0]
        )

        try
            cache = init(prob, MadNLPOptimizer())
            sol = OptimizationBase.solve!(cache)
            @test SciMLBase.successful_retcode(sol) broken = true
        catch e
            @test e isa MethodError broken = true
        end
    end

    # Dense tests with SecondOrder AD combinations. See GitHub issues #1137 and #1140 for tracking.
    @testset "Dense KKT with $ad - broken" for ad in [
            SecondOrder(AutoForwardDiff(), AutoZygote()),
            SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
            SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
        ]
        optfunc = OptimizationFunction(objective, ad, cons = constraints)
        prob = OptimizationProblem(
            optfunc, x0; sense = OptimizationBase.MinSense,
            lb = [1.0, 1.0, 1.0, 1.0],
            ub = [5.0, 5.0, 5.0, 5.0],
            lcons = [25.0, 40.0],
            ucons = [Inf, 40.0]
        )

        try
            cache = init(
                prob,
                MadNLPOptimizer(
                    kkt_system = MadNLP.DenseKKTSystem,
                    linear_solver = LapackCPUSolver
                )
            )

            sol = OptimizationBase.solve!(cache)
            @test SciMLBase.successful_retcode(sol) broken = true
            @test isapprox(sol.objective, 17.014017145179164; atol = 1.0e-6) broken = true
        catch e
            @test e isa Exception broken = true
        end
    end
end

@testset "Larger sparse Hessian" begin
    # Test with a 4x4 sparse Hessian matrix
    # min x1^2 + 2*x2^2 + x3^2 + x1*x3 + x2*x4
    # s.t. x1 + x2 + x3 + x4 = 4
    #      x1*x3 >= 1

    function objective_sparse(x, p)
        return x[1]^2 + 2 * x[2]^2 + x[3]^2 + x[1] * x[3] + x[2] * x[4]
    end

    function cons_sparse(res, x, p)
        res[1] = x[1] + x[2] + x[3] + x[4]  # Equality constraint
        res[2] = x[1] * x[3]                 # Inequality constraint
    end

    function lag_hess_sparse(res, x, sigma, mu, p)
        # Sparse Hessian structure (symmetric):
        # [2    0    1    0  ]
        # [0    4    0    1  ]
        # [1    0    2    0  ]
        # [0    1    0    0  ]
        #
        # Lower triangle indices: [1,1], [3,1], [2,2], [4,2], [3,3]
        # Total: 5 non-zero elements in lower triangle

        # Objective Hessian contribution
        res[1] = sigma * 2.0   # H[1,1] from x1^2
        res[2] = sigma * 1.0   # H[3,1] from x1*x3
        res[3] = sigma * 4.0   # H[2,2] from 2*x2^2
        res[4] = sigma * 1.0   # H[4,2] from x2*x4
        res[5] = sigma * 2.0   # H[3,3] from x3^2

        # Constraint contributions
        # First constraint (x1+x2+x3+x4=4) has zero Hessian
        # Second constraint (x1*x3>=1) has d²c/dx1dx3 = 1
        res[2] += mu[2] * 1.0  # Add to H[3,1]
    end

    # Create sparse prototype with the correct structure
    # We need 1s at positions: [1,1], [1,3], [2,2], [2,4], [3,1], [3,3], [4,2]
    hess_proto_4x4 = sparse(
        [1, 3, 2, 4, 1, 3, 2],  # row indices
        [1, 1, 2, 2, 3, 3, 4],  # column indices
        [1, 1, 1, 1, 1, 1, 1]   # values (just placeholder 1s)
    )

    x0 = [1.0, 1.0, 1.0, 1.0]
    p = Float64[]

    # Use SecondOrder AD for MadNLP
    ad = SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())
    optprob = OptimizationFunction(
        objective_sparse, ad;
        cons = cons_sparse, lag_h = lag_hess_sparse, lag_hess_prototype = hess_proto_4x4
    )

    prob = OptimizationProblem(
        optprob, x0, p,
        lcons = [4.0, 1.0],     # x1+x2+x3+x4 = 4, x1*x3 >= 1
        ucons = [4.0, Inf]
    )      # x1+x2+x3+x4 = 4, x1*x3 <= Inf

    sol = solve(prob, MadNLPOptimizer())

    @test SciMLBase.successful_retcode(sol)

    # Check constraints
    cons_vals = zeros(2)
    cons_sparse(cons_vals, sol.u, p)
    @test isapprox(cons_vals[1], 4.0, atol = 1.0e-6)  # Sum constraint
    @test cons_vals[2] >= 1.0 - 1.0e-6              # Product constraint
end

# "MadNLP Options and Common Interface" tests have SecondOrder AD compatibility issues.
# See GitHub issues #1137 and #1140 for tracking.
@testset "MadNLP Options and Common Interface - broken" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    p = [1.0, 100.0]
    ad = SecondOrder(AutoForwardDiff(), AutoZygote())

    @testset "MadNLP struct options" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        try
            # Test with MadNLP-specific struct fields
            opt = MadNLPOptimizer(
                acceptable_tol = 1.0e-6,
                acceptable_iter = 10,
                blas_num_threads = 2,
                mu_init = 0.01
            )
            sol = solve(prob, opt)
            @test SciMLBase.successful_retcode(sol) broken = true
        catch e
            @test e isa Exception broken = true
        end

        try
            # Test with hessian approximation
            opt2 = MadNLPOptimizer(
                hessian_approximation = MadNLP.CompactLBFGS,
                jacobian_constant = false,
                hessian_constant = false
            )
            sol2 = solve(prob, opt2)
            @test SciMLBase.successful_retcode(sol2) broken = true
        catch e
            @test e isa Exception broken = true
        end
    end

    @testset "additional_options dictionary" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        try
            # Test passing MadNLP options via additional_options
            opt = MadNLPOptimizer(
                linear_solver = MadNLP.UmfpackSolver,
                additional_options = Dict{Symbol, Any}(
                    :max_iter => 200,
                    :tol => 1.0e-7
                )
            )
            sol = solve(prob, opt)
            @test SciMLBase.successful_retcode(sol) broken = true
        catch e
            @test e isa Exception broken = true
        end

        try
            # Test with different options
            opt2 = MadNLPOptimizer(
                additional_options = Dict{Symbol, Any}(
                    :inertia_correction_method => MadNLP.InertiaFree,
                    :fixed_variable_treatment => MadNLP.RelaxBound
                )
            )
            sol2 = solve(prob, opt2)
            @test SciMLBase.successful_retcode(sol2) broken = true
        catch e
            @test e isa Exception broken = true
        end
    end

    @testset "Common interface arguments" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        try
            # Test that abstol overrides default tolerance
            sol1 = solve(prob, MadNLPOptimizer(); abstol = 1.0e-12)
            @test SciMLBase.successful_retcode(sol1) broken = true
            @test (sol1.u ≈ [1.0, 1.0]) broken = true
        catch e
            @test e isa Exception broken = true
        end

        try
            # Test that maxiters limits iterations
            sol2 = solve(prob, MadNLPOptimizer(); maxiters = 5)
            @test (sol2.stats.iterations <= 5) broken = true
        catch e
            @test e isa Exception broken = true
        end

        try
            # Test verbose options (MadNLP supports bool and LogLevels)
            for verbose in [false, true, MadNLP.ERROR, MadNLP.WARN, MadNLP.INFO]
                sol = solve(prob, MadNLPOptimizer(); verbose = verbose, maxiters = 20)
                @test (sol isa SciMLBase.OptimizationSolution) broken = true
            end
        catch e
            @test e isa Exception broken = true
        end
    end

    @testset "Priority: struct < additional_options < common solve args" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        try
            # Struct field is overridden by additional_options and solve arguments
            opt = MadNLPOptimizer(
                acceptable_tol = 1.0e-4,  # Struct field
                additional_options = Dict{Symbol, Any}(
                    :max_iter => 10,    # Will be overridden by maxiters
                    :tol => 1.0e-6        # Will be overridden by abstol
                )
            )

            sol = solve(
                prob, opt;
                maxiters = 5,   # Should override additional_options[:max_iter]
                abstol = 1.0e-10
            )  # Should override additional_options[:tol]

            @test (sol.stats.iterations <= 5) broken = true
            @test (sol.retcode == SciMLBase.ReturnCode.MaxIters) broken = true
        catch e
            @test e isa Exception broken = true
        end
    end
end

@testset verbose = true "LBFGS Hessian Approximation" begin
    # Based on https://madsuite.org/MadNLP.jl/dev/tutorials/lbfgs/

    @testset "Unconstrained LBFGS" begin
        # Extended Rosenbrock function (n-dimensional)
        function extended_rosenbrock(x, p)
            n = length(x)
            sum(100 * (x[2i] - x[2i - 1]^2)^2 + (1 - x[2i - 1])^2 for i in 1:div(n, 2))
        end

        n = 10  # Problem dimension
        x0 = zeros(n)
        x0[1:2:end] .= -1.2  # Starting point from tutorial
        x0[2:2:end] .= 1.0

        # Test CompactLBFGS (working)
        @testset "LBFGS variant: CompactLBFGS" begin
            ad = AutoForwardDiff()
            optfunc = OptimizationFunction(extended_rosenbrock, ad)
            prob = OptimizationProblem(optfunc, x0, nothing)

            opt = MadNLPOptimizer(
                hessian_approximation = MadNLP.CompactLBFGS
            )

            sol = solve(prob, opt; maxiters = 100, verbose = false)

            @test SciMLBase.successful_retcode(sol)
            @test all(isapprox.(sol.u, 1.0, atol = 1.0e-6))
            @test sol.objective < 1.0e-10
        end

        # Test ExactHessian (broken due to SecondOrder AD issues, see #1137 and #1140)
        @testset "LBFGS variant: ExactHessian - broken" begin
            ad = SecondOrder(AutoForwardDiff(), AutoZygote())
            optfunc = OptimizationFunction(extended_rosenbrock, ad)
            prob = OptimizationProblem(optfunc, x0, nothing)

            try
                opt = MadNLPOptimizer(
                    hessian_approximation = MadNLP.ExactHessian
                )

                sol = solve(prob, opt; maxiters = 100, verbose = false)
                @test SciMLBase.successful_retcode(sol) broken = true
                @test all(isapprox.(sol.u, 1.0, atol = 1.0e-6)) broken = true
                @test (sol.objective < 1.0e-10) broken = true
            catch e
                @test e isa Exception broken = true
            end
        end

        @testset "LBFGS memory size $memory_size" for memory_size in [5, 10, 20]
            # Test different memory sizes for L-BFGS
            ad = AutoForwardDiff()
            optfunc = OptimizationFunction(extended_rosenbrock, ad)
            prob = OptimizationProblem(optfunc, x0, nothing)

            opt = MadNLPOptimizer(
                hessian_approximation = MadNLP.CompactLBFGS,
                quasi_newton_options = MadNLP.QuasiNewtonOptions(max_history = memory_size)
            )

            sol = solve(prob, opt; maxiters = 100, verbose = false)

            @test SciMLBase.successful_retcode(sol)
            @test all(isapprox.(sol.u, 1.0, atol = 1.0e-6))
        end
    end

    @testset verbose = true "Constrained LBFGS - Electrons on Sphere" begin
        # Quasi-uniform distribution of electrons on a unit sphere
        # Minimize electrostatic potential energy (Coulomb potential)
        # Variables are organized as [x1, x2, ..., xn, y1, y2, ..., yn, z1, z2, ..., zn]
        # based on https://madsuite.org/MadNLP.jl/dev/tutorials/lbfgs

        function coulomb_potential(vars, p)
            # vars = [x1...xn, y1...yn, z1...zn]
            np = div(length(vars), 3)
            x = @view vars[1:np]
            y = @view vars[(np + 1):(2 * np)]
            z = @view vars[(2 * np + 1):(3 * np)]

            # Sum of 1/r_ij for all electron pairs
            energy = 0.0
            for i in 1:(np - 1)
                for j in (i + 1):np
                    dist_sq = (x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2
                    energy += 1.0 / sqrt(dist_sq)
                end
            end
            return energy
        end

        function unit_sphere_constraints(res, vars, p)
            # Each electron must lie on the unit sphere
            np = div(length(vars), 3)
            x = @view vars[1:np]
            y = @view vars[(np + 1):(2 * np)]
            z = @view vars[(2 * np + 1):(3 * np)]

            for i in 1:np
                res[i] = x[i]^2 + y[i]^2 + z[i]^2 - 1.0
            end
        end

        # Function to generate initial points on sphere
        function init_electrons_on_sphere(np)
            # Random.seed!(1)
            theta = 2π .* rand(np)
            phi = π .* rand(np)

            x0 = zeros(3 * np)
            # x coordinates
            x0[1:np] = cos.(theta) .* sin.(phi)
            # y coordinates
            x0[(np + 1):(2 * np)] = sin.(theta) .* sin.(phi)
            # z coordinates
            x0[(2 * np + 1):(3 * np)] = cos.(phi)

            return x0
        end

        # Test CompactLBFGS (working)
        @testset "N=5 electrons with CompactLBFGS" begin
            np = 5
            x0 = init_electrons_on_sphere(np)

            ad = AutoForwardDiff()

            optfunc = OptimizationFunction(
                coulomb_potential, ad,
                cons = unit_sphere_constraints
            )

            lcons = zeros(np)
            ucons = zeros(np)

            prob = OptimizationProblem(
                optfunc, x0;
                lcons = lcons,
                ucons = ucons
            )

            opt = MadNLPOptimizer(
                linear_solver = LapackCPUSolver,
                hessian_approximation = MadNLP.CompactLBFGS
            )

            sol = solve(prob, opt; abstol = 1.0e-7, maxiters = 200, verbose = false)

            @test SciMLBase.successful_retcode(sol)

            cons_vals = zeros(np)
            unit_sphere_constraints(cons_vals, sol.u, nothing)
            @test all(abs.(cons_vals) .< 1.0e-5)

            expected_energy = 6.474691495
            @test isapprox(sol.objective, expected_energy, rtol = 1.0e-3)

            x = sol.u[1:np]
            y = sol.u[(np + 1):(2 * np)]
            z = sol.u[(2 * np + 1):(3 * np)]

            min_dist = Inf
            for i in 1:(np - 1)
                for j in (i + 1):np
                    dist = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2)
                    min_dist = min(min_dist, dist)
                end
            end
            @test min_dist > 0.5
        end

        # Test ExactHessian (broken due to SecondOrder AD issues, see #1137 and #1140)
        @testset "N=5 electrons with ExactHessian - broken" begin
            np = 5
            x0 = init_electrons_on_sphere(np)

            ad = SecondOrder(AutoForwardDiff(), AutoZygote())

            optfunc = OptimizationFunction(
                coulomb_potential, ad,
                cons = unit_sphere_constraints
            )

            lcons = zeros(np)
            ucons = zeros(np)

            prob = OptimizationProblem(
                optfunc, x0;
                lcons = lcons,
                ucons = ucons
            )

            try
                opt = MadNLPOptimizer(
                    linear_solver = LapackCPUSolver,
                    hessian_approximation = MadNLP.ExactHessian
                )

                sol = solve(prob, opt; abstol = 1.0e-7, maxiters = 200, verbose = false)
                @test SciMLBase.successful_retcode(sol) broken = true
            catch e
                @test e isa Exception broken = true
            end
        end

        # "LBFGS vs Exact Hessian" test - broken due to SecondOrder AD issues.
        # See GitHub issues #1137 and #1140 for tracking.
        @testset verbose = true "LBFGS vs Exact Hessian - broken" begin
            np = 10
            x0 = init_electrons_on_sphere(np)

            results = Dict{String, NamedTuple}()

            # CompactLBFGS should work
            try
                optfunc = OptimizationFunction(
                    coulomb_potential, AutoForwardDiff(),
                    cons = unit_sphere_constraints
                )

                prob = OptimizationProblem(
                    optfunc, x0;
                    lcons = zeros(np),
                    ucons = zeros(np)
                )

                opt = MadNLPOptimizer(
                    hessian_approximation = MadNLP.CompactLBFGS
                )

                sol = solve(prob, opt; abstol = 1.0e-6, maxiters = 300, verbose = false)
                results["CompactLBFGS"] = (
                    objective = sol.objective,
                    iterations = sol.stats.iterations,
                    success = SciMLBase.successful_retcode(sol),
                )
                @test results["CompactLBFGS"].success
            catch e
                @test false  # CompactLBFGS should work
            end

            # ExactHessian is expected to fail
            try
                optfunc = OptimizationFunction(
                    coulomb_potential, SecondOrder(AutoForwardDiff(), AutoZygote()),
                    cons = unit_sphere_constraints
                )

                prob = OptimizationProblem(
                    optfunc, x0;
                    lcons = zeros(np),
                    ucons = zeros(np)
                )

                opt = MadNLPOptimizer(
                    hessian_approximation = MadNLP.ExactHessian
                )

                sol = solve(prob, opt; abstol = 1.0e-6, maxiters = 300, verbose = false)
                results["ExactHessian"] = (
                    objective = sol.objective,
                    iterations = sol.stats.iterations,
                    success = SciMLBase.successful_retcode(sol),
                )
                @test results["ExactHessian"].success broken = true
            catch e
                @test e isa Exception broken = true
            end
        end

        # "Exact Hessian and sparse KKT" test - broken due to SecondOrder AD issues.
        # See GitHub issues #1137 and #1140 for tracking.
        @testset "Exact Hessian and sparse KKT that hits σ == 0 in lag_h - broken" begin
            np = 12
            x0 = [
                -0.10518691576929745, 0.051771801773795686, -0.9003045175547166,
                0.23213937667116594, -0.02874270928423086, -0.652270178114126,
                -0.5918025628300999, 0.2511988210810674, -0.016535391659614228,
                0.5949770074227214, -0.4492781383448046, -0.29581324890382626,
                -0.8989309486672202, 0.10678505987872657, -0.4351575519144031,
                -0.9589360279618278, 0.02680807390998832, 0.40670966862867725,
                0.08594698464206306, -0.9646178134393677, -0.004187961953999249,
                -0.09107912492873807, -0.6973104772728601, 0.40182616259664583,
                0.4252750430946946, -0.9929333469713824, 0.009469988512801456,
                0.1629509253594941, -0.9992272933803594, -0.6396333795127627,
                -0.8014878928958706, 0.08007263129768477, -0.9998545103150432,
                0.7985655600140281, -0.5584865734204564, -0.8666200187082093,
            ]

            approx = MadNLP.ExactHessian
            ad = SecondOrder(AutoForwardDiff(), AutoZygote())

            optfunc = OptimizationFunction(
                coulomb_potential, ad,
                cons = unit_sphere_constraints
            )

            prob = OptimizationProblem(
                optfunc, x0;
                lcons = zeros(np),
                ucons = zeros(np)
            )

            try
                opt = MadNLPOptimizer(
                    hessian_approximation = approx,
                    kkt_system = MadNLP.SparseKKTSystem
                )

                sol = solve(prob, opt; abstol = 1.0e-6, maxiters = 300, verbose = false)

                @test SciMLBase.successful_retcode(sol) broken = true
                @test (sol.objective ≈ 49.165253058) broken = true
            catch e
                @test e isa Exception broken = true
            end
        end
    end

    @testset "LBFGS with damped update" begin
        # Test the damped BFGS update option
        function simple_quadratic(x, p)
            return sum(x .^ 2)
        end

        x0 = randn(5)

        ad = AutoForwardDiff()
        optfunc = OptimizationFunction(simple_quadratic, ad)
        prob = OptimizationProblem(optfunc, x0, nothing)

        opt = MadNLPOptimizer(
            hessian_approximation = MadNLP.DampedBFGS,  # Use damped BFGS variant
            linear_solver = MadNLP.LapackCPUSolver,
            kkt_system = MadNLP.DenseKKTSystem
        )

        sol = solve(prob, opt; maxiters = 50, verbose = false)

        @test SciMLBase.successful_retcode(sol)
        @test all(abs.(sol.u) .< 1.0e-6)  # Solution should be at origin
        @test sol.objective < 1.0e-10
    end
end
