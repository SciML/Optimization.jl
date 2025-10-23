using OptimizationMadNLP
using OptimizationBase
using MadNLP
using Test
import Zygote, ForwardDiff, ReverseDiff
using SparseArrays
using DifferentiationInterface
using Random

@testset "rosenbrock" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    # MadNLP requires second-order derivatives
    ad = SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())
    optfunc = OptimizationFunction(
        (x, p) -> -rosenbrock(x, p), ad)
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
        lH = sigma * [2 + 8(x[1]^2) * p[2]-4(x[2] - (x[1]^2)) * p[2] -4p[2]*x[1]
              -4p[2]*x[1] 2p[2]] .+ [2mu[1] mu[2]
              mu[2] 2mu[1]]
        # MadNLP uses lower triangle. For symmetric sparse([1 1; 1 1]), lower triangle has [1,1], [2,1], and [2,2]
        res[1] = lH[1, 1]  # Position [1,1]
        res[2] = lH[2, 1]  # Position [2,1] (off-diagonal)
        res[3] = lH[2, 2]  # Position [2,2]
    end
    lag_hess_prototype = sparse([1 1; 1 1])  # Symmetric sparse pattern for Hessian

    # Use SecondOrder AD for MadNLP
    ad = SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoZygote())
    optprob = OptimizationFunction(rosenbrock, ad;
        cons = cons, lag_h = lagh, lag_hess_prototype)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])

    opts = [
        MadNLPOptimizer(),
        MadNLPOptimizer(linear_solver = LapackCPUSolver)
    ]

    for opt in opts
        sol = solve(prob, opt)
        @test SciMLBase.successful_retcode(sol)

        # compare against Ipopt results
        @test sol≈[0.7071678163428006, 0.7070457460302945] rtol=1e-4
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
        SecondOrder(AutoForwardDiff(), AutoReverseDiff())
    ]
        optf = OptimizationFunction(objective, ad)
        prob = OptimizationProblem(optf, x0, p)
        cache = OptimizationBase.init(prob, MadNLPOptimizer())
        sol = OptimizationBase.solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u≈[1.0] atol=1e-3

        cache = OptimizationBase.reinit!(cache; p = [2.0])
        sol = OptimizationBase.solve!(cache)
        # @test sol.retcode == ReturnCode.Success
        @test sol.u≈[2.0] atol=1e-3
    end
end

@testset "constraints & AD" begin
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
        ]
    end

    x0 = [1.0, 5.0, 5.0, 1.0]

    @testset "$ad" for ad in [
        AutoSparse(SecondOrder(AutoForwardDiff(), AutoZygote())),
        AutoSparse(SecondOrder(AutoForwardDiff(), AutoForwardDiff())),
        AutoSparse(SecondOrder(AutoForwardDiff(), AutoReverseDiff()))
    ]
        optfunc = OptimizationFunction(objective, ad, cons = constraints)
        prob = OptimizationProblem(optfunc, x0; sense = OptimizationBase.MinSense,
            lb = [1.0, 1.0, 1.0, 1.0],
            ub = [5.0, 5.0, 5.0, 5.0],
            lcons = [25.0, 40.0],
            ucons = [Inf, 40.0])

        cache = init(prob, MadNLPOptimizer())

        sol = OptimizationBase.solve!(cache)

        @test SciMLBase.successful_retcode(sol)

        @test isapprox(sol.objective, 17.014017145179164; atol = 1e-6)
        x = [1.0, 4.7429996418092970, 3.8211499817883077, 1.3794082897556983]
        @test isapprox(sol.u, x; atol = 1e-6)
        @test prod(sol.u) >= 25.0 - 1e-6
        @test isapprox(sum(sol.u .^ 2), 40.0; atol = 1e-6)
    end

    # dense
    @testset "$ad" for ad in [
        SecondOrder(AutoForwardDiff(), AutoZygote()),
        SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
        SecondOrder(AutoForwardDiff(), AutoReverseDiff())
    ]
        optfunc = OptimizationFunction(objective, ad, cons = constraints)
        prob = OptimizationProblem(optfunc, x0; sense = OptimizationBase.MinSense,
            lb = [1.0, 1.0, 1.0, 1.0],
            ub = [5.0, 5.0, 5.0, 5.0],
            lcons = [25.0, 40.0],
            ucons = [Inf, 40.0])

        cache = init(prob,
            MadNLPOptimizer(kkt_system = MadNLP.DenseKKTSystem,
                linear_solver = LapackCPUSolver))

        sol = OptimizationBase.solve!(cache)

        @test SciMLBase.successful_retcode(sol)

        @test isapprox(sol.objective, 17.014017145179164; atol = 1e-6)
        x = [1.0, 4.7429996418092970, 3.8211499817883077, 1.3794082897556983]
        @test isapprox(sol.u, x; atol = 1e-6)
        @test prod(sol.u) >= 25.0 - 1e-6
        @test isapprox(sum(sol.u .^ 2), 40.0; atol = 1e-6)
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
    optprob = OptimizationFunction(objective_sparse, ad;
        cons = cons_sparse, lag_h = lag_hess_sparse, lag_hess_prototype = hess_proto_4x4)

    prob = OptimizationProblem(optprob, x0, p,
        lcons = [4.0, 1.0],     # x1+x2+x3+x4 = 4, x1*x3 >= 1
        ucons = [4.0, Inf])      # x1+x2+x3+x4 = 4, x1*x3 <= Inf

    sol = solve(prob, MadNLPOptimizer())

    @test SciMLBase.successful_retcode(sol)

    # Check constraints
    cons_vals = zeros(2)
    cons_sparse(cons_vals, sol.u, p)
    @test isapprox(cons_vals[1], 4.0, atol = 1e-6)  # Sum constraint
    @test cons_vals[2] >= 1.0 - 1e-6              # Product constraint
end

@testset "MadNLP Options and Common Interface" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    p = [1.0, 100.0]
    ad = SecondOrder(AutoForwardDiff(), AutoForwardDiff())

    @testset "MadNLP struct options" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        # Test with MadNLP-specific struct fields
        opt = MadNLPOptimizer(
            acceptable_tol = 1e-6,
            acceptable_iter = 10,
            blas_num_threads = 2,
            mu_init = 0.01
        )
        sol = solve(prob, opt)
        @test SciMLBase.successful_retcode(sol)

        # Test with hessian approximation
        opt2 = MadNLPOptimizer(
            hessian_approximation = MadNLP.CompactLBFGS,
            jacobian_constant = false,
            hessian_constant = false
        )
        sol2 = solve(prob, opt2)
        @test SciMLBase.successful_retcode(sol2)
    end

    @testset "additional_options dictionary" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        # Test passing MadNLP options via additional_options
        opt = MadNLPOptimizer(
            linear_solver = MadNLP.UmfpackSolver,
            additional_options = Dict{Symbol, Any}(
                :max_iter => 200,
                :tol => 1e-7
            )
        )
        sol = solve(prob, opt)
        @test SciMLBase.successful_retcode(sol)

        # Test with different options
        opt2 = MadNLPOptimizer(
            additional_options = Dict{Symbol, Any}(
            :inertia_correction_method => MadNLP.InertiaFree,
            :fixed_variable_treatment => MadNLP.RelaxBound
        )
        )
        sol2 = solve(prob, opt2)
        @test SciMLBase.successful_retcode(sol2)
    end

    @testset "Common interface arguments" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        # Test that abstol overrides default tolerance
        sol1 = solve(prob, MadNLPOptimizer(); abstol = 1e-12)
        @test SciMLBase.successful_retcode(sol1)
        @test sol1.u≈[1.0, 1.0] atol=1e-10

        # Test that maxiters limits iterations
        sol2 = solve(prob, MadNLPOptimizer(); maxiters = 5)
        # May not converge with only 5 iterations
        @test sol2.stats.iterations <= 5

        # Test verbose options (MadNLP supports bool and LogLevels)
        for verbose in [false, true, MadNLP.ERROR, MadNLP.WARN, MadNLP.INFO]
            sol = solve(prob, MadNLPOptimizer(); verbose = verbose, maxiters = 20)
            @test sol isa SciMLBase.OptimizationSolution
        end
    end

    @testset "Priority: struct < additional_options < common solve args" begin
        optfunc = OptimizationFunction(rosenbrock, ad)
        prob = OptimizationProblem(optfunc, x0, p)

        # Struct field is overridden by additional_options and solve arguments
        opt = MadNLPOptimizer(
            acceptable_tol = 1e-4,  # Struct field
            additional_options = Dict{Symbol, Any}(
                :max_iter => 10,    # Will be overridden by maxiters
                :tol => 1e-6        # Will be overridden by abstol
            )
        )

        sol = solve(prob, opt;
            maxiters = 5,   # Should override additional_options[:max_iter]
            abstol = 1e-10)  # Should override additional_options[:tol]

        @test sol.stats.iterations <= 5
        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
    end
end

@testset "LBFGS Hessian Approximation" begin
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

        # Test different LBFGS configurations
        @testset "LBFGS variant: $variant" for variant in [
            MadNLP.CompactLBFGS,
            MadNLP.ExactHessian  # For comparison
        ]
            # Only provide gradients, no Hessian needed for LBFGS
            ad = AutoForwardDiff()  # First-order AD is sufficient
            optfunc = OptimizationFunction(extended_rosenbrock, ad)
            prob = OptimizationProblem(optfunc, x0, nothing)

            if variant == MadNLP.ExactHessian
                # Use second-order AD for exact Hessian
                ad = SecondOrder(AutoForwardDiff(), AutoForwardDiff())
                optfunc = OptimizationFunction(extended_rosenbrock, ad)
                prob = OptimizationProblem(optfunc, x0, nothing)
            end

            opt = MadNLPOptimizer(
                hessian_approximation = variant
            )

            sol = solve(prob, opt; maxiters = 100, verbose = false)

            @test SciMLBase.successful_retcode(sol)
            @test all(isapprox.(sol.u, 1.0, atol = 1e-6))  # Solution should be all ones
            @test sol.objective < 1e-10  # Should be close to zero
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
            @test all(isapprox.(sol.u, 1.0, atol = 1e-6))
        end
    end

    @testset "Constrained LBFGS - Electrons on Sphere" begin
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

        @testset "N=$np electrons with $approx" for np in [6, 8, 10],
            approx in [MadNLP.CompactLBFGS, MadNLP.ExactHessian]

            x0 = init_electrons_on_sphere(np)

            if approx == MadNLP.CompactLBFGS
                # For LBFGS variants, only first-order derivatives needed
                ad = AutoForwardDiff()
            else
                # For exact Hessian, need second-order
                ad = SecondOrder(AutoForwardDiff(), AutoForwardDiff())
            end

            optfunc = OptimizationFunction(
                coulomb_potential, ad,
                cons = unit_sphere_constraints
            )

            # Equality constraints: each electron on unit sphere
            lcons = zeros(np)
            ucons = zeros(np)

            prob = OptimizationProblem(optfunc, x0;
                lcons = lcons,
                ucons = ucons
            )

            opt = MadNLPOptimizer(
                linear_solver = LapackCPUSolver,
                hessian_approximation = approx
            )

            sol = solve(prob, opt; abstol = 1e-7, maxiters = 200, verbose = false)

            @test SciMLBase.successful_retcode(sol)

            # Check that all electrons are on the unit sphere
            cons_vals = zeros(np)
            unit_sphere_constraints(cons_vals, sol.u, nothing)
            @test all(abs.(cons_vals) .< 1e-5)

            # Known optimal energies for small electron numbers
            # Reference: https://en.wikipedia.org/wiki/Thomson_problem
            # Note: These are the minimum Coulomb potential energies for N electrons on unit sphere
            expected_energies = Dict(
                6 => 9.985281374,    # Octahedron (Oh symmetry)
                8 => 19.675287861,   # Square antiprism (D4d)
                10 => 32.716949460   # Gyroelongated square dipyramid (D4d)
            )

            if haskey(expected_energies, np)
                @test isapprox(sol.objective, expected_energies[np], rtol = 1e-3)
            end

            # Verify minimum distance between electrons
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
            @test min_dist > 0.5  # Electrons should be well-separated
        end

        @testset "LBFGS vs Exact Hessian" begin
            # Test with moderate size to show LBFGS efficiency
            np = 12  # Icosahedron configuration
            x0 = init_electrons_on_sphere(np)

            results = Dict()

            for (name, approx, ad) in [("CompactLBFGS", MadNLP.CompactLBFGS,
                                           AutoForwardDiff())
                                       ("ExactHessian",
                                           MadNLP.ExactHessian,
                                           SecondOrder(
                                               AutoForwardDiff(), AutoForwardDiff()))]
                optfunc = OptimizationFunction(
                    coulomb_potential, ad,
                    cons = unit_sphere_constraints
                )

                prob = OptimizationProblem(optfunc, x0;
                    lcons = zeros(np),
                    ucons = zeros(np)
                )

                opt = MadNLPOptimizer(
                    hessian_approximation = approx
                )

                sol = solve(prob, opt; abstol = 1e-6, maxiters = 300, verbose = false)
                results[name] = (
                    objective = sol.objective,
                    iterations = sol.stats.iterations,
                    success = SciMLBase.successful_retcode(sol)
                )
            end

            # All methods should converge
            @test all(r.success for r in values(results))

            # All should find similar objective values (icosahedron energy)
            # Reference: https://en.wikipedia.org/wiki/Thomson_problem
            objectives = [r.objective for r in values(results)]
            @test all(abs.(objectives .- 49.165253058) .< 0.1)

            # LBFGS methods typically need more iterations but less cost per iteration
            @test results["CompactLBFGS"].iterations > 0
            @test results["ExactHessian"].iterations > 0
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
        @test all(abs.(sol.u) .< 1e-6)  # Solution should be at origin
        @test sol.objective < 1e-10
    end
end
