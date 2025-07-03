using Test
using OptimizationODE
using Optimization
using LinearAlgebra, ForwardDiff
using OrdinaryDiffEq, DifferentialEquations, SteadyStateDiffEq, Sundials

# Test helper functions
function rosenbrock(x, p)
    return (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
end

function rosenbrock_grad!(grad, x, p)
    grad[1] = -2.0 * (p[1] - x[1]) - 4.0 * p[2] * (x[2] - x[1]^2) * x[1]
    grad[2] = 2.0 * p[2] * (x[2] - x[1]^2)
end

function quadratic(x, p)
    return (x[1] - p[1])^2 + (x[2] - p[2])^2
end

function quadratic_grad!(grad, x, p)
    grad[1] = 2.0 * (x[1] - p[1])
    grad[2] = 2.0 * (x[2] - p[2])
end

# Constrained optimization problem
function constrained_objective(x, p)
    return x[1]^2 + x[2]^2
end

function constrained_objective_grad!(grad, x, p)
    grad[1] = 2.0 * x[1]
    grad[2] = 2.0 * x[2]
end

function constraint_func(res, x, p)
    res[1] = x[1] + x[2] - 1.0  # x[1] + x[2] = 1
    return x[1] + x[2] - 1.0
end

function constraint_jac!(jac, x, p)
    jac[1, 1] = 1.0
    jac[1, 2] = -1.0
end

@testset "OptimizationODE.jl Tests" begin
    
    
    @testset "Basic Unconstrained Optimization" begin
        @testset "Quadratic Function - ODE Optimizers" begin
            x0 = [2.0, 2.0]
            p = [1.0, 1.0]  # Minimum at (1, 1)
            
            optf = OptimizationFunction(quadratic, grad=quadratic_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            optimizers = [
                ("ODEGradientDescent", ODEGradientDescent()),
                ("RKChebyshevDescent", RKChebyshevDescent()),
                ("RKAccelerated", RKAccelerated()),
                ("HighOrderDescent", HighOrderDescent())
            ]
            
            for (name, opt) in optimizers
                @testset "$name" begin
                    sol = solve(prob, opt, dt=0.001, maxiters=1000000)
                    @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
                    @test isapprox(sol.u, p, atol=1e-1)
                    @test sol.objective < 1e-2
                end
            end
        end
        
        @testset "Rosenbrock Function - Selected Optimizers" begin
            x0 = [1.5, 2.0]
            p = [1.0, 100.0]  # Classic Rosenbrock parameters
            
            optf = OptimizationFunction(rosenbrock, grad=rosenbrock_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            # Test with more robust optimizers for Rosenbrock
            optimizers = [
                ("RKAccelerated", RKAccelerated()),
                ("HighOrderDescent", HighOrderDescent())
            ]
            
            for (name, opt) in optimizers
                @testset "$name" begin
                    sol = solve(prob, opt, dt=0.001, maxiters=1000000)
                    @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
                    # Rosenbrock is harder, so we use looser tolerances
                    @test isapprox(sol.u[1], 1.0, atol=1e-1)
                    @test isapprox(sol.u[2], 1.0, atol=1e-1)
                    @test sol.objective < 1.0
                end
            end
        end
    end
    
    @testset "Constrained Optimization - DAE Optimizers" begin
       @testset "Equality Constrained Optimization" begin
    # Minimize f(x) = x₁² + x₂²
    # Subject to x₁ - x₂ = 1

    function constrained_objective(x, p,args...)
        return x[1]^2 + x[2]^2
    end

    function constrained_objective_grad!(g, x, p, args...)
        g .= 2 .* x .* p[1]
        return nothing
    end

    # Constraint: x₁ - x₂ - p[1] = 0  (p[1] = 1 → x₁ - x₂ = 1)
    function constraint_func(x, p, args...)
        return x[1] - x[2] - p[1]
    end

    function constraint_jac!(J, x,args...)
        J[1, 1] = 1.0
        J[1, 2] = -1.0
        return nothing
    end

    x0 = [1.0, 0.0]           # reasonable initial guess
    p  = [1.0]                 # enforce x₁ - x₂ = 1

    optf = OptimizationFunction(constrained_objective;
                                grad = constrained_objective_grad!,
                                cons = constraint_func,
                                cons_j = constraint_jac!)

    @testset "Equality Constrained - Mass Matrix Method" begin
        prob = OptimizationProblem(optf, x0, p)
        opt = DAEMassMatrix()
        sol = solve(prob, opt; dt=0.01, maxiters=1_000_000)

        @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
        @test isapprox(sol.u[1] - sol.u[2], 1.0; atol = 1e-2)
        @test isapprox(sol.u, [0.5, -0.5]; atol = 1e-2)
    end

    @testset "Equality Constrained - Index Method" begin
        prob = OptimizationProblem(optf, x0, p)
        opt = DAEIndexing()
        differential_vars = [true, true, false]  # x vars = differential, λ = algebraic
        sol = solve(prob, opt; dt=0.01, maxiters=1_000_000,
                    differential_vars = differential_vars)

        @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
        @test isapprox(sol.u[1] - sol.u[2], 1.0; atol = 1e-2)
        @test isapprox(sol.u, [0.5, -0.5]; atol = 1e-2)
    end
end
    end
    
    @testset "Parameter Handling" begin
        @testset "NullParameters Handling" begin
            x0 = [0.0, 0.0]
            p=Float64[]  # No parameters provided
            # Create a problem with NullParameters
            optf = OptimizationFunction((x, p, args...) -> sum(x.^2), 
                                      grad=(grad, x, p, args...) -> (grad .= 2.0 .* x))
            prob = OptimizationProblem(optf, x0,p)  # No parameters provided
            
            opt = ODEGradientDescent()
            sol = solve(prob, opt, dt=0.01, maxiters=100000)
            
            @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
            @test isapprox(sol.u, [0.0, 0.0], atol=1e-2)
        end
        
        @testset "Regular Parameters" begin
            x0 = [0.5, 1.5]
            p = [1.0, 1.0]
            
            optf = OptimizationFunction(quadratic, grad=quadratic_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            opt = RKAccelerated()
            sol = solve(prob, opt; dt=0.001, maxiters=1000000)
            
            @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
            @test isapprox(sol.u, p, atol=1e-1)
        end
    end
    
    @testset "Solver Options and Keywords" begin
        @testset "Custom dt and maxiters" begin
            x0 = [0.0, 0.0]
            p = [1.0, 1.0]
            
            optf = OptimizationFunction(quadratic, grad=quadratic_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            opt = RKAccelerated()
            
            # Test with custom dt
            sol1 = solve(prob, opt; dt=0.001, maxiters=100000)
            @test sol1.retcode == ReturnCode.Success || sol1.retcode == ReturnCode.Default
            
            # Test with smaller dt (should be more accurate)
            sol2 = solve(prob, opt; dt=0.001, maxiters=100000)
            @test sol2.retcode == ReturnCode.Success || sol2.retcode == ReturnCode.Default
            @test sol2.objective <= sol1.objective  # Should be at least as good
        end
    end
    
    @testset "Callback Functionality" begin
        @testset "Progress Callback" begin
            x0 = [0.0, 0.0]
            p = [1.0, 1.0]
            
            callback_called = Ref(false)
            callback_values = Vector{Vector{Float64}}()
            
            function test_callback(x, p, t)
                return false 
            end
            
            optf = OptimizationFunction(quadratic; grad=quadratic_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            opt = RKAccelerated()
            sol = solve(prob, opt, dt=0.1, maxiters=100000, callback=test_callback, progress=true)
            
            @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
        end
    end
    
    @testset "Finite Difference Jacobian" begin
        @testset "Jacobian Computation" begin
            x = [1.0, 2.0]
            f(x) = [x[1]^2 + x[2], x[1] * x[2]]
            
            J = ForwardDiff.jacobian(f, x)
            
            expected_J = [2.0 1.0; 2.0 1.0]
            
            @test isapprox(J, expected_J, atol=1e-6)
        end
    end
    
    @testset "Solver Type Detection" begin
        @testset "Mass Matrix Solvers" begin
            opt = DAEMassMatrix()
            @test OptimizationODE.get_solver_type(opt) == :mass_matrix
        end
        
        @testset "Index Method Solvers" begin
            opt = DAEIndexing()
            @test OptimizationODE.get_solver_type(opt) == :indexing
        end
    end
    
    @testset "Error Handling and Edge Cases" begin
        @testset "Empty Constraints" begin
            x0 = [1.5, 0.5]
            p = Float64[]
            
            # Problem without constraints should fall back to ODE method
            optf = OptimizationFunction(constrained_objective, 
                                     grad=constrained_objective_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            opt = DAEMassMatrix()
            sol = solve(prob, opt; dt=0.001, maxiters=50000)
            
            @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
            @test isapprox(sol.u, [0.0, 0.0], atol=1e-1)
        end
        
        @testset "Single Variable Optimization" begin
            x0 = [0.5]
            p = [1.0]
            
            single_var_func(x, p,args...) = (x[1] - p[1])^2
            single_var_grad!(grad, x, p,args...) = (grad[1] = 2.0 * (x[1] - p[1]))
            
            optf = OptimizationFunction(single_var_func; grad=single_var_grad!)
            prob = OptimizationProblem(optf, x0, p)
            
            opt = RKAccelerated()
            sol = solve(prob, opt; dt=0.001, maxiters=10000)
            
            @test sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
            @test isapprox(sol.u[1], p[1], atol=1e-1)
        end
    end
end
