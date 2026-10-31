using OptimizationMOI, OptimizationBase, ModelingToolkitBase, HiGHS, AmplNLWriter, Ipopt_jll
using MathOptInterface, SciMLBase, Test

@testset "Integer powers in MOI expressions" begin
    @variables x
    @parameters a
    for symbolic_system in (true, false)
        @testset "quadratic, system=$symbolic_system" begin
            prob = if symbolic_system
                sys = complete(
                    OptimizationSystem(
                        a * x^2, [x], [a]; constraints = [x ≲ a^2], name = :quadratic_power
                    )
                )
                OptimizationProblem(sys, [x => 0.5, a => 2.0]; grad = false, hess = false)
            else
                f = OptimizationFunction(
                    (u, p) -> p[1] * u[1]^2, AutoSymbolics();
                    cons = (res, u, p) -> (res[1] = u[1] - p[1]^2)
                )
                OptimizationProblem(f, [0.5], [2.0]; lcons = [-Inf], ucons = [0.0])
            end
            sol = solve(prob, HiGHS.Optimizer(); output_flag = false)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [0.0] atol = 1.0e-8
            @test sol.objective ≈ 0.0 atol = 1.0e-8
        end
        @testset "nonlinear, system=$symbolic_system" begin
            prob = if symbolic_system
                sys = complete(
                    OptimizationSystem(
                        x^4, [x], []; constraints = [x^3 ~ -8], name = :nonlinear_power
                    )
                )
                OptimizationProblem(sys, [x => -1.5]; lb = [-3.0], ub = [-1.0])
            else
                f = OptimizationFunction(
                    (u, p) -> u[1]^4, AutoSymbolics();
                    cons = (res, u, p) -> (res[1] = u[1]^3)
                )
                OptimizationProblem(
                    f, [-1.5]; lb = [-3.0], ub = [-1.0], lcons = [-8.0], ucons = [-8.0]
                )
            end
            sol = solve(prob, AmplNLWriter.Optimizer(Ipopt_jll.amplexe); tol = 1.0e-10)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ [-2.0] atol = 1.0e-7
            @test sol.objective ≈ 16.0 atol = 1.0e-7
        end
    end
end

@testset "Literal-power constants and exponents" begin
    for (base, exponent, expected) in (
            (2, -2, 0.25), (Int32(2), 2, Int32(4)), (2.0f0, 3, 8.0f0),
            (-2.0, 3, -8.0), (0.0, -2, Inf), (-0.0, 3, -0.0), (2, 0, 1),
        )
        for op in (Base.literal_pow, :literal_pow),
                val in (Val(exponent), Expr(:call, Val, exponent), Expr(:call, :Val, exponent))
            expr = Expr(:call, op, ^, base, val)
            @test OptimizationMOI._normalize_integer_powers!(expr) === expected
        end
    end
    for (op, power) in ((Base.literal_pow, ^), (:literal_pow, :^))
        expr = Expr(:call, op, power, :(x[1]), Val(Int32(3)))
        normalized = OptimizationMOI._normalize_integer_powers!(expr)
        @test normalized == :(x[1]^$(Int32(3)))
        @test normalized.args[3] === Int32(3)
    end
    fractional = :(x[1]^2.5)
    @test OptimizationMOI._normalize_integer_powers!(deepcopy(fractional)) == fractional
end

@testset "Quadratic power collection" begin
    for op in (Base.literal_pow, :literal_pow)
        expr = Expr(:call, op, ^, :(x[1] + x[2]), Val(2))
        expr = OptimizationMOI.fixpoint_simplify_and_expand!(expr)
        func, constant = OptimizationMOI.get_moi_function(expr)
        @test func isa MathOptInterface.ScalarQuadraticFunction{Float64}
        @test constant == 0.0
        for (x, expected) in (([2.0, -3.0], 1.0), ([-2.0, -3.0], 25.0))
            @test MathOptInterface.Utilities.eval_variables(v -> x[v.value], func) == expected
        end
    end
    expr = Expr(:call, Base.literal_pow, ^, :(x[1]), Val(3))
    expr = OptimizationMOI.fixpoint_simplify_and_expand!(expr)
    @test_throws OptimizationMOI.MalformedExprException OptimizationMOI.get_moi_function(expr)
end
