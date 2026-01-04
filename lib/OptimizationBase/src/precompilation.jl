using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Basic optimization function creation with NoAD
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
        x0 = zeros(2)
        p = [1.0, 100.0]

        # Create OptimizationFunction with NoAD
        optf_noad = SciMLBase.OptimizationFunction(rosenbrock)

        # Create OptimizationProblem
        prob = SciMLBase.OptimizationProblem(optf_noad, x0, p)

        # Create problem with bounds
        prob_bounded = SciMLBase.OptimizationProblem(
            optf_noad, x0, p;
            lb = [-1.0, -1.0], ub = [1.0, 1.0]
        )

        # Test ReInitCache creation
        cache = ReInitCache(x0, p)
    end
end
