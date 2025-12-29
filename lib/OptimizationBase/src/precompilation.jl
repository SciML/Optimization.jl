using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Precompile basic OptimizationFunction and OptimizationProblem creation
        # These are the most common operations users perform

        # Simple objective function
        f_simple(x, p) = sum(abs2, x .- p)

        # Create OptimizationFunction with NoAD (most basic case)
        optf = OptimizationFunction(f_simple)

        # Create OptimizationProblem with common type combinations
        x0 = zeros(2)
        p = [1.0, 2.0]
        prob = OptimizationProblem(optf, x0, p)

        # Also precompile with bounds
        prob_bounded = OptimizationProblem(optf, x0, p; lb = zeros(2), ub = ones(2))
    end
end
