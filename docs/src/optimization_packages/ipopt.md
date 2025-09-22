# OptimizationIpopt.jl

[`OptimizationIpopt.jl`](https://github.com/SciML/Optimization.jl/tree/master/lib/OptimizationIpopt) is a wrapper package that integrates [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl) with the [`Optimization.jl`](https://github.com/SciML/Optimization.jl) ecosystem. This allows you to use the powerful Ipopt (Interior Point OPTimizer) solver through Optimization.jl's unified interface.

Ipopt is a software package for large-scale nonlinear optimization designed to find (local) solutions of mathematical optimization problems of the form:

```math
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \quad & g_L \leq g(x) \leq g_U \\
& x_L \leq x \leq x_U
\end{aligned}
```

where ``f(x): \mathbb{R}^n \to \mathbb{R}`` is the objective function, ``g(x): \mathbb{R}^n \to \mathbb{R}^m`` are the constraint functions, and the vectors ``g_L`` and ``g_U`` denote the lower and upper bounds on the constraints, and the vectors ``x_L`` and ``x_U`` are the bounds on the variables ``x``.

## Installation: OptimizationIpopt.jl

To use this package, install the OptimizationIpopt package:

```julia
import Pkg;
Pkg.add("OptimizationIpopt");
```

## Methods

OptimizationIpopt.jl provides the `IpoptOptimizer` algorithm, which wraps the Ipopt.jl solver for use with Optimization.jl. This is an interior-point algorithm that uses line search filter methods and is particularly effective for:
- Large-scale nonlinear problems
- Problems with nonlinear constraints
- Problems requiring high accuracy solutions

### Algorithm Requirements

`IpoptOptimizer` requires:
- Gradient information (via automatic differentiation or user-provided)
- Hessian information (can be approximated or provided)
- Constraint Jacobian (for constrained problems)
- Constraint Hessian (for constrained problems)

The algorithm supports:
- Box constraints via `lb` and `ub` in the `OptimizationProblem`
- General nonlinear equality and inequality constraints via `lcons` and `ucons`

### Basic Usage

```julia
using Optimization, OptimizationIpopt

# Create optimizer with default settings
opt = IpoptOptimizer()

# Or configure Ipopt-specific options
opt = IpoptOptimizer(
    acceptable_tol = 1e-8,
    mu_strategy = "adaptive"
)

# Solve the problem
sol = solve(prob, opt)
```

## Options and Parameters

### Common Interface Options

The following options can be passed as keyword arguments to `solve` and follow the common Optimization.jl interface:

- `maxiters`: Maximum number of iterations (overrides Ipopt's `max_iter`)
- `maxtime`: Maximum wall time in seconds (overrides Ipopt's `max_wall_time`)
- `abstol`: Absolute tolerance (not directly used by Ipopt)
- `reltol`: Convergence tolerance (overrides Ipopt's `tol`)
- `verbose`: Control output verbosity (overrides Ipopt's `print_level`)
  - `false` or `0`: No output
  - `true` or `5`: Standard output
  - Integer values 0-12: Different verbosity levels

### IpoptOptimizer Constructor Options

Ipopt-specific options are passed to the `IpoptOptimizer` constructor. The most commonly used options are available as struct fields:

#### Termination Options
- `acceptable_tol::Float64 = 1e-6`: Acceptable convergence tolerance (relative)
- `acceptable_iter::Int = 15`: Number of acceptable iterations before termination
- `dual_inf_tol::Float64 = 1.0`: Desired threshold for dual infeasibility
- `constr_viol_tol::Float64 = 1e-4`: Desired threshold for constraint violation
- `compl_inf_tol::Float64 = 1e-4`: Desired threshold for complementarity conditions

#### Linear Solver Options
- `linear_solver::String = "mumps"`: Linear solver to use
  - Default: "mumps" (included with Ipopt)
  - HSL solvers: "ma27", "ma57", "ma86", "ma97" (require [separate installation](https://github.com/jump-dev/Ipopt.jl?tab=readme-ov-file#linear-solvers))
  - Others: "pardiso", "spral" (require [separate installation](https://github.com/jump-dev/Ipopt.jl?tab=readme-ov-file#linear-solvers))
- `linear_system_scaling::String = "none"`: Method for scaling linear system. Use "mc19" for HSL solvers.

#### NLP Scaling Options
- `nlp_scaling_method::String = "gradient-based"`: Scaling method for NLP
  - Options: "none", "user-scaling", "gradient-based", "equilibration-based"
- `nlp_scaling_max_gradient::Float64 = 100.0`: Maximum gradient after scaling

#### Barrier Parameter Options
- `mu_strategy::String = "monotone"`: Update strategy for barrier parameter ("monotone", "adaptive")
- `mu_init::Float64 = 0.1`: Initial value for barrier parameter
- `mu_oracle::String = "quality-function"`: Oracle for adaptive mu strategy

#### Hessian Options
- `hessian_approximation::String = "exact"`: How to approximate the Hessian
  - `"exact"`: Use exact Hessian
  - `"limited-memory"`: Use L-BFGS approximation
- `limited_memory_max_history::Int = 6`: History size for L-BFGS
- `limited_memory_update_type::String = "bfgs"`: Quasi-Newton update formula ("bfgs", "sr1")

#### Line Search Options
- `line_search_method::String = "filter"`: Line search method ("filter", "penalty")
- `accept_every_trial_step::String = "no"`: Accept every trial step (disables line search)

#### Output Options
- `print_timing_statistics::String = "no"`: Print detailed timing information
- `print_info_string::String = "no"`: Print algorithm info string

#### Warm Start Options
- `warm_start_init_point::String = "no"`: Use warm start from previous solution

#### Restoration Phase Options
- `expect_infeasible_problem::String = "no"`: Enable if problem is expected to be infeasible

### Additional Options Dictionary

For Ipopt options not available as struct fields, use the `additional_options` dictionary:

```julia
opt = IpoptOptimizer(
    linear_solver = "ma57",
    additional_options = Dict(
        "derivative_test" => "first-order",
        "derivative_test_tol" => 1e-4,
        "fixed_variable_treatment" => "make_parameter",
        "alpha_for_y" => "primal"
    )
)
```

The full list of available options is documented in the [Ipopt Options Reference](https://coin-or.github.io/Ipopt/OPTIONS.html).

### Option Priority

Options follow this priority order (highest to lowest):
1. Common interface arguments passed to `solve` (e.g., `reltol`, `maxiters`)
2. Options in `additional_options` dictionary
3. Struct field values in `IpoptOptimizer`

Example with multiple option sources:

```julia
opt = IpoptOptimizer(
    acceptable_tol = 1e-6,           # Struct field
    mu_strategy = "adaptive",        # Struct field
    linear_solver = "ma57",          # Struct field (needs HSL)
    print_timing_statistics = "yes", # Struct field
    additional_options = Dict(
        "alpha_for_y" => "primal",   # Not a struct field
        "max_iter" => 500            # Will be overridden by maxiters below
    )
)

sol = solve(prob, opt;
    maxiters = 1000,  # Overrides max_iter in additional_options
    reltol = 1e-8     # Sets Ipopt's tol
)
```

## Examples

### Basic Unconstrained Optimization

The Rosenbrock function can be minimized using `IpoptOptimizer`:

```@example Ipopt1
using Optimization, OptimizationIpopt
using Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

# Ipopt requires gradient information
optfunc = OptimizationFunction(rosenbrock, AutoZygote())
prob = OptimizationProblem(optfunc, x0, p)
sol = solve(prob, IpoptOptimizer())
```

### Box-Constrained Optimization

Adding box constraints to limit the search space:

```@example Ipopt2
using Optimization, OptimizationIpopt
using Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

optfunc = OptimizationFunction(rosenbrock, AutoZygote())
prob = OptimizationProblem(optfunc, x0, p;
                          lb = [-1.0, -1.0],
                          ub = [1.5, 1.5])
sol = solve(prob, IpoptOptimizer())
```

### Nonlinear Constrained Optimization

Solving problems with nonlinear equality and inequality constraints:

```@example Ipopt3
using Optimization, OptimizationIpopt
using Zygote

# Objective: minimize x[1]^2 + x[2]^2
objective(x, p) = x[1]^2 + x[2]^2

# Constraint: x[1]^2 + x[2]^2 - 2*x[1] = 0 (equality)
# and x[1] + x[2] >= 1 (inequality)
function constraints(res, x, p)
    res[1] = x[1]^2 + x[2]^2 - 2*x[1]  # equality constraint
    res[2] = x[1] + x[2]                # inequality constraint
end

x0 = [0.5, 0.5]
optfunc = OptimizationFunction(objective, AutoZygote(); cons = constraints)

# First constraint is equality (lcons = ucons = 0)
# Second constraint is inequality (lcons = 1, ucons = Inf)
prob = OptimizationProblem(optfunc, x0;
                          lcons = [0.0, 1.0],
                          ucons = [0.0, Inf])

sol = solve(prob, IpoptOptimizer())
```

### Using Limited-Memory BFGS Approximation

For large-scale problems where computing the exact Hessian is expensive:

```@example Ipopt4
using Optimization, OptimizationIpopt
using Zygote

# Large-scale problem
n = 100
rosenbrock_nd(x, p) = sum(p[2] * (x[i+1] - x[i]^2)^2 + (p[1] - x[i])^2 for i in 1:n-1)

x0 = zeros(n)
p = [1.0, 100.0]

# Using automatic differentiation for gradients only
optfunc = OptimizationFunction(rosenbrock_nd, AutoZygote())
prob = OptimizationProblem(optfunc, x0, p)

# Use L-BFGS approximation for Hessian
sol = solve(prob, IpoptOptimizer(
           hessian_approximation = "limited-memory",
           limited_memory_max_history = 10);
           maxiters = 1000)
```

### Portfolio Optimization Example

A practical example of portfolio optimization with constraints:

```@example Ipopt5
using Optimization, OptimizationIpopt
using Zygote
using LinearAlgebra

# Portfolio optimization: minimize risk subject to return constraint
n_assets = 5
μ = [0.05, 0.10, 0.15, 0.08, 0.12]  # Expected returns
Σ = [0.05 0.01 0.02 0.01 0.00;      # Covariance matrix
     0.01 0.10 0.03 0.02 0.01;
     0.02 0.03 0.15 0.02 0.03;
     0.01 0.02 0.02 0.08 0.02;
     0.00 0.01 0.03 0.02 0.06]

target_return = 0.10

# Objective: minimize portfolio variance
portfolio_risk(w, p) = dot(w, Σ * w)

# Constraints: sum of weights = 1, expected return >= target
function portfolio_constraints(res, w, p)
    res[1] = sum(w) - 1.0                    # Sum to 1 (equality)
    res[2] = dot(μ, w) - target_return       # Minimum return (inequality)
end

optfunc = OptimizationFunction(portfolio_risk, AutoZygote();
                              cons = portfolio_constraints)
w0 = fill(1.0/n_assets, n_assets)

prob = OptimizationProblem(optfunc, w0;
                          lb = zeros(n_assets),     # No short selling
                          ub = ones(n_assets),      # No single asset > 100%
                          lcons = [0.0, 0.0],       # Equality and inequality constraints
                          ucons = [0.0, Inf])

sol = solve(prob, IpoptOptimizer();
           reltol = 1e-8,
           verbose = 5)

println("Optimal weights: ", sol.u)
println("Expected return: ", dot(μ, sol.u))
println("Portfolio variance: ", sol.objective)
```

## Tips and Best Practices

1. **Scaling**: Ipopt performs better when variables and constraints are well-scaled. Consider normalizing your problem if variables have very different magnitudes.

2. **Initial Points**: Provide good initial guesses when possible. Ipopt is a local optimizer and the solution quality depends on the starting point.

3. **Hessian Approximation**: For large problems or when Hessian computation is expensive, use `hessian_approximation = "limited-memory"` in the `IpoptOptimizer` constructor.

4. **Linear Solver Selection**: The choice of linear solver can significantly impact performance. For large problems, consider using HSL solvers (ma27, ma57, ma86, ma97). Note that HSL solvers require [separate installation](https://github.com/jump-dev/Ipopt.jl?tab=readme-ov-file#linear-solvers) - see the Ipopt.jl documentation for setup instructions. The default MUMPS solver works well for small to medium problems.

5. **Constraint Formulation**: Ipopt handles equality constraints well. When possible, formulate constraints as equalities rather than pairs of inequalities.

6. **Warm Starting**: When solving a sequence of similar problems, use the solution from the previous problem as the initial point for the next. You can enable warm starting with `IpoptOptimizer(warm_start_init_point = "yes")`.

## References

For more detailed information about Ipopt's algorithms and options, consult:
- [Ipopt Documentation](https://coin-or.github.io/Ipopt/)
- [Ipopt Options Reference](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [Ipopt Implementation Paper](https://link.springer.com/article/10.1007/s10107-004-0559-y)
