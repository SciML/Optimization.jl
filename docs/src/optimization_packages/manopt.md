# Manopt.jl

[Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl) is a package with implementations of a variety of optimization solvers on manifolds supported by
[Manifolds](https://github.com/JuliaManifolds/Manifolds.jl).

## Installation: OptimizationManopt.jl

To use the Optimization.jl interface to Manopt, install the OptimizationManopt package:

```julia
import Pkg;
Pkg.add("OptimizationManopt");
```

## Methods

The following methods are available for the `OptimizationManopt` package:

  - `GradientDescentOptimizer`: Corresponds to the [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent/) method in Manopt.
  - `NelderMeadOptimizer` : Corresponds to the [`NelderMead`](https://manoptjl.org/stable/solvers/NelderMead/) method in Manopt.
  - `ConjugateGradientDescentOptimizer`: Corresponds to the [`conjugate_gradient_descent`](https://manoptjl.org/stable/solvers/conjugate_gradient_descent/) method in Manopt.
  - `ParticleSwarmOptimizer`: Corresponds to the [`particle_swarm`](https://manoptjl.org/stable/solvers/particle_swarm/) method in Manopt.
  - `QuasiNewtonOptimizer`: Corresponds to the [`quasi_Newton`](https://manoptjl.org/stable/solvers/quasi_Newton/) method in Manopt.
  - `CMAESOptimizer`: Corresponds to the [`cma_es`](https://manoptjl.org/stable/solvers/cma_es/) method in Manopt.
  - `ConvexBundleOptimizer`: Corresponds to the [`convex_bundle_method`](https://manoptjl.org/stable/solvers/convex_bundle_method/) method in Manopt.
  - `FrankWolfeOptimizer`: Corresponds to the [`FrankWolfe`](https://manoptjl.org/stable/solvers/FrankWolfe/) method in Manopt.

The common kwargs `maxiters`, `maxtime` and `abstol` are supported by all the optimizers. Solver specific kwargs from Manopt can be passed to the `solve`
function or `OptimizationProblem`.

!!! note
    
    The `OptimizationProblem` has to be passed the manifold as the `manifold` keyword argument.

## Examples

The Rosenbrock function on the Euclidean manifold can be optimized using the `GradientDescentOptimizer` as follows:

```@example Manopt1
using Optimization, OptimizationManopt, Manifolds
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

R2 = Euclidean(2)

stepsize = Manopt.ArmijoLinesearch(R2)
opt = OptimizationManopt.GradientDescentOptimizer()

optf = OptimizationFunction(rosenbrock, Optimization.AutoZygote())

prob = OptimizationProblem(
    optf, x0, p; manifold = R2, stepsize = stepsize)

sol = Optimization.solve(prob, opt)
```

The box-constrained Karcher mean problem on the SPD manifold with the Frank-Wolfe algorithm can be solved as follows:

```@example Manopt2
M = SymmetricPositiveDefinite(5)
m = 100
σ = 0.005
q = Matrix{Float64}(I, 5, 5) .+ 2.0
data2 = [exp(M, q, σ * rand(M; vector_at = q)) for i in 1:m]

f(x, p = nothing) = sum(distance(M, x, data2[i])^2 for i in 1:m)
optf = OptimizationFunction(f, Optimization.AutoZygote())
prob = OptimizationProblem(optf, data2[1]; manifold = M, maxiters = 1000)

function closed_form_solution!(M::SymmetricPositiveDefinite, q, L, U, p, X)
    # extract p^1/2 and p^{-1/2}
    (p_sqrt_inv, p_sqrt) = Manifolds.spd_sqrt_and_sqrt_inv(p)
    # Compute D & Q
    e2 = eigen(p_sqrt_inv * X * p_sqrt_inv) # decompose Sk  = QDQ'
    D = Diagonal(1.0 .* (e2.values .< 0))
    Q = e2.vectors

    Uprime = Q' * p_sqrt_inv * U * p_sqrt_inv * Q
    Lprime = Q' * p_sqrt_inv * L * p_sqrt_inv * Q
    P = cholesky(Hermitian(Uprime - Lprime))
    z = P.U' * D * P.U + Lprime
    copyto!(M, q, p_sqrt * Q * z * Q' * p_sqrt)
    return q
end
N = m
U = mean(data2)
L = inv(sum(1 / N * inv(matrix) for matrix in data2))

optf = OptimizationFunction(f, Optimization.AutoZygote())
prob = OptimizationProblem(optf, U; manifold = M, maxiters = 1000)

sol = Optimization.solve(
    prob, opt, sub_problem = (M, q, p, X) -> closed_form_solution!(M, q, L, U, p, X))
```

This example is based on the [example](https://juliamanifolds.github.io/ManoptExamples.jl/stable/examples/Riemannian-mean/) in the Manopt and [Weber and Sra'22](https://doi.org/10.1007/s10107-022-01840-5).

The following example is adapted from the Rayleigh Quotient example in ManoptExamples.jl.
We solve the Rayleigh quotient problem on the Sphere manifold:

```@example Manopt3
using Optimization, OptimizationManopt
using Manifolds, LinearAlgebra
using Manopt

n = 1000
A = Symmetric(randn(n, n) / n)
manifold = Sphere(n - 1)

cost(x, p = nothing) = -x' * A * x
egrad(G, x, p = nothing) = (G .= -2 * A * x)

optf = OptimizationFunction(cost, grad = egrad)
x0 = rand(manifold)
prob = OptimizationProblem(optf, x0, manifold = manifold)

sol = solve(prob, GradientDescentOptimizer())
```

Let's check that this indeed corresponds to the minimum eigenvalue of the matrix `A`.

```@example Manopt3
@show eigmin(A)
@show sol.objective
```
