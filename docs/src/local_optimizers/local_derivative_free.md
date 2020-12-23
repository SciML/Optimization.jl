# Local Derivative-Free Optimization

Derivative-free optimizers are optimizers that can be used even in cases
where no derivatives or automatic differentiation is specified. While
they tend to be less efficient than derivative-based optimizers, they
can be easily applied to cases where defining derivatives is difficult.

## Recommended Methods

NLOpt COBYLA

## Optim.jl

- [`Optim.NelderMead`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/nelder_mead/): **Nelder-Mead optimizer**

    * `solve(problem, NelderMead(parameters, initial_simplex))`
    * `parameters = AdaptiveParameters()` or `parameters = FixedParameters()`
    * `initial_simplex = AffineSimplexer()`
    * defaults to: `parameters = AdaptiveParameters(), initial_simplex = AffineSimplexer()`

- [`Optim.SimulatedAnnealing`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/simulated_annealing/): **Simulated Annealing**

    * `solve(problem, SimulatedAnnealing(neighbor, T, p))`
    * `neighbor` is a mutating function of the current and proposed `x`
    * `T` is a function of the current iteration that returns a temperature
    * `p` is a function of the current temperature
    * defaults to: `neighbor = default_neighbor!, T = default_temperature, p = kirkpatrick`

## NLopt.jl
