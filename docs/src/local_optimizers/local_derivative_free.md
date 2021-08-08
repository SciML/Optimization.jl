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
    * Defaults:
        * `parameters = AdaptiveParameters()`
        * `initial_simplex = AffineSimplexer()`

- [`Optim.SimulatedAnnealing`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/simulated_annealing/): **Simulated Annealing**

    * `solve(problem, SimulatedAnnealing(neighbor, T, p))`
    * `neighbor` is a mutating function of the current and proposed `x`
    * `T` is a function of the current iteration that returns a temperature
    * `p` is a function of the current temperature
    * Defaults:
        * `neighbor = default_neighbor!`
        * `T = default_temperature`
        * `p = kirkpatrick`

### Optim Keyword Arguments

The following special keyword arguments can be used with Optim.jl optimizers:

* `x_tol`: Absolute tolerance in changes of the input vector `x`, in infinity norm. Defaults to `0.0`.
* `f_tol`: Relative tolerance in changes of the objective value. Defaults to `0.0`.
* `g_tol`: Absolute tolerance in the gradient, in infinity norm. Defaults to `1e-8`. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
* `f_calls_limit`: A soft upper limit on the number of objective calls. Defaults to `0` (unlimited).
* `g_calls_limit`: A soft upper limit on the number of gradient calls. Defaults to `0` (unlimited).
* `h_calls_limit`: A soft upper limit on the number of Hessian calls. Defaults to `0` (unlimited).
* `allow_f_increases`: Allow steps that increase the objective value. Defaults to `false`. Note that, when setting this to `true`, the last iterate will be returned as the minimizer even if the objective increased.
* `iterations`: How many iterations will run before the algorithm gives up? Defaults to `1_000`.
* `store_trace`: Should a trace of the optimization algorithm's state be stored? Defaults to `false`.
* `show_trace`: Should a trace of the optimization algorithm's state be shown on `stdout`? Defaults to `false`.
* `extended_trace`: Save additional information. Solver dependent. Defaults to `false`.
* `trace_simplex`: Include the full simplex in the trace for `NelderMead`. Defaults to `false`.
* `show_every`: Trace output is printed every `show_every`th iteration.
* `time_limit`: A soft upper limit on the total run time. Defaults to `NaN` (unlimited).

## NLopt.jl
