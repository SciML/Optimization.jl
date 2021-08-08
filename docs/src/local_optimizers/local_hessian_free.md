# Local Hessian-Free Second Order Optimization

Hessian-free methods are methods which perform second order optimization
by direct computation of Hessian-vector products (`Hv`) without requiring
the construction of the full Hessian. As such, these methods can perform
well for large second order optimization problems, but can require
special case when considering conditioning of the Hessian.

## Recommended Methods

`KrylovTrustRegion` is the only choice and it's pretty good.

## Optim.jl

* `KrylovTrustRegion`: **A Newton-Krylov method with Trust Regions**
    * `initial_delta`: The starting trust region radius
    * `delta_hat`: The largest allowable trust region radius
    * `eta`: When rho is at least eta, accept the step.
    * `rho_lower`: When rho is less than rho_lower, shrink the trust region.
    * `rho_upper`: When rho is greater than rho_upper, grow the trust region (though no greater than delta_hat).
    * Defaults:
        * `initial_delta = 1.0`
        * `delta_hat = 100.0`
        * `eta = 0.1`
        * `rho_lower = 0.25`
        * `rho_upper = 0.75`

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
