# Global Constrained Optimization

## Recommended Methods

`SAMIN` or `NLopt.GNISRES`

## Optim.jl

- [`Optim.SAMIN`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/samin/): **Simulated Annealing with bounds**

    * `solve(problem, SAMIN(nt, ns, rt, neps, f_tol, x_tol, coverage_ok, verbosity))`
    * defaults to:
    ```julia
    SAMIN(; nt::Int = 5  # reduce temperature every nt*ns*dim(x_init) evaluations
            ns::Int = 5  # adjust bounds every ns*dim(x_init) evaluations
            rt::T = 0.9  # geometric temperature reduction factor: when temp changes, new temp is t=rt*t
            neps::Int = 5  # number of previous best values the final result is compared to
            f_tol::T = 1e-12  # the required tolerance level for function value comparisons
            x_tol::T = 1e-6  # the required tolerance level for x
            coverage_ok::Bool = false,  # if false, increase temperature until initial parameter space is covered
            verbosity::Int = 0)  # scalar: 0, 1, 2 or 3 (default = 0).

    # copied verbatim from https://julianlsolvers.github.io/Optim.jl/stable/#algo/samin/#constructor
    ```

## NLopt.jl
