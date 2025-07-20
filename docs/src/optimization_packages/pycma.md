# PyCMA.jl

[`PyCMA`](https://github.com/CMA-ES/pycma) is a Python implementation of CMA-ES and a few related numerical optimization tools. `OptimizationPyCMA.jl` gives access to the CMA-ES optimizer through the unified `Optimization.jl` interface just like any native Julia optimizer.

`OptimizationPyCMA.jl` relies on [`PythonCall`](https://github.com/cjdoris/PythonCall.jl). A minimal Python distribution containing PyCMA will be installed automatically on first use, so no manual Python set-up is required.

## Installation: OptimizationPyCMA.jl

```julia
import Pkg
Pkg.add("OptimizationPyCMA")
```

## Methods

`PyCMAOpt` supports the usual keyword arguments `maxiters`, `maxtime`, `abstol`, `reltol`, `callback` in addition to any PyCMA-specific options (passed verbatim via keyword arguments to `solve`).

## Example

```@example PyCMA
using OptimizationPyCMA

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]
l1 = rosenbrock(x0, _p)
f = OptimizationFunction(rosenbrock)
prob = OptimizationProblem(f, x0, _p, lb = [-1.0, -1.0], ub = [0.8, 0.8])
sol = solve(prob, PyCMAOpt())
```

## Passing solver-specific options

Any keyword that `Optimization.jl` does not interpret is forwarded directly to PyCMA. 

In the event an `Optimization.jl` keyword overlaps with a `PyCMA` keyword, the `Optimization.jl` keyword takes precedence. 

An exhaustive list of keyword arguments can be found by running the following python script:

```python
import cma
options = cma.CMAOptions()
print(options)
```

An example passing the `PyCMA` keywords "verbose" and "seed":
```julia
sol = solve(prob, PyCMA(), verbose = -9, seed = 42)
```

## Troubleshooting

The original Python result object is attached to the solution in the `original` field:

```julia
sol = solve(prob, PyCMAOpt())
println(sol.original) 
```

## Contributing

Bug reports and feature requests are welcome in the [Optimization.jl](https://github.com/SciML/Optimization.jl) issue tracker.  Pull requests that improve either the Julia wrapper or the documentation are highly appreciated. 

