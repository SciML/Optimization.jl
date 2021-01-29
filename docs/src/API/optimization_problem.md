# Defining OptimizationProblems

All optimizations start by defining an `OptimizationProblem` as follows:

```julia
OptimizationProblem(f, x, p = DiffEqBase.NullParameters(),;
                    lb = nothing,
                    ub = nothing,
                    lcons = nothing,
                    ucons = nothing,
                    kwargs...)
```

Formally, the `OptimizationProblem` finds the minimum of `f(x,p)` with an
initial condition `x`. The parameters `p` are optional. `lb` and `ub`
are arrays matching the size of `x`, which stand for the lower and upper
bounds of `x`, respectively.

`f` is an `OptimizationFunction`, as defined [here](@ref optfunction).
If `f` is a standard Julia function, it is automatically converted into an
`OptimizationFunction` with `NoAD()`, i.e., no automatic generation
of the derivative functions.

Any extra keyword arguments are captured to be sent to the optimizers.
