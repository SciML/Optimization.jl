# Frequently Asked Questions

## The Solver Seems to Violate Constraints During the Optimization, Causing `DomainError`s, What Can I Do About That?

During the optimization, optimizers use slack variables to relax the solution to the constraints. Because of this,
there is no guarentee that for an arbitrary optimizer the steps will all satisfy the constraints during the 
optimization. In many cases, this can cause one's objective function code throw a `DomainError` if it is evaluated
outside of its acceptable zone. For example, `log(-1)` gives:

```
julia> log(-1)
ERROR: DomainError with -1.0:
log will only return a complex result if called with a complex argument. Try log(Complex(x)).
```

To handle this, one should not assume that the variables will always satisfy the constraints on each step. There
are three general ways to handle this better:

1. Use NaNMath.jl
2. Process variables before domain-restricted calls
3. Use a domain transformation

NaNMath.jl gives alternative implementations of standard math functions like `log` and `sqrt` in forms that do not
throw `DomainError`s but rather return `NaN`s. The optimizers will be able to handle the NaNs gracefully and recover,
allowing for many of these cases to be solved without further modification. Note that this is done [internally in
JuMP.jl, and thus if a case is working with JuMP and not Optimization.jl
](https://discourse.julialang.org/t/optimizationmoi-ipopt-violating-inequality-constraint/92608/) this may be the 
reason for the difference.

Alternatively, one can pre-process the values directly. For example, `log(abs(x))` is guaranteed to work. If one does
this, there are two things to make note of. One is that the solution will not be transformed, and thus the transformation
should be applied on `sol.u` as well. I.e., the solution could fine an optima for `x = -2`, and one should manually
change this to `x = 2` if the `abs` version is used within the objective function. Note that many functions for this will
introduce a disocontinuity in the derivative which can effect the optimization process.

Finally and relatedly, one can write the optimization with domain transformations in order to allow the optimization to
take place in the full real set. For example, instead of optimizing `x in [0,Inf]`, one can optimize `exp(x) in [0,Inf]`
and thus `x in [-Inf, Inf]` is allowed without any bounds. To do this, you would simply add the transformations to the
top of the objective function:

```julia
function my_objective(u)
    x = exp(u[1])
    # ... use x
end
```

When the optimization is done, `sol.u[1]` will be `exp(x)` and thus `log(sol.u[1])` will be the optimal value for `x`.
There exist packages in the Julia ecosystem which make it easier to keep track of these domain transformations and their
inverses for more general domains. See [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) and
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) for high level interfaces for this.

While this can allow an optimization with constraints to be rewritten as one without constraints, note that this can change
the numerical properties of the solve which can either improve or decrease the numerical stability in a case-by-case
basis. Thus while a solution, one should be aware that it could make the optimization more difficult in some cases.

## What are the advantages and disadvantages of using the ModelingToolkit.jl or other symbolic interfaces (JuMP)?

The purely numerical function interfaces of Optimization.jl has its pros and cons. The major pro of the direct
Optimization.jl interface is that it can take arbitrary Julia programs. If you have an optimization defined over a
program, like a Neural ODE or something that calls out to web servers, then these advanced setups rarely work within
specialized symbolic environments for optimization. Direct usage of Optimization.jl is thus the preferred route for
this kind of problem, and is the popular choice in the Julia ecosystem for these cases due to the simplicity of use.

However, symbolic interfaces are smart, and they may know more than you for how to make this optimization faster.
And symbolic interfaces are willing to do "tedious work" in order to make the optimization more efficient. For
example, the ModelingToolkit integration with Optimization.jl will do many simplifications when `structural_simplify`
is called. One of them is tearing on the constraints. To understand the tearing process, assume that we had
nonlinear constraints of the form:

```julia
    0 ~ u1 - sin(u5) * h,
    0 ~ u2 - cos(u1),
    0 ~ u3 - hypot(u1, u2),
    0 ~ u4 - hypot(u2, u3),
```

If these were the constraints, one can write `u1 = sin(u5) * h` and substitute `u1` for this value in the objective
function. If this is done, then `u1` does not need to be solved for, the optimization has one less state variable and
one less constraint. One can continue this process all the way to a bunch of functions:

```julia
u1 = f1(u5)
u2 = f2(u1)
u3 = f3(u1, u2)
u4 = f4(u2, u3)
```

and thus if the objective function was the function of these 5 variables and 4 constraints, ModelingToolkit.jl will 
transform it into system of 1 variable with no constraints, allowing unconstrained optimization on a smaller system.
This will both be faster and numerically easier.

[JuMP.jl](https://jump.dev/JuMP.jl/stable/) is another symbolic interface. While it does not include these tearing
and symbolic simplification passes, it does include the ability to specialize the solution process. For example,
it can treat linear optimization problems, quadratic optimization problem, convex optimization problems, etc.
in specific ways that are more efficient than a general nonlinear interface. For more information on the types of
special solves that are allowed with JuMP, see [this page](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers).